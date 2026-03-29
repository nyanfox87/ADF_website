# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


def get_mask_from_lengths(lengths, max_len=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def get_mask_from_sep(input, max_len=None, sep_value=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = input.shape[0]
    src_length = input.shape[1]
    sep_indices = (input == sep_value).nonzero()[:, 1:2]  # one-element slicing
    input_ids = (
        torch.arange(-(max_len - src_length), src_length)
        .unsqueeze(0)
        .expand(batch_size, -1)
    ).to(device)
    return input_ids >= sep_indices


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    class_weights: Optional[List[float]] = field(
        default=None, metadata={"help": "weights for each class"}
    )
    focal_loss_gamma: float = field(
        default=0.0, metadata={"help": "gamma parameter for focal loss (0 disables focal loss)"}
    )


@register_criterion("cross_entropy_prompt", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, class_weights=None, focal_loss_gamma=0.0):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.class_weights = class_weights
        self.focal_loss_gamma = focal_loss_gamma
        # Handle case where class_weights might be passed as a string representation of a list
        if isinstance(self.class_weights, str):
            try:
                import ast
                self.class_weights = ast.literal_eval(self.class_weights)
            except:
                pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # [B, L, V]
        targets = model.get_targets(sample, net_output)  # [B, L]

        # ==================#
        # !!! Prompting !!! #
        # Only Calculate cross entropy on labels
        prompt_length = net_output[1]["prompt_length"]
        prob_masks = get_mask_from_sep(targets, lprobs.shape[1])
        label_masks = get_mask_from_sep(targets, lprobs.shape[1] - prompt_length)
        
        label_probs = lprobs[prob_masks]
        label = targets[label_masks]
        # ==================#

        weights = None
        if self.class_weights is not None:
            # Assuming classes start at index 4 (after <s>, </s>, <pad>, <unk>)
            # and class_weights corresponds to these classes.
            # We create a weight tensor for the full vocabulary (or output dim)
            vocab_size = lprobs.size(-1)
            weights = torch.ones(vocab_size, device=lprobs.device)
            for i, w in enumerate(self.class_weights):
                if 4 + i < vocab_size:
                    weights[4 + i] = w

        if self.focal_loss_gamma > 0:
            # Focal loss: - (1 - p)^gamma * log(p)
            # For NLL, we need to modify the probs
            # But since we're using nll_loss, we can compute focal weights per sample
            # Get the probability of the correct class
            correct_probs = label_probs[torch.arange(label.size(0)), label]
            focal_weights = (1 - correct_probs.exp()).pow(self.focal_loss_gamma)
            # Apply to the loss
            loss = F.nll_loss(
                label_probs,
                label,
                ignore_index=self.padding_idx,
                reduction="none",
            )
            loss = (focal_weights * loss).sum()
        else:
            loss = F.nll_loss(
                label_probs,
                label,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
                weight=weights,
            )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
