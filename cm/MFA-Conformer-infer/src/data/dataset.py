import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from typing import Tuple


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def collate_fn(batch):
    """
    Collate a batch of audio samples.

    Handles:
        - (waveform, length, label) tuples
        - (waveform, key) tuples
    """
    if not batch:
        return torch.tensor([])

    elem = batch[0]

    if len(elem) == 3:
        waveforms, lengths, labels = zip(*batch)
        return torch.stack(waveforms), torch.tensor(lengths), torch.tensor(labels)

    elif len(elem) == 2:
        waveforms, keys = zip(*batch)
        return torch.stack(waveforms), keys

    else:
        return torch.utils.data.dataloader.default_collate(batch)
