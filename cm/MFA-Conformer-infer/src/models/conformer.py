import torch
import nemo.collections.asr as nemo_asr
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling


class LayerwiseConformerClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model_name: str = "stt_en_conformer_ctc_small",
        num_classes: int = 2,
        trainable_encoder: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=base_model_name
        )
        self.base_model.eval()
        if not trainable_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.encoder = self.base_model.encoder
        self.preprocessor = self.base_model.preprocessor
        self.num_layers = len(self.encoder.layers)
        self.hidden_dim = self.encoder.d_model

        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim * self.num_layers)
        self.pooling = AttentiveStatisticsPooling(self.hidden_dim * self.num_layers)

        feature_dim = self.hidden_dim * self.num_layers * 2
        self.feature_norm = torch.nn.BatchNorm1d(feature_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

        self._layer_outputs = [None] * self.num_layers
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_layer_hooks()

    def _register_layer_hooks(self) -> None:
        for idx, layer in enumerate(self.encoder.layers):
            hook = layer.register_forward_hook(self._build_layer_hook(idx))
            self._hooks.append(hook)

    def _build_layer_hook(self, idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            self._layer_outputs[idx] = hidden
        return hook

    def forward(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> torch.Tensor:
        self._layer_outputs = [None] * self.num_layers

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )

        _encoded, _encoded_length = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        collected = [h for h in self._layer_outputs if h is not None]
        if not collected:
            raise RuntimeError("No layer outputs were captured by hooks.")

        stacked_features = torch.cat(collected, dim=-1)

        pooled_input = stacked_features.transpose(1, 2)
        pooled = self.pooling(pooled_input).squeeze(-1)

        features = self.feature_norm(pooled)
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class LayerwiseTimeASPConcatLayerASP(torch.nn.Module):
    def __init__(
        self,
        base_model_name: str = "stt_en_conformer_ctc_small",
        num_classes: int = 2,
        trainable_encoder: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=base_model_name
        )
        self.base_model.eval()
        if not trainable_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.encoder = self.base_model.encoder
        self.preprocessor = self.base_model.preprocessor
        self.num_layers = len(self.encoder.layers)
        self.hidden_dim = self.encoder.d_model

        self.time_asp = AttentiveStatisticsPooling(self.hidden_dim)
        self.layer_asp = AttentiveStatisticsPooling(self.hidden_dim * 2)

        feature_dim = self.hidden_dim * 4
        self.feature_norm = torch.nn.BatchNorm1d(feature_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(feature_dim, num_classes)

        self._layer_outputs = [None] * self.num_layers
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_layer_hooks()

    def _register_layer_hooks(self) -> None:
        for idx, layer in enumerate(self.encoder.layers):
            hook = layer.register_forward_hook(self._build_layer_hook(idx))
            self._hooks.append(hook)

    def _build_layer_hook(self, idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            self._layer_outputs[idx] = hidden
        return hook

    def forward(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> torch.Tensor:
        self._layer_outputs = [None] * self.num_layers

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )

        _encoded, _encoded_length = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        collected = [h for h in self._layer_outputs if h is not None]
        if not collected:
            raise RuntimeError("No layer outputs were captured by hooks.")

        stacked = torch.stack(collected, dim=1)
        B, L, T, D = stacked.shape

        flat_input = stacked.view(B * L, T, D).transpose(1, 2)
        time_pooled = self.time_asp(flat_input)
        time_pooled = time_pooled.view(B, L, D * 2)

        layer_input = time_pooled.transpose(1, 2)
        layer_pooled = self.layer_asp(layer_input).squeeze(-1)

        features = self.feature_norm(layer_pooled)
        features = self.dropout(features)
        logits = self.classifier(features)

        return logits

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class MFAConformerClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model_name: str = "stt_en_conformer_ctc_small",
        num_classes: int = 2,
        embedding_dim: int = 256,
        trainable_encoder: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=base_model_name
        )
        self.base_model.eval()
        if not trainable_encoder:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.encoder = self.base_model.encoder
        self.preprocessor = self.base_model.preprocessor
        self.num_layers = len(self.encoder.layers)
        self.hidden_dim = self.encoder.d_model

        concat_dim = self.hidden_dim * self.num_layers
        self.layer_norm = torch.nn.LayerNorm(concat_dim)
        self.pooling = AttentiveStatisticsPooling(concat_dim)

        asp_output_dim = concat_dim * 2
        self.batch_norm = torch.nn.BatchNorm1d(asp_output_dim)
        self.fc_layer = torch.nn.Linear(asp_output_dim, embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)

        self._layer_outputs = [None] * self.num_layers
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_layer_hooks()

    def _register_layer_hooks(self) -> None:
        for idx, layer in enumerate(self.encoder.layers):
            hook = layer.register_forward_hook(self._build_layer_hook(idx))
            self._hooks.append(hook)

    def _build_layer_hook(self, idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            self._layer_outputs[idx] = hidden
        return hook

    def forward(
        self,
        input_signal: torch.Tensor,
        input_signal_length: torch.Tensor,
    ) -> torch.Tensor:
        self._layer_outputs = [None] * self.num_layers

        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length
        )

        _encoded, _encoded_length = self.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )

        collected = [h for h in self._layer_outputs if h is not None]
        if not collected:
            raise RuntimeError("No layer outputs were captured.")

        stacked_features = torch.cat(collected, dim=-1)
        normed_features = self.layer_norm(stacked_features)
        pooled_input = normed_features.transpose(1, 2)
        pooled = self.pooling(pooled_input).squeeze(-1)

        embeddings = self.batch_norm(pooled)
        embeddings = self.fc_layer(embeddings)
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        return logits

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
