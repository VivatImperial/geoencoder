"""Модель: USER2-small (encoder) + регрессионная голова на 2 выхода (lat_norm, lon_norm)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class GeocodingModel(nn.Module):
    """Encoder (USER2-small) + mean pooling по токенам + линейный слой -> (lat_norm, lon_norm)."""

    def __init__(
        self,
        encoder_name: str = "deepvk/USER2-small",
        hidden_size: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        logits = self.regressor(pooled)
        return torch.sigmoid(logits)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder_layer(self, layer_index: int) -> list[torch.Tensor]:
        """Размораживает один слой энкодера по индексу (11 = верхний, 0 = нижний). Возвращает список параметров слоя."""
        params = []
        prefix = f"layers.{layer_index}."
        for name, p in self.encoder.named_parameters():
            if name.startswith(prefix) or f".layers.{layer_index}." in name:
                p.requires_grad = True
                params.append(p)
        return params

    def num_encoder_layers(self) -> int:
        if hasattr(self.encoder, "config") and getattr(self.encoder.config, "num_hidden_layers", None) is not None:
            return self.encoder.config.num_hidden_layers
        n = 0
        for name in self.encoder.state_dict():
            for sep in (".layer.", ".layers."):
                if sep in name:
                    part = name.split(sep, 1)[-1]
                    idx = part.split(".")[0]
                    if idx.isdigit():
                        n = max(n, int(idx) + 1)
                    break
        return n


def get_tokenizer(model_name: str = "deepvk/USER2-small") -> Any:
    return AutoTokenizer.from_pretrained(model_name)
