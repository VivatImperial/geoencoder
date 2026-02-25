"""Дифференцируемый лосс для геокодинга: MSE по нормализованным координатам [0,1]."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeocodingMSELoss(nn.Module):
    """MSE по паре (lat_norm, lon_norm). pred и target в [0, 1]. Общий лосс = MSE по обоим выходам."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        lat_norm: torch.Tensor,
        lon_norm: torch.Tensor,
    ) -> torch.Tensor:
        # pred: (B, 2) — [lat_norm, lon_norm]
        if pred.shape[-1] != 2:
            raise ValueError("pred must have last dim 2")
        target = torch.stack([lat_norm, lon_norm], dim=1)
        return F.mse_loss(pred, target, reduction=self.reduction)
