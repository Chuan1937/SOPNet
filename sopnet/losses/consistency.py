"""Physical inversion consistency: ``f(-x) = -f(x)``."""

from __future__ import annotations

import torch
import torch.nn as nn


class InversionConsistencyLoss(nn.Module):
    """Penalises ``|f(-x) + f(x)|`` so polarity inversion is exactly equivariant."""

    def forward(self, prediction: torch.Tensor, prediction_inverted: torch.Tensor) -> torch.Tensor:
        return (prediction + prediction_inverted).abs().mean()
