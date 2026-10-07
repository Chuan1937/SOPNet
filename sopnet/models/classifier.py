"""SOPNet-Cls: the ablation baseline sharing SOPNet's encoder but classifying U/D/X directly."""

from __future__ import annotations

import torch
import torch.nn as nn

from sopnet.models.sopnet import SOPNetConfig, SOPNetEncoder


class SOPNetCls(nn.Module):
    """Same encoder as :class:`~sopnet.models.sopnet.SOPNet`, global average pool, linear head.

    Comparing this model against SOPNet isolates the contribution of the
    signed-onset formulation from the capacity of the backbone.
    """

    def __init__(self, config: SOPNetConfig | None = None, num_classes: int = 3):
        super().__init__()
        self.config = config or SOPNetConfig()
        self.encoder = SOPNetEncoder(self.config)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.config.bottleneck_channels, num_classes)

    def forward(self, x: torch.Tensor, x_pol: torch.Tensor | None = None) -> torch.Tensor:
        bottleneck, _ = self.encoder(x)
        pooled = self.pool(bottleneck).squeeze(-1)
        return self.classifier(pooled)
