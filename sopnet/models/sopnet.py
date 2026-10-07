"""SOPNet: a compact 1-D multi-scale encoder-decoder predicting a signed onset field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sopnet.models.blocks import (
    DilatedResidualBlock,
    DownBlock,
    MultiScaleStem,
    ResidualBlock,
    UpBlock,
    fuse_and_project,
)


@dataclass
class SOPNetConfig:
    stem_channels: int = 48
    encoder_channels: Tuple[int, ...] = (64, 96, 128)
    bottleneck_channels: int = 192
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
    window_length: int = 400
    dropout: float = 0.0


class SOPNetEncoder(nn.Module):
    """Shared convolutional encoder used by SOPNet and the SOPNet-Cls baseline."""

    def __init__(self, config: SOPNetConfig):
        super().__init__()
        self.config = config
        self.stem = MultiScaleStem(out_channels=config.stem_channels)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_residuals = nn.ModuleList()

        in_channels = config.stem_channels
        for out_channels in config.encoder_channels:
            self.encoder_blocks.append(DownBlock(in_channels, out_channels))
            self.encoder_residuals.append(ResidualBlock(out_channels))
            in_channels = out_channels

        self.bottleneck_proj = nn.Sequential(
            nn.Conv1d(in_channels, config.bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(config.bottleneck_channels),
            nn.GELU(),
        )
        self.dilated = nn.ModuleList(
            [DilatedResidualBlock(config.bottleneck_channels, d) for d in config.dilations]
        )
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        stem = self.stem(x)
        skips.append(stem)
        features = stem
        for down, residual in zip(self.encoder_blocks, self.encoder_residuals):
            features = residual(down(features))
            skips.append(features)
        bottleneck = self.bottleneck_proj(features)
        for block in self.dilated:
            bottleneck = block(bottleneck)
        return self.dropout(bottleneck), skips


class SOPNet(nn.Module):
    """Predicts ``f(x) in [-1, 1]^window``; P onset and polarity are read from the field."""

    def __init__(self, config: SOPNetConfig | None = None):
        super().__init__()
        self.config = config or SOPNetConfig()
        channels = (self.config.stem_channels,) + tuple(self.config.encoder_channels)
        self.encoder = SOPNetEncoder(self.config)

        self.decoder_blocks = nn.ModuleList()
        decoder_channels = list(
            reversed((self.config.stem_channels,) + tuple(self.config.encoder_channels)[:-1])
        )
        previous_out = self.config.bottleneck_channels
        for index, out_channels in enumerate(decoder_channels):
            skip_channels = channels[-2 - index]
            self.decoder_blocks.append(UpBlock(previous_out, out_channels))
            self.decoder_blocks.append(fuse_and_project(out_channels + skip_channels, out_channels))
            previous_out = out_channels

        self.head = nn.Conv1d(decoder_channels[-1], 1, kernel_size=1)

    def _decode(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        features = bottleneck
        for step, index in enumerate(range(0, len(self.decoder_blocks), 2)):
            up = self.decoder_blocks[index]
            fuse = self.decoder_blocks[index + 1]
            features = up(features)
            skip = skips[-2 - step]
            features = fuse(torch.cat([features, skip], dim=1))
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encoder(x)
        return torch.tanh(self.head(self._decode(bottleneck, skips)))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return ``p_position``, ``polarity`` (+1/-1/0) and ``confidence``."""
        field = self.forward(x)
        magnitude = field.abs()
        confidence, position = magnitude.max(dim=-1)
        signed = field.gather(-1, position.unsqueeze(1)).squeeze(1).squeeze(-1)
        position = position.squeeze(-1)
        confidence = confidence.squeeze(-1)
        polarity = torch.sign(signed)
        polarity[confidence == 0] = 0
        return {"field": field, "p_position": position, "polarity": polarity, "confidence": confidence}


class SOPNetMulti(SOPNet):
    """SOPNet plus a polarity classification head on the shared encoder.

    The signed field still predicts the P onset (position and shape), while the
    pooled encoder features feed a known-only binary head whose logits are the
    polarity output. This decouples polarity accuracy from the regression
    objective without giving up the field's localisation and inversion
    consistency.
    """

    def __init__(self, config: SOPNetConfig | None = None, num_polarity_classes: int = 2):
        super().__init__(config)
        self.num_polarity_classes = int(num_polarity_classes)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.polarity_head = nn.Linear(self.config.bottleneck_channels, self.num_polarity_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck, skips = self.encoder(x)
        field = torch.tanh(self.head(self._decode(bottleneck, skips)))
        logits = self.polarity_head(self.pool(bottleneck).squeeze(-1))
        return field, logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        field, logits = self.forward(x)
        position = field.abs().argmax(dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        class_index = probabilities.argmax(dim=-1)
        polarity = torch.where(
            class_index == 1,
            torch.ones_like(class_index),
            torch.full_like(class_index, -1),
        )
        return {
            "field": field,
            "p_position": position.squeeze(1),
            "polarity": polarity,
            "confidence": probabilities.max(dim=-1).values,
            "logits": logits,
        }
