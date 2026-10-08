"""Lightweight 1-D building blocks for SOPNet."""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )


class MultiScaleStem(nn.Module):
    """Parallel convolutions with kernels 5/11/21 capture different P slopes."""

    def __init__(self, out_channels: int = 48, branch_channels: int = 16):
        super().__init__()
        kernels = (5, 11, 21)
        self.branches = nn.ModuleList(
            [nn.Conv1d(1, branch_channels, kernel_size=k, padding=k // 2, bias=False) for k in kernels]
        )
        self.project = conv_norm_act(branch_channels * len(kernels), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(features)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 1):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            conv_norm_act(channels, hidden, kernel_size=3, padding=1),
            nn.Conv1d(hidden, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_norm_act(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DownBlock(nn.Module):
    """Strided convolution to halve the temporal resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = conv_norm_act(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpBlock(nn.Module):
    """Nearest-neighbour upsampling followed by a 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = conv_norm_act(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


def fuse_and_project(in_channels: int, out_channels: int) -> nn.Sequential:
    """1x1 fusion after a skip-connection concatenation."""
    return nn.Sequential(
        conv_norm_act(in_channels, out_channels, kernel_size=1),
        ResidualBlock(out_channels),
    )
