"""Signed-onset-field losses.

The field target is zero almost everywhere, so a plain MSE would be dominated
by empty samples. ``WeightedSignedFieldLoss`` up-weights the neighbourhood of
the P onset; ``PolarityConsistencyLoss`` converts the field itself into a
differentiable polarity score without adding a classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSignedFieldLoss(nn.Module):
    """``mean((1 + beta * |target|) * SmoothL1(prediction, target))``.

    ``sample_weight`` (shape ``(batch,)``) rescales each sample's contribution,
    which lets training down-weight unlabelled (unknown-polarity) samples whose
    all-zero target would otherwise dominate the gradient.
    """

    def __init__(self, beta: float = 8.0, smooth_l1_beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.dim() == 3 and prediction.shape[1] == 1:
            prediction = prediction.squeeze(1)
        if target.dim() == 3 and target.shape[1] == 1:
            target = target.squeeze(1)
        elementwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.smooth_l1_beta)
        weights = 1.0 + self.beta * target.abs()
        per_sample = (weights * elementwise).flatten(1).mean(dim=1)
        if sample_weight is None:
            return per_sample.mean()
        normalizer = sample_weight.sum().clamp_min(1e-8)
        return (per_sample * sample_weight).sum() / normalizer


class PolarityConsistencyLoss(nn.Module):
    """Hinge-style loss on the signed field's attention-weighted polarity score."""

    def __init__(self, kappa: float = 4.0, gamma: float = 1.0):
        super().__init__()
        self.kappa = float(kappa)
        self.gamma = float(gamma)

    def forward(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        field = prediction.squeeze(1)
        known = label != 0
        if not bool(known.any()):
            return prediction.sum() * 0.0

        weights = torch.softmax(self.kappa * field.abs(), dim=-1)
        score = (weights * field).sum(dim=-1)
        sign = torch.sign(label.float())
        return F.softplus(-self.gamma * sign[known] * score[known]).mean()
