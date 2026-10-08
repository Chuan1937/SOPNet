"""Optimizer and learning-rate schedule (AdamW + warmup + cosine decay)."""

from __future__ import annotations

import math
from typing import Optional

import torch


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    betas=(0.9, 0.999),
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def current_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    if not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0]["lr"])
