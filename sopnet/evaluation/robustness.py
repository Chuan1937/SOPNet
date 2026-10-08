"""Robustness protocols: P-window perturbation and additive noise.

All experiments run on the *unified* test split; performance is never broken
down by data source. Every model implements a common callable interface::

    model(x) -> (ud_score, confidence)

where ``ud_score`` is positive for UP. The same fixed test subset
(``subset_size`` samples drawn with ``subset_seed``) is used for every model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from sopnet.data.augment import AugmentConfig
from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.data.dataset import DEFAULT_SIGMA, UnifiedPolarityDataset, make_dataloader
from sopnet.training.metrics import binary_metrics, macro_f1

DEFAULT_P_SHIFTS_SECONDS = (0.0, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2)
DEFAULT_SNR_LEVELS_DB = (None, 20.0, 10.0, 5.0, 0.0, -5.0)

UDPrediction = Tuple[torch.Tensor, torch.Tensor]


def _evaluate_single(
    model: Callable[[torch.Tensor], UDPrediction],
    cache_dir: Path,
    device,
    *,
    p_shift_samples: int = 0,
    augment: Optional[AugmentConfig] = None,
    sigma: float = DEFAULT_SIGMA,
    window_length: int = 600,
    subset_size: Optional[int] = None,
    subset_seed: int = 20261004,
    batch_size: int = 1024,
    num_workers: int = 4,
):
    dataset = UnifiedPolarityDataset(
        cache_dir,
        split="test",
        jitter=None,
        p_shift_samples=p_shift_samples,
        sigma=sigma,
        augment=augment,
        max_samples=subset_size,
        seed=subset_seed,
        window_length=window_length,
    )
    loader = make_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    labels, scores = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            ud_score, _ = model(x)
            labels.append(batch["label"].numpy())
            scores.append(ud_score.cpu().numpy())
    dataset.close()

    y_true = np.concatenate(labels)
    ud_score = np.concatenate(scores)
    known = y_true != UNKNOWN
    predictions = np.where(ud_score >= 0, UP, DOWN)
    metrics = binary_metrics(y_true[known], predictions[known]) if known.any() else {}
    metrics["macro_f1_ud"] = (
        macro_f1(y_true[known], predictions[known], labels=(DOWN, UP)) if known.any() else 0.0
    )
    metrics["n"] = int(y_true.size)
    metrics["n_known"] = int(known.sum())
    return metrics


def evaluate_p_shift(
    model: Callable[[torch.Tensor], UDPrediction],
    cache_dir: Path,
    device="cuda",
    shifts_seconds: Sequence[float] = DEFAULT_P_SHIFTS_SECONDS,
    sigma: float = DEFAULT_SIGMA,
    batch_size: int = 1024,
    num_workers: int = 4,
    fs: int = 100,
    window_length: int = 600,
    subset_size: Optional[int] = None,
    subset_seed: int = 20261004,
) -> pd.DataFrame:
    """Shift the reference window relative to the labelled P and re-evaluate."""
    rows = []
    for shift in shifts_seconds:
        metrics = _evaluate_single(
            model,
            Path(cache_dir),
            device,
            p_shift_samples=int(round(shift * fs)),
            sigma=sigma,
            window_length=window_length,
            subset_size=subset_size,
            subset_seed=subset_seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        metrics["shift_seconds"] = float(shift)
        rows.append(metrics)
    return pd.DataFrame(rows)


def evaluate_noise(
    model: Callable[[torch.Tensor], UDPrediction],
    cache_dir: Path,
    device="cuda",
    snr_levels_db: Sequence[Optional[float]] = DEFAULT_SNR_LEVELS_DB,
    sigma: float = DEFAULT_SIGMA,
    batch_size: int = 1024,
    num_workers: int = 4,
    fs: int = 100,
    window_length: int = 600,
    subset_size: Optional[int] = None,
    subset_seed: int = 20261004,
) -> pd.DataFrame:
    """Add white Gaussian noise at fixed SNR levels (``None`` = clean test set)."""
    rows = []
    for level in snr_levels_db:
        augment = None
        if level is not None:
            augment = AugmentConfig(
                p_noise=1.0,
                snr_db_range=(float(level), float(level)),
                p_invert=0.0,
                p_scale=0.0,
            )
        metrics = _evaluate_single(
            model,
            Path(cache_dir),
            device,
            augment=augment,
            sigma=sigma,
            window_length=window_length,
            subset_size=subset_size,
            subset_seed=subset_seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        metrics["snr_db"] = level if level is not None else float("inf")
        rows.append(metrics)
    return pd.DataFrame(rows)
