"""Robustness protocols: P-window perturbation and additive noise.

All experiments run on the *unified* test split; performance is never broken
down by data source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch

from sopnet.data.augment import AugmentConfig
from sopnet.data.cache import load_cache
from sopnet.data.dataset import DEFAULT_SIGMA, UnifiedPolarityDataset, make_dataloader
from sopnet.evaluation.evaluate import evaluate_field

DEFAULT_P_SHIFTS_SECONDS = (0.0, -0.1, 0.1, -0.2, 0.2, -0.5, 0.5, -1.0, 1.0)
DEFAULT_SNR_LEVELS_DB = (None, 20.0, 10.0, 5.0, 0.0, -5.0)


def evaluate_p_shift(
    model: torch.nn.Module,
    cache_dir: Path,
    device="cuda",
    shifts_seconds: Sequence[float] = DEFAULT_P_SHIFTS_SECONDS,
    sigma: float = DEFAULT_SIGMA,
    batch_size: int = 1024,
    num_workers: int = 4,
    fs: int = 100,
) -> pd.DataFrame:
    """Shift the reference window relative to the labelled P and re-evaluate."""
    manifest, index = load_cache(Path(cache_dir))
    rows = []
    for shift in shifts_seconds:
        dataset = UnifiedPolarityDataset(
            cache_dir,
            split="test",
            jitter=None,
            p_shift_samples=int(round(shift * fs)),
            sigma=sigma,
            manifest=manifest,
            index=index,
        )
        loader = make_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        metrics = evaluate_field(model, loader, device, fs=fs)
        metrics["shift_seconds"] = float(shift)
        rows.append(metrics)
        dataset.close()
    return pd.DataFrame(rows)


def evaluate_noise(
    model: torch.nn.Module,
    cache_dir: Path,
    device="cuda",
    snr_levels_db: Sequence[Optional[float]] = DEFAULT_SNR_LEVELS_DB,
    sigma: float = DEFAULT_SIGMA,
    batch_size: int = 1024,
    num_workers: int = 4,
    fs: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Add white Gaussian noise at fixed SNR levels (``None`` = clean test set)."""
    manifest, index = load_cache(Path(cache_dir))
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
        dataset = UnifiedPolarityDataset(
            cache_dir,
            split="test",
            jitter=None,
            sigma=sigma,
            augment=augment,
            seed=seed,
            manifest=manifest,
            index=index,
        )
        loader = make_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        metrics = evaluate_field(model, loader, device, fs=fs)
        metrics["snr_db"] = level if level is not None else float("inf")
        rows.append(metrics)
        dataset.close()
    return pd.DataFrame(rows)
