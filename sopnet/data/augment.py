"""Waveform augmentations. Every transform keeps the label in sync."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentConfig:
    p_noise: float = 0.35
    snr_db_range: Tuple[float, float] = (0.0, 30.0)
    p_invert: float = 0.5
    p_scale: float = 0.2
    scale_range: Tuple[float, float] = (0.5, 2.0)


def add_noise_snr(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise at the requested signal-to-noise ratio."""
    arr = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr**2)))
    if rms <= 0.0:
        return arr.astype(np.float32)
    noise_std = rms / (10.0 ** (snr_db / 20.0))
    return (arr + rng.normal(0.0, noise_std, size=arr.shape)).astype(np.float32)


def random_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> np.ndarray:
    if rng.random() >= config.p_noise:
        return x
    snr_db = rng.uniform(*config.snr_db_range)
    return add_noise_snr(x, snr_db, rng)


def invert(x: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
    """Polarity inversion rewrites U<->D; unknown stays unknown."""
    return -np.asarray(x, dtype=np.float32), -int(label)


def scale_amplitude(
    x: np.ndarray,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> np.ndarray:
    if rng.random() >= config.p_scale:
        return x
    factor = rng.uniform(*config.scale_range)
    return (np.asarray(x, dtype=np.float32) * factor).astype(np.float32)


def augment_trace(
    x: np.ndarray,
    label: int,
    rng: np.random.Generator,
    config: AugmentConfig,
) -> Tuple[np.ndarray, int]:
    """Apply noise, random inversion and amplitude scaling to a training trace."""
    wave = random_noise(x, rng, config)
    if rng.random() < config.p_invert:
        wave, label = invert(wave, label)
    wave = scale_amplitude(wave, rng, config)
    return wave.astype(np.float32), int(label)
