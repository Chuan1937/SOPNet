"""Paired bootstrap confidence intervals for metric differences."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from sopnet.training.metrics import macro_f1


def paired_bootstrap_difference(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261004,
) -> Dict[str, float]:
    """Bootstrap the paired difference ``metric(A) - metric(B)`` on the same samples."""
    metric = metric or macro_f1
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must have equal length")

    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = metric(y_true, pred_a) - metric(y_true, pred_b)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        deltas[i] = metric(y_true[indices], pred_a[indices]) - metric(y_true[indices], pred_b[indices])

    alpha = (1.0 - confidence) / 2.0
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "p_improvement": float(np.mean(deltas > 0)),
    }
