"""Paired bootstrap confidence intervals for metric differences."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from sopnet.training.metrics import macro_f1


def _weighted_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    labels: tuple = (-1, 1),
) -> float:
    scores = []
    for label in labels:
        true = y_true == label
        pred = y_pred == label
        tp = float(weights[true & pred].sum())
        fp = float(weights[~true & pred].sum())
        fn = float(weights[true & ~pred].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def paired_bootstrap_difference_clustered(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    labels: tuple = (-1, 1),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20261004,
) -> Dict[str, float]:
    """Event-clustered paired bootstrap of ``macro_f1(A) - macro_f1(B)``.

    Whole events (``groups``) are resampled with replacement so correlated
    waveforms from the same event can never be split across resamples. The
    weighted-macro-F1 formulation avoids materialising index arrays.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    groups = np.asarray(groups)
    if not (len(y_true) == len(pred_a) == len(pred_b) == len(groups)):
        raise ValueError("y_true, pred_a, pred_b and groups must have equal length")

    _, group_ids = np.unique(groups, return_inverse=True)
    n_groups = int(group_ids.max()) + 1
    unit = np.ones(len(y_true), dtype=np.float64)
    observed = _weighted_macro_f1(y_true, pred_a, unit, labels) - _weighted_macro_f1(
        y_true, pred_b, unit, labels
    )

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        chosen = rng.integers(0, n_groups, size=n_groups)
        counts = np.bincount(chosen, minlength=n_groups)
        weights = counts[group_ids].astype(np.float64)
        deltas[i] = _weighted_macro_f1(y_true, pred_a, weights, labels) - _weighted_macro_f1(
            y_true, pred_b, weights, labels
        )

    alpha = (1.0 - confidence) / 2.0
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
        "p_improvement": float(np.mean(deltas > 0)),
        "n_groups": int(n_groups),
    }


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
