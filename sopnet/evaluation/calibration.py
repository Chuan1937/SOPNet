"""Confidence calibration and selective-prediction utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sopnet.training.metrics import coverage_accuracy


def reliability_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """Bin mean confidence against empirical accuracy."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = (confidence >= edges[i]) & (
            confidence < edges[i + 1] if i < n_bins - 1 else confidence <= edges[i + 1]
        )
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            accuracy[i] = float(correct[mask].mean())
    return {"bin_centre": centres, "accuracy": accuracy, "counts": counts}


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    curve = reliability_curve(confidence, correct, n_bins=n_bins)
    total = curve["counts"].sum()
    if total == 0:
        return float("nan")
    weights = curve["counts"] / total
    valid = ~np.isnan(curve["accuracy"])
    return float(np.sum(weights[valid] * np.abs(curve["accuracy"][valid] - curve["bin_centre"][valid])))


def coverage_curve(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    return coverage_accuracy(confidence, correct, thresholds)
