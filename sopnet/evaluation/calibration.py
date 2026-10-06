"""Confidence calibration and selective-prediction utilities."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

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


def fit_platt_scaling(
    scores: np.ndarray,
    correct: np.ndarray,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Fit ``P(correct) = sigmoid(a * score + b)`` by minimising log loss.

    Field models emit an unnormalised peak magnitude, so it must be mapped to a
    probability on validation before ECE/reliability are meaningful.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    if scores.size == 0 or scores.size != correct.size:
        raise ValueError("scores and correct must be non-empty and of equal length")
    if np.unique(correct).size < 2:
        raise ValueError("Platt scaling needs both correct and incorrect samples")

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        logits = parameters[0] * scores + parameters[1]
        return float(np.mean(np.logaddexp(0.0, logits) - correct * logits))

    scale = float(np.std(scores)) or 1.0
    initial = np.array([1.0 / scale, -float(np.mean(scores)) / scale])
    result = minimize(
        negative_log_likelihood,
        initial,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    if not result.success:
        raise RuntimeError(f"Platt scaling did not converge: {result.message}")
    return {"a": float(result.x[0]), "b": float(result.x[1])}


def apply_platt_scaling(scores: np.ndarray, parameters: Dict[str, float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    logits = parameters["a"] * scores + parameters["b"]
    return expit(logits)
