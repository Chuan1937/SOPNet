"""Evaluation metrics for polarity and P-onset localisation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from sopnet.data.canonical import UP


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Binary confusion matrix with positive class = UP (+1)."""
    true_up = y_true == UP
    pred_up = y_pred == UP
    tp = int(np.sum(true_up & pred_up))
    fp = int(np.sum(~true_up & pred_up))
    fn = int(np.sum(true_up & ~pred_up))
    tn = int(np.sum(~true_up & ~pred_up))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    matrix = binary_confusion(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    denominator = float(
        np.sqrt(
            (float(tp + fp))
            * float(tp + fn)
            * float(tn + fp)
            * float(tn + fn)
        )
    )
    mcc = (float(tp) * float(tn) - float(fp) * float(fn)) / denominator if denominator else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "support": total,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels=(-1, 0, 1)) -> float:
    """Macro F1 over the requested label set (ignores labels without support)."""
    scores = []
    for label in labels:
        true = y_true == label
        pred = y_pred == label
        if true.sum() == 0:
            continue
        tp = int(np.sum(true & pred))
        fp = int(np.sum(~true & pred))
        fn = int(np.sum(true & ~pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def p_error_metrics(predicted: np.ndarray, true: np.ndarray, fs: int = 100) -> Dict[str, float]:
    errors = np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(true, dtype=np.float64))
    if errors.size == 0:
        return {
            "mae_samples": 0.0,
            "median_ae_samples": 0.0,
            "p95_ae_samples": 0.0,
            "mae_ms": 0.0,
            "median_ae_ms": 0.0,
            "p95_ae_ms": 0.0,
        }
    scale = 1000.0 / fs
    return {
        "mae_samples": float(errors.mean()),
        "median_ae_samples": float(np.median(errors)),
        "p95_ae_samples": float(np.percentile(errors, 95)),
        "mae_ms": float(errors.mean() * scale),
        "median_ae_ms": float(np.median(errors) * scale),
        "p95_ae_ms": float(np.percentile(errors, 95) * scale),
    }


def coverage_accuracy(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Accuracy as a function of a confidence threshold (selective prediction)."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    coverage = np.array([np.mean(confidence >= t) for t in thresholds])
    accuracy = np.array(
        [correct[confidence >= t].mean() if np.any(confidence >= t) else np.nan for t in thresholds]
    )
    return {"thresholds": thresholds, "coverage": coverage, "accuracy": accuracy}


def best_threshold(
    confidence: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric=macro_f1,
) -> float:
    """Pick the confidence threshold that maximises macro-F1 with an unknown class."""
    thresholds = np.linspace(0.05, 0.95, 19)
    best_score, best_t = -1.0, 0.5
    for threshold in thresholds:
        labels = np.where(confidence >= threshold, y_pred, 0)
        score = metric(y_true, labels)
        if score > best_score:
            best_score, best_t = score, float(threshold)
    return best_t
