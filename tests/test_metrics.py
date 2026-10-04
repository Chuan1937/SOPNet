from __future__ import annotations

import numpy as np

from sopnet.data.canonical import DOWN, UP
from sopnet.training.metrics import binary_metrics, macro_f1, p_error_metrics


def test_binary_metrics_perfect_prediction():
    labels = np.array([UP, DOWN, UP, DOWN])
    metrics = binary_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0


def test_binary_metrics_inverted_prediction():
    labels = np.array([UP, DOWN, UP, DOWN])
    metrics = binary_metrics(labels, -labels)
    assert metrics["accuracy"] == 0.0
    assert metrics["mcc"] == -1.0


def test_mcc_does_not_overflow_on_large_support():
    labels = np.array([UP, DOWN] * 1_000_000)
    predictions = np.array([UP, DOWN] * 1_000_000)
    metrics = binary_metrics(labels, predictions)
    assert metrics["mcc"] == 1.0


def test_macro_f1_ignores_absent_labels():
    labels = np.array([UP, UP, UP])
    predictions = np.array([UP, UP, DOWN])
    assert macro_f1(labels, predictions, labels=(-1, 1)) > 0.0


def test_p_error_metrics_units():
    predicted = np.array([100, 200])
    true = np.array([100, 210])
    metrics = p_error_metrics(predicted, true, fs=100)
    assert metrics["mae_samples"] == 5.0
    assert metrics["mae_ms"] == 50.0
    assert metrics["p95_ae_ms"] >= metrics["median_ae_ms"]
