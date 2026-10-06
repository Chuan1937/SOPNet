from __future__ import annotations

import numpy as np
import torch

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.evaluation.calibration import (
    apply_platt_scaling,
    expected_calibration_error,
    fit_platt_scaling,
)
from sopnet.evaluation.evaluate import evaluate_field


def test_platt_scaling_recovers_synthetic_probability():
    rng = np.random.default_rng(0)
    scores = rng.normal(2.0, 1.0, size=20_000)
    probability = 1.0 / (1.0 + np.exp(-(1.5 * scores - 2.0)))
    correct = rng.random(scores.size) < probability

    parameters = fit_platt_scaling(scores, correct)
    assert abs(parameters["a"] - 1.5) < 0.3
    assert abs(parameters["b"] + 2.0) < 0.3

    calibrated = apply_platt_scaling(scores, parameters)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    assert np.all(np.diff(calibrated[np.argsort(scores)]) >= 0.0)
    assert expected_calibration_error(calibrated, correct) < 0.03


class _StubModel(torch.nn.Module):
    def __init__(self, fields):
        super().__init__()
        self._fields = fields

    def forward(self, x):
        return self._fields.pop(0)


def _field(peak: float) -> torch.Tensor:
    field = torch.zeros(1, 1, 400)
    field[0, 0, 300] = peak
    return field


def _batch(label: int) -> dict:
    return {
        "x": torch.zeros(1, 1, 400),
        "label": torch.tensor([label]),
        "p_pick": torch.tensor([300]),
        "target": torch.zeros(1, 400),
    }


def test_evaluate_field_known_only_selective_metrics():
    batches = [_batch(UP), _batch(DOWN), _batch(UNKNOWN)]
    model = _StubModel([_field(2.0), _field(-0.1), _field(2.0)])

    metrics = evaluate_field(model, batches, device="cpu", threshold=1.0)

    # covered = {sample 1 (correct), sample 3 (unknown truth)}; the known sample
    # 2 is below threshold, so mixing X into "covered accuracy" halves it.
    assert metrics["coverage"] == 2 / 3
    assert metrics["covered_accuracy"] == 0.5
    assert metrics["coverage_known"] == 0.5
    assert metrics["covered_accuracy_known"] == 1.0
    assert metrics["known_accuracy"] == 1.0
