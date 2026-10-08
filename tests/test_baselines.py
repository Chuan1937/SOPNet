from __future__ import annotations

import torch

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.training.baselines import BASELINE_SPECS, _predict


def test_binary_du_maps_class_one_to_down():
    logits = torch.tensor([[10.0, 0.0]])
    predictions, _ = _predict(BASELINE_SPECS["rpnet"], logits)
    assert predictions.item() == UP

    logits = torch.tensor([[0.0, 10.0]])
    predictions, _ = _predict(BASELINE_SPECS["rpnet"], logits)
    assert predictions.item() == DOWN


def test_binary_ud_sigmoid_threshold():
    predictions, _ = _predict(BASELINE_SPECS["eqpolarity"], torch.tensor([[5.0]]))
    assert predictions.item() == UP
    predictions, _ = _predict(BASELINE_SPECS["eqpolarity"], torch.tensor([[-5.0]]))
    assert predictions.item() == DOWN


def test_three_class_mapping():
    for index, expected in enumerate((UP, DOWN, UNKNOWN)):
        logits = torch.zeros(1, 3)
        logits[0, index] = 10.0
        predictions, _ = _predict(BASELINE_SPECS["ross"], logits)
        assert predictions.item() == expected
