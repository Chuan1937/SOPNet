from __future__ import annotations

import torch

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.training.baselines import (
    BASELINE_SPECS,
    _predict,
    _ud_score,
    prepare_baseline_input,
)


def test_binary_du_follows_official_rpnet_convention():
    # Official RPNet: index 1 = UP, index 0 = DOWN.
    predictions, _ = _predict(BASELINE_SPECS["rpnet"], torch.tensor([[10.0, 0.0]]))
    assert predictions.item() == DOWN
    predictions, _ = _predict(BASELINE_SPECS["rpnet"], torch.tensor([[0.0, 10.0]]))
    assert predictions.item() == UP


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


def test_diting_motion_second_channel_is_sign_of_forward_difference():
    waveform = torch.tensor([[[0.0, 1.0, 3.0, 2.0, 0.0]]])
    prepared = prepare_baseline_input(waveform, BASELINE_SPECS["diting_motion"])
    # forward difference, zero padded at the start, then sign
    expected = torch.sign(torch.tensor([0.0, 1.0, 2.0, -1.0, -2.0]))
    assert torch.equal(prepared[0, 0], waveform[0, 0])
    assert torch.equal(prepared[0, 1], expected)


def test_ud_score_conventions():
    assert float(_ud_score(BASELINE_SPECS["ross"], torch.tensor([[2.0, 1.0, 0.0]]))) == 1.0
    assert float(_ud_score(BASELINE_SPECS["rpnet"], torch.tensor([[1.0, 2.0]]))) == 1.0
    assert float(_ud_score(BASELINE_SPECS["cfm"], torch.tensor([[0.5]]))) == 0.5
