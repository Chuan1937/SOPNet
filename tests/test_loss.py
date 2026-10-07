from __future__ import annotations

import torch

from sopnet.data.canonical import DOWN, UNKNOWN, UP, signed_field_target
from sopnet.losses import (
    InversionConsistencyLoss,
    PolarityConsistencyLoss,
    WeightedSignedFieldLoss,
)


def _targets():
    return torch.stack(
        [
            torch.from_numpy(signed_field_target(400, 200, UP, 10.0)),
            torch.from_numpy(signed_field_target(400, 220, DOWN, 10.0)),
            torch.from_numpy(signed_field_target(400, 180, UNKNOWN, 10.0)),
        ]
    ).unsqueeze(1)


def test_weighted_field_loss_zero_for_perfect_prediction():
    target = _targets()
    loss = WeightedSignedFieldLoss()(target, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_weighted_field_loss_handles_channel_dimension():
    target = _targets()
    prediction = target + 0.1 * torch.randn_like(target)
    criterion = WeightedSignedFieldLoss()
    with_channel = criterion(prediction, target.squeeze(1))
    without_channel = criterion(prediction.squeeze(1), target.squeeze(1))
    assert torch.allclose(with_channel, without_channel)
    assert float(with_channel) < 0.1


def test_weighted_field_loss_positive_for_wrong_prediction():
    target = _targets()
    loss = WeightedSignedFieldLoss()(torch.zeros_like(target), target)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_polarity_loss_prefers_correct_sign():
    target = _targets()
    labels = torch.tensor([UP, DOWN, UNKNOWN])
    criterion = PolarityConsistencyLoss()
    with torch.no_grad():
        good = criterion(target, labels)
        bad = criterion(-target, labels)
    assert float(good) < float(bad)


def test_polarity_loss_finite_with_only_unknown():
    target = _targets()
    labels = torch.tensor([UNKNOWN, UNKNOWN, UNKNOWN])
    loss = PolarityConsistencyLoss()(target, labels)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_inversion_consistency_zero_for_antisymmetric_prediction():
    prediction = torch.randn(4, 1, 400)
    loss = InversionConsistencyLoss()(prediction, -prediction)
    assert float(loss) == 0.0


def test_inversion_consistency_positive_otherwise():
    prediction = torch.randn(4, 1, 400)
    loss = InversionConsistencyLoss()(prediction, prediction)
    assert float(loss) > 0.0


def test_weighted_field_loss_sample_weight_selects_known_samples():
    torch.manual_seed(0)
    targets = _targets()
    prediction = torch.randn_like(targets)
    loss_fn = WeightedSignedFieldLoss(beta=8.0)
    weight = torch.tensor([1.0, 1.0, 0.0])

    weighted = loss_fn(prediction, targets, sample_weight=weight)
    known_only = loss_fn(prediction[:2], targets[:2])
    assert torch.allclose(weighted, known_only, atol=1e-6)
    assert not torch.allclose(weighted, loss_fn(prediction, targets))


def test_weighted_field_loss_unit_weights_match_plain_loss():
    torch.manual_seed(1)
    targets = _targets()
    prediction = torch.randn_like(targets)
    loss_fn = WeightedSignedFieldLoss(beta=8.0)
    plain = loss_fn(prediction, targets)
    unit = loss_fn(prediction, targets, sample_weight=torch.ones(3))
    assert torch.allclose(plain, unit, atol=1e-6)
