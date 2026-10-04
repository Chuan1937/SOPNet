from __future__ import annotations

import numpy as np
import pytest

from sopnet.data.canonical import (
    DOWN,
    UNKNOWN,
    UP,
    canonical_to_name,
    first_motion_from_field,
    map_raw_label,
    signed_field_target,
)


@pytest.mark.parametrize(
    "source,raw,expected",
    [
        # SCSN: 0=Down, 1=Up, 2=Unknown (empirically verified)
        ("scsn", 0, DOWN),
        ("scsn", 1, UP),
        ("scsn", 2, UNKNOWN),
        ("scsn", 7, UNKNOWN),
        # TXED
        ("txed", "U", UP),
        ("txed", "D", DOWN),
        ("txed", "unknown", UNKNOWN),
        ("txed", "junk", UNKNOWN),
        # INSTANCE
        ("instance", "positive", UP),
        ("instance", "negative", DOWN),
        ("instance", "undecidable", UNKNOWN),
        # PNW
        ("pnw", "positive", UP),
        ("pnw", "negative", DOWN),
        ("pnw", "undecidable", UNKNOWN),
        ("pnw", float("nan"), UNKNOWN),
        # DiTing: C=compression=Up, R=rarefaction=Down
        ("diting", "U", UP),
        ("diting", "C", UP),
        ("diting", "D", DOWN),
        ("diting", "R", DOWN),
        ("diting", " ", UNKNOWN),
        ("diting", "n", UNKNOWN),
    ],
)
def test_canonical_label_mapping(source, raw, expected):
    assert map_raw_label(source, raw) == expected


def test_unknown_source_rejected():
    with pytest.raises(KeyError):
        map_raw_label("not_a_dataset", "U")


def test_canonical_to_name_roundtrip():
    assert canonical_to_name(UP) == "U"
    assert canonical_to_name(DOWN) == "D"
    assert canonical_to_name(UNKNOWN) == "X"
    names = canonical_to_name(np.array([UP, DOWN, UNKNOWN]))
    assert list(names) == ["U", "D", "X"]


@pytest.mark.parametrize("label,sign", [(UP, 1.0), (DOWN, -1.0)])
def test_signed_field_target_peak(label, sign):
    target = signed_field_target(400, p_position=200, label=label, sigma=10.0)
    assert target.shape == (400,)
    assert target.dtype == np.float32
    assert np.argmax(np.abs(target)) == 200
    assert np.isclose(target[200], sign, atol=1e-6)
    assert (target * sign >= 0).all()


def test_signed_field_unknown_is_zero():
    target = signed_field_target(400, p_position=200, label=UNKNOWN)
    assert np.all(target == 0.0)


def test_signed_field_rejects_bad_position():
    with pytest.raises(ValueError):
        signed_field_target(100, p_position=100, label=UP)


def test_first_motion_from_field_reads_sign_and_position():
    target = signed_field_target(400, 123, DOWN, sigma=10)
    result = first_motion_from_field(target)
    assert result["p_position"] == 123
    assert result["polarity"] == DOWN
    assert result["confidence"] > 0.99


def test_inversion_flips_target_exactly():
    up = signed_field_target(400, 210, UP)
    down = signed_field_target(400, 210, DOWN)
    assert np.allclose(up, -down)
