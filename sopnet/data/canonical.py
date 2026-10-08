"""Canonical polarity labels and signed onset-field targets.

Polarity is represented by its physical meaning, never by dataset-specific codes:

    UP      = +1  (first motion up / compression)
    DOWN    = -1  (first motion down / rarefaction)
    UNKNOWN =  0  (undetermined)

Every raw dataset label must pass through an explicit mapping in ``SOURCE_LABEL_MAPS``.
Concatenating raw integer labels across datasets is forbidden because SCSN uses
``0=Down, 1=Up`` while other datasets use ``0=Up, 1=Down`` conventions.
"""

from __future__ import annotations

from typing import Dict, Mapping, Union

import numpy as np

UP = 1
DOWN = -1
UNKNOWN = 0

LABEL_NAMES: Dict[int, str] = {UP: "U", DOWN: "D", UNKNOWN: "X"}
NAME_TO_CANONICAL: Dict[str, int] = {"U": UP, "UP": UP, "D": DOWN, "DOWN": DOWN, "X": UNKNOWN}

# SCSN numeric encoding is verified empirically in docs/data_protocol.md:
#   label 0 -> negative post-P deflection -> Down
#   label 1 -> positive post-P deflection -> Up
#   label 2 -> no consistent deflection      -> Unknown
SCSN_LABELS: Dict[int, int] = {0: DOWN, 1: UP, 2: UNKNOWN}

# TXED / INSTANCE / PNW CSV label strings.
STRING_LABELS: Dict[str, int] = {
    "U": UP,
    "UP": UP,
    "POSITIVE": UP,
    "D": DOWN,
    "DOWN": DOWN,
    "NEGATIVE": DOWN,
}

# DiTing p_motion codes. ``R`` (rarefaction) and ``C`` (compression) are real
# polarity labels; their assignment is verified empirically in
# docs/data_protocol.md (R first-break sign matches D, C matches U;
# U+R+D+C = 641,025 which equals the published DiTing polarity-label count).
DITING_LABELS: Dict[str, int] = {
    "U": UP,
    "C": UP,
    "D": DOWN,
    "R": DOWN,
}

SOURCE_LABEL_MAPS: Dict[str, Mapping] = {
    "scsn": SCSN_LABELS,
    "txed": STRING_LABELS,
    "instance": STRING_LABELS,
    "pnw": STRING_LABELS,
    "diting": DITING_LABELS,
}


def canonical_to_name(labels: Union[int, np.ndarray]) -> Union[str, np.ndarray]:
    """Map canonical labels to ``U``/``D``/``X`` names."""
    if np.isscalar(labels):
        return LABEL_NAMES.get(int(labels), "X")
    arr = np.asarray(labels)
    return np.array([LABEL_NAMES.get(int(value), "X") for value in arr.ravel()]).reshape(arr.shape)


def map_raw_label(source: str, raw: object) -> int:
    """Map a single raw label of ``source`` to the canonical encoding."""
    mapping = SOURCE_LABEL_MAPS.get(source)
    if mapping is None:
        raise KeyError(f"Unknown data source '{source}'. Known: {sorted(SOURCE_LABEL_MAPS)}")

    if source == "scsn":
        try:
            key = int(float(raw))
        except (TypeError, ValueError):
            return UNKNOWN
        return int(mapping.get(key, UNKNOWN))

    key = str(raw).strip()
    if key in ("", "nan", "None", "-"):
        return UNKNOWN
    return int(mapping.get(key.upper(), UNKNOWN))


def map_raw_labels(source: str, raws) -> np.ndarray:
    """Vectorised :func:`map_raw_label` returning ``int8`` canonical labels."""
    return np.array([map_raw_label(source, raw) for raw in raws], dtype=np.int8)


def signed_field_target(
    length: int,
    p_position: int,
    label: int,
    sigma: float = 10.0,
) -> np.ndarray:
    """Signed Gaussian onset target of shape ``(length,)`` and dtype ``float32``.

    ``p_position`` is the sample index of the target peak inside the window.
    The amplitude equals ``sign(label)``; unknown samples (``label == 0``)
    produce an all-zero field.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if not (0 <= p_position < length):
        raise ValueError(f"p_position {p_position} outside window of length {length}")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    sign = float(np.sign(label))
    if sign == 0.0:
        return np.zeros(length, dtype=np.float32)

    samples = np.arange(length, dtype=np.float32)
    field = sign * np.exp(-((samples - float(p_position)) ** 2) / (2.0 * sigma**2))
    return field.astype(np.float32)


def first_motion_from_field(
    field: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Union[int, float]]:
    """Read P position, polarity and confidence from a signed onset field."""
    arr = np.asarray(field, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("field must not be empty")
    index = int(np.argmax(np.abs(arr)))
    confidence = float(abs(arr[index]))
    sign = float(arr[index])
    if confidence <= threshold or sign == 0.0:
        label = UNKNOWN
    else:
        label = UP if sign > 0 else DOWN
    return {"p_position": index, "polarity": label, "confidence": confidence}
