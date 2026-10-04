"""SOPNet: Signed-Onset Polarity Network for P-wave first-motion polarity picking."""

__version__ = "0.1.0"

from sopnet.data.canonical import DOWN, UNKNOWN, UP, canonical_to_name, signed_field_target

__all__ = [
    "__version__",
    "UP",
    "DOWN",
    "UNKNOWN",
    "canonical_to_name",
    "signed_field_target",
]
