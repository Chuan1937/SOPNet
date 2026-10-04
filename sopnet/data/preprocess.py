"""Deterministic waveform preprocessing shared by the cache builder and inference.

Pipeline (matching the SOPNet data protocol):

    component Z -> resample to 100 Hz -> demean -> linear detrend
    -> zero-phase 1-45 Hz Butterworth bandpass -> crop around P
    -> max-abs normalisation

Normalisation never changes the waveform sign, which is essential because the
sign carries the first-motion polarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class PreprocessConfig:
    fs: int = 100
    lowcut: float = 1.0
    highcut: float = 45.0
    order: int = 4
    cache_length: int = 600
    p_position: int = 300
    epsilon: float = 1e-12
    _sos: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    def sos(self) -> np.ndarray:
        if self._sos is None:
            nyquist = 0.5 * self.fs
            if not 0 < self.lowcut < self.highcut < nyquist:
                raise ValueError(f"Invalid band {self.lowcut}-{self.highcut} Hz for fs={self.fs} Hz")
            self._sos = signal.butter(
                self.order,
                [self.lowcut / nyquist, self.highcut / nyquist],
                btype="bandpass",
                output="sos",
            )
        return self._sos


def resample_trace(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational resampling to the target rate (no-op when rates match)."""
    if np.isclose(fs_in, fs_out):
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def extract_cache_window(
    x: np.ndarray,
    fs_in: float,
    p_pick_in: float,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, int]:
    """Extract a standardised ``cache_length`` window with P at ``p_position``.

    Only a short segment around the pick is resampled, so this is cheap even for
    DiTing's 180 s traces. Out-of-range regions are zero-padded.
    """
    config = config or PreprocessConfig()
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("waveform is empty")

    ratio = config.fs / float(fs_in)
    half_native = int(round((config.cache_length / 2.0 + 0.5) / ratio))
    start = int(np.floor(p_pick_in)) - half_native
    end = int(np.floor(p_pick_in)) + half_native

    segment = np.zeros(2 * half_native, dtype=np.float64)
    src_start = max(0, start)
    src_end = min(x.size, end)
    if src_end > src_start:
        segment[src_start - start : src_end - start] = x[src_start:src_end]

    segment = resample_trace(segment, fs_in, config.fs)
    segment = segment - segment.mean()
    segment = signal.detrend(segment, type="linear")
    segment = signal.sosfiltfilt(config.sos(), segment)

    centre_target = len(segment) // 2
    half_out = config.cache_length // 2
    out_start = centre_target - half_out
    out_end = out_start + config.cache_length
    window = np.zeros(config.cache_length, dtype=np.float64)
    src_start = max(0, out_start)
    src_end = min(len(segment), out_end)
    if src_end > src_start:
        offset = src_start - out_start
        window[offset : offset + (src_end - src_start)] = segment[src_start:src_end]

    peak = float(np.max(np.abs(window)))
    if peak > config.epsilon:
        window = window / peak
    return window.astype(np.float32), config.p_position


def normalize_max_abs(x: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(arr)))
    if peak > epsilon:
        arr = arr / peak
    return arr.astype(np.float32)


def waveform_hash(x: np.ndarray, digest_size: int = 8) -> int:
    """Stable 64-bit blake2b hash of a standardised waveform."""
    import hashlib

    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes()
    digest = hashlib.blake2b(arr, digest_size=digest_size).digest()
    return int.from_bytes(digest, "little", signed=False)
