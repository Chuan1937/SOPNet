from __future__ import annotations

import numpy as np

from sopnet.data.preprocess import PreprocessConfig, extract_cache_window, waveform_hash


def test_extract_cache_window_shape_and_normalisation():
    rng = np.random.default_rng(0)
    trace = rng.normal(size=3000)
    window, p_position = extract_cache_window(trace, fs_in=50, p_pick_in=1500)
    assert window.shape == (600,)
    assert window.dtype == np.float32
    assert p_position == 300
    assert np.isfinite(window).all()
    assert np.isclose(np.max(np.abs(window)), 1.0, atol=1e-5)


def test_p_onset_lands_at_cache_position():
    trace = np.zeros(4000)
    trace[2000] = 1.0
    window, _ = extract_cache_window(trace, fs_in=100, p_pick_in=2000)
    assert abs(int(np.argmax(np.abs(window))) - 300) <= 5


def test_resampling_from_50hz_preserves_p_position():
    trace = np.zeros(1800)
    trace[900] = 1.0
    window, _ = extract_cache_window(trace, fs_in=50, p_pick_in=900)
    assert abs(int(np.argmax(np.abs(window))) - 300) <= 6


def test_pick_near_edge_is_padded():
    trace = np.ones(200)
    window, _ = extract_cache_window(trace, fs_in=100, p_pick_in=5)
    assert window.shape == (600,)
    assert np.isfinite(window).all()


def test_waveform_hash_is_stable_and_sensitive():
    x = np.linspace(-1, 1, 600).astype(np.float32)
    assert waveform_hash(x) == waveform_hash(x.copy())
    assert waveform_hash(x) != waveform_hash(-x)


def test_config_rejects_invalid_band():
    config = PreprocessConfig(fs=100, lowcut=1.0, highcut=60.0)
    try:
        config.sos()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
