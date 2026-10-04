from __future__ import annotations

import numpy as np
import pandas as pd

from sopnet.data.split import assign_splits, check_split_leakage, stable_hash64


def _manifest(n=400, n_events=100, duplicate=False):
    events = [f"event{i % n_events}" for i in range(n)]
    hashes = np.arange(n, dtype=np.uint64) + np.uint64(1)
    if duplicate:
        hashes[10] = hashes[0]
    return pd.DataFrame(
        {
            "sample_id": [f"s:{i}" for i in range(n)],
            "event_key": events,
            "waveform_hash": hashes,
        }
    )


def test_split_is_deterministic():
    first = assign_splits(_manifest())
    second = assign_splits(_manifest())
    assert (first["split"] == second["split"]).all()


def test_events_never_cross_splits():
    manifest = assign_splits(_manifest())
    assert check_split_leakage(manifest)["event_key_leaks"] == 0
    ratio = manifest["split"].value_counts(normalize=True)
    assert abs(ratio["train"] - 0.8) < 0.05


def test_duplicate_waveforms_stay_together():
    manifest = assign_splits(_manifest(duplicate=True))
    duplicated = manifest[manifest["waveform_hash"] == manifest["waveform_hash"].iloc[0]]
    assert len(duplicated) >= 2
    assert duplicated["split"].nunique() == 1
    assert check_split_leakage(manifest)["waveform_hash_leaks"] == 0


def test_stable_hash_is_seed_sensitive():
    assert stable_hash64("a", seed=1) != stable_hash64("a", seed=2)
    assert stable_hash64("a", seed=1) == stable_hash64("a", seed=1)


def test_zero_hashes_are_ignored():
    manifest = _manifest()
    manifest["waveform_hash"] = np.uint64(0)
    assigned = assign_splits(manifest)
    assert check_split_leakage(assigned)["waveform_hash_leaks"] == 0
