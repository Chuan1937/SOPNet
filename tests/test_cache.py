from __future__ import annotations

import h5py
import numpy as np

from sopnet.data.cache import CacheConfig, build_cache
from sopnet.data.manifest import build_manifest, read_waveform
from sopnet.data.preprocess import PreprocessConfig, extract_cache_window


def test_cached_rows_match_their_sources(fake_root, tmp_path):
    manifest = build_manifest(fake_root)
    config = CacheConfig(
        data_root=fake_root,
        cache_dir=tmp_path / "cache",
        shard_size=3,
        chunk_size=4,
        workers=1,
        preprocess=PreprocessConfig(),
    )
    cached, index = build_cache(manifest, config)
    assert len(cached) == len(manifest)
    assert index["manifest_index"].tolist() == list(range(len(cached)))

    handles = {}
    for position, (_, row) in enumerate(cached.iterrows()):
        raw = read_waveform(row["source"], row, fake_root)
        expected, _ = extract_cache_window(raw, row["sampling_rate"], float(row["p_pick"]))
        shard = int(index.iloc[position]["shard"])
        shard_index = int(index.iloc[position]["shard_index"])
        if shard not in handles:
            handles[shard] = h5py.File(tmp_path / "cache" / f"shard_{shard:05d}.h5", "r")
        actual = handles[shard]["X"][shard_index]
        assert np.allclose(actual, expected, atol=1e-5), f"row {position} does not match its source"
        stored_label = int(handles[shard]["label"][shard_index])
        assert stored_label == int(row["canonical_label"])
        assert int(handles[shard]["p_pick"][shard_index]) == 300
