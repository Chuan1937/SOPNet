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


def test_resume_keeps_completed_shards_and_rebuilds_missing(fake_root, tmp_path):
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
    shards = sorted(config.cache_dir.glob("shard_*.h5"))
    assert len(shards) > 1

    missing = shards[-1]
    missing.unlink()
    resumed, resumed_index = build_cache(manifest, config, resume=True)

    assert len(resumed) == len(cached)
    assert resumed_index["manifest_index"].tolist() == list(range(len(resumed)))
    assert len(sorted(config.cache_dir.glob("shard_*.h5"))) >= len(shards)

    handles = {}
    for position, (_, row) in enumerate(resumed.iterrows()):
        raw = read_waveform(row["source"], row, fake_root)
        expected, _ = extract_cache_window(raw, row["sampling_rate"], float(row["p_pick"]))
        shard = int(resumed_index.iloc[position]["shard"])
        shard_index = int(resumed_index.iloc[position]["shard_index"])
        if shard not in handles:
            handles[shard] = h5py.File(config.cache_dir / f"shard_{shard:05d}.h5", "r")
        assert np.allclose(handles[shard]["X"][shard_index], expected, atol=1e-5)


def test_resume_on_complete_cache_adds_no_shards(fake_root, tmp_path):
    manifest = build_manifest(fake_root)
    config = CacheConfig(
        data_root=fake_root,
        cache_dir=tmp_path / "cache",
        shard_size=3,
        chunk_size=4,
        workers=1,
        preprocess=PreprocessConfig(),
    )
    build_cache(manifest, config)
    shard_count = len(list(config.cache_dir.glob("shard_*.h5")))
    build_cache(manifest, config, resume=True)
    assert len(list(config.cache_dir.glob("shard_*.h5"))) == shard_count
