from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from sopnet.data.split import assign_splits


def _write_shard(cache_dir, shard_id, rows):
    path = cache_dir / f"shard_{shard_id:05d}.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=np.stack([row["x"] for row in rows]))
        handle.create_dataset("p_pick", data=np.array([300] * len(rows), dtype=np.int16))
        handle.create_dataset("label", data=np.array([row["label"] for row in rows], dtype=np.int8))
        handle.create_dataset("sample_index", data=np.arange(len(rows), dtype=np.int64))
    return path


@pytest.fixture
def synthetic_cache(tmp_path):
    """A 120-sample cache spanning the five sources with deterministic splits."""
    rng = np.random.default_rng(0)
    n = 120
    sources = ["scsn", "txed", "instance", "pnw", "diting"]
    events = [f"ev{i // 4}" for i in range(n)]
    labels = rng.choice([-1, 0, 1], size=n)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    rows = []
    for index in range(n):
        rows.append({"x": rng.normal(size=600).astype(np.float32), "label": int(labels[index])})
    _write_shard(cache_dir, 0, rows[:60])
    _write_shard(cache_dir, 1, rows[60:])

    manifest = pd.DataFrame(
        {
            "sample_id": [f"x:{i:06d}" for i in range(n)],
            "source": pd.Categorical([sources[i % 5] for i in range(n)], categories=sources),
            "source_path": "synthetic",
            "trace_id": "",
            "row_index": np.arange(n, dtype=np.int32),
            "p_pick": np.full(n, 300.0, dtype=np.float32),
            "sampling_rate": np.full(n, 100, dtype=np.int16),
            "component": "Z",
            "raw_label": ["U" if v > 0 else "D" if v < 0 else "X" for v in labels],
            "canonical_label": labels.astype(np.int8),
            "event_key": events,
            "split_native": pd.NA,
            "waveform_hash": np.zeros(n, dtype=np.uint64),
        }
    )
    manifest = assign_splits(manifest, seed=20261004)
    manifest.to_parquet(cache_dir / "manifest.parquet", index=False)

    records = [(i, 0 if i < 60 else 1, i if i < 60 else i - 60) for i in range(n)]
    rng.shuffle(records)
    index = pd.DataFrame(
        {
            "manifest_index": np.array([r[0] for r in records], dtype=np.int64),
            "shard": np.array([r[1] for r in records], dtype=np.int32),
            "shard_index": np.array([r[2] for r in records], dtype=np.int32),
        }
    )
    index["sample_id"] = manifest["sample_id"].to_numpy()[index["manifest_index"]]
    index["waveform_hash"] = np.zeros(n, dtype=np.uint64)
    index.to_parquet(cache_dir / "index.parquet", index=False)
    return cache_dir, manifest, index


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "data"
    (root / "scsn").mkdir(parents=True)
    (root / "txed").mkdir()
    (root / "Instance").mkdir()
    (root / "pnw").mkdir()
    (root / "谛听1.0-50HZ" / "Diting50hz").mkdir(parents=True)

    rng = np.random.default_rng(0)
    with h5py.File(root / "scsn/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5", "w") as handle:
        handle.create_dataset("X", data=rng.normal(size=(8, 600)).astype(np.float32))
        handle.create_dataset("Y", data=np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.uint8))
        handle.create_dataset("evids", data=np.array([10, 10, 11, 11, 12, 12, 13, 13], dtype=np.uint32))

    with h5py.File(root / "txed/Texd.hdf5", "w") as handle:
        handle.create_dataset("data/bucket0", data=rng.normal(size=(2, 3, 6000)))
        handle.create_dataset("data_format/component_order", data=np.bytes_("ZNE"))
        handle.create_dataset("data_format/sampling_rate", data=100)

    txed_csv = pd.DataFrame(
        {
            "trace_name": ["bucket0$0,:3,:6000", "bucket0$1,:3,:6000"],
            "trace_p_arrival_sample": [1000.0, 900.0],
            "trace_polarity": ["U", "unknown"],
            "trace_name_original": ["eventA_STA1_EV", "eventA_STA2_EV"],
            "source_origin_time": ["2020-01-01", "2020-01-01"],
            "source_latitude_deg": [30.0, 30.0],
            "source_longitude_deg": [100.0, 100.0],
            "source_depth_km": [10.0, 10.0],
            "split": ["train", "test"],
        }
    )
    txed_csv.to_csv(root / "txed/Texd_filtered.csv", index=False)

    with h5py.File(root / "Instance/INSTANCE.hdf5", "w") as handle:
        handle.create_dataset("data/bucket0", data=rng.normal(size=(2, 3, 12000)).astype(np.float32))
        handle.create_dataset("data_format/component_order", data=np.bytes_("ZNE"))

    instance_csv = pd.DataFrame(
        {
            "trace_name": ["bucket0$0,:3,:12000", "bucket0$1,:3,:12000"],
            "trace_P_arrival_sample": [2000.0, 2100.0],
            "trace_polarity": ["positive", "undecidable"],
            "source_id": [100, 100],
            "split": ["train", "train"],
        }
    )
    instance_csv.to_csv(root / "Instance/INSTANCE_filtered.csv", index=False)

    with h5py.File(root / "pnw/pnw.hdf5", "w") as handle:
        handle.create_dataset("data/bucket0", data=rng.normal(size=(2, 3, 15001)))
        handle.create_dataset("data_format/component_order", data=np.bytes_("ENZ"))

    pnw_csv = pd.DataFrame(
        {
            "trace_name": ["bucket0$0,:3,:15001", "bucket0$1,:3,:15001"],
            "trace_P_arrival_sample": [5000.0, 5200.0],
            "trace_P_polarity": ["positive", "negative"],
            "event_id": ["uw1", "uw1"],
        }
    )
    pnw_csv.to_csv(root / "pnw/pnw.csv", index=False)

    with h5py.File(root / "谛听1.0-50HZ/Diting50hz/DiTing330km_part_0.hdf5", "w") as handle:
        handle.create_dataset("earthquake/000001.0001", data=rng.normal(size=(9000, 3)))
        handle.create_dataset("earthquake/000001.0002", data=rng.normal(size=(9000, 3)))

    diting_csv = pd.DataFrame(
        {
            "key": [1.0001, 1.0002],
            "p_pick": [1500.0, 1600.0],
            "p_motion": ["U", "R"],
            "ev_id": [1, 1],
        }
    )
    diting_csv.to_csv(root / "谛听1.0-50HZ/Diting50hz/DiTing330km_part_0.csv", index=False)
    return root
