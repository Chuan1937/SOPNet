"""Sharded HDF5 cache of standardised 600-sample windows (P at sample 300).

The raw datasets stay untouched; only compact float32 windows are copied into a
Linux-native cache directory so training never fights WSL/NTFS random-read
latency. Each shard stores ``X [N, 600]``, ``p_pick``, ``label`` and the global
manifest index; the companion ``index.parquet`` maps manifest rows to shards and
carries the waveform hash used for leakage-safe splits.
"""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from sopnet.data.manifest import SourceReader
from sopnet.data.preprocess import PreprocessConfig, extract_cache_window, waveform_hash
from sopnet.data.split import DEFAULT_SEED, assign_splits

WORKER: Dict[str, object] = {}


@dataclass
class CacheConfig:
    data_root: Path
    cache_dir: Path
    shard_size: int = 50_000
    chunk_size: int = 2_000
    workers: int = 8
    seed: int = DEFAULT_SEED
    compression: str = "lzf"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _init_worker(data_root: str, preprocess: PreprocessConfig) -> None:
    WORKER["data_root"] = Path(data_root)
    WORKER["preprocess"] = preprocess
    WORKER["readers"] = {}


def _get_reader(source: str) -> SourceReader:
    readers = WORKER.setdefault("readers", {})
    if source not in readers:
        readers[source] = SourceReader(source, WORKER["data_root"])
    return readers[source]


def _process_chunk(records: List[dict]) -> List[dict]:
    preprocess: PreprocessConfig = WORKER["preprocess"]
    results = []
    for record in records:
        try:
            reader = _get_reader(record["source"])
            raw = reader.read(record)
            window, p_position = extract_cache_window(
                raw, record["sampling_rate"], record["p_pick"], preprocess
            )
            results.append(
                {
                    "ok": True,
                    "orig_index": record["orig_index"],
                    "x": window,
                    "p_pick": np.int16(p_position),
                    "label": np.int8(record["canonical_label"]),
                    "hash": np.uint64(waveform_hash(window)),
                }
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the caller
            results.append(
                {
                    "ok": False,
                    "orig_index": record["orig_index"],
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(limit=2),
                }
            )
    return results


def locality_key(source: str, trace_id: str, row_index: int) -> str:
    """Build a sort key that keeps reads physically sequential.

    Trace ids such as ``bucket2$1000`` sort *before* ``bucket2$999``
    lexicographically, which turns sorted reads into random seeks. Bucket row
    indices are therefore zero padded so numeric order is preserved.
    """
    if source == "diting":
        part, _, key = str(trace_id).partition(":")
        return f"0:{int(part):03d}:{key}"
    if str(trace_id) == "":
        return f"0:000:{int(row_index):09d}"
    group, _, spec = str(trace_id).partition("$")
    first = spec.split(",", 1)[0].lstrip(":")
    index = int(first) if first else 0
    return f"{group}:{index:09d}"


def _sort_for_locality(manifest: pd.DataFrame) -> pd.DataFrame:
    frame = manifest.reset_index(drop=True)
    keys = [
        f"{source_path}|{locality_key(source, trace_id, row_index)}"
        for source_path, source, trace_id, row_index in zip(
            frame["source_path"].astype(str),
            frame["source"].astype(str),
            frame["trace_id"].astype(str),
            frame["row_index"],
        )
    ]
    frame["_order"] = keys
    frame = frame.sort_values("_order", kind="stable").drop(columns="_order")
    return frame.reset_index(drop=True)


def _make_records(frame: pd.DataFrame) -> List[dict]:
    columns = [
        "source",
        "source_path",
        "trace_id",
        "row_index",
        "p_pick",
        "sampling_rate",
        "canonical_label",
    ]
    records = []
    for orig_index, row in zip(frame["_orig"].to_numpy(), frame[columns].to_dict("records")):
        record = {key: (str(value) if key in ("source", "source_path", "trace_id") else value)
                  for key, value in row.items()}
        record["orig_index"] = int(orig_index)
        record["row_index"] = int(row["row_index"])
        record["sampling_rate"] = float(row["sampling_rate"])
        record["p_pick"] = float(row["p_pick"])
        record["canonical_label"] = int(row["canonical_label"])
        records.append(record)
    return records


def build_cache(
    manifest: pd.DataFrame,
    config: CacheConfig,
    limit: Optional[int] = None,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the cache and return ``(manifest_with_splits, cache_index)``."""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = manifest.reset_index(drop=True)
    if limit is not None and limit < len(frame):
        frame = frame.sample(n=int(limit), random_state=config.seed).reset_index(drop=True)
    frame["_orig"] = np.arange(len(frame), dtype=np.int64)

    ordered = _sort_for_locality(frame)
    records = _make_records(ordered)
    chunks = [
        records[start : start + config.chunk_size]
        for start in range(0, len(records), config.chunk_size)
    ]

    x_buffer: List[np.ndarray] = []
    p_buffer: List[np.int16] = []
    label_buffer: List[np.int8] = []
    index_buffer: List[int] = []
    shard_records: List[Tuple[int, int, int, int]] = []
    hash_by_orig: Dict[int, int] = {}
    failures: List[dict] = []
    shard_count = 0
    position_in_shard = 0

    def flush_shard() -> None:
        nonlocal x_buffer, p_buffer, label_buffer, index_buffer, shard_count, position_in_shard
        if not x_buffer:
            return
        path = cache_dir / f"shard_{shard_count:05d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "X", data=np.stack(x_buffer), compression=config.compression
            )
            handle.create_dataset("p_pick", data=np.array(p_buffer, dtype=np.int16))
            handle.create_dataset("label", data=np.array(label_buffer, dtype=np.int8))
            handle.create_dataset("sample_index", data=np.array(index_buffer, dtype=np.int64))
            handle.attrs["cache_length"] = config.preprocess.cache_length
            handle.attrs["p_position"] = config.preprocess.p_position
            handle.attrs["sampling_rate"] = config.preprocess.fs
        for offset in range(len(index_buffer)):
            shard_records.append((index_buffer[offset], shard_count, offset))
        shard_count += 1
        position_in_shard = 0
        x_buffer, p_buffer, label_buffer, index_buffer = [], [], [], []

    def handle_result(result: dict) -> None:
        nonlocal position_in_shard
        if not result["ok"]:
            failures.append(result)
            return
        x_buffer.append(result["x"])
        p_buffer.append(result["p_pick"])
        label_buffer.append(result["label"])
        index_buffer.append(result["orig_index"])
        hash_by_orig[result["orig_index"]] = int(result["hash"])
        position_in_shard += 1
        if position_in_shard >= config.shard_size:
            flush_shard()

    waves = max(1, config.workers) * 4
    with ProcessPoolExecutor(
        max_workers=max(1, config.workers),
        initializer=_init_worker,
        initargs=(str(config.data_root), config.preprocess),
    ) as executor:
        for start in range(0, len(chunks), waves):
            wave = chunks[start : start + waves]
            for results in executor.map(_process_chunk, wave):
                for result in results:
                    handle_result(result)
    flush_shard()

    success = np.zeros(len(manifest), dtype=bool)
    success[np.fromiter(hash_by_orig.keys(), dtype=np.int64, count=len(hash_by_orig))] = True
    manifest_out = manifest.reset_index(drop=True)[success].reset_index(drop=True)
    old_to_new = np.full(len(manifest), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(success)] = np.arange(int(success.sum()))

    manifest_out["waveform_hash"] = np.array(
        [hash_by_orig[int(old)] for old in np.flatnonzero(success)], dtype=np.uint64
    )
    manifest_out = assign_splits(manifest_out, seed=config.seed)

    cache_index = pd.DataFrame(
        {
            "manifest_index": np.array([old_to_new[row[0]] for row in shard_records], dtype=np.int64),
            "shard": np.array([row[1] for row in shard_records], dtype=np.int32),
            "shard_index": np.array([row[2] for row in shard_records], dtype=np.int32),
        }
    )
    cache_index["sample_id"] = manifest_out["sample_id"].to_numpy()[cache_index["manifest_index"]]
    cache_index["waveform_hash"] = manifest_out["waveform_hash"].to_numpy()[
        cache_index["manifest_index"]
    ]
    # Rows are processed in locality-sorted order; the dataset indexes
    # positionally, so the index must be stored in manifest order.
    cache_index = cache_index.sort_values("manifest_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        cache_index["manifest_index"].to_numpy(), np.arange(len(manifest_out), dtype=np.int64)
    ):
        raise RuntimeError("cache index does not cover the manifest exactly")

    manifest_out.to_parquet(cache_dir / "manifest.parquet", index=False)
    cache_index.to_parquet(cache_dir / "index.parquet", index=False)

    meta = {
        "config": {
            "shard_size": config.shard_size,
            "chunk_size": config.chunk_size,
            "workers": config.workers,
            "seed": config.seed,
            "compression": config.compression,
            "data_root": str(config.data_root),
            "preprocess": asdict(config.preprocess),
        },
        "n_requested": int(len(frame)),
        "n_cached": int(len(manifest_out)),
        "n_shards": int(shard_count),
        "failures": int(len(failures)),
        "failure_examples": failures[:10],
        "sources": manifest_out["source"].value_counts().to_dict(),
        "labels": manifest_out["canonical_label"].value_counts().to_dict(),
        "splits": manifest_out["split"].value_counts().to_dict(),
    }
    meta["config"]["preprocess"].pop("_sos", None)
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)

    if logger is not None:
        logger.info(
            "cache built: %d/%d samples, %d shards, %d failures",
            len(manifest_out),
            len(frame),
            shard_count,
            len(failures),
        )
        if failures:
            logger.warning("first failure: %s", failures[0]["error"])
    return manifest_out, cache_index


def load_cache(cache_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    manifest = pd.read_parquet(cache_dir / "manifest.parquet")
    index = pd.read_parquet(cache_dir / "index.parquet")
    return manifest, index
