"""Sharded HDF5 cache of standardised 600-sample windows (P at sample 300).

The raw datasets stay untouched; only compact float32 windows are copied into a
Linux-native cache directory so training never fights WSL/NTFS random-read
latency. Each shard stores ``X [N, 600]``, ``p_pick``, ``label``, the global
manifest index and the waveform hash; the companion ``index.parquet`` maps
manifest rows to shards.

The build is resumable: every shard is self-describing, so an interrupted run
can continue with ``--resume`` without losing completed shards. Chunks are
generated lazily and workers are spawned (not forked), keeping peak memory flat
even for the full 7.8 M-sample manifest.
"""

from __future__ import annotations

import ctypes
import gc
import json
import multiprocessing as mp
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

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
    tasks_per_child: int = 50
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def _init_worker(data_root: str, preprocess: PreprocessConfig) -> None:
    WORKER["data_root"] = Path(data_root)
    WORKER["preprocess"] = preprocess
    WORKER["readers"] = {}
    WORKER["trim"] = _malloc_trim()


def _malloc_trim():
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except OSError:
        return None


def _release_memory() -> None:
    gc.collect()
    trim = WORKER.get("trim")
    if trim is not None:
        trim(0)


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
    _release_memory()
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


def _sort_for_locality(frame: pd.DataFrame) -> pd.DataFrame:
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
    frame.sort_values("_order", kind="stable", inplace=True)
    frame.drop(columns="_order", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


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
        record = {
            key: (str(value) if key in ("source", "source_path", "trace_id") else value)
            for key, value in row.items()
        }
        record["orig_index"] = int(orig_index)
        record["row_index"] = int(row["row_index"])
        record["sampling_rate"] = float(row["sampling_rate"])
        record["p_pick"] = float(row["p_pick"])
        record["canonical_label"] = int(row["canonical_label"])
        records.append(record)
    return records


def _existing_shards(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.glob("shard_*.h5"))


def _read_existing(cache_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(sample_index, waveform_hash)`` of every completed shard.

    Shards interrupted mid-write are unreadable and are removed so a resumed run
    starts from a consistent state.
    """
    positions, hashes = [], []
    for path in _existing_shards(cache_dir):
        try:
            with h5py.File(path, "r") as handle:
                positions.append(np.asarray(handle["sample_index"][:], dtype=np.int64))
                hashes.append(np.asarray(handle["waveform_hash"][:], dtype=np.uint64))
        except (OSError, KeyError):
            path.unlink(missing_ok=True)
    if not positions:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
    return np.concatenate(positions), np.concatenate(hashes)


def build_cache(
    manifest: pd.DataFrame,
    config: CacheConfig,
    limit: Optional[int] = None,
    resume: bool = False,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build (or resume) the cache; returns ``(manifest_with_splits, cache_index)``."""
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if limit is None:
        frame = manifest
    else:
        frame = manifest.sample(n=min(int(limit), len(manifest)), random_state=config.seed).reset_index(
            drop=True
        )
    frame = frame.copy()
    frame["_orig"] = np.arange(len(frame), dtype=np.int64)

    completed_positions, completed_hashes = (
        _read_existing(cache_dir) if resume else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64))
    )
    already_done = np.zeros(len(frame), dtype=bool)
    already_done[completed_positions] = True
    next_shard = (
        max(int(path.stem.split("_")[1]) for path in _existing_shards(cache_dir)) + 1
        if resume and _existing_shards(cache_dir)
        else 0
    )

    ordered = _sort_for_locality(frame)
    n_total = len(ordered)

    x_buffer: List[np.ndarray] = []
    p_buffer: List[np.int16] = []
    label_buffer: List[np.int8] = []
    hash_buffer: List[np.uint64] = []
    index_buffer: List[int] = []
    failures: List[dict] = []
    shard_count = next_shard
    cached_count = int(already_done.sum())
    position_in_shard = 0

    def flush_shard() -> None:
        nonlocal x_buffer, p_buffer, label_buffer, hash_buffer, index_buffer, shard_count
        nonlocal position_in_shard
        if not x_buffer:
            return
        path = cache_dir / f"shard_{shard_count:05d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("X", data=np.stack(x_buffer), compression=config.compression)
            handle.create_dataset("p_pick", data=np.array(p_buffer, dtype=np.int16))
            handle.create_dataset("label", data=np.array(label_buffer, dtype=np.int8))
            handle.create_dataset("sample_index", data=np.array(index_buffer, dtype=np.int64))
            handle.create_dataset("waveform_hash", data=np.array(hash_buffer, dtype=np.uint64))
            handle.attrs["cache_length"] = config.preprocess.cache_length
            handle.attrs["p_position"] = config.preprocess.p_position
            handle.attrs["sampling_rate"] = config.preprocess.fs
        shard_count += 1
        position_in_shard = 0
        x_buffer, p_buffer, label_buffer, hash_buffer, index_buffer = [], [], [], [], []

    def handle_result(result: dict) -> None:
        nonlocal position_in_shard, cached_count
        if not result["ok"]:
            failures.append(result)
            return
        x_buffer.append(result["x"])
        p_buffer.append(result["p_pick"])
        label_buffer.append(result["label"])
        hash_buffer.append(np.uint64(result["hash"]))
        index_buffer.append(int(result["orig_index"]))
        position_in_shard += 1
        cached_count += 1
        if logger is not None and cached_count % 500_000 == 0:
            logger.info("cache progress: %d/%d samples", cached_count, n_total)
        if position_in_shard >= config.shard_size:
            flush_shard()

    waves = max(1, config.workers) * 4
    step = config.chunk_size * waves
    context = mp.get_context("spawn")
    # Worker processes recycle after ``tasks_per_child`` chunks: h5py/HDF5 holds
    # some per-dataset state that cannot be fully reclaimed in-process, so a
    # bounded worker lifetime is the only way to guarantee flat memory.
    pool = context.Pool(
        processes=max(1, config.workers),
        initializer=_init_worker,
        initargs=(str(config.data_root), config.preprocess),
        maxtasksperchild=max(1, config.tasks_per_child),
    )
    try:
        for start in range(0, n_total, step):
            wave = []
            for offset in range(0, step, config.chunk_size):
                begin = start + offset
                if begin >= n_total:
                    break
                piece = ordered.iloc[begin : begin + config.chunk_size]
                if cached_count:
                    keep = ~already_done[piece["_orig"].to_numpy()]
                    piece = piece[keep]
                if len(piece):
                    wave.append(_make_records(piece))
            if not wave:
                continue
            for results in pool.imap(_process_chunk, wave, chunksize=1):
                for result in results:
                    handle_result(result)
    finally:
        pool.close()
        pool.join()
    flush_shard()

    completed_positions, completed_hashes = _read_existing(cache_dir)
    if len(np.unique(completed_positions)) != len(completed_positions):
        raise RuntimeError("duplicate sample indices across cache shards")

    success = np.zeros(len(frame), dtype=bool)
    success[completed_positions] = True
    positions = np.flatnonzero(success)
    hash_by_position = np.zeros(len(frame), dtype=np.uint64)
    hash_by_position[completed_positions] = completed_hashes

    # ``frame`` is sorted by read locality in ``ordered``; restore original order
    # so row k of the output corresponds to ``manifest_index == k``.
    manifest_out = frame.sort_values("_orig", kind="stable").drop(columns="_orig").reset_index(drop=True)
    manifest_out = manifest_out.iloc[positions].reset_index(drop=True)
    manifest_out["waveform_hash"] = hash_by_position[positions]
    manifest_out = assign_splits(manifest_out, seed=config.seed)

    old_to_new = np.full(len(frame), -1, dtype=np.int64)
    old_to_new[positions] = np.arange(len(positions), dtype=np.int64)
    orig_parts, shard_parts, shard_index_parts = [], [], []
    for path in _existing_shards(cache_dir):
        shard_id = int(path.stem.split("_")[1])
        with h5py.File(path, "r") as handle:
            sample_index = np.asarray(handle["sample_index"][:], dtype=np.int64)
        orig_parts.append(sample_index)
        shard_parts.append(np.full(len(sample_index), shard_id, dtype=np.int32))
        shard_index_parts.append(np.arange(len(sample_index), dtype=np.int32))
    if orig_parts:
        orig = np.concatenate(orig_parts)
        shard_column = np.concatenate(shard_parts)
        shard_position = np.concatenate(shard_index_parts)
    else:
        orig = np.empty(0, dtype=np.int64)
        shard_column = np.empty(0, dtype=np.int32)
        shard_position = np.empty(0, dtype=np.int32)
    cache_index = pd.DataFrame(
        {"manifest_index": old_to_new[orig], "shard": shard_column, "shard_index": shard_position}
    )
    cache_index["sample_id"] = manifest_out["sample_id"].to_numpy()[cache_index["manifest_index"]]
    cache_index["waveform_hash"] = manifest_out["waveform_hash"].to_numpy()[cache_index["manifest_index"]]
    # Rows are processed in locality-sorted order; the dataset indexes
    # positionally, so the index must be stored in manifest order.
    cache_index.sort_values("manifest_index", kind="stable", inplace=True)
    cache_index.reset_index(drop=True, inplace=True)
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
            "tasks_per_child": config.tasks_per_child,
            "data_root": str(config.data_root),
            "preprocess": asdict(config.preprocess),
        },
        "n_requested": int(len(frame)),
        "n_cached": int(len(manifest_out)),
        "n_shards": int(len(_existing_shards(cache_dir))),
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
            meta["n_shards"],
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
