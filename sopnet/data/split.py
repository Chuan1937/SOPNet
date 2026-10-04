"""Deterministic, leakage-safe train/validation/test splits.

Two constraints are enforced:

1. **Group split** - all samples from the same event (``event_key``) stay together.
2. **Duplicate split** - samples sharing an identical standardised waveform
   (``waveform_hash``) stay together, even across events or datasets.

The split is derived from a fixed seed (default ``20261004``) and is therefore
reproducible without storing extra state.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SEED = 20261004
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def stable_hash64(text: str, seed: int = 0, digest_size: int = 8) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=digest_size).digest(), "little")


def _uniform(value: int) -> float:
    return value / float(2**64)


def _base_split(keys: np.ndarray, seed: int, ratios: Tuple[float, float, float]) -> np.ndarray:
    boundaries = np.cumsum(ratios)
    splits = np.empty(len(keys), dtype=np.int8)
    for i, key in enumerate(keys):
        u = _uniform(stable_hash64(str(key), seed))
        splits[i] = int(np.searchsorted(boundaries, u, side="right"))
    return np.clip(splits, 0, len(SPLIT_NAMES) - 1)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def assign_splits(
    manifest: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    event_column: str = "event_key",
) -> pd.DataFrame:
    """Return a copy of ``manifest`` with a deterministic ``split`` column."""
    if event_column not in manifest.columns:
        raise KeyError(f"manifest is missing '{event_column}'")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("ratios must sum to 1")

    frame = manifest.reset_index(drop=True)
    event_codes, event_values = pd.factorize(frame[event_column].astype(str), sort=True)
    base = _base_split(event_values.to_numpy(), seed, ratios)

    union = _UnionFind(len(event_values))
    if "waveform_hash" in frame.columns:
        hashes = frame["waveform_hash"].to_numpy(dtype=np.uint64)
        valid = hashes != np.uint64(0)
        if valid.any():
            hash_codes, _ = pd.factorize(hashes[valid])
            event_of_row = event_codes[valid]
            order = np.argsort(hash_codes, kind="stable")
            sorted_hashes = hash_codes[order]
            sorted_events = event_of_row[order]
            boundaries = np.flatnonzero(np.diff(sorted_hashes)) + 1
            for group in np.split(sorted_events, boundaries):
                for code in group[1:]:
                    union.union(int(group[0]), int(code))

    roots = np.array([union.find(code) for code in range(len(event_values))])
    votes = np.zeros((len(event_values), len(SPLIT_NAMES)), dtype=np.int64)
    for code, root in enumerate(roots):
        votes[root, base[code]] += 1

    root_split: Dict[int, int] = {}
    for root in np.unique(roots):
        counts = votes[root]
        best = np.flatnonzero(counts == counts.max())
        if len(best) == 1:
            root_split[int(root)] = int(best[0])
        else:
            tie = min(best, key=lambda s: stable_hash64(f"tie:{root}:{s}", seed))
            root_split[int(root)] = int(tie)

    event_splits = np.array([root_split[int(root)] for root in roots], dtype=np.int8)
    row_splits = event_splits[event_codes]
    frame["split"] = pd.Categorical(
        [SPLIT_NAMES[code] for code in row_splits], categories=SPLIT_NAMES
    )
    return frame


def check_split_leakage(manifest: pd.DataFrame) -> Dict[str, int]:
    """Verify event and duplicate isolation; raises on violation."""
    report: Dict[str, int] = {}
    for column in ("event_key",):
        groups = manifest.groupby(column, observed=True)["split"].nunique()
        leaked = int((groups > 1).sum())
        report[f"{column}_leaks"] = leaked
        if leaked:
            raise ValueError(f"{leaked} {column} values span multiple splits")

    if "waveform_hash" in manifest.columns:
        hashes = manifest[manifest["waveform_hash"] != np.uint64(0)]
        groups = hashes.groupby("waveform_hash", observed=True)["split"].nunique()
        leaked = int((groups > 1).sum())
        report["waveform_hash_leaks"] = leaked
        if leaked:
            raise ValueError(f"{leaked} duplicate waveforms span multiple splits")

    return report
