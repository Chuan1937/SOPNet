"""Unified polarity dataset over the sharded HDF5 cache.

``source`` is deliberately *not* exposed to the model: after caching, every
sample looks identical (600 standardised samples, P at index 300, canonical
label). Training crops a random 400-sample window so the network cannot rely on
P always sitting at the same position.
"""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sopnet.data.augment import AugmentConfig, augment_trace
from sopnet.data.cache import load_cache
from sopnet.data.canonical import signed_field_target

CACHE_POSITION = 300
DEFAULT_WINDOW = 400
DEFAULT_SIGMA = 10.0
DEFAULT_JITTER: Tuple[int, int] = (120, 280)


class UnifiedPolarityDataset(Dataset):
    """Maps manifest rows to ``(waveform, signed onset field)`` pairs."""

    def __init__(
        self,
        cache_dir: Path,
        split: Optional[str] = None,
        window_length: int = DEFAULT_WINDOW,
        sigma: float = DEFAULT_SIGMA,
        jitter: Optional[Tuple[int, int]] = DEFAULT_JITTER,
        augment: Optional[AugmentConfig] = None,
        p_shift_samples: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 0,
        manifest=None,
        index=None,
    ):
        self.cache_dir = Path(cache_dir)
        loaded_manifest, loaded_index = load_cache(self.cache_dir)
        frame = loaded_manifest if manifest is None else manifest
        cache_index = loaded_index if index is None else index

        if "manifest_index" in cache_index.columns:
            cache_index = cache_index.sort_values("manifest_index", kind="stable").reset_index(drop=True)
            if not np.array_equal(
                cache_index["manifest_index"].to_numpy(), np.arange(len(frame), dtype=np.int64)
            ):
                raise ValueError("cache index rows do not match the manifest rows")
        if split is not None:
            mask = (frame["split"] == split).to_numpy()
        else:
            mask = np.ones(len(frame), dtype=bool)

        positions = np.flatnonzero(mask)
        if max_samples is not None and max_samples < len(positions):
            rng = np.random.default_rng(seed)
            positions = np.sort(rng.choice(positions, size=int(max_samples), replace=False))
        selected = frame.iloc[positions].reset_index(drop=True)
        cache_index = cache_index.iloc[positions].reset_index(drop=True)

        self.labels = selected["canonical_label"].to_numpy(dtype=np.int64)
        self.p_picks = selected["p_pick"].to_numpy(dtype=np.int64)
        self.shards = cache_index["shard"].to_numpy(dtype=np.int64)
        self.shard_indices = cache_index["shard_index"].to_numpy(dtype=np.int64)
        self.sample_ids = cache_index["sample_id"].to_numpy()

        # Release the (multi-GB) manifest/index frames; only the compact arrays
        # above are needed at training time. Keeping them would also be shared
        # with forked workers and copied on refcount updates.
        del frame, selected, cache_index, loaded_manifest, loaded_index

        self.window_length = int(window_length)
        self.sigma = float(sigma)
        self.jitter = jitter
        self.augment = augment
        self.p_shift_samples = int(p_shift_samples)
        self.seed = int(seed)
        self._handles: Dict[int, h5py.File] = {}
        self._x_datasets: Dict[int, h5py.Dataset] = {}
        self._rng = np.random.default_rng(seed)
        self._reads = 0
        try:
            self._trim = ctypes.CDLL("libc.so.6").malloc_trim
        except OSError:
            self._trim = None

        if self.window_length > 600:
            raise ValueError("window_length cannot exceed the 600-sample cache window")

    def __len__(self) -> int:
        return len(self.labels)

    def _handle(self, shard: int) -> h5py.File:
        handle = self._handles.get(int(shard))
        if handle is None:
            handle = h5py.File(self.cache_dir / f"shard_{int(shard):05d}.h5", "r")
            self._handles[int(shard)] = handle
        return handle

    def _x_dataset(self, shard: int) -> h5py.Dataset:
        """Cache the ``X`` dataset object; creating one per read leaks HDF5 state."""
        dataset = self._x_datasets.get(int(shard))
        if dataset is None:
            dataset = self._handle(int(shard))["X"]
            self._x_datasets[int(shard)] = dataset
        return dataset

    def _crop_start(self) -> int:
        base = (600 - self.window_length) // 2
        if self.jitter is None:
            start = base
        else:
            low, high = self.jitter
            target_position = int(self._rng.integers(low, high + 1))
            start = CACHE_POSITION - target_position
        start += self.p_shift_samples
        return int(np.clip(start, 0, 600 - self.window_length))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        label = int(self.labels[index])
        shard = int(self.shards[index])
        position = int(self.shard_indices[index])
        waveform = np.asarray(self._x_dataset(shard)[position], dtype=np.float32)

        self._reads += 1
        if self._reads % 20_000 == 0:
            gc.collect()
            if self._trim is not None:
                self._trim(0)

        start = self._crop_start()
        window = waveform[start : start + self.window_length].copy()
        p_position = CACHE_POSITION - start

        if self.augment is not None:
            window, label = augment_trace(window, label, self._rng, self.augment)

        target = signed_field_target(self.window_length, p_position, label, self.sigma)
        return {
            "x": torch.from_numpy(window).unsqueeze(0),
            "target": torch.from_numpy(target),
            "label": torch.tensor(label, dtype=torch.long),
            "p_pick": torch.tensor(p_position, dtype=torch.long),
        }

    def close(self) -> None:
        self._x_datasets.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_x_datasets"] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown safety
            pass


def make_dataloader(
    dataset: UnifiedPolarityDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 4,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with worker-local RNG seeding for reproducible augmentation."""

    def worker_init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        dataset._rng = np.random.default_rng(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init if num_workers > 0 else None,
        drop_last=drop_last,
    )
