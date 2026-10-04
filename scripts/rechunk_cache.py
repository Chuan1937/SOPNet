#!/usr/bin/env python
"""Rewrite cache shards with contiguous, uncompressed X for fast random reads.

h5py's auto-chunking for compressed datasets produced chunks like (782, 19),
so reading a single 600-sample row decompressed ~1.9 MB. LZF saved almost
nothing (7.78 M x 600 x float32 = 18.7 GB raw), so contiguous storage is both
smaller to read and dramatically faster.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rechunk_shard(path: Path) -> None:
    temporary = path.with_suffix(".tmp.h5")
    with h5py.File(path, "r") as source, h5py.File(temporary, "w") as target:
        x = source["X"][:]
        target.create_dataset("X", data=x)
        for key in ("p_pick", "label", "sample_index", "waveform_hash"):
            target.create_dataset(key, data=source[key][:])
        for key, value in source.attrs.items():
            target.attrs[key] = value
    shutil.move(str(temporary), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--workers", type=int, default=1, help="reserved for future use")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    shards = sorted(cache_dir.glob("shard_*.h5"))
    if not shards:
        raise SystemExit(f"no shards found in {cache_dir}")
    for path in tqdm(shards, desc="rechunk", unit="shard"):
        rechunk_shard(path)
    print(f"rechunked {len(shards)} shards in {cache_dir}")


if __name__ == "__main__":
    main()
