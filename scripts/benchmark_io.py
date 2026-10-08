#!/usr/bin/env python
"""Benchmark raw /mnt read throughput and cache DataLoader throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.cache import load_cache  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.data.manifest import SOURCE_PATHS, SourceReader  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def timed(label: str, function) -> float:
    started = time.perf_counter()
    value = function()
    elapsed = time.perf_counter() - started
    return value, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/d/AI_Seismic_Data")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--sequential", type=int, default=20_000)
    parser.add_argument("--random", type=int, default=2_000)
    parser.add_argument("--output", default="outputs/benchmark_io.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    import h5py

    logger = get_logger("sopnet.benchmark")
    root = Path(args.data_root)
    results: dict = {}
    scsn_path = root / SOURCE_PATHS["scsn"]["flat"]

    with h5py.File(scsn_path, "r") as handle:
        n = min(args.sequential, handle["X"].shape[0])
        _, sequential_time = timed("sequential", lambda: handle["X"][:n])
        results["scsn_sequential_samples_per_s"] = n / sequential_time
        logger.info("SCSN sequential: %.0f samples/s", n / sequential_time)

        rng = np.random.default_rng(0)
        count = min(args.random, handle["X"].shape[0])
        indices = np.sort(rng.choice(handle["X"].shape[0], count, replace=False))

        def read_random():
            for index in indices:
                handle["X"][int(index)]

        if count <= 20_000:
            _, random_time = timed("random", read_random)
            results["scsn_random_samples_per_s"] = count / random_time
            logger.info("SCSN random: %.0f samples/s", count / random_time)

    manifest, index = (None, None)
    cache_dir = Path(args.cache_dir)
    if (cache_dir / "manifest.parquet").exists():
        manifest, index = load_cache(cache_dir)
        dataset = UnifiedPolarityDataset(cache_dir, split="train", manifest=manifest, index=index)
        if len(dataset) > 0:
            for workers in {4, args.workers, 12}:
                loader = make_dataloader(
                    dataset, args.batch_size, shuffle=True, num_workers=workers, prefetch_factor=4
                )
                batches = 0
                samples = 0
                started = time.perf_counter()
                for batch in loader:
                    batches += 1
                    samples += batch["x"].shape[0]
                    if batches >= 40:
                        break
                elapsed = time.perf_counter() - started
                results[f"cache_workers_{workers}_samples_per_s"] = samples / elapsed
                logger.info("cache DataLoader workers=%d: %.0f samples/s", workers, samples / elapsed)

    raw_samples = []
    if manifest is not None:
        for source in ["txed", "instance", "pnw", "diting"]:
            subset = manifest[manifest["source"] == source]
            if len(subset) == 0:
                continue
            subset = subset.sample(min(500, len(subset)), random_state=0)
            reader = SourceReader(source, root)
            started = time.perf_counter()
            for _, row in subset.iterrows():
                reader.read(row)
            elapsed = time.perf_counter() - started
            reader.close()
            results[f"{source}_raw_samples_per_s"] = len(subset) / elapsed
            raw_samples.append(len(subset) / elapsed)
            logger.info("%s raw: %.0f samples/s", source, len(subset) / elapsed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("benchmark written to %s", output)


if __name__ == "__main__":
    main()
