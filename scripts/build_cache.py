#!/usr/bin/env python
"""Build the sharded 600-sample HDF5 cache and finalise leakage-safe splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.cache import CacheConfig, build_cache  # noqa: E402
from sopnet.data.manifest import build_manifest, load_manifest  # noqa: E402
from sopnet.data.preprocess import PreprocessConfig  # noqa: E402
from sopnet.data.split import check_split_leakage  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/d/AI_Seismic_Data")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--manifest", default="outputs/manifest_raw.parquet")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--limit", type=int, default=None, help="cap total cached samples")
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=45.0)
    args = parser.parse_args()

    logger = get_logger("sopnet.cache")
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        logger.info("loading manifest %s", manifest_path)
        manifest = load_manifest(manifest_path)
    else:
        logger.info("manifest missing, building it first")
        manifest = build_manifest(
            Path(args.data_root),
            limit_per_source=args.limit_per_source,
            seed=args.seed,
        )

    preprocess = PreprocessConfig(
        fs=args.fs,
        lowcut=args.lowcut,
        highcut=args.highcut,
        cache_length=600,
        p_position=300,
    )
    config = CacheConfig(
        data_root=Path(args.data_root),
        cache_dir=Path(args.cache_dir),
        shard_size=args.shard_size,
        chunk_size=args.chunk_size,
        workers=args.workers,
        seed=args.seed,
        preprocess=preprocess,
    )
    manifest_out, index = build_cache(manifest, config, limit=args.limit, logger=logger)
    report = check_split_leakage(manifest_out)
    logger.info("split leakage check: %s", report)
    logger.info("split sizes:\n%s", manifest_out["split"].value_counts().to_string())
    logger.info(
        "label x split:\n%s",
        manifest_out.groupby(["split", "canonical_label"], observed=True).size().to_string(),
    )
    logger.info("cache index: %d rows in %s", len(index), args.cache_dir)


if __name__ == "__main__":
    main()
