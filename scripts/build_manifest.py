#!/usr/bin/env python
"""Build the unified polarity manifest (metadata only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.manifest import (  # noqa: E402
    SOURCES,
    build_manifest,
    manifest_summary,
    save_manifest,
)
from sopnet.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/d/AI_Seismic_Data")
    parser.add_argument("--output", default="outputs/manifest_raw.parquet")
    parser.add_argument("--datasets", nargs="+", default=SOURCES)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    logger = get_logger("sopnet.manifest")
    manifest = build_manifest(
        Path(args.data_root),
        datasets=args.datasets,
        limit_per_source=args.limit_per_source,
        seed=args.seed,
    )
    path = save_manifest(manifest, Path(args.output))
    summary = manifest_summary(manifest)
    print(summary.to_string())
    summary.to_csv(Path(args.output).with_suffix(".summary.csv"))
    logger.info("manifest saved: %s (%d rows)", path, len(manifest))


if __name__ == "__main__":
    main()
