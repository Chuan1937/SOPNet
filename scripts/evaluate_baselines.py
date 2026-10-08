#!/usr/bin/env python
"""Re-evaluate existing baseline checkpoints with the strict U/D protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.training.baselines import BASELINE_SPECS, evaluate_baseline  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--window-length", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--predictions-dir", default="outputs/paper/predictions")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    args = parser.parse_args()

    logger = get_logger("sopnet.baselines")
    for name in args.baselines:
        run_dir = Path(args.output_root) / f"baseline_{name}"
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            logger.warning("skip %s: no checkpoint at %s", name, checkpoint)
            continue
        spec = BASELINE_SPECS[name]
        predictions_dir = Path(args.predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            metrics = evaluate_baseline(
                name,
                checkpoint,
                Path(args.cache_dir),
                split=split,
                num_workers=args.num_workers,
                window_length=args.window_length,
                save_predictions=predictions_dir / f"{name}_{split}.npz",
            )
            (run_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            logger.info("%s %s metrics: %s", name, split, metrics)

        protocol_path = run_dir / "protocol.json"
        if not protocol_path.exists():
            summary = {}
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                summary = json.loads(metrics_path.read_text())
            protocol = {
                "name": name,
                "factory": spec.factory,
                "input_length": spec.input_length,
                "input_channels": spec.input_channels,
                "output": spec.output,
                "derivative": spec.derivative,
                "window_length": args.window_length,
                "parameters": summary.get("parameters"),
                "checkpoint": str(checkpoint),
                "note": "re-evaluated with the strict U/D protocol; training config in config.json",
            }
            protocol_path.write_text(json.dumps(protocol, indent=2, default=str))


if __name__ == "__main__":
    main()
