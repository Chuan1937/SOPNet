#!/usr/bin/env python
"""Run the A-E ablation matrix by invoking train.py per experiment config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "A_cls": "configs/experiments/classification.yaml",
    "B_field": "configs/experiments/signed_field.yaml",
    "C_jitter": "configs/experiments/signed_field_jitter.yaml",
    "D_polarity": "configs/experiments/signed_field_polarity.yaml",
    "E_full": "configs/experiments/sopnet_full.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output-root", default="outputs/ablation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--output-csv", default="outputs/paper/tables/ablation.csv")
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = []
    for experiment in args.experiments:
        config_path = REPO / EXPERIMENTS[experiment]
        run_dir = root / experiment
        command = [
            sys.executable,
            str(REPO / "scripts/train.py"),
            "--config",
            str(config_path),
            "--cache-dir",
            args.cache_dir,
            "--output",
            str(run_dir),
            "--seed",
            str(args.seed),
        ]
        if args.epochs:
            command += ["--epochs", str(args.epochs)]
        if args.limit_train:
            command += ["--limit-train", str(args.limit_train)]
        if args.limit_val:
            command += ["--limit-val", str(args.limit_val)]
        if args.batch_size:
            command += ["--batch-size", str(args.batch_size)]
        print("running:", " ".join(command))
        subprocess.run(command, check=True)

        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            summary = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "experiment": experiment,
                    "config": EXPERIMENTS[experiment],
                    "best_epoch": summary.get("best_epoch"),
                    "monitor": summary.get("monitor"),
                    "best_metric": summary.get("best_metric"),
                    "parameters": summary.get("parameters"),
                    "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(pd.DataFrame(rows).to_string())
    print("ablation table written to", output_csv)


if __name__ == "__main__":
    main()
