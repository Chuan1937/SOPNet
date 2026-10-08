#!/usr/bin/env python
"""Aggregate per-run metrics into paper-ready tables and a summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    summary = json.loads(metrics_path.read_text())
    row = {
        "run": run_dir.name,
        "path": str(run_dir),
        "parameters": summary.get("parameters"),
        "peak_gpu_memory_gb": summary.get("peak_gpu_memory_gb"),
        "best_epoch": summary.get("best_epoch"),
        "epochs_run": summary.get("epochs_run") or len(summary.get("history", [])),
    }
    if "baseline" in summary:
        row["name"] = summary["baseline"]
        row["best_val_f1"] = summary.get("best_f1")
    else:
        row["name"] = run_dir.name
        row["best_metric"] = summary.get("best_metric")
        row["monitor"] = summary.get("monitor")
    for extra in ("test_metrics.json", "val_metrics.json"):
        path = run_dir / extra
        if path.exists():
            values = json.loads(path.read_text())
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    row[f"{extra.split('_')[0]}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", nargs="+", default=["outputs/runs", "outputs/ablation", "outputs/smoke"]
    )
    parser.add_argument("--output-dir", default="outputs/paper")
    parser.add_argument("--patterns", nargs="+", default=["*/metrics.json", "*/*/metrics.json"])
    args = parser.parse_args()

    run_dirs = set()
    for root in args.runs_root:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in args.patterns:
            for metrics in root_path.glob(pattern):
                run_dirs.add(metrics.parent)

    rows = [row for row in (load_run(path) for path in sorted(run_dirs)) if row]
    tables_dir = Path(args.output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "main_results.csv", index=False)

    complexity_columns = [
        c for c in ("name", "run", "parameters", "peak_gpu_memory_gb", "epochs_run") if c in frame.columns
    ]
    if complexity_columns:
        frame[complexity_columns].to_csv(tables_dir / "complexity.csv", index=False)

    lines = ["# SOPNet experiment summary", "", f"{len(frame)} runs", ""]
    if len(frame):
        lines.append(frame.to_markdown(index=False))
    (Path(args.output_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(frame.to_string())
    print("tables written to", tables_dir)


if __name__ == "__main__":
    main()
