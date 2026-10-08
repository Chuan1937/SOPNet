#!/usr/bin/env python
"""Assemble the G1 paper tables: main results, per-source results and baseline protocols."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import DOWN, UP  # noqa: E402
from sopnet.training.metrics import binary_metrics, macro_f1  # noqa: E402

MODEL_ORDER = ["ross", "diting_motion", "rpnet", "eqpolarity", "cfm", "sopnet"]
LABELS = {
    "ross": "Ross",
    "diting_motion": "DiTingMotion",
    "rpnet": "RPNet",
    "eqpolarity": "EQPolarity",
    "cfm": "CFM",
    "sopnet": "SOPNet",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def standard_test_metrics(name: str, run_dir: Path) -> dict:
    metrics = _load_json(run_dir / "test_metrics.json")
    if not metrics:
        return {}
    if name == "sopnet":
        row = {
            "accuracy": metrics.get("known_accuracy"),
            "macro_f1": metrics.get("macro_f1_ud"),
            "precision": metrics.get("known_precision"),
            "recall": metrics.get("known_recall"),
            "mcc": metrics.get("known_mcc"),
            "n": metrics.get("n"),
            "n_known": metrics.get("n_known"),
            # fraction of all test samples below the validation-tuned threshold
            "unknown_rate": (1.0 - metrics["coverage"] if metrics.get("coverage") is not None else None),
        }
    else:
        row = {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1_ud"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mcc": metrics.get("mcc"),
            "n": metrics.get("n"),
            "n_known": metrics.get("n_known"),
            "unknown_rate": metrics.get("unknown_rate"),
        }
    row["model"] = LABELS[name]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="outputs/runs")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--predictions-dir", default="outputs/paper/predictions")
    parser.add_argument("--output-dir", default="outputs/paper/tables")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(Path(args.cache_dir) / "manifest.parquet", columns=["sample_id", "source"])
    source_map = dict(zip(manifest["sample_id"], manifest["source"]))

    main_rows, source_rows, protocol_rows = [], [], []
    for name in MODEL_ORDER:
        if name == "sopnet":
            run_dir = runs_root / "sopnet_nojitter_36"
            proto = _load_json(run_dir / "metrics.json")
            protocol = {
                "model": "SOPNet",
                "input_length": 400,
                "input_channels": 1,
                "output": "signed field",
                "derivative": False,
                "window_length": 400,
                "epochs_run": proto.get("epochs_run"),
                "best_epoch": proto.get("best_epoch"),
                "parameters": proto.get("parameters"),
                "checkpoint": str(run_dir / "best.pt"),
            }
        else:
            run_dir = runs_root / f"baseline_{name}"
            protocol = _load_json(run_dir / "protocol.json")
            if protocol:
                protocol["model"] = LABELS[name]
        row = standard_test_metrics(name, run_dir)
        if row:
            main_rows.append(row)
        if protocol:
            protocol_rows.append(protocol)

        pred_path = predictions_dir / f"{name}_test.npz"
        if not pred_path.exists():
            continue
        data = np.load(pred_path, allow_pickle=True)
        if "sample_ids" not in data.files:
            continue
        labels = data["labels"]
        if "ud_score" in data:
            predictions = np.where(data["ud_score"] >= 0, UP, DOWN)
        else:
            predictions = data["predictions"]
        ids = data["sample_ids"]
        sources = np.array([source_map[i] for i in ids])
        known = labels != 0
        for source in np.unique(sources):
            mask = known & (sources == source)
            if mask.sum() == 0:
                continue
            metrics = binary_metrics(labels[mask], predictions[mask])
            source_rows.append(
                {
                    "model": LABELS[name],
                    "source": source,
                    "n_known": int(mask.sum()),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": macro_f1(labels[mask], predictions[mask], labels=(DOWN, UP)),
                    "mcc": metrics["mcc"],
                }
            )

    main_columns = [
        "model",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "mcc",
        "n_known",
        "n",
        "unknown_rate",
    ]
    main = pd.DataFrame(main_rows)[main_columns]
    main.to_csv(output_dir / "main_results.csv", index=False)

    if source_rows:
        pd.DataFrame(source_rows).to_csv(output_dir / "per_source_results.csv", index=False)
    if protocol_rows:
        pd.DataFrame(protocol_rows).to_csv(output_dir / "baseline_protocol.csv", index=False)

    print(main.to_string(index=False))
    if source_rows:
        print()
        print(pd.DataFrame(source_rows).to_string(index=False))
    print("\ntables written to", output_dir)


if __name__ == "__main__":
    main()
