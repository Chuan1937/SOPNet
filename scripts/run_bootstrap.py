#!/usr/bin/env python
"""Paired bootstrap of SOPNet vs every trained baseline on the fixed test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import DOWN, UNKNOWN, UP  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.evaluation.bootstrap import paired_bootstrap_difference  # noqa: E402
from sopnet.evaluation.evaluate import collect_predictions  # noqa: E402
from sopnet.models import build_model  # noqa: E402
from sopnet.training.baselines import collect_baseline_predictions  # noqa: E402
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402
from sopnet.training.metrics import macro_f1  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SOPNet checkpoint")
    parser.add_argument("--baselines", nargs="+", default=None)
    parser.add_argument("--baseline-root", default="outputs/runs")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", default="outputs/paper/tables/bootstrap.csv")
    args = parser.parse_args()

    logger = get_logger("sopnet.bootstrap")
    payload = load_checkpoint(Path(args.checkpoint))
    config = payload.get("config", {})
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    device = args.device
    import torch

    model.to(torch.device(device) if torch.cuda.is_available() or device == "cpu" else torch.device("cpu"))
    task = config.get("train", {}).get("task", "field")
    sigma = float(config.get("dataset", {}).get("sigma", 10.0))
    window = int(config.get("dataset", {}).get("window_length", 400))

    dataset = UnifiedPolarityDataset(
        args.cache_dir, split=args.split, jitter=None, sigma=sigma, window_length=window
    )
    loader = make_dataloader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    outputs = collect_predictions(model, loader, device, task=task)
    known = outputs["labels"] != UNKNOWN
    y_true = outputs["labels"][known]
    sopnet_pred = outputs["predictions"][known]

    def score(y: np.ndarray, p: np.ndarray) -> float:
        return macro_f1(y, p, labels=(DOWN, UP))

    sopnet_f1 = score(y_true, sopnet_pred)
    logger.info("SOPNet macro-F1 (U/D, n=%d): %.4f", len(y_true), sopnet_f1)

    if args.baselines:
        names = list(args.baselines)
    else:
        names = sorted(
            path.name.replace("baseline_", "")
            for path in Path(args.baseline_root).glob("baseline_*")
            if (path / "best.pt").exists()
        )
    if not names:
        raise SystemExit("no baseline checkpoints found")

    rows = []
    for name in names:
        checkpoint = Path(args.baseline_root) / f"baseline_{name}" / "best.pt"
        baseline = collect_baseline_predictions(
            name,
            checkpoint,
            Path(args.cache_dir),
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        if not np.array_equal(baseline["labels"], outputs["labels"]):
            raise RuntimeError(f"{name}: baseline predictions are not aligned with SOPNet")
        baseline_pred = baseline["predictions"][known]
        result = paired_bootstrap_difference(
            y_true, sopnet_pred, baseline_pred, metric=score, n_resamples=args.n_resamples
        )
        row = {
            "baseline": name,
            "sopnet_macro_f1": sopnet_f1,
            "baseline_macro_f1": score(y_true, baseline_pred),
            **result,
        }
        rows.append(row)
        logger.info(
            "%s: delta=%+.4f [%+.4f, %+.4f], p(improvement)=%.3f",
            name,
            row["delta"],
            row["ci_low"],
            row["ci_high"],
            row["p_improvement"],
        )

    frame = pd.DataFrame(rows).sort_values("baseline_macro_f1", ascending=False)
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print("bootstrap table written to", output)


if __name__ == "__main__":
    main()
