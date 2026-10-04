#!/usr/bin/env python
"""Evaluate a trained checkpoint on the unified validation/test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import UNKNOWN  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.evaluation.calibration import expected_calibration_error, reliability_curve  # noqa: E402
from sopnet.evaluation.evaluate import (  # noqa: E402
    choose_threshold,
    collect_predictions,
    evaluate_field,
    plot_confusion_matrix,
    plot_prediction_examples,
    plot_reliability,
    save_metrics,
)
from sopnet.models import build_model  # noqa: E402
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint_path)
    config = payload.get("config", {})
    model = build_model(config)
    model.load_state_dict(payload["model_state"])

    output_dir = Path(args.output) if args.output else checkpoint_path.parent
    logger = get_logger("sopnet.evaluate", output_dir / "evaluate.log")
    task = config.get("train", {}).get("task", "field")
    sigma = float(config.get("dataset", {}).get("sigma", 10.0))
    window = int(config.get("dataset", {}).get("window_length", 400))

    device = args.device
    import torch

    model.to(torch.device(device) if torch.cuda.is_available() or device == "cpu" else torch.device("cpu"))

    dataset = UnifiedPolarityDataset(
        args.cache_dir, split=args.split, jitter=None, sigma=sigma, window_length=window
    )
    loader = make_dataloader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("evaluating %d samples on split=%s", len(dataset), args.split)

    threshold = args.threshold
    if task == "field":
        if args.auto_threshold:
            val_dataset = UnifiedPolarityDataset(
                args.cache_dir, split="val", jitter=None, sigma=sigma, window_length=window
            )
            val_loader = make_dataloader(
                val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
            )
            threshold = choose_threshold(model, val_loader, device)
            logger.info("auto threshold from validation: %.3f", threshold)
        metrics = evaluate_field(model, loader, device, threshold=threshold)
        outputs = collect_predictions(
            model,
            loader,
            device,
            task="field",
            max_plot_samples=max(1, args.examples) if args.examples else 0,
        )
        correct = outputs["predictions"] == outputs["labels"]
        metrics["ece"] = expected_calibration_error(outputs["confidence"], correct)
        metrics["reliability"] = {
            key: value.tolist() for key, value in reliability_curve(outputs["confidence"], correct).items()
        }
        if args.examples:
            plot_prediction_examples(outputs, output_dir / "examples", n=args.examples)
        import numpy as np

        if threshold is not None:
            final_predictions = np.where(outputs["confidence"] >= threshold, outputs["predictions"], UNKNOWN)
        else:
            final_predictions = outputs["predictions"]
        plot_confusion_matrix(outputs["labels"], final_predictions, output_dir / "fig_confusion_matrix")
        plot_reliability(outputs["confidence"], correct, output_dir / "fig_calibration")
    else:
        metrics = {}
        outputs = collect_predictions(model, loader, device, task="classify")
        metrics["accuracy"] = float((outputs["predictions"] == outputs["labels"]).mean())
        metrics["n"] = int(len(outputs["labels"]))

    save_metrics(metrics, output_dir / f"{args.split}_metrics.json")
    logger.info("metrics: %s", {k: v for k, v in metrics.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
