#!/usr/bin/env python
"""Evaluate a trained checkpoint on the unified validation/test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import DOWN, UNKNOWN, UP  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.evaluation.calibration import (  # noqa: E402
    apply_platt_scaling,
    expected_calibration_error,
    fit_platt_scaling,
    reliability_curve,
)
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
from sopnet.training.metrics import binary_metrics, macro_f1  # noqa: E402
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
    parser.add_argument(
        "--save-predictions",
        default=None,
        help="write per-sample predictions to this .npz path",
    )
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
    if task in ("field", "field_multi"):
        val_dataset = UnifiedPolarityDataset(
            args.cache_dir, split="val", jitter=None, sigma=sigma, window_length=window
        )
        val_loader = make_dataloader(
            val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        if args.auto_threshold:
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
        known = outputs["labels"] != UNKNOWN
        correct = outputs["predictions"] == outputs["labels"]

        if task == "field_multi":
            # The polarity head already emits softmax probabilities.
            calibrated = outputs["confidence"]
            metrics["calibration"] = {"type": "softmax", "n_eval": int(known.sum())}
        else:
            # The field model emits an unnormalised peak magnitude, not a
            # probability. Fit Platt scaling on validation and report ECE on
            # the calibrated probability over labelled U/D samples only.
            val_outputs = collect_predictions(model, val_loader, device, task="field")
            val_known = val_outputs["labels"] != UNKNOWN
            parameters = fit_platt_scaling(
                val_outputs["confidence"][val_known],
                val_outputs["predictions"][val_known] == val_outputs["labels"][val_known],
            )
            calibrated = apply_platt_scaling(outputs["confidence"], parameters)
            metrics["calibration"] = {
                **parameters,
                "n_fit": int(val_known.sum()),
                "n_eval": int(known.sum()),
            }
        metrics["ece"] = expected_calibration_error(calibrated[known], correct[known])
        metrics["reliability"] = {
            key: value.tolist() for key, value in reliability_curve(calibrated[known], correct[known]).items()
        }
        logger.info("calibrated ECE (known U/D only): %.4f", metrics["ece"])
        if args.examples:
            plot_prediction_examples(outputs, output_dir / "examples", n=args.examples)
        import numpy as np

        if threshold is not None:
            final_predictions = np.where(outputs["confidence"] >= threshold, outputs["predictions"], UNKNOWN)
        else:
            final_predictions = outputs["predictions"]
        plot_confusion_matrix(outputs["labels"], final_predictions, output_dir / "fig_confusion_matrix")
        plot_reliability(calibrated[known], correct[known], output_dir / "fig_calibration")
    else:
        metrics = {}
        if task == "classify_ud":
            outputs = collect_predictions(model, loader, device, task="classify_ud")
            known = outputs["labels"] != UNKNOWN
            metrics = {"n": int(len(outputs["labels"])), "n_known": int(known.sum())}
            if known.any():
                metrics.update(binary_metrics(outputs["labels"][known], outputs["predictions"][known]))
                metrics["macro_f1_ud"] = macro_f1(
                    outputs["labels"][known], outputs["predictions"][known], labels=(DOWN, UP)
                )
                correct = outputs["predictions"] == outputs["labels"]
                metrics["ece"] = expected_calibration_error(outputs["confidence"][known], correct[known])
                metrics["reliability"] = {
                    key: value.tolist()
                    for key, value in reliability_curve(outputs["confidence"][known], correct[known]).items()
                }
            plot_confusion_matrix(
                outputs["labels"], outputs["predictions"], output_dir / "fig_confusion_matrix"
            )
        else:
            outputs = collect_predictions(model, loader, device, task="classify")
            metrics["accuracy"] = float((outputs["predictions"] == outputs["labels"]).mean())
            metrics["n"] = int(len(outputs["labels"]))

    save_metrics(metrics, output_dir / f"{args.split}_metrics.json")
    logger.info("metrics: %s", {k: v for k, v in metrics.items() if not isinstance(v, dict)})

    if args.save_predictions and "labels" in outputs:
        import numpy as np

        save_path = Path(args.save_predictions)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path,
            sample_ids=outputs.get("sample_ids", np.array([])),
            labels=outputs["labels"],
            predictions=outputs["predictions"],
            confidence=outputs["confidence"],
            p_pred=outputs.get("p_pred", np.array([])),
        )
        logger.info("predictions saved to %s", save_path)


if __name__ == "__main__":
    main()
