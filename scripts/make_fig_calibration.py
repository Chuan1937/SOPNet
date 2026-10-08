#!/usr/bin/env python
"""Figure 5: confidence calibration and selective prediction from saved predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import DOWN, UNKNOWN, UP  # noqa: E402


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int = 10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers, accuracies, fractions = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if not mask.any():
            continue
        centers.append((low + high) / 2)
        accuracies.append(correct[mask].mean())
        fractions.append(mask.mean())
    return np.array(centers), np.array(accuracies), np.array(fractions)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10):
    centers, accuracies, fractions = reliability(confidence, correct, bins)
    return float(np.sum(fractions * np.abs(accuracies - centers)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="outputs/paper/predictions/sopnet_test.npz")
    parser.add_argument("--metrics", default="outputs/runs/sopnet_nojitter_36/test_metrics.json")
    parser.add_argument("--output", default="outputs/paper/figures/fig5_calibration")
    parser.add_argument("--model-label", default="SOPNet")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.predictions, allow_pickle=True)
    labels = data["labels"]
    if "predictions" in data.files:
        predictions = data["predictions"]
    else:
        predictions = np.where(data["ud_score"] >= 0, UP, DOWN)
    confidence = data["confidence"].astype(np.float64)
    known = labels != UNKNOWN
    labels, predictions, confidence = labels[known], predictions[known], confidence[known]
    correct = (predictions == labels).astype(np.float64)

    # optional Platt calibration from the metrics file
    calibrated = None
    metrics = {}
    if args.metrics and Path(args.metrics).exists():
        metrics = json.loads(Path(args.metrics).read_text())
        calibration = metrics.get("calibration")
        if calibration:
            a, b = float(calibration["a"]), float(calibration["b"])
            logit = np.log(np.clip(confidence, 1e-6, 1 - 1e-6) / (1 - np.clip(confidence, 1e-6, 1 - 1e-6)))
            calibrated = 1.0 / (1.0 + np.exp(-(a * logit + b)))

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=200)

    centers, accuracies, _ = reliability(confidence, correct)
    ece_raw = expected_calibration_error(confidence, correct)
    ax1.plot([0, 1], [0, 1], color="#999999", linewidth=0.7, linestyle="--", label="ideal")
    ax1.plot(
        centers,
        accuracies,
        marker="o",
        markersize=3,
        linewidth=1.0,
        label=f"raw (ECE {ece_raw:.3f})",
    )
    if calibrated is not None:
        c_centers, c_acc, _ = reliability(calibrated, correct)
        ece_cal = expected_calibration_error(calibrated, correct)
        ax1.plot(
            c_centers,
            c_acc,
            marker="s",
            markersize=3,
            linewidth=1.0,
            label=f"Platt (ECE {ece_cal:.3f})",
        )
    ax1.set_xlabel("confidence")
    ax1.set_ylabel("accuracy")
    ax1.set_title(f"{args.model_label}: reliability (known U/D)", fontsize=7)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=5.6, frameon=False)

    thresholds = np.quantile(confidence, np.linspace(0.0, 0.95, 200))
    coverages, accuracies_sel = [], []
    for threshold in np.unique(thresholds):
        mask = confidence >= threshold
        if mask.sum() < 100:
            continue
        coverages.append(mask.mean())
        accuracies_sel.append(correct[mask].mean())
    ax2.plot(coverages, accuracies_sel, color="#7b2d26", linewidth=1.2)
    ax2.axhline(correct.mean(), color="#999999", linewidth=0.7, linestyle="--", label="all known")
    if metrics.get("threshold") is not None:
        operating = float(metrics["threshold"])
        mask = confidence >= operating
        if mask.any():
            ax2.plot(
                [mask.mean()],
                [correct[mask].mean()],
                marker="*",
                markersize=8,
                color="#1f4e79",
                label=f"$\\tau$={operating:g}",
            )
    ax2.set_xlabel("coverage (known U/D)")
    ax2.set_ylabel("accuracy")
    ax2.set_title(f"{args.model_label}: selective prediction", fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=5.6, frameon=False)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
