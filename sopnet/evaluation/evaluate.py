"""Model evaluation, confidence-thresholded unknown handling, and example plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.training.engine import CLASS_ORDER
from sopnet.training.metrics import (
    best_threshold,
    binary_metrics,
    macro_f1,
    p_error_metrics,
)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: str | torch.device = "cuda",
    task: str = "field",
    max_plot_samples: int = 0,
) -> Dict[str, np.ndarray]:
    """Run inference and return all arrays needed for metrics and figures."""
    device = torch.device(device)
    model.eval()
    labels, predictions, confidences = [], [], []
    p_pred, p_true = [], []
    keep = {"x": [], "target": [], "field": []}
    kept = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        label = batch["label"].numpy()
        if task == "classify":
            logits = model(x)
            pred_class = logits.argmax(dim=-1).cpu().numpy()
            pred = np.array([CLASS_ORDER[c] for c in pred_class])
            confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
        else:
            field = model(x)
            magnitude = field.abs()
            confidence_t, position = magnitude.max(dim=-1)
            signed = field.gather(-1, position.unsqueeze(1)).squeeze(1).squeeze(-1)
            pred = torch.sign(signed).cpu().numpy()
            confidence = confidence_t.squeeze(-1).cpu().numpy()
            position = position.squeeze(-1)
            pred[confidence == 0] = 0
            p_pred.append(position.cpu().numpy())
            p_true.append(batch["p_pick"].numpy())
        labels.append(label)
        predictions.append(pred)
        confidences.append(confidence)

        if max_plot_samples and kept < max_plot_samples:
            needed = min(max_plot_samples - kept, x.shape[0])
            keep["x"].append(x[:needed].cpu().numpy())
            keep["target"].append(batch["target"][:needed].numpy())
            if task == "field":
                keep["field"].append(field[:needed].cpu().numpy())
            kept += needed

    result: Dict[str, np.ndarray] = {
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(predictions),
        "confidence": np.concatenate(confidences),
    }
    if p_pred:
        result["p_pred"] = np.concatenate(p_pred)
        result["p_true"] = np.concatenate(p_true)
    for key, values in keep.items():
        if values:
            result[key] = np.concatenate(values)
    return result


def evaluate_field(
    model: torch.nn.Module,
    loader,
    device: str | torch.device = "cuda",
    threshold: Optional[float] = None,
    fs: int = 100,
) -> Dict[str, float]:
    """Full evaluation: known-subset U/D metrics, optional U/D/X metrics, P errors."""
    outputs = collect_predictions(model, loader, device, task="field")
    labels = outputs["labels"]
    predictions = outputs["predictions"]
    confidence = outputs["confidence"]
    known = labels != UNKNOWN

    metrics: Dict[str, float] = {"n": int(labels.size), "n_known": int(known.sum())}
    if known.any():
        metrics.update(
            {f"known_{k}": v for k, v in binary_metrics(labels[known], predictions[known]).items()}
        )
        metrics["macro_f1_ud"] = macro_f1(labels[known], predictions[known], labels=(DOWN, UP))
    if "p_pred" in outputs:
        metrics.update(
            {f"p_{k}": v for k, v in p_error_metrics(outputs["p_pred"], outputs["p_true"], fs=fs).items()}
        )

    if threshold is not None:
        pred_with_unknown = np.where(confidence >= threshold, predictions, UNKNOWN)
        metrics["threshold"] = float(threshold)
        metrics["macro_f1_udx"] = macro_f1(labels, pred_with_unknown)
        metrics["macro_f1_ud"] = macro_f1(labels[known], pred_with_unknown[known], labels=(DOWN, UP))
        metrics["coverage"] = float(np.mean(confidence >= threshold))
        covered = confidence >= threshold
        if covered.any():
            metrics["covered_accuracy"] = float(np.mean(labels[covered] == pred_with_unknown[covered]))
        # Selective prediction restricted to labelled U/D samples: X samples have
        # no ground-truth polarity, so mixing them into "covered accuracy" makes
        # the number meaningless.
        covered_known = covered & known
        metrics["coverage_known"] = float(covered_known.sum() / max(1, int(known.sum())))
        if covered_known.any():
            metrics["covered_accuracy_known"] = float(
                np.mean(labels[covered_known] == pred_with_unknown[covered_known])
            )
    return metrics


def choose_threshold(model: torch.nn.Module, val_loader, device, fs: int = 100) -> float:
    """Select the field-confidence threshold that maximises U/D/X macro-F1 on validation."""
    outputs = collect_predictions(model, val_loader, device, task="field")
    return best_threshold(outputs["confidence"], outputs["labels"], outputs["predictions"])


def save_metrics(metrics: Dict[str, float], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    labels=(-1, 0, 1),
    names=("Down", "Unknown", "Up"),
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = len(labels)
    matrix = np.zeros((size, size), dtype=np.int64)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = int(np.sum((y_true == true_label) & (y_pred == pred_label)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.2, 2.9), dpi=150)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(size), names, fontsize=7)
    axis.set_yticks(range(size), names, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for i in range(size):
        for j in range(size):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)
    return path


def plot_reliability(
    confidence: np.ndarray,
    correct: np.ndarray,
    path: Path,
    n_bins: int = 10,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sopnet.evaluation.calibration import reliability_curve

    curve = reliability_curve(confidence, correct, n_bins=n_bins)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.2, 2.9), dpi=150)
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.6")
    valid = ~np.isnan(curve["accuracy"])
    axis.plot(curve["bin_centre"][valid], curve["accuracy"][valid], marker="o", linewidth=1.4)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)
    return path


def plot_prediction_examples(
    outputs: Dict[str, np.ndarray],
    path: Path,
    n: int = 12,
    fs: int = 100,
) -> Path:
    """Plot waveform + target field + predicted field for random samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    available = outputs["x"].shape[0]
    n = min(n, available)
    if n == 0:
        raise ValueError("no samples available for plotting")
    indices = np.linspace(0, available - 1, n, dtype=int)
    columns = 3
    rows = int(np.ceil(n / columns))

    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 2.6 * rows), squeeze=False)
    time = np.arange(outputs["x"].shape[-1]) / fs
    for slot, index in enumerate(indices):
        axis = axes[slot // columns][slot % columns]
        waveform = outputs["x"][index, 0]
        target = outputs["target"][index]
        field = outputs["field"][index, 0]
        true_label = int(outputs["labels"][index])
        pred_label = int(outputs["predictions"][index])
        confidence = float(outputs["confidence"][index])
        p_true = int(outputs["p_true"][index])
        p_pred = int(outputs["p_pred"][index])

        axis.plot(time, waveform, color="0.55", linewidth=0.8, label="waveform")
        axis.plot(time, target, color="tab:green", linewidth=1.0, label="target")
        axis.plot(time, field, color="tab:red", linestyle="--", linewidth=1.0, label="prediction")
        axis.axvline(p_true / fs, color="tab:blue", linewidth=0.8, alpha=0.7)
        axis.axvline(p_pred / fs, color="tab:orange", linewidth=0.8, linestyle=":")
        axis.set_title(
            f"true={'UDX'[1 - true_label] if true_label else 'X'} "
            f"pred={'UDX'[1 - pred_label] if pred_label else 'X'} "
            f"conf={confidence:.2f} | dt={abs(p_pred - p_true)} smp",
            fontsize=8,
        )
        axis.set_xlabel("time (s)", fontsize=7)
        axis.tick_params(labelsize=7)
        if slot == 0:
            axis.legend(fontsize=6, loc="upper left")

    for slot in range(n, rows * columns):
        axes[slot // columns][slot % columns].axis("off")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=600)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path
