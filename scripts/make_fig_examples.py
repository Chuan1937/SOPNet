#!/usr/bin/env python
"""Figure 2: qualitative examples (waveform + signed field, correct and hard cases)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.canonical import DOWN, UNKNOWN, UP  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.models import build_model  # noqa: E402
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402

LABEL_TEXT = {UP: "up", DOWN: "down", UNKNOWN: "unknown"}


def collect(model, loader, device):
    xs, fields, labels, confidence = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            field = model(x).cpu()
            magnitude = field.abs().squeeze(1)
            peak, _ = magnitude.max(dim=-1)
            xs.append(x.cpu())
            fields.append(field)
            labels.append(batch["label"])
            confidence.append(peak)
    return (
        torch.cat(xs),
        torch.cat(fields),
        torch.cat(labels).numpy(),
        torch.cat(confidence).numpy(),
    )


def pick_examples(labels, confidence, predictions):
    known = labels != UNKNOWN
    correct = (predictions == labels) & known
    picks = []
    # one confident up, one confident down
    for label, name in ((UP, "up"), (DOWN, "down")):
        mask = correct & (labels == label) & (confidence >= 0.9)
        if mask.any():
            picks.append(int(np.argmax(np.where(mask, confidence, -1))))
    # one low-confidence correct (hard but right), avoiding degenerate near-zero fields
    mask = correct & (confidence > 0.1) & (confidence < 0.65)
    if mask.any():
        picks.append(int(np.argmax(np.where(mask, confidence, -1))))
    # one error if present
    errors = known & ~correct
    if errors.any():
        picks.append(int(np.argmax(np.where(errors, confidence, -1))))
    return picks[:4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/runs/sopnet_nojitter_36/best.pt")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output", default="outputs/paper/figures/fig2_examples")
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = load_checkpoint(Path(args.checkpoint))
    config = payload.get("config", {})
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    window = int(config.get("dataset", {}).get("window_length", 400))
    sigma = float(config.get("dataset", {}).get("sigma", 10.0))

    dataset = UnifiedPolarityDataset(
        args.cache_dir,
        split="test",
        jitter=None,
        sigma=sigma,
        window_length=window,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    loader = make_dataloader(dataset, args.batch_size, shuffle=False, num_workers=0)
    xs, fields, labels, confidence = collect(model, loader, device)
    dataset.close()

    field_np = fields.squeeze(1).numpy()
    predictions = np.where(field_np[np.arange(len(field_np)), np.abs(field_np).argmax(1)] >= 0, UP, DOWN)
    picks = pick_examples(labels, confidence, predictions)
    if not picks:
        raise SystemExit("no examples found; increase --max-samples")

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, len(picks), figsize=(1.9 * len(picks), 2.6), dpi=200, sharex="col")
    if len(picks) == 1:
        axes = axes.reshape(2, 1)
    t = np.arange(window)
    for column, index in enumerate(picks):
        trace = xs[index, 0].numpy()
        field = field_np[index]
        label = labels[index]
        pred = predictions[index]
        conf = confidence[index]
        correct = label == pred

        ax_top = axes[0, column]
        ax_top.plot(t, trace, color="#1f4e79", linewidth=0.6)
        ax_top.axvline(window / 2, color="#cc4444", linestyle="--", linewidth=0.6)
        ax_top.set_title(
            f"true {LABEL_TEXT[label]} / pred {LABEL_TEXT[pred]}\nconfidence {conf:.2f}",
            fontsize=6.4,
            color="#1a7a1a" if correct else "#aa2222",
        )
        ax_top.set_yticks([])
        ax_top.tick_params(axis="x", labelsize=5.5)
        if column == 0:
            ax_top.set_ylabel("amplitude", fontsize=6.2)

        ax_bottom = axes[1, column]
        ax_bottom.plot(t, field, color="#7b2d26", linewidth=0.8)
        ax_bottom.fill_between(t, 0, field, where=field >= 0, color="#e8a5a5", alpha=0.6)
        ax_bottom.fill_between(t, 0, field, where=field < 0, color="#a5c8e8", alpha=0.6)
        ax_bottom.axvline(window / 2, color="#cc4444", linestyle="--", linewidth=0.6)
        ax_bottom.axhline(0, color="#bbbbbb", linewidth=0.4)
        ax_bottom.set_ylim(-1.05, 1.05)
        peak = int(np.argmax(np.abs(field)))
        ax_bottom.plot([peak], [field[peak]], marker="o", markersize=2.4, color="#333333")
        ax_bottom.set_xlabel("sample", fontsize=6.2)
        ax_bottom.tick_params(axis="y", labelsize=5.5)
        ax_bottom.tick_params(axis="x", labelsize=5.5)
        if column == 0:
            ax_bottom.set_ylabel("signed field $f(t)$", fontsize=6.2)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "and", out.with_suffix(".png"))
    print("examples:", [int(i) for i in picks], "labels:", [int(labels[i]) for i in picks])


if __name__ == "__main__":
    main()
