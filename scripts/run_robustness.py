#!/usr/bin/env python
"""Six-model robustness (P-shift, SNR) on a fixed unified-test subset.

Each ``--models name=checkpoint`` entry is either a SeisPolarity baseline
(``name`` in ``BASELINE_SPECS``) or a SOPNet signed-field checkpoint. All
models are evaluated on the same ``--subset-size`` samples drawn with
``--subset-seed`` and receive the same SNR / shift grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.evaluation.robustness import evaluate_noise, evaluate_p_shift  # noqa: E402
from sopnet.models import build_model  # noqa: E402
from sopnet.training.baselines import (  # noqa: E402
    BASELINE_SPECS,
    BaselineWrapper,
    _ud_score,
    build_baseline,
)
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


class FieldAdapter:
    """SOPNet signed-field checkpoint as a ``(ud_score, confidence)`` callable."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __call__(self, x: torch.Tensor):
        field = self.model(x)
        magnitude = field.abs()
        peak, position = magnitude.max(dim=-1)
        signed = field.gather(-1, position.unsqueeze(1)).squeeze(1).squeeze(-1)
        return signed, peak.squeeze(-1)


class BaselineAdapter:
    """SeisPolarity baseline as a ``(ud_score, confidence)`` callable."""

    def __init__(self, name: str, wrapper: torch.nn.Module):
        self.name = name
        self.spec = BASELINE_SPECS[name]
        self.wrapper = wrapper

    def __call__(self, x: torch.Tensor):
        output = self.wrapper(x)
        confidence = torch.softmax(output, dim=-1).max(dim=-1).values
        return _ud_score(self.spec, output), confidence


def load_model(name: str, path: Path, device: torch.device):
    if name in BASELINE_SPECS:
        spec = BASELINE_SPECS[name]
        model = BaselineWrapper(build_baseline(name), spec)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.to(device).eval()
        return BaselineAdapter(name, model), 600
    payload = load_checkpoint(path)
    config = payload.get("config", {})
    task = config.get("train", {}).get("task", "field")
    if task not in ("field", "field_multi"):
        raise ValueError(f"{name}: only signed-field SOPNet checkpoints are supported")
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    window = int(config.get("dataset", {}).get("window_length", 400))
    return FieldAdapter(model), window


def plot_models(frame: pd.DataFrame, x_column: str, path: Path, xlabel: str, categorical=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.6, 3.0), dpi=150)
    for model, group in frame.groupby("model"):
        group = group.sort_values(x_column)
        x = list(range(len(group))) if categorical else group[x_column]
        axis.plot(x, group["accuracy"], marker="o", linewidth=1.4, label=model)
    if categorical:
        ticks = frame.sort_values(x_column)["snr_db"].unique()
        labels = ["clean" if v == float("inf") else f"{v:g}" for v in ticks]
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("U/D accuracy")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="name=checkpoint; repeatable")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output-dir", default="outputs/paper")
    parser.add_argument("--subset-size", type=int, default=30000)
    parser.add_argument("--subset-seed", type=int, default=20261004)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--p-shift", action="store_true")
    args = parser.parse_args()

    logger = get_logger("sopnet.robustness")
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    adapters = {}
    for entry in args.models:
        name, _, path = entry.partition("=")
        adapter, window = load_model(name, Path(path or name), device)
        adapters[name] = (adapter, window)
        logger.info("loaded %s (window %d) from %s", name, window, path or name)

    do_all = not (args.noise or args.p_shift)
    shift_frames, snr_frames = [], []
    for name, (adapter, window) in adapters.items():
        if do_all or args.p_shift:
            frame = evaluate_p_shift(
                adapter,
                Path(args.cache_dir),
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                subset_size=args.subset_size,
                subset_seed=args.subset_seed,
                window_length=window,
            )
            frame.insert(0, "model", name)
            frame.to_csv(tables_dir / f"p_shift_{name}.csv", index=False)
            shift_frames.append(frame)
            logger.info("P-shift done for %s", name)
        if do_all or args.noise:
            frame = evaluate_noise(
                adapter,
                Path(args.cache_dir),
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                subset_size=args.subset_size,
                subset_seed=args.subset_seed,
                window_length=window,
            )
            frame.insert(0, "model", name)
            frame.to_csv(tables_dir / f"snr_{name}.csv", index=False)
            snr_frames.append(frame)
            logger.info("SNR done for %s", name)

    if shift_frames:
        combined = pd.concat(shift_frames, ignore_index=True)
        combined.to_csv(tables_dir / "p_shift_all.csv", index=False)
        plot_models(combined, "shift_seconds", figures_dir / "fig_p_shift_all", "P-window shift (s)")
        logger.info("combined P-shift table written to %s", tables_dir / "p_shift_all.csv")
    if snr_frames:
        combined = pd.concat(snr_frames, ignore_index=True)
        combined.to_csv(tables_dir / "snr_all.csv", index=False)
        plot_models(combined, "snr_db", figures_dir / "fig_snr_all", "SNR (dB)", categorical=True)
        logger.info("combined SNR table written to %s", tables_dir / "snr_all.csv")


if __name__ == "__main__":
    main()
