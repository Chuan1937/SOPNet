#!/usr/bin/env python
"""P-shift and noise robustness on the unified test split, with paper figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.evaluation.evaluate import save_metrics  # noqa: E402
from sopnet.evaluation.robustness import evaluate_noise, evaluate_p_shift  # noqa: E402
from sopnet.models import build_model  # noqa: E402
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402


def load_model(checkpoint: Path, device: str):
    payload = load_checkpoint(checkpoint)
    config = payload.get("config", {})
    model = build_model(config)
    model.load_state_dict(payload["model_state"])
    import torch

    model.to(torch.device(device) if torch.cuda.is_available() or device == "cpu" else torch.device("cpu"))
    return model, config


def plot_curve(table, x_column, y_column, path: Path, xlabel: str, ylabel: str = "Macro-F1 (U/D/X)") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    axis.plot(table[x_column], table[y_column], marker="o", linewidth=1.5)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, action="append", help="name=path; repeatable")
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output-dir", default="outputs/paper")
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

    models: Dict[str, tuple] = {}
    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        model, config = load_model(Path(path or name), args.device)
        models[name if path else Path(name).parent.name] = (model, config)

    do_all = not (args.noise or args.p_shift)
    for name, (model, config) in models.items():
        if do_all or args.p_shift:
            frame = evaluate_p_shift(
                model,
                Path(args.cache_dir),
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            frame.insert(0, "model", name)
            path = tables_dir / f"p_shift_{name}.csv"
            frame.to_csv(path, index=False)
            logger.info("P-shift table written to %s", path)
            if "macro_f1_ud" in frame.columns:
                plot_curve(
                    frame.sort_values("shift_seconds"),
                    "shift_seconds",
                    "macro_f1_ud",
                    figures_dir / f"fig_p_shift_{name}",
                    "P-window shift (s)",
                    "Macro-F1 (U/D)",
                )
        if do_all or args.noise:
            frame = evaluate_noise(
                model,
                Path(args.cache_dir),
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            frame.insert(0, "model", name)
            path = tables_dir / f"snr_{name}.csv"
            frame.to_csv(path, index=False)
            logger.info("noise table written to %s", path)
            finite = frame[frame["snr_db"] != float("inf")].sort_values("snr_db")
            if "macro_f1_ud" in finite.columns and len(finite):
                plot_curve(
                    finite,
                    "snr_db",
                    "macro_f1_ud",
                    figures_dir / f"fig_snr_{name}",
                    "SNR (dB)",
                    "Macro-F1 (U/D)",
                )
        save_metrics(
            {"model": name, "config": config if isinstance(config, dict) else {}},
            output_dir / f"robustness_{name}.json",
        )


if __name__ == "__main__":
    main()
