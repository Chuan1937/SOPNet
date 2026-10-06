#!/usr/bin/env python
"""Train SeisPolarity baselines on the same unified train/val splits as SOPNet."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.augment import AugmentConfig  # noqa: E402
from sopnet.training.baselines import (  # noqa: E402
    BASELINE_SPECS,
    BaselineTrainConfig,
    build_baseline,
    evaluate_baseline,
    prepare_baseline_input,
    train_baseline,
)
from sopnet.utils.logging import get_logger  # noqa: E402


def dry_run(names) -> None:
    import torch

    for name in names:
        spec = BASELINE_SPECS[name]
        model = build_baseline(name)
        model.eval()
        dummy = torch.zeros(2, 1, 400)
        with torch.no_grad():
            output = model(prepare_baseline_input(dummy, spec))
        shape = output.shape if hasattr(output, "shape") else [tuple(o.shape) for o in output]
        parameters = sum(p.numel() for p in model.parameters())
        print(
            f"{name:14s} spec={spec.output:11s} input={spec.input_length} channels={spec.input_channels} "
            f"output={shape} params={parameters:,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", nargs="+", default=list(BASELINE_SPECS))
    parser.add_argument("--cache-dir", default="outputs/cache_v1")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip the val/test evaluation after training",
    )
    args = parser.parse_args()

    logger = get_logger("sopnet.baselines")
    if args.dry_run:
        dry_run(args.baselines)
        return

    augment = AugmentConfig()
    for name in args.baselines:
        output_dir = Path(args.output_root) / f"baseline_{name}"
        config = BaselineTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            augment=augment,
        )
        logger.info("training baseline %s -> %s", name, output_dir)
        summary = train_baseline(
            name,
            Path(args.cache_dir),
            output_dir,
            config,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            logger=logger,
        )
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))
        logger.info("%s: best val F1 %.4f (epoch %d)", name, summary["best_f1"], summary["best_epoch"])

        if not args.skip_eval:
            for split in ("val", "test"):
                metrics = evaluate_baseline(
                    name,
                    output_dir / "best.pt",
                    Path(args.cache_dir),
                    split=split,
                    num_workers=args.num_workers,
                )
                (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
                logger.info("%s %s metrics: %s", name, split, metrics)


if __name__ == "__main__":
    main()
