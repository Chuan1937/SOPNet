#!/usr/bin/env python
"""Train SOPNet (signed onset field) or SOPNet-Cls on the unified cache."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.augment import AugmentConfig  # noqa: E402
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader  # noqa: E402
from sopnet.evaluation.evaluate import (  # noqa: E402
    choose_threshold,
    collect_predictions,
    evaluate_field,
    plot_prediction_examples,
    save_metrics,
)
from sopnet.models import build_model, count_parameters  # noqa: E402
from sopnet.training.checkpoint import load_checkpoint  # noqa: E402
from sopnet.training.engine import TrainConfig, Trainer  # noqa: E402
from sopnet.utils.config import deep_update, load_config, save_yaml  # noqa: E402
from sopnet.utils.logging import get_logger  # noqa: E402
from sopnet.utils.seed import set_seed  # noqa: E402
from sopnet.utils.system import format_env_report  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def resolve_config(args) -> dict:
    config = load_config(REPO / "configs/train/main.yaml")
    config = deep_update(config, load_config(REPO / "configs/model/sopnet_s.yaml"))
    config = deep_update(config, load_config(REPO / "configs/data/unified.yaml"))
    config = deep_update(config, load_config(args.config))
    if args.cache_dir:
        config["data"]["cache_dir"] = args.cache_dir
    if args.output:
        config["output"]["dir"] = args.output
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = args.learning_rate
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.no_amp:
        config["train"]["amp"] = False
    if args.device:
        config["train"]["device"] = args.device
    return config


def make_train_config(section: dict) -> TrainConfig:
    allowed = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in section.items() if key in allowed})


def build_datasets(config: dict, limit_train=None, limit_val=None):
    cache_dir = Path(config["data"]["cache_dir"])
    dataset_config = config.get("dataset", {})
    augment = None
    if dataset_config.get("augment"):
        augment = AugmentConfig(**dataset_config["augment"])
    jitter = dataset_config.get("jitter")
    jitter = tuple(jitter) if jitter else None
    common = dict(
        cache_dir=cache_dir,
        window_length=int(dataset_config.get("window_length", 400)),
        sigma=float(dataset_config.get("sigma", 10.0)),
        seed=int(config["train"].get("seed", 36)),
    )
    train_dataset = UnifiedPolarityDataset(
        split="train", jitter=jitter, augment=augment, max_samples=limit_train, **common
    )
    val_dataset = UnifiedPolarityDataset(
        split="val", jitter=None, augment=None, max_samples=limit_val, **common
    )
    return train_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sopnet.train", output_dir / "train.log")
    save_yaml(config, output_dir / "config_resolved.yaml")

    seed = int(config["train"].get("seed", 36))
    set_seed(seed)
    logger.info("seed=%d", seed)

    model = build_model(config)
    logger.info("model: %s, %s parameters", type(model).__name__, f"{count_parameters(model):,}")

    train_dataset, val_dataset = build_datasets(
        config, limit_train=args.limit_train, limit_val=args.limit_val
    )
    logger.info("train samples=%d, val samples=%d", len(train_dataset), len(val_dataset))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("empty train/val split; build the cache first")

    trainer = Trainer(
        model,
        make_train_config(config["train"]),
        output_dir,
        logger=logger,
        run_config=config,
    )
    logger.info(format_env_report(trainer.env))
    summary = trainer.fit(train_dataset, val_dataset)
    logger.info(
        "training finished: best epoch %s (%s=%.4f)",
        summary["best_epoch"],
        summary["monitor"],
        summary["best_metric"],
    )

    best_path = output_dir / "best.pt"
    if best_path.exists():
        payload = load_checkpoint(best_path)
        model.load_state_dict(payload["model_state"])
        model.to(trainer.device)
    elif hasattr(model, "to"):
        model.to(trainer.device)

    val_loader = make_dataloader(
        val_dataset, batch_size=trainer.config.batch_size, shuffle=False, num_workers=2
    )
    threshold = None
    if trainer.config.task == "field":
        threshold = choose_threshold(model, val_loader, trainer.device)
        logger.info("selected confidence threshold on validation: %.3f", threshold)
        val_metrics = evaluate_field(model, val_loader, trainer.device, threshold=threshold)
        save_metrics(val_metrics, output_dir / "val_metrics.json")
        logger.info("val metrics: %s", val_metrics)
        outputs = collect_predictions(
            model, val_loader, trainer.device, task="field", max_plot_samples=max(1, args.examples)
        )
        plot_prediction_examples(outputs, output_dir / "examples", n=args.examples)
        logger.info("example figure written to %s", output_dir / "examples.png")
    else:
        val_metrics = trainer.validate(val_loader)
        save_metrics(val_metrics, output_dir / "val_metrics.json")


if __name__ == "__main__":
    main()
