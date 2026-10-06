"""Adapters for the SeisPolarity baseline models.

Baselines are imported from the external ``seispolarity`` package (never
copied). They share exactly the same manifest, splits and 400-sample cache
windows as SOPNet; only the last input crop / input channels differ, following
each model's published convention.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from sopnet.data.augment import AugmentConfig
from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader
from sopnet.training.metrics import binary_metrics
from sopnet.training.optimizer import build_optimizer, build_scheduler
from sopnet.utils.system import collect_env, peak_gpu_memory_gb

LN2 = float(np.log(2.0))


@dataclass
class BaselineSpec:
    name: str
    factory: str
    input_length: int
    input_channels: int = 1
    output: str = "binary_ud"  # binary_ud (positive logit = UP), binary_du, three_class
    derivative: bool = False
    kwargs: Dict = field(default_factory=dict)


BASELINE_SPECS: Dict[str, BaselineSpec] = {
    "ross": BaselineSpec("ross", "SCSN", 400, output="three_class"),
    "rpnet": BaselineSpec("rpnet", "RPNet", 400, output="binary_du"),
    "eqpolarity": BaselineSpec("eqpolarity", "EQPolarityCCT", 200, output="binary_ud"),
    "cfm": BaselineSpec("cfm", "CFM", 160, output="binary_ud"),
    "diting_motion": BaselineSpec(
        "diting_motion", "DitingMotion", 128, input_channels=2, output="three_class", derivative=True
    ),
}


def build_baseline(name: str) -> nn.Module:
    import seispolarity.models as models

    spec = BASELINE_SPECS[name]
    factory = getattr(models, spec.factory)
    return factory(**spec.kwargs)


def _center_crop(waveform: torch.Tensor, length: int) -> torch.Tensor:
    total = waveform.shape[-1]
    if length >= total:
        return waveform
    start = (total - length) // 2
    return waveform[..., start : start + length]


def prepare_baseline_input(waveform: torch.Tensor, spec: BaselineSpec) -> torch.Tensor:
    """Crop to the model's window and add the derivative channel when required."""
    cropped = _center_crop(waveform, spec.input_length)
    if spec.derivative:
        derivative = torch.gradient(cropped, dim=-1)[0]
        cropped = torch.cat([cropped, derivative], dim=1)
    return cropped


class BaselineWrapper(nn.Module):
    """Normalises every baseline to a common interface for training and metrics."""

    def __init__(self, model: nn.Module, spec: BaselineSpec):
        super().__init__()
        self.model = model
        self.spec = spec

    def forward(self, waveform: torch.Tensor):
        x = prepare_baseline_input(waveform, self.spec)
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[3] if len(output) >= 4 else output[0]
        return output


def _targets(spec: BaselineSpec, labels: torch.Tensor):
    """Return (loss_target, mask) for the model's output convention."""
    if spec.output == "three_class":
        mapping = {UP: 0, DOWN: 1, UNKNOWN: 2}
        target = torch.tensor([mapping[int(v)] for v in labels.tolist()], device=labels.device)
        return target, torch.ones_like(target, dtype=torch.bool)
    mask = labels != UNKNOWN
    target = (labels == UP).float()
    if spec.output == "binary_du":
        target = (labels == DOWN).long()
    else:
        target = target
    return target, mask


def _loss(spec: BaselineSpec, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target, mask = _targets(spec, labels)
    if spec.output == "three_class":
        return nn.functional.cross_entropy(output, target)
    if not bool(mask.any()):
        return output.sum() * 0.0
    if spec.output == "binary_du":
        return nn.functional.cross_entropy(output[mask], target[mask])
    return nn.functional.binary_cross_entropy_with_logits(output[mask].squeeze(-1), target[mask])


def _predict(spec: BaselineSpec, output: torch.Tensor):
    if spec.output == "three_class":
        predicted_class = output.argmax(dim=-1)
        mapping = torch.tensor([UP, DOWN, UNKNOWN], device=output.device)
        predictions = mapping[predicted_class]
        confidence = torch.softmax(output, dim=-1).max(dim=-1).values
        return predictions, confidence
    if spec.output == "binary_du":
        predicted_class = output.argmax(dim=-1)
        predictions = torch.where(
            predicted_class == 1, torch.full_like(predicted_class, UP), torch.full_like(predicted_class, DOWN)
        )
        confidence = torch.softmax(output, dim=-1).max(dim=-1).values
        return predictions, confidence
    probability = torch.sigmoid(output.squeeze(-1))
    predictions = torch.where(
        probability >= 0.5,
        torch.ones_like(probability, dtype=torch.long),
        torch.full_like(probability, DOWN, dtype=torch.long),
    )
    confidence = torch.maximum(probability, 1 - probability)
    return predictions, confidence


@dataclass
class BaselineTrainConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    patience: int = 6
    num_workers: int = 8
    seed: int = 36
    device: str = "cuda"
    augment: Optional[AugmentConfig] = None


def train_baseline(
    name: str,
    cache_dir: Path,
    output_dir: Path,
    config: BaselineTrainConfig,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    logger=None,
) -> Dict:
    from sopnet.utils.seed import set_seed

    spec = BASELINE_SPECS[name]
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = BaselineWrapper(build_baseline(name), spec).to(device)

    train_dataset = UnifiedPolarityDataset(
        cache_dir,
        split="train",
        jitter=None,
        augment=config.augment,
        max_samples=limit_train,
        seed=config.seed,
    )
    val_dataset = UnifiedPolarityDataset(
        cache_dir,
        split="val",
        jitter=None,
        augment=None,
        max_samples=limit_val,
        seed=config.seed,
    )
    train_loader = make_dataloader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    scheduler = build_scheduler(optimizer, config.epochs * max(1, len(train_loader)), config.warmup_ratio)

    best = float("-inf")
    best_epoch = 0
    patience = 0
    history = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                output = model(x)
                loss = _loss(spec, output, label)
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())
            steps += 1

        model.eval()
        labels, predictions, confidences = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                output = model(x)
                pred, confidence = _predict(spec, output)
                labels.append(batch["label"].numpy())
                predictions.append(pred.cpu().numpy())
                confidences.append(confidence.cpu().numpy())
        y_true = np.concatenate(labels)
        y_pred = np.concatenate(predictions)
        known = y_true != UNKNOWN
        validation = binary_metrics(y_true[known], y_pred[known]) if known.any() else {"accuracy": 0.0}
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, steps),
            "val": validation,
            "seconds": round(time.time() - started, 2),
        }
        history.append(record)
        if logger is not None:
            logger.info(
                "epoch %03d | loss %.4f | val U/D accuracy %.4f",
                epoch,
                record["train_loss"],
                validation.get("accuracy", 0.0),
            )
        score = validation.get("f1", 0.0)
        if score > best:
            best, best_epoch, patience = score, epoch, 0
            torch.save({"model_state": model.state_dict(), "spec": name}, output_dir / "best.pt")
        else:
            patience += 1
        if patience >= config.patience:
            break

    summary = {
        "baseline": name,
        "best_epoch": best_epoch,
        "best_f1": best,
        "history": history,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "peak_gpu_memory_gb": peak_gpu_memory_gb(),
        "env": collect_env(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


@torch.no_grad()
def collect_baseline_predictions(
    name: str,
    checkpoint: Path,
    cache_dir: Path,
    split: str = "test",
    batch_size: int = 1024,
    num_workers: int = 4,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Load a baseline checkpoint and return per-sample labels/predictions/confidence."""
    spec = BASELINE_SPECS[name]
    model = BaselineWrapper(build_baseline(name), spec)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(torch.device(device) if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()

    dataset = UnifiedPolarityDataset(cache_dir, split=split, jitter=None)
    loader = make_dataloader(dataset, batch_size, shuffle=False, num_workers=num_workers)
    labels, predictions, confidences = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        output = model(x)
        pred, confidence = _predict(spec, output)
        labels.append(batch["label"].numpy())
        predictions.append(pred.cpu().numpy())
        confidences.append(confidence.cpu().numpy())
    return {
        "labels": np.concatenate(labels),
        "predictions": np.concatenate(predictions),
        "confidence": np.concatenate(confidences),
    }


@torch.no_grad()
def evaluate_baseline(
    name: str,
    checkpoint: Path,
    cache_dir: Path,
    split: str = "test",
    batch_size: int = 1024,
    num_workers: int = 4,
    device: str = "cuda",
) -> Dict[str, float]:
    outputs = collect_baseline_predictions(
        name,
        checkpoint,
        cache_dir,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    y_true = outputs["labels"]
    y_pred = outputs["predictions"]
    known = y_true != UNKNOWN
    metrics = binary_metrics(y_true[known], y_pred[known]) if known.any() else {}
    metrics["n"] = int(len(y_true))
    metrics["n_known"] = int(known.sum())
    return metrics
