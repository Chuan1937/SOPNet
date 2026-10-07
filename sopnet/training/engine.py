"""Training engine for SOPNet (signed field) and SOPNet-Cls (classification)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader
from sopnet.losses import (
    InversionConsistencyLoss,
    PolarityConsistencyLoss,
    WeightedSignedFieldLoss,
)
from sopnet.training.checkpoint import save_checkpoint
from sopnet.training.metrics import binary_metrics, macro_f1, p_error_metrics
from sopnet.training.optimizer import build_optimizer, build_scheduler, current_lr
from sopnet.utils.system import collect_env, peak_gpu_memory_gb

CLASS_ORDER = (DOWN, UNKNOWN, UP)


@dataclass
class TrainConfig:
    task: str = "field"
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    beta: float = 8.0
    lambda_pol: float = 0.5
    lambda_inv: float = 0.1
    lambda_cls: float = 0.0
    inv_batch_prob: float = 0.25
    unknown_weight: float = 1.0
    patience: int = 10
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 4
    seed: int = 36
    device: str = "cuda"


def canonical_to_class(label: torch.Tensor) -> torch.Tensor:
    mapping = {DOWN: 0, UNKNOWN: 1, UP: 2}
    return torch.tensor([mapping[int(v)] for v in label.tolist()], dtype=torch.long, device=label.device)


class Trainer:
    """Train either a signed-field model or a classifier on the unified cache."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        output_dir: Path,
        logger=None,
        run_config: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.run_config = run_config or asdict(config)
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        self.field_loss = WeightedSignedFieldLoss(beta=config.beta)
        self.polarity_loss = PolarityConsistencyLoss()
        self.inversion_loss = InversionConsistencyLoss()
        self.classification_loss = nn.CrossEntropyLoss()

        self.optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.amp and self.device.type == "cuda")
        self.scheduler = None
        self.history: list = []
        self.env = collect_env()

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)

    def _build_loader(self, dataset: UnifiedPolarityDataset, shuffle: bool):
        return make_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            seed=self.config.seed,
            drop_last=shuffle,
        )

    def _forward_loss(self, batch: Dict[str, torch.Tensor], training: bool) -> torch.Tensor:
        x = batch["x"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        label = batch["label"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            if self.config.task == "classify":
                logits = self.model(x)
                loss = self.classification_loss(logits, canonical_to_class(label))
            else:
                x_pol = batch.get("x_pol")
                if x_pol is not None:
                    x_pol = x_pol.to(self.device, non_blocking=True)
                if self.config.task == "field_multi":
                    output = self.model(x, x_pol=x_pol)
                else:
                    output = self.model(x)
                if isinstance(output, tuple):
                    field, polarity_logits = output
                else:
                    field, polarity_logits = output, None
                sample_weight = None
                if self.config.unknown_weight != 1.0:
                    sample_weight = torch.where(
                        label != UNKNOWN,
                        torch.ones_like(label, dtype=field.dtype),
                        torch.full_like(label, self.config.unknown_weight, dtype=field.dtype),
                    )
                loss = self.field_loss(field, target, sample_weight=sample_weight)
                if self.config.lambda_pol > 0:
                    loss = loss + self.config.lambda_pol * self.polarity_loss(field, label)
                if polarity_logits is not None and self.config.lambda_cls > 0:
                    loss = loss + self.config.lambda_cls * self._polarity_cross_entropy(
                        polarity_logits, label
                    )
                if (
                    training
                    and self.config.lambda_inv > 0
                    and np.random.random() < self.config.inv_batch_prob
                ):
                    if self.config.task == "field_multi":
                        inverted = self.model(-x, x_pol=x_pol)
                    else:
                        inverted = self.model(-x)
                    if isinstance(inverted, tuple):
                        inverted = inverted[0]
                    loss = loss + self.config.lambda_inv * self.inversion_loss(field, inverted)
        return loss

    def _polarity_cross_entropy(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy on labelled U/D samples only (0=DOWN, 1=UP)."""
        known = label != UNKNOWN
        if not bool(known.any()):
            return logits.sum() * 0.0
        target = (label[known] == UP).long()
        return F.cross_entropy(logits[known], target)

    def train_epoch(self, loader, epoch: int = 0) -> Dict[str, float]:
        self.model.train()
        total, count = 0.0, 0
        seen = 0
        total_steps = len(loader)
        log_every = max(1, total_steps // 10)
        started = time.time()
        progress = tqdm(
            loader,
            desc=f"epoch {epoch:03d}",
            unit="step",
            mininterval=5.0,
            dynamic_ncols=True,
            file=sys.stderr,
            leave=True,
        )
        for step, batch in enumerate(progress):
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(batch, training=True)
            self.scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total += float(loss.detach())
            count += 1
            seen += int(batch["x"].shape[0])
            current_loss = total / count
            progress.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr(self.optimizer):.2e}")

            if step % log_every == 0 or step == total_steps - 1:
                elapsed = max(1e-6, time.time() - started)
                step_seconds = elapsed / (step + 1)
                remaining = step_seconds * (total_steps - step - 1)
                utilization = float("nan")
                if self.device.type == "cuda":
                    try:
                        utilization = float(torch.cuda.utilization())
                    except Exception:  # noqa: BLE001 - NVML may be unavailable
                        pass
                self._log(
                    f"epoch {epoch:03d} step {step + 1:5d}/{total_steps} "
                    f"loss {current_loss:.4f} | {seen / elapsed:7.0f} samples/s | "
                    f"ETA {remaining / 60:5.1f} min | GPU {utilization:4.0f}%"
                )
        progress.close()
        return {"loss": total / max(1, count), "lr": current_lr(self.optimizer)}

    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        labels, predictions, confidences, p_pred, p_true = [], [], [], [], []
        total, count = 0.0, 0
        for batch in loader:
            loss = self._forward_loss(batch, training=False)
            total += float(loss.detach())
            count += 1
            label = batch["label"].numpy()
            if self.config.task == "classify":
                x = batch["x"].to(self.device, non_blocking=True)
                logits = self.model(x)
                pred_class = logits.argmax(dim=-1).cpu().numpy()
                pred = np.array([CLASS_ORDER[c] for c in pred_class])
                confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
                labels.append(label)
                predictions.append(pred)
                confidences.append(confidence)
            else:
                x = batch["x"].to(self.device, non_blocking=True)
                x_pol = batch.get("x_pol")
                if x_pol is not None:
                    x_pol = x_pol.to(self.device, non_blocking=True)
                if self.config.task == "field_multi":
                    output = self.model(x, x_pol=x_pol)
                else:
                    output = self.model(x)
                if isinstance(output, tuple):
                    field, polarity_logits = output
                else:
                    field, polarity_logits = output, None
                position = field.abs().argmax(dim=-1).squeeze(-1)
                if polarity_logits is not None:
                    probabilities = torch.softmax(polarity_logits, dim=-1)
                    class_index = probabilities.argmax(dim=-1)
                    pred = (
                        torch.where(
                            class_index == 1,
                            torch.ones_like(class_index),
                            torch.full_like(class_index, DOWN),
                        )
                        .cpu()
                        .numpy()
                    )
                    confidence = probabilities.max(dim=-1).values.cpu().numpy()
                else:
                    magnitude = field.abs()
                    field_confidence, position = magnitude.max(dim=-1)
                    signed = field.gather(-1, position.unsqueeze(1)).squeeze(1).squeeze(-1)
                    confidence = field_confidence.squeeze(-1).cpu().numpy()
                    position = position.squeeze(-1)
                    pred = torch.sign(signed).cpu().numpy()
                    pred[confidence == 0] = 0
                labels.append(label)
                predictions.append(pred)
                confidences.append(confidence)
                p_pred.append(position.cpu().numpy())
                p_true.append(batch["p_pick"].numpy())

        labels = np.concatenate(labels) if labels else np.array([])
        predictions = np.concatenate(predictions) if predictions else np.array([])
        confidences = np.concatenate(confidences) if confidences else np.array([])

        metrics: Dict[str, float] = {"loss": total / max(1, count)}
        if self.config.task == "classify":
            if labels.size:
                metrics["accuracy"] = float(np.mean(predictions == labels))
                metrics["macro_f1"] = macro_f1(labels, predictions, labels=(DOWN, UNKNOWN, UP))
            return metrics
        known = labels != UNKNOWN
        if known.any():
            metrics.update(
                {
                    "known_" + key: value
                    for key, value in binary_metrics(labels[known], predictions[known]).items()
                }
            )
            metrics["macro_f1"] = macro_f1(labels[known], predictions[known], labels=(DOWN, UP))
        if p_pred:
            metrics.update(p_error_metrics(np.concatenate(p_pred), np.concatenate(p_true), fs=100))
        return metrics

    def fit(
        self,
        train_dataset: UnifiedPolarityDataset,
        val_dataset: UnifiedPolarityDataset,
        resume_payload: Optional[Dict] = None,
    ) -> Dict[str, object]:
        train_loader = self._build_loader(train_dataset, shuffle=True)
        val_loader = self._build_loader(val_dataset, shuffle=False)
        total_steps = max(1, self.config.epochs * len(train_loader))
        self.scheduler = build_scheduler(self.optimizer, total_steps, self.config.warmup_ratio)

        monitor = "known_accuracy" if self.config.task in ("field", "field_multi") else "macro_f1"
        best = float("-inf")
        best_epoch = 0
        patience = 0
        start_epoch = 1
        if resume_payload:
            if resume_payload.get("optimizer_state") is not None:
                self.optimizer.load_state_dict(resume_payload["optimizer_state"])
            if resume_payload.get("scheduler_state") is not None:
                self.scheduler.load_state_dict(resume_payload["scheduler_state"])
            start_epoch = int(resume_payload.get("epoch", 0)) + 1
            best = float(resume_payload.get("best_metric", float("-inf")))
            best_epoch = int(resume_payload.get("best_epoch", 0))
            patience = int(resume_payload.get("patience", 0))
            self.history = list(resume_payload.get("history", []))
            self._log(f"resuming at epoch {start_epoch} (best {best:.4f} @ epoch {best_epoch})")

        parameters = sum(p.numel() for p in self.model.parameters())
        self._log(
            f"training {self.config.task} model: {parameters:,} parameters, "
            f"device={self.device}, train={len(train_dataset)}, val={len(val_dataset)}"
        )
        for epoch in range(start_epoch, self.config.epochs + 1):
            started = time.time()
            train_metrics = self.train_epoch(train_loader, epoch=epoch)
            val_metrics = self.validate(val_loader)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "seconds": round(time.time() - started, 2),
            }
            self.history.append(record)
            self._log(
                f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"val U/D accuracy {val_metrics.get('known_accuracy', float('nan')):.4f}"
            )

            score = val_metrics.get(monitor, float("-inf"))
            if score > best:
                best, best_epoch, patience = score, epoch, 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    best,
                    best_epoch=best_epoch,
                    patience=patience,
                    config=self.run_config,
                    env=self.env,
                    history=self.history,
                )
            else:
                patience += 1
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                best,
                best_epoch=best_epoch,
                patience=patience,
                config=self.run_config,
                env=self.env,
                history=self.history,
            )
            if patience >= self.config.patience:
                self._log(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

        summary = {
            "best_epoch": best_epoch,
            "best_metric": best,
            "monitor": monitor,
            "epochs_run": len(self.history),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "peak_gpu_memory_gb": peak_gpu_memory_gb(),
            "env": self.env,
            "config": self.run_config,
            "history": self.history,
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        return summary
