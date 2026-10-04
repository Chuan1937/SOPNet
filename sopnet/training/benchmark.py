"""Batch-size and throughput benchmarking to keep the GPU busy.

Picks the batch size with the highest training throughput whose peak memory
stays below a safety fraction of VRAM. VRAM does not need to be full; what
matters is samples/second and GPU utilisation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from sopnet.data.dataset import UnifiedPolarityDataset, make_dataloader
from sopnet.losses import WeightedSignedFieldLoss


def gpu_utilization() -> float:
    try:
        return float(torch.cuda.utilization())
    except Exception:  # noqa: BLE001 - NVML not always available
        return float("nan")


def select_best_batch(
    results: List[dict],
    budget_gb: float,
    fallback: int = 512,
    throughput_margin: float = 0.9,
) -> int:
    """Fastest batch size, preferring the smallest one when throughput ties.

    Throughput is the wall-clock metric that matters; GPU utilisation and VRAM
    are secondary. Candidates within ``throughput_margin`` of the best
    throughput are treated as equivalent and the smallest batch is chosen,
    which keeps VRAM well below the budget. Peak VRAM must still fit within
    ``budget_gb``.
    """
    safe = [
        row
        for row in results
        if row.get("status") == "ok" and row.get("peak_vram_gb", float("inf")) <= budget_gb
    ]
    if not safe:
        return fallback
    best_throughput = max(row["samples_per_s"] for row in safe)
    eligible = [row for row in safe if row["samples_per_s"] >= throughput_margin * best_throughput]
    return int(min(row["batch_size"] for row in eligible))


def benchmark_batch_sizes(
    model: torch.nn.Module,
    dataset: UnifiedPolarityDataset,
    device: torch.device,
    candidates: Iterable[int] = (512, 1024, 1536, 2048),
    num_workers: int = 4,
    amp: bool = True,
    steps: int = 8,
    warmup: int = 2,
    vram_fraction: float = 0.9,
    seed: int = 0,
    logger=None,
    output: Optional[Path] = None,
) -> Dict[str, object]:
    """Return ``{"batch_size", "results", ...}`` for the fastest safe batch size."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"batch_size": next(iter(candidates), 1024), "results": [], "reason": "cpu"}

    model.to(device)
    criterion = WeightedSignedFieldLoss()
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
    budget = vram_fraction * total_vram
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    results: List[dict] = []

    for batch_size in candidates:
        loader = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=max(0, min(num_workers, 8)),
            persistent_workers=False,
            prefetch_factor=2,
            seed=seed,
            drop_last=True,
        )
        total_steps = warmup + max(1, min(steps, len(loader) - warmup))
        iterator = iter(loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        timings: List[float] = []
        utilization: List[float] = []
        try:
            for step in range(total_steps):
                batch = next(iterator)
                x = batch["x"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=amp):
                    loss = criterion(model(x), target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                if step >= warmup:
                    timings.append(time.perf_counter() - started)
                    utilization.append(gpu_utilization())
            seconds = sum(timings) / max(1, len(timings))
            peak = torch.cuda.max_memory_allocated() / 1024**3
            results.append(
                {
                    "batch_size": int(batch_size),
                    "samples_per_s": float(batch_size / seconds),
                    "seconds_per_step": float(seconds),
                    "peak_vram_gb": float(peak),
                    "gpu_utilization": float(sum(utilization) / max(1, len(utilization))),
                    "status": "ok",
                }
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"batch_size": int(batch_size), "status": "oom"})
            model.zero_grad(set_to_none=True)
        finally:
            del loader, iterator

    selection = {
        "batch_size": select_best_batch(results, budget, fallback=512),
        "total_vram_gb": round(total_vram, 2),
        "budget_gb": round(budget, 2),
        "results": results,
    }
    if logger is not None:
        for row in results:
            if row.get("status") == "ok":
                logger.info(
                    "batch %5d: %.0f samples/s, %.1f%% GPU, peak %.2f GB",
                    row["batch_size"],
                    row["samples_per_s"],
                    row["gpu_utilization"],
                    row["peak_vram_gb"],
                )
            else:
                logger.info("batch %5d: %s", row["batch_size"], row["status"])
        logger.info("selected batch size %d", selection["batch_size"])
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2)
    return selection
