from __future__ import annotations

from sopnet.training.benchmark import select_best_batch


def _results():
    return [
        {
            "batch_size": 1024,
            "samples_per_s": 6800.0,
            "peak_vram_gb": 2.35,
            "gpu_utilization": 60.0,
            "status": "ok",
        },
        {
            "batch_size": 1536,
            "samples_per_s": 5870.0,
            "peak_vram_gb": 3.49,
            "gpu_utilization": 84.0,
            "status": "ok",
        },
        {
            "batch_size": 2048,
            "samples_per_s": 5929.0,
            "peak_vram_gb": 4.67,
            "gpu_utilization": 98.0,
            "status": "ok",
        },
    ]


def test_prefers_highest_gpu_utilisation_near_peak_throughput():
    assert select_best_batch(_results(), budget_gb=8.0) == 2048


def test_respects_memory_budget():
    assert select_best_batch(_results(), budget_gb=3.0) == 1024
    assert select_best_batch(_results(), budget_gb=1.0) == 512


def test_fallback_when_all_oom():
    results = [{"batch_size": 512, "status": "oom"}, {"batch_size": 1024, "status": "oom"}]
    assert select_best_batch(results, budget_gb=8.0, fallback=256) == 256
