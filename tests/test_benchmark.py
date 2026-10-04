from __future__ import annotations

from sopnet.training.benchmark import select_best_batch


def _results():
    return [
        {"batch_size": 512, "samples_per_s": 4000.0, "peak_vram_gb": 2.0, "status": "ok"},
        {"batch_size": 1024, "samples_per_s": 5200.0, "peak_vram_gb": 3.5, "status": "ok"},
        {"batch_size": 1536, "samples_per_s": 5100.0, "peak_vram_gb": 5.8, "status": "ok"},
        {"batch_size": 2048, "status": "oom"},
    ]


def test_selects_fastest_safe_batch():
    assert select_best_batch(_results(), budget_gb=8.0) == 1024


def test_excludes_batches_over_memory_budget():
    assert select_best_batch(_results(), budget_gb=4.0) == 1024
    assert select_best_batch(_results(), budget_gb=2.5) == 512


def test_fallback_when_all_oom():
    results = [{"batch_size": 512, "status": "oom"}, {"batch_size": 1024, "status": "oom"}]
    assert select_best_batch(results, budget_gb=8.0, fallback=256) == 256
