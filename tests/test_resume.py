from __future__ import annotations

from sopnet.data.dataset import UnifiedPolarityDataset
from sopnet.models import build_model
from sopnet.training.checkpoint import load_checkpoint
from sopnet.training.engine import TrainConfig, Trainer


def _trainer(tmp_path, epochs):
    model = build_model({"model": {"name": "sopnet"}})
    config = TrainConfig(epochs=epochs, batch_size=16, num_workers=0, amp=False, device="cpu", seed=0)
    return Trainer(model, config, tmp_path / "run", run_config={"train": {"task": "field"}})


def test_resume_continues_from_last_epoch(synthetic_cache, tmp_path):
    cache_dir, manifest, index = synthetic_cache
    train_dataset = UnifiedPolarityDataset(
        cache_dir,
        split="train",
        jitter=(120, 280),
        max_samples=48,
        seed=0,
        manifest=manifest,
        index=index,
    )
    val_dataset = UnifiedPolarityDataset(
        cache_dir, split="val", max_samples=24, seed=0, manifest=manifest, index=index
    )

    first = _trainer(tmp_path, epochs=1)
    summary_first = first.fit(train_dataset, val_dataset)
    assert summary_first["epochs_run"] == 1

    payload = load_checkpoint(tmp_path / "run" / "last.pt")
    assert payload["epoch"] == 1
    assert "best_epoch" in payload and "patience" in payload

    second = _trainer(tmp_path, epochs=2)
    second.model.load_state_dict(payload["model_state"])
    summary_second = second.fit(train_dataset, val_dataset, resume_payload=payload)

    assert summary_second["epochs_run"] == 2
    assert [record["epoch"] for record in summary_second["history"]] == [1, 2]


def test_resume_without_payload_starts_at_epoch_one(synthetic_cache, tmp_path):
    cache_dir, manifest, index = synthetic_cache
    dataset = UnifiedPolarityDataset(
        cache_dir, split="train", max_samples=32, seed=0, manifest=manifest, index=index
    )
    trainer = _trainer(tmp_path, epochs=1)
    summary = trainer.fit(dataset, dataset)
    assert summary["history"][0]["epoch"] == 1
