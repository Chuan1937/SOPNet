from __future__ import annotations

import torch

from sopnet.data.dataset import UnifiedPolarityDataset
from sopnet.evaluation.evaluate import collect_predictions
from sopnet.models import build_model
from sopnet.training.engine import TrainConfig, Trainer


def test_multitask_forward_shapes():
    model = build_model({"model": {"name": "sopnet_multi"}})
    field, logits = model(torch.randn(2, 1, 400))
    assert field.shape == (2, 1, 400)
    assert logits.shape == (2, 2)
    assert float(field.abs().max()) <= 1.0

    field, logits = model(torch.randn(2, 1, 400), x_pol=torch.randn(2, 1, 160))
    assert logits.shape == (2, 2)
    prediction = model.predict(torch.randn(2, 1, 400))
    assert prediction["polarity"].shape == (2,)
    assert prediction["p_position"].shape == (2,)


def test_multitask_training_and_collect_predictions(synthetic_cache, tmp_path):
    cache_dir, manifest, index = synthetic_cache
    model = build_model({"model": {"name": "sopnet_multi"}})
    config = TrainConfig(
        task="field_multi",
        epochs=1,
        batch_size=16,
        num_workers=0,
        amp=False,
        device="cpu",
        lambda_cls=1.0,
        unknown_weight=0.0,
        seed=0,
    )
    trainer = Trainer(model, config, tmp_path / "run", run_config={"train": {"task": "field_multi"}})
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
    summary = trainer.fit(train_dataset, val_dataset)
    assert "known_accuracy" in summary["history"][0]["val"]

    loader = [
        {
            "x": torch.randn(8, 1, 400),
            "label": torch.tensor([1, -1, 0, 1, -1, 0, 1, -1]),
            "p_pick": torch.full((8,), 200),
            "target": torch.zeros(8, 400),
        }
    ]
    outputs = collect_predictions(model, loader, device="cpu", task="field")
    assert set(torch.unique(torch.from_numpy(outputs["predictions"])).tolist()) <= {1, -1}
    assert outputs["confidence"].min() >= 0.0
    assert outputs["confidence"].max() <= 1.0
    assert outputs["p_pred"].shape == (8,)
