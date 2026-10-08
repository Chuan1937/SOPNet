from __future__ import annotations

import numpy as np

from sopnet.data.augment import AugmentConfig
from sopnet.data.dataset import DEFAULT_JITTER, UnifiedPolarityDataset


def _first_known_index(dataset):
    for index in range(len(dataset)):
        if int(dataset.labels[index]) != 0:
            return index
    raise AssertionError("synthetic cache has no known polarity labels")


def test_fixed_center_crop_puts_p_at_window_center(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    dataset = UnifiedPolarityDataset(cache_dir, split=None, jitter=None, manifest=manifest, index=index)
    assert len(dataset) == len(manifest)
    item = dataset[_first_known_index(dataset)]
    assert item["x"].shape == (1, 400)
    assert item["target"].shape == (400,)
    assert item["p_pick"].item() == 200
    assert int(item["target"].abs().argmax()) == 200


def test_jitter_keeps_p_inside_requested_range(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    dataset = UnifiedPolarityDataset(
        cache_dir, split=None, jitter=DEFAULT_JITTER, sigma=10.0, manifest=manifest, index=index
    )
    for i in range(30):
        item = dataset[i]
        assert 120 <= item["p_pick"].item() <= 280
        if item["label"].item() != 0:
            assert int(item["target"].abs().argmax()) == item["p_pick"].item()


def test_target_sign_matches_label(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    dataset = UnifiedPolarityDataset(cache_dir, split=None, jitter=None, manifest=manifest, index=index)
    for i in range(len(dataset)):
        item = dataset[i]
        label = item["label"].item()
        target = item["target"].numpy()
        if label == 0:
            assert np.all(target == 0.0)
        else:
            assert np.sign(target[int(item["p_pick"])]) == np.sign(label)


def test_inversion_augmentation_flips_label_and_target(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    augment = AugmentConfig(p_noise=0.0, p_invert=1.0, p_scale=0.0)
    dataset = UnifiedPolarityDataset(
        cache_dir, split=None, jitter=None, augment=augment, seed=1, manifest=manifest, index=index
    )
    flipped = 0
    for i in range(len(dataset)):
        item = dataset[i]
        original = int(manifest["canonical_label"].iloc[i])
        if original == 0:
            continue
        assert item["label"].item() == -original
        flipped += 1
    assert flipped > 0


def test_p_shift_moves_reference_window(synthetic_cache):
    # Positive shift moves the analysis window later, so the P appears earlier.
    cache_dir, manifest, index = synthetic_cache
    dataset = UnifiedPolarityDataset(
        cache_dir, split=None, jitter=None, p_shift_samples=50, manifest=manifest, index=index
    )
    assert dataset[0]["p_pick"].item() == 150


def test_p_shift_applies_to_full_cache_window(synthetic_cache):
    """Regression: with ``window_length=600`` the shift used to be clipped away."""
    cache_dir, manifest, index = synthetic_cache
    reference = UnifiedPolarityDataset(
        cache_dir, split=None, jitter=None, window_length=600, manifest=manifest, index=index
    )[0]["x"][0].numpy()

    plus = UnifiedPolarityDataset(
        cache_dir,
        split=None,
        jitter=None,
        window_length=600,
        p_shift_samples=20,
        manifest=manifest,
        index=index,
    )
    item_plus = plus[0]
    assert item_plus["x"].shape == (1, 600)
    assert item_plus["p_pick"].item() == 280
    shifted = item_plus["x"][0].numpy()
    assert np.allclose(shifted[:580], reference[20:])
    assert np.all(shifted[580:] == 0.0)

    minus = UnifiedPolarityDataset(
        cache_dir,
        split=None,
        jitter=None,
        window_length=600,
        p_shift_samples=-20,
        manifest=manifest,
        index=index,
    )
    item_minus = minus[0]
    assert item_minus["p_pick"].item() == 320
    shifted_minus = item_minus["x"][0].numpy()
    assert np.allclose(shifted_minus[20:], reference[:580])
    assert np.all(shifted_minus[:20] == 0.0)


def test_max_samples_subsets_deterministically(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    first = UnifiedPolarityDataset(
        cache_dir, split="train", max_samples=5, seed=7, manifest=manifest, index=index
    )
    second = UnifiedPolarityDataset(
        cache_dir, split="train", max_samples=5, seed=7, manifest=manifest, index=index
    )
    assert len(first) == 5
    assert first.sample_ids.tolist() == second.sample_ids.tolist()


def test_split_filtering(synthetic_cache):
    cache_dir, manifest, index = synthetic_cache
    for split in ("train", "val", "test"):
        dataset = UnifiedPolarityDataset(cache_dir, split=split, manifest=manifest, index=index)
        expected = int((manifest["split"] == split).sum())
        assert len(dataset) == expected
