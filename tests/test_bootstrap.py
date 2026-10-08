from __future__ import annotations

import numpy as np

from sopnet.evaluation.bootstrap import paired_bootstrap_difference_clustered


def test_clustered_bootstrap_detects_improvement():
    rng = np.random.default_rng(0)
    n_groups, per_group = 100, 20
    groups = np.repeat(np.arange(n_groups), per_group)
    y_true = rng.choice([-1, 1], size=n_groups * per_group)
    pred_a = y_true.copy()
    pred_b = np.where(rng.random(y_true.size) < 0.2, -y_true, y_true)

    result = paired_bootstrap_difference_clustered(y_true, pred_a, pred_b, groups, n_resamples=200)
    assert result["delta"] > 0
    assert result["ci_low"] > 0
    assert result["n_groups"] == n_groups
    assert result["p_improvement"] > 0.95


def test_clustered_bootstrap_handles_identical_predictions():
    groups = np.repeat(np.arange(10), 5)
    y_true = np.ones(50, dtype=int)
    y_true[::2] = -1
    result = paired_bootstrap_difference_clustered(y_true, y_true, y_true, groups, n_resamples=50)
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0
