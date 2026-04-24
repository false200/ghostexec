"""Smoke checks for the random-policy held-out baseline."""

from __future__ import annotations

import math

from training.random_baseline import evaluate_random_policy


def test_random_baseline_returns_finite_metrics():
    report = evaluate_random_policy(episodes_per_scenario=1, max_steps=3, seed=7)
    agg = report.aggregate()
    assert int(agg.get("episodes", 0)) > 0

    expected_keys = (
        "mean_return",
        "format_valid_rate",
        "vip_critical_first_reply_rate",
        "conflicts_resolved_rate",
        "mean_channel_conflict",
        "mean_channel_relationship",
        "mean_channel_task",
    )
    for key in expected_keys:
        value = float(agg.get(key, 0.0))
        assert math.isfinite(value), f"{key} must be finite, got {value!r}"
