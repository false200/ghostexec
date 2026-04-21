# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/eval_harness.py using the scripted smart policy."""

from __future__ import annotations

import json
from pathlib import Path

from training.eval_harness import (
    EVAL_SCENARIOS,
    evaluate,
    scripted_smart_policy,
    write_report,
)


def test_evaluate_returns_metrics_for_eval_pool(tmp_path: Path) -> None:
    policy = scripted_smart_policy()
    report = evaluate(policy, episodes_per_scenario=1, max_steps=4)
    assert len(report.episodes) == len(EVAL_SCENARIOS)
    agg = report.aggregate()
    assert agg["episodes"] == len(EVAL_SCENARIOS)
    assert 0.0 <= agg["format_valid_rate"] <= 1.0
    assert 0.0 <= agg["vip_critical_first_reply_rate"] <= 1.0
    out = write_report(report, tmp_path / "eval.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "aggregate" in payload and "episodes" in payload


def test_evaluate_aggregate_shape() -> None:
    report = evaluate(
        scripted_smart_policy(),
        scenarios=("vip_meltdown.json",),
        episodes_per_scenario=2,
        max_steps=3,
    )
    agg = report.aggregate()
    for key in (
        "mean_return",
        "format_valid_rate",
        "vip_critical_first_reply_rate",
        "conflicts_resolved_rate",
        "mean_channel_conflict",
        "mean_channel_relationship",
        "mean_channel_task",
    ):
        assert key in agg
