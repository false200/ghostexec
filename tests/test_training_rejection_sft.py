# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/rejection_sft.py."""

from __future__ import annotations

import json
from pathlib import Path

from training.rejection_sft import (
    build_dataset,
    filter_top,
    generate_samples,
    write_jsonl,
)


def test_generate_samples_produces_briefing_and_json() -> None:
    rows = generate_samples(4, seed=0)
    assert len(rows) == 4
    for r in rows:
        assert "prompt" in r and r["prompt"]
        js = json.loads(r["completion"])  # must be valid JSON
        assert "action_type" in js
        assert "reward" in r and isinstance(r["reward"], float)


def test_filter_top_keeps_upper_quantile() -> None:
    samples = [{"prompt": "p", "completion": "{}", "reward": float(i)} for i in range(10)]
    kept = filter_top(samples, quantile=0.3, min_reward=None)
    # 30% of 10 = 3 samples, top rewards should be 7,8,9.
    assert len(kept) == 3
    assert min(s["reward"] for s in kept) >= 7


def test_filter_top_respects_min_reward_floor() -> None:
    samples = [{"prompt": "p", "completion": "{}", "reward": float(i)} for i in range(5)]
    kept = filter_top(samples, quantile=1.0, min_reward=3.0)
    assert all(s["reward"] >= 3.0 for s in kept)


def test_build_dataset_and_write_jsonl(tmp_path: Path) -> None:
    rows = build_dataset(6, quantile=0.5, min_reward=None, seed=1)
    out = write_jsonl(rows, tmp_path / "sft.jsonl")
    text = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == len(rows)
    for line in text:
        obj = json.loads(line)
        assert "prompt" in obj and "completion" in obj and "reward" in obj
