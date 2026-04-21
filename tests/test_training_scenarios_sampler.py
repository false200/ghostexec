# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/scenarios_sampler.py (pure: no GPU / TRL)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from training.scenarios_sampler import (
    CURRICULUM,
    EVAL_SCENARIOS,
    SCENARIOS_ROOT,
    TRAIN_SCENARIOS,
    load_perturbed_scenario,
    pick_scenario,
    scenario_path,
)


def test_scenario_path_resolves_existing_file() -> None:
    for name in TRAIN_SCENARIOS + EVAL_SCENARIOS:
        p = scenario_path(name)
        assert p.is_file(), f"Scenario missing on disk: {p}"
        assert p.parent == SCENARIOS_ROOT


def test_pick_scenario_honors_curriculum() -> None:
    rng = random.Random(0)
    for _ in range(20):
        p = pick_scenario(rng, "easy")
        assert p.name in CURRICULUM["easy"]


def test_pick_scenario_falls_back_to_train_pool() -> None:
    rng = random.Random(1)
    p = pick_scenario(rng, "nope-unknown-level")
    assert p.name in TRAIN_SCENARIOS


def test_train_and_eval_sets_are_disjoint() -> None:
    assert set(TRAIN_SCENARIOS).isdisjoint(set(EVAL_SCENARIOS))


def test_perturbed_scenario_keeps_invariants() -> None:
    src = scenario_path("phase2_core.json")
    rng = random.Random(42)
    out = load_perturbed_scenario(src, rng, hours_shift=0)
    try:
        assert out.is_file() and out != src
        original = json.loads(src.read_text(encoding="utf-8"))
        mutated = json.loads(out.read_text(encoding="utf-8"))
        # Same set of emails / ids, only order can change.
        assert {e["id"] for e in original["emails"]} == {e["id"] for e in mutated["emails"]}
        assert {c["name"] for c in original["contacts"]} == {c["name"] for c in mutated["contacts"]}
        assert original["simulation_time"] == mutated["simulation_time"]
    finally:
        Path(out).unlink(missing_ok=True)


def test_perturbed_scenario_shifts_time_when_requested() -> None:
    src = scenario_path("phase2_core.json")
    rng = random.Random(7)
    out = load_perturbed_scenario(src, rng, hours_shift=3)
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["simulation_time"] != json.loads(src.read_text(encoding="utf-8"))["simulation_time"]
    finally:
        Path(out).unlink(missing_ok=True)
