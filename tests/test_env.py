"""OpenEnv Phase 2 submission guardrails (graders + manifest wiring)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graders import (
    dinner_disaster_grader,
    monday_morning_grader,
    phase2_core_grader,
)

PUBLIC_GRADERS = (phase2_core_grader, monday_morning_grader, dinner_disaster_grader)


@pytest.mark.parametrize("grader", PUBLIC_GRADERS)
def test_public_graders_are_strictly_bounded(grader):
    assert grader({"rewards": [1.0]}) == 0.99
    assert grader({"rewards": [0.0]}) == 0.01
    assert grader({"rewards": [-5.0]}) == 0.01
    assert grader({"score": 1.5}) == 0.99
    assert grader({"score": -0.5}) == 0.01
    assert grader({"reward": {"total": 1.0}}) == 0.99
    v = grader(None)
    assert 0.0 < v < 1.0
    v = grader({})
    assert 0.0 < v < 1.0


def test_openenv_yaml_declares_three_tasks_with_graders():
    import yaml

    root = Path(__file__).resolve().parent.parent
    with (root / "openenv.yaml").open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    tasks = spec.get("tasks", [])
    assert len(tasks) >= 3, "Phase 2 requires >= 3 tasks"
    for t in tasks:
        assert "grader" in t, f"Task {t.get('id')} missing grader"
        module_path, _, func_name = t["grader"].rpartition(".")
        mod = importlib.import_module(module_path)
        assert callable(getattr(mod, func_name)), f"{t['grader']} not callable"
