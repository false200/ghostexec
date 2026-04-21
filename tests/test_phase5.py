# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Phase 5: training log, demo scenarios, Colab notebook structure, optional HF health.

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ghostexec.models import GhostexecAction
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SC_MONDAY = ROOT / "scenarios" / "monday_morning.json"
SC_VIP = ROOT / "scenarios" / "vip_meltdown.json"
SC_VIP_DRIFT = ROOT / "scenarios" / "vip_meltdown_drift.json"
TRAIN = ROOT / "training" / "train.py"
NOTEBOOKS = (
    ROOT / "training" / "ghostexec_colab.ipynb",
    ROOT / "training" / "ghostexec_unsloth_grpo_colab.ipynb",
)


def _train_subprocess(train_args: list[str]) -> tuple[list[str], dict[str, str] | None]:
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python", str(TRAIN), *train_args], None
    prev = os.environ.get("PYTHONPATH", "")
    merged = os.pathsep.join([str(ROOT), prev]).strip(os.pathsep)
    return [sys.executable, str(TRAIN), *train_args], {**os.environ, "PYTHONPATH": merged}


def test_training_script_writes_reward_log(tmp_path: Path) -> None:
    logf = tmp_path / "episodes.jsonl"
    cmd, env = _train_subprocess(
        [
            "--backend",
            "local",
            "--agent",
            "smart",
            "--episodes",
            "22",
            "--max-steps",
            "10",
            "--log-path",
            str(logf),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--scenario",
            str(ROOT / "scenarios" / "phase2_core.json"),
        ]
    )
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    assert logf.is_file()
    lines = [ln for ln in logf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 20
    row = json.loads(lines[0])
    assert "return" in row and "episode" in row and row["agent"] == "smart"


def test_monday_morning_scenario_loads_and_steps() -> None:
    assert SC_MONDAY.is_file()
    env = GhostexecEnvironment(SC_MONDAY)
    obs = env.reset()
    assert "BRIEFING" in obs.echoed_message
    r = env.step(GhostexecAction(action_type="do_nothing")).reward
    assert isinstance(r, float)


def test_vip_meltdown_drift_escalates_mood() -> None:
    env = GhostexecEnvironment(SC_VIP, schema_drift_events_path=SC_VIP_DRIFT)
    env.reset()
    env.step(GhostexecAction(action_type="do_nothing"))
    taylor = env.get_contact("Taylor Brooks")
    assert taylor is not None
    assert taylor.mood == "annoyed"
    env.step(GhostexecAction(action_type="do_nothing"))
    assert env.get_contact("Taylor Brooks").mood == "angry"
    env.step(GhostexecAction(action_type="do_nothing"))
    assert env.get_contact("Taylor Brooks").mood == "furious"


def test_colab_notebooks_are_valid_json_with_runnable_cells() -> None:
    for path in NOTEBOOKS:
        assert path.is_file(), f"missing notebook: {path}"
        nb = json.loads(path.read_text(encoding="utf-8"))
        assert nb.get("nbformat") == 4
        cells = nb.get("cells", [])
        min_cells = 4 if path.name == "ghostexec_colab.ipynb" else 8
        assert len(cells) >= min_cells, f"{path.name}: expected at least {min_cells} cells"
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        assert code_cells, f"{path.name} should include code cells"
        for c in code_cells:
            ast.parse("".join(c.get("source", [])))


def test_hf_space_health_if_url_configured() -> None:
    url = os.environ.get("GHOSTEXEC_HF_SPACE_URL", "").strip().rstrip("/")
    if not url:
        pytest.skip("Set GHOSTEXEC_HF_SPACE_URL to your public Space for live HTTP check.")
    health = f"{url}/health"
    req = urllib.request.Request(health, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            assert resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 308):
            pytest.skip(f"Redirect from {health}; follow redirects in browser.")
        raise
