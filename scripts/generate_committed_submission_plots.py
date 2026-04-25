#!/usr/bin/env python3
"""Regenerate ``docs/submission_results/*.png`` from real local runs (no GPU).

1. Random-policy rollouts on ``GhostexecEnvironment`` → ``outputs/logs/episode_rewards.jsonl``
2. ``training/train.py`` REINFORCE → ``outputs/training/smoke/reinforce_returns.jsonl``
3. ``scripts/export_submission_plots.py`` → committed PNGs under ``docs/submission_results/``

Run from repo root: ``uv run python scripts/generate_committed_submission_plots.py``
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG = ROOT / "outputs" / "logs" / "episode_rewards.jsonl"
SMOKE = ROOT / "outputs" / "training" / "smoke"
RETURNS = SMOKE / "reinforce_returns.jsonl"
SCENARIO = ROOT / "scenarios" / "phase2_core.json"


def _write_random_rollout_jsonl(*, episodes: int, seed: int) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from ghostexec.models import GhostexecAction, GhostexecObservation
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
    from training.train import random_valid_action

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.unlink(missing_ok=True)
    rng = random.Random(seed)
    for _ in range(episodes):
        env = GhostexecEnvironment(SCENARIO)
        obs = env.reset()
        for _step in range(128):
            act = random_valid_action(obs, rng)
            obs = env.step(act)
            if obs.done:
                break


def _run_reinforce_smoke(*, episodes: int, seed: int) -> None:
    SMOKE.mkdir(parents=True, exist_ok=True)
    RETURNS.unlink(missing_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "training" / "train.py"),
        "--backend",
        "local",
        "--agent",
        "reinforce",
        "--episodes",
        str(episodes),
        "--max-steps",
        "14",
        "--seed",
        str(seed),
        "--scenario",
        str(SCENARIO),
        "--log-path",
        str(RETURNS),
        "--checkpoint-dir",
        str(SMOKE / "checkpoints"),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    _write_random_rollout_jsonl(episodes=36, seed=2026)
    _run_reinforce_smoke(episodes=48, seed=4242)
    exp = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_submission_plots.py")],
        cwd=str(ROOT),
    )
    return int(exp.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
