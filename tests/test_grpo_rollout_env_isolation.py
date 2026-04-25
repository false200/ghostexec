# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Ensures each GRPO-style episode starts from the same scenario baseline
# (reset / fresh env) so parallel samples are not correlated by leaked state.

from __future__ import annotations

from pathlib import Path

from ghostexec.models import GhostexecAction
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"


def _unreplied_critical_count(env: GhostexecEnvironment) -> int:
    return sum(1 for e in env.world.emails if e.priority == "critical" and not e.replied)


def test_reset_restores_world_after_reply() -> None:
    """Same env + reset() must restore initial unreplied critical count."""
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    before = _unreplied_critical_count(env)
    env.step(
        GhostexecAction(
            action_type="reply_email",
            email_id="e01",
            message_body="Working on it now.",
        )
    )
    assert _unreplied_critical_count(env) < before
    env.reset()
    assert _unreplied_critical_count(env) == before


def test_fresh_env_instances_match_baseline() -> None:
    """Two fresh environments after reset() must match the same baseline."""
    a = GhostexecEnvironment(SCENARIO)
    b = GhostexecEnvironment(SCENARIO)
    a.reset()
    b.reset()
    assert _unreplied_critical_count(a) == _unreplied_critical_count(b)
