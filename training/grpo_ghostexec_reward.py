# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Callable reward for TRL GRPOTrainer: first Ghostexec step from a deterministic reset.

from __future__ import annotations

import os
from pathlib import Path

from .llm_action_parse import parse_completion_to_action

try:
    from ghostexec.models import GhostexecAction
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction
    from server.ghostexec_environment import GhostexecEnvironment


def _default_scenario_path() -> Path:
    env_path = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[1] / "scenarios" / "phase2_core.json"


def ghostexec_env_step_reward(prompts: list[str], completions: list[str], **kwargs: object) -> list[float]:
    """
    TRL-compatible reward: for each (prompt, completion), fresh env reset + one step.

    Same deterministic scenario yields identical initial state so GRPO group comparisons
    are meaningful for the first transition.
    """
    scenario = _default_scenario_path()
    rewards: list[float] = []
    for _p, completion in zip(prompts, completions):
        act = parse_completion_to_action(completion)
        if act is None:
            act = GhostexecAction(action_type="do_nothing")
        env = GhostexecEnvironment(scenario)
        env.reset()
        obs = env.step(act)
        rewards.append(float(obs.reward or 0.0))
    return rewards
