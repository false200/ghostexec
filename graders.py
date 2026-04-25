"""
Public trajectory graders for OpenEnv Phase 2 / HF deep validation.

These are **episode-level** scores (strictly inside (0, 1)), separate from per-step
rewards in `server/reward.py`. The hackathon validator reads `openenv.yaml`
`tasks[].grader` and calls these functions with trajectory dicts.
"""
from __future__ import annotations

from typing import Iterable, List

STRICT_MIN = 0.01
STRICT_MAX = 0.99


def _bounded(value: float) -> float:
    return min(max(round(float(value), 4), STRICT_MIN), STRICT_MAX)


def _as_reward_list(trajectory: dict | None) -> List[float]:
    payload = trajectory or {}
    rewards = payload.get("rewards")
    if isinstance(rewards, list) and rewards:
        return [float(r) for r in rewards]
    if "score" in payload:
        return [float(payload["score"])]
    reward = payload.get("reward")
    if isinstance(reward, dict) and "total" in reward:
        return [float(reward["total"])]
    if reward is not None:
        return [float(reward)]
    return []


def _profile(reward: float) -> str:
    if reward <= 0.05:
        return "unsafe_miss"
    if reward <= 0.20:
        return "bad_call"
    if reward < 0.50:
        return "weak"
    if reward < 0.80:
        return "workable"
    if reward < 0.95:
        return "strong"
    return "expert"


def _score_episode(
    rewards: List[float],
    *,
    miss_cost: float,
    overcall_cost: float,
    stability_gain: float,
    expertise_gain: float,
) -> float:
    if not rewards:
        return _bounded(0.5)
    labels = [_profile(r) for r in rewards]
    mean_r = sum(rewards) / len(rewards)
    n = len(rewards)
    miss = labels.count("unsafe_miss")
    bad = labels.count("bad_call")
    weak = labels.count("weak")
    strong = labels.count("strong") + labels.count("expert")
    expert = labels.count("expert")

    downward = (
        min(miss * miss_cost, 0.35)
        + min(bad * overcall_cost, 0.15)
        + min(weak * 0.015, 0.06)
    )
    upward = 0.0
    if strong / n >= 0.80:
        upward += stability_gain
    if expert / n >= 0.60:
        upward += expertise_gain

    return _bounded(mean_r - downward + upward)


def phase2_core_grader(trajectory: dict | None = None) -> float:
    """Easy tier — dense default inbox (scenarios/phase2_core.json)."""
    return _score_episode(
        _as_reward_list(trajectory),
        miss_cost=0.12,
        overcall_cost=0.03,
        stability_gain=0.05,
        expertise_gain=0.01,
    )


def monday_morning_grader(trajectory: dict | None = None) -> float:
    """Medium tier — stacked Monday conflicts (scenarios/monday_morning.json)."""
    return _score_episode(
        _as_reward_list(trajectory),
        miss_cost=0.09,
        overcall_cost=0.04,
        stability_gain=0.03,
        expertise_gain=0.02,
    )


def dinner_disaster_grader(trajectory: dict | None = None) -> float:
    """Hard tier — personal/professional collision (scenarios/dinner_disaster.json)."""
    return _score_episode(
        _as_reward_list(trajectory),
        miss_cost=0.07,
        overcall_cost=0.03,
        stability_gain=0.02,
        expertise_gain=0.04,
    )


__all__ = [
    "phase2_core_grader",
    "monday_morning_grader",
    "dinner_disaster_grader",
    "STRICT_MIN",
    "STRICT_MAX",
]
