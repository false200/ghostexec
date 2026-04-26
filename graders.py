"""
Public trajectory graders for OpenEnv Phase 2 / HF deep validation.

These are **episode-level** scores (strictly inside (0, 1)), separate from per-step
rewards in `server/reward.py`. The hackathon validator reads `openenv.yaml`
`tasks[].grader` and calls these functions with trajectory dicts.
"""
from __future__ import annotations

import math
from typing import List

STRICT_MIN = 0.01
STRICT_MAX = 0.99


def _bounded(value: float) -> float:
    try:
        v = round(float(value), 4)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return min(max(v, STRICT_MIN), STRICT_MAX)


def _as_reward_list(trajectory: dict | None) -> List[float]:
    payload = trajectory or {}
    if not isinstance(payload, dict):
        return []
    rewards = payload.get("rewards")
    if isinstance(rewards, list) and rewards:
        out: List[float] = []
        for r in rewards:
            try:
                rv = float(r)
            except (TypeError, ValueError):
                continue
            if math.isfinite(rv):
                out.append(rv)
        return out
    if "score" in payload:
        try:
            v = float(payload["score"])
            return [v] if math.isfinite(v) else []
        except (TypeError, ValueError):
            return []
    reward = payload.get("reward")
    if isinstance(reward, dict) and "total" in reward:
        try:
            v = float(reward["total"])
            return [v] if math.isfinite(v) else []
        except (TypeError, ValueError):
            return []
    if reward is not None:
        try:
            v = float(reward)
            return [v] if math.isfinite(v) else []
        except (TypeError, ValueError):
            return []
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
