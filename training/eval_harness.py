# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Held-out evaluation for Ghostexec policies.
#
# A policy is any callable ``(obs, rng) -> GhostexecAction``. The harness runs
# N episodes per scenario, aggregates mean return, format-valid rate,
# VIP-critical-reply rate, conflicts-resolved rate, and per-channel reward
# averages (conflict / relationship / task) pulled from ``reward_breakdown``.
#
# Use with ``EVAL_SCENARIOS`` from ``scenarios_sampler`` to keep train / eval
# splits separate.

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .llm_action_parse import parse_completion_to_action
from .scenarios_sampler import EVAL_SCENARIOS, scenario_path

try:
    from ghostexec.models import GhostexecAction, GhostexecObservation
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction, GhostexecObservation
    from server.ghostexec_environment import GhostexecEnvironment


Policy = Callable[[GhostexecObservation, random.Random], GhostexecAction]
TextPolicy = Callable[[GhostexecObservation, random.Random], str]


def scripted_smart_policy() -> Policy:
    """Reference baseline: the scripted ``smart_action`` from ``train``."""
    from .train import smart_action

    return smart_action


def text_policy_as_action(text_policy: TextPolicy) -> Policy:
    """Adapt a text-emitting policy (LLM) into a ``GhostexecAction`` policy."""

    def _wrap(obs: GhostexecObservation, rng: random.Random) -> GhostexecAction:
        raw = text_policy(obs, rng)
        act = parse_completion_to_action(raw)
        return act or GhostexecAction(action_type="do_nothing")

    return _wrap


@dataclass
class EpisodeMetrics:
    scenario: str
    episode: int
    steps: int
    total_reward: float
    first_action: dict[str, Any] | None
    first_action_valid: bool
    vip_critical_first_reply: bool
    initial_conflicts: int
    final_conflicts: int
    conflicts_resolved: int
    channel_conflict: float = 0.0
    channel_relationship: float = 0.0
    channel_task: float = 0.0


@dataclass
class EvalReport:
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def aggregate(self) -> dict[str, float]:
        n = len(self.episodes)
        if not n:
            return {
                "episodes": 0,
                "mean_return": 0.0,
                "format_valid_rate": 0.0,
                "vip_critical_first_reply_rate": 0.0,
                "conflicts_resolved_rate": 0.0,
                "mean_channel_conflict": 0.0,
                "mean_channel_relationship": 0.0,
                "mean_channel_task": 0.0,
            }
        total_initial = sum(e.initial_conflicts for e in self.episodes) or 1
        return {
            "episodes": n,
            "mean_return": sum(e.total_reward for e in self.episodes) / n,
            "format_valid_rate": sum(1.0 for e in self.episodes if e.first_action_valid) / n,
            "vip_critical_first_reply_rate": sum(
                1.0 for e in self.episodes if e.vip_critical_first_reply
            )
            / n,
            "conflicts_resolved_rate": sum(e.conflicts_resolved for e in self.episodes)
            / total_initial,
            "mean_channel_conflict": sum(e.channel_conflict for e in self.episodes) / n,
            "mean_channel_relationship": sum(e.channel_relationship for e in self.episodes) / n,
            "mean_channel_task": sum(e.channel_task for e in self.episodes) / n,
        }


def _first_action_valid_and_vip(
    first_action: dict[str, Any] | None,
    vip_crit_unreplied: set[str],
) -> tuple[bool, bool]:
    if first_action is None:
        return False, False
    try:
        act = GhostexecAction.model_validate(first_action)
    except Exception:
        return False, False
    valid = True
    hit = (
        act.action_type == "reply_email"
        and bool((act.message_body or "").strip())
        and act.email_id in vip_crit_unreplied
    )
    return valid, hit


def _vip_critical_unreplied(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {
        e.get("id")
        for e in data.get("emails", [])
        if isinstance(e, dict)
        and e.get("priority") == "critical"
        and e.get("sender_relationship") == "VIP"
        and not e.get("replied")
    }


def run_episode(
    scenario: Path,
    policy: Policy,
    *,
    max_steps: int = 16,
    seed: int = 0,
) -> EpisodeMetrics:
    rng = random.Random(seed)
    env = GhostexecEnvironment(scenario)
    obs = env.reset()
    initial_conflicts = len(env.world.active_conflicts)
    vip_crit = _vip_critical_unreplied(scenario)
    total = 0.0
    steps = 0
    first_action: dict[str, Any] | None = None
    ch_conflict = ch_relationship = ch_task = 0.0
    for _ in range(max_steps):
        act = policy(obs, rng)
        if first_action is None:
            first_action = act.model_dump(mode="json")
        obs = env.step(act)
        total += float(obs.reward or 0.0)
        steps += 1
        bd = (obs.metadata or {}).get("reward_breakdown") or {}
        ch_conflict += float(bd.get("conflict", 0.0))
        ch_relationship += float(bd.get("relationship", 0.0))
        ch_task += float(bd.get("task", 0.0))
        if obs.done:
            break
    valid, vip_hit = _first_action_valid_and_vip(first_action, vip_crit)
    final_conflicts = len(env.world.active_conflicts)
    return EpisodeMetrics(
        scenario=scenario.name,
        episode=seed,
        steps=steps,
        total_reward=total,
        first_action=first_action,
        first_action_valid=valid,
        vip_critical_first_reply=vip_hit,
        initial_conflicts=initial_conflicts,
        final_conflicts=final_conflicts,
        conflicts_resolved=max(0, initial_conflicts - final_conflicts),
        channel_conflict=ch_conflict,
        channel_relationship=ch_relationship,
        channel_task=ch_task,
    )


def evaluate(
    policy: Policy,
    *,
    scenarios: tuple[str, ...] = EVAL_SCENARIOS,
    episodes_per_scenario: int = 3,
    max_steps: int = 16,
    base_seed: int = 1000,
) -> EvalReport:
    """Run ``episodes_per_scenario`` rollouts on every held-out scenario."""
    report = EvalReport()
    for scen_name in scenarios:
        path = scenario_path(scen_name)
        for i in range(episodes_per_scenario):
            report.episodes.append(
                run_episode(path, policy, max_steps=max_steps, seed=base_seed + i)
            )
    return report


def write_report(report: EvalReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": report.aggregate(),
        "episodes": [asdict(e) for e in report.episodes],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "EpisodeMetrics",
    "EvalReport",
    "Policy",
    "TextPolicy",
    "evaluate",
    "run_episode",
    "scripted_smart_policy",
    "text_policy_as_action",
    "write_report",
]
