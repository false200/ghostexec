# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Callable rewards for TRL GRPOTrainer.
#
# ``ghostexec_env_step_reward`` is the core scalar: fresh env reset + one step
# of the parsed action (optional k-step scripted lookahead via smart_action).
# The shaping reward functions below (format / diversity / id-relevance /
# VIP-critical) are additive side-channels: GRPO averages across ``reward_funcs``
# entries, so the 0.35 / 0.35 / 0.30 core blend stays intact.
#
# For the OpenEnv + TRL pattern (``rollout_func`` + ``generate_rollout_completions``
# + multi-key kwargs), see Meta's tutorial and ``training/openenv_grpo_rollout.py``:
# https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

from .llm_action_parse import completion_to_str, parse_completion_to_action
from .scenarios_sampler import (
    SCENARIOS_ROOT,
    load_perturbed_scenario,
    pick_scenario,
    scenario_path,
)

try:
    from ghostexec.models import GhostexecAction
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction
    from server.ghostexec_environment import GhostexecEnvironment


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _bounded_squash(raw: float, scale: float) -> float:
    """PAR-style bounded reward transform (``tanh(raw / scale)``).

    The "Reward Shaping to Mitigate Reward Hacking in RLHF" paper (PAR, 2025)
    argues an RL reward should be (1) bounded and (2) exhibit rapid initial
    growth followed by gradual convergence. ``tanh`` has exactly those two
    properties, keeps the output in ``(-1, 1)``, and preserves sign / ordering
    of the underlying raw reward, so GRPO advantages stay well-conditioned.
    """
    if scale <= 0:
        return raw
    return math.tanh(raw / scale)


def _tie_break_jitter(texts: list[str], magnitude: float) -> list[float]:
    """Return deterministic micro-jitter per completion text.

    GRPO computes advantage as ``(reward - group_mean) / group_std``. When a
    group's completions all produce an identical reward, ``group_std = 0`` so
    the gradient signal is zero — the "dead signal" pattern behind the flat
    reward curves we observed before. Adding a text-dependent jitter of order
    1e-3 (much smaller than any real reward channel) breaks ties without
    teaching the model anything spurious: identical completions still get
    identical jitter, so the advantage only becomes non-zero when the
    completions genuinely differ.
    """
    if magnitude <= 0:
        return [0.0] * len(texts)
    out: list[float] = []
    for t in texts:
        h = hashlib.md5(t.encode("utf-8", "replace")).digest()
        val = int.from_bytes(h[:8], "big") / float(1 << 64)
        out.append((val - 0.5) * 2.0 * magnitude)
    return out


def _default_scenario_path() -> Path:
    env_path = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if env_path:
        return Path(env_path)
    return SCENARIOS_ROOT / "phase2_core.json"


def _pick_scenario_for_call(rng: random.Random) -> Path:
    """Pick a scenario honoring ``GHOSTEXEC_GRPO_SCENARIO`` (pin) then curriculum then default."""
    env_path = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if env_path:
        return Path(env_path)
    level = os.environ.get("GHOSTEXEC_CURRICULUM", "").strip()
    if level:
        return pick_scenario(rng, level)
    return SCENARIOS_ROOT / "phase2_core.json"


def _load_world_data(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _episode_reward(scen: Path, act: GhostexecAction, rng: random.Random) -> float:
    k_steps = _env_int("GHOSTEXEC_REWARD_KSTEPS", 0)
    gamma = _env_float("GHOSTEXEC_REWARD_GAMMA", 0.9)
    env = GhostexecEnvironment(scen)
    env.reset()
    obs = env.step(act)
    total = float(obs.reward or 0.0)
    if k_steps > 0:
        # Lazy import to avoid argparse-heavy module at package load time.
        from .train import smart_action

        discount = gamma
        for _ in range(k_steps):
            if obs.done:
                break
            follow = smart_action(obs, rng)
            obs = env.step(follow)
            total += discount * float(obs.reward or 0.0)
            discount *= gamma
    return total


def ghostexec_env_step_reward(
    prompts: list[Any],
    completions: list[Any],
    **kwargs: object,
) -> list[float]:
    """Core scalar reward: parse completion -> action -> env reward.

    Behavior knobs (env vars):
    - ``GHOSTEXEC_GRPO_SCENARIO``: hard-pin a scenario path (back-compat default).
    - ``GHOSTEXEC_CURRICULUM``: rotate over ``easy|mid|hard|all`` scenarios.
    - ``GHOSTEXEC_PERTURB=1``: shuffle list order / sim-time inside the chosen scenario.
    - ``GHOSTEXEC_REWARD_KSTEPS``: extend each rollout with N scripted follow-ups.
    - ``GHOSTEXEC_REWARD_GAMMA``: discount for those follow-ups.
    """
    perturb = os.environ.get("GHOSTEXEC_PERTURB", "0").strip() not in ("", "0", "false", "False")
    rewards: list[float] = []
    texts: list[str] = []
    for i, completion in enumerate(completions):
        rng = random.Random(1_000_003 * (i + 1))
        scen = _pick_scenario_for_call(rng)
        cleanup: Path | None = None
        if perturb:
            try:
                cleanup = load_perturbed_scenario(scen, rng)
                scen = cleanup
            except Exception:
                cleanup = None
        try:
            act = parse_completion_to_action(completion) or GhostexecAction(action_type="do_nothing")
            rewards.append(_episode_reward(scen, act, rng))
            texts.append(completion_to_str(completion))
        finally:
            if cleanup is not None:
                try:
                    cleanup.unlink(missing_ok=True)
                except Exception:
                    pass

    # PAR-style bounded squash: keeps rewards in (-1, 1), preserves ordering.
    if _env_bool("GHOSTEXEC_REWARD_SQUASH", True):
        scale = _env_float("GHOSTEXEC_REWARD_SQUASH_SCALE", 2.0)
        rewards = [_bounded_squash(r, scale) for r in rewards]

    # Anti dead-signal jitter: only kicks in when the group actually collapses
    # to (near) zero variance. Preserves learning on genuinely diverse groups.
    if _env_bool("GHOSTEXEC_REWARD_JITTER", True) and len(rewards) >= 2:
        try:
            sigma = statistics.pstdev(rewards)
        except statistics.StatisticsError:
            sigma = 0.0
        if sigma < 1e-6:
            magnitude = _env_float("GHOSTEXEC_REWARD_JITTER_MAG", 1e-3)
            rewards = [r + j for r, j in zip(rewards, _tie_break_jitter(texts, magnitude))]

    return rewards


# --- Shaping reward channels (plug into GRPOConfig.reward_funcs as a list) ---


def reward_format_valid(
    prompts: list[Any], completions: list[Any], **kwargs: object
) -> list[float]:
    """+1 if the completion parses into a valid ``GhostexecAction``, else -1."""
    return [1.0 if parse_completion_to_action(c) is not None else -1.0 for c in completions]


def reward_group_diversity(
    prompts: list[Any], completions: list[Any], **kwargs: object
) -> list[float]:
    """Penalize duplicate completions within a GRPO group to prevent collapse."""
    counts: dict[str, int] = {}
    texts: list[str] = []
    for c in completions:
        t = completion_to_str(c).strip()
        texts.append(t)
        counts[t] = counts.get(t, 0) + 1
    return [(1.0 / counts[t]) - 1.0 if counts[t] > 1 else 0.0 for t in texts]


def reward_id_relevance(
    prompts: list[Any], completions: list[Any], **kwargs: object
) -> list[float]:
    """Small penalty when the action cites an ID that doesn't exist in the scenario.

    Uses the same scenario-pick logic as ``ghostexec_env_step_reward`` so the
    channel stays aligned with what the env actually saw.
    """
    rng = random.Random(0)
    scen = _pick_scenario_for_call(rng)
    data = _load_world_data(scen)
    email_ids = {e.get("id") for e in data.get("emails", []) if isinstance(e, dict)}
    task_ids = {t.get("id") for t in data.get("tasks", []) if isinstance(t, dict)}
    meeting_ids = {m.get("id") for m in data.get("meetings", []) if isinstance(m, dict)}
    out: list[float] = []
    for c in completions:
        act = parse_completion_to_action(c)
        if act is None:
            out.append(0.0)
            continue
        score = 0.0
        if act.email_id and act.email_id not in email_ids:
            score -= 0.25
        if act.task_id and act.task_id not in task_ids:
            score -= 0.25
        if act.meeting_id and act.meeting_id not in meeting_ids:
            score -= 0.25
        out.append(score)
    return out


def reward_vip_critical_reply_bonus(
    prompts: list[Any], completions: list[Any], **kwargs: object
) -> list[float]:
    """Bonus when the action is a non-empty reply to a VIP + critical + unreplied email."""
    rng = random.Random(0)
    scen = _pick_scenario_for_call(rng)
    data = _load_world_data(scen)
    vip_critical_unreplied = {
        e.get("id")
        for e in data.get("emails", [])
        if isinstance(e, dict)
        and e.get("priority") == "critical"
        and e.get("sender_relationship") == "VIP"
        and not e.get("replied")
    }
    out: list[float] = []
    for c in completions:
        act = parse_completion_to_action(c)
        if (
            act is not None
            and act.action_type == "reply_email"
            and act.email_id in vip_critical_unreplied
            and (act.message_body or "").strip()
        ):
            out.append(0.5)
        else:
            out.append(0.0)
    return out


REWARD_FUNCS_ALL = [
    ghostexec_env_step_reward,
    reward_format_valid,
    reward_group_diversity,
    reward_id_relevance,
    reward_vip_critical_reply_bonus,
]
