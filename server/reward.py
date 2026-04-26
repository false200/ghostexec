# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Phase 4 reward: weighted (0.35 / 0.35 / 0.30) with potential-style deltas, critical-queue
shaping, full sub-scores even on invalid steps (+ explicit invalid penalty), and mild output
scaling.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from ..models import GhostexecAction, RewardBreakdown, WorldState
except ImportError:
    from models import GhostexecAction, RewardBreakdown, WorldState

W_CONFLICT = 0.35
W_REL = 0.35
W_TASK = 0.30

# Raw conflict units (pre-weight) are clamped to keep invalid / idle steps from exploding.
CONFLICT_RAW_CAP: float = 6.0

# Scales the weighted sum of the three channels (weights stay fixed per hackathon rules).
WEIGHTED_OUTPUT_SCALE: float = 0.48

# Tone misfit penalties kept small vs outcome terms (~<20% of a strong +2 conflict step after weights).
TONE_PENALTY_CASUAL_ANGRY_BOARD: float = 0.35
TONE_PENALTY_FORMAL_PERSONAL: float = 0.08

_RESOLVE_MICRO_BONUS: float = 0.12
_CRITICAL_PER_EMAIL_BONUS: float = 0.22
_RESCHEDULE_VALID_MICRO_BONUS: float = 0.10
_SEND_MESSAGE_VALID_MICRO_BONUS: float = 0.08
_COMPLETE_TASK_VALID_MICRO_BONUS: float = 0.06
_DELEGATE_TASK_VALID_MICRO_BONUS: float = 0.10
_DO_NOTHING_STRICT_PENALTY: float = -0.15
_SYNERGY_CAP: float = 0.40
_TRADEOFF_CAP: float = 0.30
_POTENTIAL_CAP: float = 0.25
_SCAFFOLD_CAP: float = 0.35
_SHAPING_TO_BASE_BUDGET: float = 1.25
_QUALITY_CAP: float = 0.28
_REPLY_PRIORITY_MICRO_BONUS: dict[str, float] = {
    "critical": 0.30,
    "high": 0.15,
    "normal": 0.04,
    "low": 0.02,
}

_MOOD_RANK: dict[str, int] = {
    "happy": 4,
    "neutral": 3,
    "annoyed": 2,
    "angry": 1,
    "furious": 0,
}


def _parse_dt(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _meeting_end(m: Any) -> datetime:
    start = _parse_dt(m.start)
    return start + timedelta(minutes=m.duration_minutes)


def _overlap(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> bool:
    return a0 < b1 and b0 < a1


def meeting_conflicts(world: WorldState) -> list[dict[str, Any]]:
    active = [m for m in world.meetings if not m.cancelled]
    out: list[dict[str, Any]] = []
    for i, a in enumerate(active):
        a0, a1 = _parse_dt(a.start), _meeting_end(a)
        for b in active[i + 1 :]:
            b0, b1 = _parse_dt(b.start), _meeting_end(b)
            if _overlap(a0, a1, b0, b1):
                o0, o1 = max(a0, b0), min(a1, b1)
                out.append(
                    {
                        "meeting_a": a.id,
                        "meeting_b": b.id,
                        "overlap_start": o0.isoformat(),
                        "overlap_end": o1.isoformat(),
                    }
                )
    return out


def _pair_set(rows: list[dict[str, Any]]) -> set[frozenset[str]]:
    return {frozenset((r["meeting_a"], r["meeting_b"])) for r in rows}


def _attendee_moods_ok(world: WorldState, pair: frozenset[str]) -> bool:
    names: set[str] = set()
    for mid in pair:
        m = next((x for x in world.meetings if x.id == mid), None)
        if m:
            names.update(m.attendees)
    for n in names:
        c = next((x for x in world.contacts if x.name == n), None)
        if c is None:
            continue
        if c.mood not in ("happy", "neutral"):
            return False
    return True


def score_conflict_resolution(
    before: WorldState,
    after: WorldState,
    action: GhostexecAction,
    *,
    action_ok: bool,
) -> float:
    b = _pair_set(meeting_conflicts(before))
    a = _pair_set(meeting_conflicts(after))
    s = 0.0
    for _p in b - a:
        s += 2.0 + _RESOLVE_MICRO_BONUS
        if _attendee_moods_ok(after, _p):
            s += 1.0
    for _ in a - b:
        s -= 3.0
    if action_ok and action.action_type == "reschedule_meeting":
        s += _RESCHEDULE_VALID_MICRO_BONUS
    return s


def critical_unreplied_count(world: WorldState) -> int:
    return sum(1 for e in world.emails if e.priority == "critical" and not e.replied)


def score_critical_queue_bonus(before: WorldState, after: WorldState) -> float:
    reduction = critical_unreplied_count(before) - critical_unreplied_count(after)
    return _CRITICAL_PER_EMAIL_BONUS * max(0, reduction)


def _classify_tone(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("sorry", "apologize", "apologies", "my mistake")):
        return "apologetic"
    if any(w in t for w in ("dear ", "sincerely", "best regards", "respectfully", "cordially")):
        return "formal"
    if any(w in t for w in ("hey", "lol", "haha", "👋", "no worries", "cheers")):
        return "casual"
    if any(w in t for w in ("must", "immediately", "asap", "non-negotiable", "demand")):
        return "assertive"
    return "neutral"


def score_relationship(
    before: WorldState,
    after: WorldState,
    action: GhostexecAction,
    *,
    action_ok: bool,
    relationship_suppressed_for_email_to: frozenset[str] | None = None,
) -> float:
    rel_sup = relationship_suppressed_for_email_to or frozenset()
    s = 0.0
    before_map = {c.name: c for c in before.contacts}
    after_map = {c.name: c for c in after.contacts}
    for name, ca in after_map.items():
        cb = before_map.get(name)
        if not cb:
            continue
        ra, rb = _MOOD_RANK[ca.mood], _MOOD_RANK[cb.mood]
        vip = ca.importance >= 4
        if ra > rb:
            s += 3.0 if vip else 1.0
        elif ra < rb:
            s -= 4.0 if vip else 2.0

    if action.action_type == "reply_email" and action.email_id:
        em = next((e for e in before.emails if e.id == action.email_id), None)
        if em and em.sender in rel_sup:
            return 0.0
        if em:
            if action_ok and (action.message_body or "").strip():
                pri = (em.priority or "").lower()
                micro = _REPLY_PRIORITY_MICRO_BONUS.get(pri, 0.0)
                if em.sender_relationship == "VIP":
                    micro *= 2.0
                s += micro
            tone = _classify_tone(action.message_body)
            contact = next((c for c in before.contacts if c.name == em.sender), None)
            if (
                contact
                and contact.relationship_type == "board_member"
                and contact.mood in ("angry", "furious", "annoyed")
                and tone == "casual"
            ):
                s -= TONE_PENALTY_CASUAL_ANGRY_BOARD
            if em.sender_relationship == "personal" and tone == "formal":
                s -= TONE_PENALTY_FORMAL_PERSONAL
    if action_ok and action.action_type == "send_message" and action.contact_name:
        known_contact = any(c.name == action.contact_name for c in before.contacts)
        if known_contact and (action.message_body or "").strip():
            s += _SEND_MESSAGE_VALID_MICRO_BONUS
    return s


def _overdue_tasks(world: WorldState) -> list[Any]:
    now = _parse_dt(world.simulation_time)
    out = []
    for t in world.tasks:
        if t.status == "done":
            continue
        if _parse_dt(t.deadline) < now:
            out.append(t)
    return out


def score_task_completion(
    before: WorldState,
    after: WorldState,
    action: GhostexecAction,
    *,
    action_ok: bool,
) -> float:
    s = 0.0
    now = _parse_dt(after.simulation_time)

    before_tasks = {t.id: t for t in before.tasks}
    after_tasks = {t.id: t for t in after.tasks}
    for tid, ta in after_tasks.items():
        tb = before_tasks.get(tid)
        if not tb:
            continue
        if tb.status != "overdue" and tb.status != "done" and ta.status == "overdue":
            s -= 2.0
        if tb.status != "done" and ta.status == "done":
            dl = _parse_dt(tb.deadline)
            if dl >= now:
                s += 2.0
            else:
                s += 0.5
        if (not tb.delegated_to) and ta.delegated_to:
            de = next((c for c in after.contacts if c.name == ta.delegated_to), None)
            if de and de.importance <= 3:
                s += 1.0
    if action_ok and action.action_type == "complete_task":
        s += _COMPLETE_TASK_VALID_MICRO_BONUS
    if action_ok and action.action_type == "delegate_task":
        s += _DELEGATE_TASK_VALID_MICRO_BONUS
    return s


def catastrophic(world: WorldState) -> bool:
    vip_furious = any(c.importance >= 4 and c.mood == "furious" for c in world.contacts)
    critical_open = sum(1 for e in world.emails if e.priority == "critical" and not e.replied)
    return vip_furious or critical_open > 3


def _scaffold_learning_signal(
    before: WorldState,
    after: WorldState,
    action: GhostexecAction,
    *,
    action_ok: bool,
    step_index: int | None,
    max_steps: int | None,
) -> float:
    if not action_ok:
        return 0.0
    if action.action_type == "do_nothing":
        return 0.0
    s = 0.0
    critical_before = critical_unreplied_count(before)
    critical_after = critical_unreplied_count(after)
    conflict_before = len(meeting_conflicts(before))
    conflict_after = len(meeting_conflicts(after))
    overdue_before = len(_overdue_tasks(before))
    overdue_after = len(_overdue_tasks(after))
    if action.action_type == "reply_email":
        if critical_after < critical_before:
            s += 0.16
        elif critical_before > 0:
            s += 0.05
    if action.action_type in ("reschedule_meeting", "cancel_meeting"):
        if conflict_after < conflict_before:
            s += 0.15
        elif conflict_before > 0:
            s += 0.04
    if action.action_type in ("complete_task", "delegate_task"):
        if overdue_after < overdue_before:
            s += 0.12
        elif overdue_before > 0:
            s += 0.03
    # Early episode shaping slightly amplified for better exploration guidance.
    if step_index is not None and max_steps and max_steps > 0:
        frac = max(0.0, min(1.0, step_index / max_steps))
        if frac <= 0.33:
            s *= 1.20
        elif frac >= 0.85:
            s *= 0.90

    return max(-_SCAFFOLD_CAP, min(_SCAFFOLD_CAP, s))


def _state_potential(world: WorldState) -> float:
    conflicts = len(meeting_conflicts(world))
    critical_open = critical_unreplied_count(world)
    overdue = len(_overdue_tasks(world))
    stress = float(world.stress)
    # Lower operational pressure => higher potential.
    return -(
        1.15 * critical_open
        + 0.90 * conflicts
        + 0.55 * overdue
        + 0.02 * stress
    )


def _budgeted_shaping_total(base_weighted_inner: float, shaping_total_inner: float) -> float:
    # Keep shaping informative but bounded against the base objective to avoid exploit loops.
    budget = _SHAPING_TO_BASE_BUDGET * (abs(base_weighted_inner) + 0.05)
    return max(-budget, min(budget, shaping_total_inner))


def _quality_separation_signal(
    *,
    c: float,
    r: float,
    t: float,
    action: GhostexecAction,
    action_ok: bool,
) -> float:
    # Amplify distance between clearly good vs clearly bad valid actions.
    if not action_ok or action.action_type == "do_nothing":
        return 0.0
    base = W_CONFLICT * c + W_REL * r + W_TASK * t
    if base >= 0.90:
        return _QUALITY_CAP
    if base >= 0.35:
        return 0.12
    if base <= -0.90:
        return -_QUALITY_CAP
    if base <= -0.35:
        return -0.12
    return 0.0


def aggregate_scores(
    conflict: float,
    relationship: float,
    task: float,
    *,
    conflict_raw: float,
    critical_queue_bonus: float,
    weighted_inner: float,
    weighted_base_only: float,
    shaping_synergy: float,
    shaping_tradeoff: float,
    shaping_potential: float,
    shaping_scaffold: float,
    shaping_quality: float,
    action_ok: bool,
    episode_done: bool,
    world_after: WorldState,
) -> RewardBreakdown:
    weighted = WEIGHTED_OUTPUT_SCALE * weighted_inner
    weighted_base_only_scaled = WEIGHTED_OUTPUT_SCALE * weighted_base_only
    shaping_total = WEIGHTED_OUTPUT_SCALE * (
        shaping_synergy + shaping_tradeoff + shaping_potential + shaping_scaffold + shaping_quality
    )
    denom = abs(weighted_base_only_scaled) + 1e-6
    shaping_ratio = min(10.0, abs(shaping_total) / denom)
    inv = 0.0
    if not action_ok:
        inv = -0.25
    bonus = 0.0
    cata = 0.0
    if episode_done:
        if world_after.stress < 40:
            bonus = 10.0
        if catastrophic(world_after):
            cata = -15.0
    final = weighted + inv + bonus + cata
    return RewardBreakdown(
        conflict_raw=conflict_raw,
        critical_queue_bonus=critical_queue_bonus,
        conflict=conflict,
        relationship=relationship,
        task=task,
        shaping_synergy=WEIGHTED_OUTPUT_SCALE * shaping_synergy,
        shaping_tradeoff=WEIGHTED_OUTPUT_SCALE * shaping_tradeoff,
        shaping_potential=WEIGHTED_OUTPUT_SCALE * shaping_potential,
        shaping_scaffold=WEIGHTED_OUTPUT_SCALE * shaping_scaffold,
        shaping_quality=WEIGHTED_OUTPUT_SCALE * shaping_quality,
        shaping_total=shaping_total,
        shaping_to_base_ratio=shaping_ratio,
        weighted_base=weighted,
        output_scale=WEIGHTED_OUTPUT_SCALE,
        invalid_step_adjustment=inv,
        episode_completion_bonus=bonus,
        catastrophic_penalty=cata,
        do_nothing_floor=0.0,
        final=final,
    )


def apply_do_nothing_penalty_floor(
    action: GhostexecAction,
    breakdown: RewardBreakdown,
) -> RewardBreakdown:
    if action.action_type != "do_nothing":
        return breakdown
    floor_delta = _DO_NOTHING_STRICT_PENALTY
    new_final = breakdown.final + floor_delta
    return breakdown.model_copy(
        update={"do_nothing_floor": floor_delta, "final": new_final},
    )


def compute_step_reward(
    before: WorldState,
    after: WorldState,
    action: GhostexecAction,
    *,
    action_ok: bool,
    episode_done: bool,
    relationship_suppressed_for_email_to: frozenset[str] | None = None,
    reward_mode: str = "full",
    step_index: int | None = None,
    max_steps: int | None = None,
) -> RewardBreakdown:
    c_core = score_conflict_resolution(before, after, action, action_ok=action_ok)
    crit_b = score_critical_queue_bonus(before, after)
    c_raw = c_core + crit_b
    c = max(-CONFLICT_RAW_CAP, min(CONFLICT_RAW_CAP, c_raw))
    r = score_relationship(
        before,
        after,
        action,
        action_ok=action_ok,
        relationship_suppressed_for_email_to=relationship_suppressed_for_email_to,
    )
    t = score_task_completion(before, after, action, action_ok=action_ok)
    weighted_base_only = W_CONFLICT * c + W_REL * r + W_TASK * t
    weighted_inner = weighted_base_only
    synergy = 0.0
    tradeoff_penalty = 0.0
    potential_progress = 0.0
    scaffold_signal = 0.0
    quality_signal = 0.0
    if reward_mode in ("full", "shaping"):
        # Bounded nonlinear shaping to speed learning without overpowering base channels.
        if c > 0.0 and r > 0.0:
            synergy += min(_SYNERGY_CAP, 0.18 * math.tanh(0.35 * c * r))
        if t > 0.0 and (c > 0.0 or r > 0.0):
            bridge = max(c, 0.0) + max(r, 0.0)
            synergy += min(_SYNERGY_CAP, 0.10 * math.tanh(0.25 * t * bridge))
        if c < -0.5 and r < -0.5:
            tradeoff_penalty -= min(_TRADEOFF_CAP, 0.12 * math.tanh(0.25 * abs(c * r)))
        if t < -0.5 and (c < 0.0 or r < 0.0):
            debt = abs(t) * (abs(min(c, 0.0)) + abs(min(r, 0.0)))
            tradeoff_penalty -= min(_TRADEOFF_CAP, 0.08 * math.tanh(0.18 * debt))
        potential_progress = max(
            -_POTENTIAL_CAP,
            min(_POTENTIAL_CAP, _state_potential(after) - _state_potential(before)),
        )
        scaffold_signal = _scaffold_learning_signal(
            before,
            after,
            action,
            action_ok=action_ok,
            step_index=step_index,
            max_steps=max_steps,
        )
        quality_signal = _quality_separation_signal(
            c=c,
            r=r,
            t=t,
            action=action,
            action_ok=action_ok,
        )
        shaping_total_inner = (
            synergy + tradeoff_penalty + potential_progress + scaffold_signal + quality_signal
        )
        weighted_inner += _budgeted_shaping_total(weighted_base_only, shaping_total_inner)
    bd = aggregate_scores(
        c,
        r,
        t,
        conflict_raw=c_raw,
        critical_queue_bonus=crit_b,
        weighted_inner=weighted_inner,
        weighted_base_only=weighted_base_only,
        shaping_synergy=synergy,
        shaping_tradeoff=tradeoff_penalty,
        shaping_potential=potential_progress,
        shaping_scaffold=scaffold_signal,
        shaping_quality=quality_signal,
        action_ok=action_ok,
        episode_done=episode_done,
        world_after=after,
    )
    return apply_do_nothing_penalty_floor(action, bd)
