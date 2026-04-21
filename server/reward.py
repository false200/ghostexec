# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Phase 4 reward: weighted (0.35 / 0.35 / 0.30) with conflict cap, critical-queue shaping,
full sub-scores even on invalid steps (+ explicit invalid penalty), and mild output scaling.
"""

from __future__ import annotations

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


def score_conflict_resolution(before: WorldState, after: WorldState) -> float:
    b = _pair_set(meeting_conflicts(before))
    a = _pair_set(meeting_conflicts(after))
    s = 0.0
    for _p in b - a:
        s += 2.0 + _RESOLVE_MICRO_BONUS
        if _attendee_moods_ok(after, _p):
            s += 1.0
    for _ in b & a:
        s -= 1.0
    for _ in a - b:
        s -= 3.0
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
    overdue_before = _overdue_tasks(before)

    if action.action_type == "do_nothing" and overdue_before:
        s -= 3.0
    elif not action_ok and overdue_before:
        # Illegal action while overdue work exists: same board pressure as idle (plus invalid add-on in aggregate).
        s -= 3.0

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
    return s


def catastrophic(world: WorldState) -> bool:
    vip_furious = any(c.importance >= 4 and c.mood == "furious" for c in world.contacts)
    critical_open = sum(1 for e in world.emails if e.priority == "critical" and not e.replied)
    return vip_furious or critical_open > 3


def aggregate_scores(
    conflict: float,
    relationship: float,
    task: float,
    *,
    conflict_raw: float,
    critical_queue_bonus: float,
    weighted_inner: float,
    action_ok: bool,
    episode_done: bool,
    world_after: WorldState,
) -> RewardBreakdown:
    weighted = WEIGHTED_OUTPUT_SCALE * weighted_inner
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
    floor_delta = 0.0
    new_final = breakdown.final
    if new_final > -0.12:
        floor_delta = -0.12 - new_final
        new_final = -0.12
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
) -> RewardBreakdown:
    c_core = score_conflict_resolution(before, after)
    crit_b = score_critical_queue_bonus(before, after)
    c_raw = c_core + crit_b
    c = max(-CONFLICT_RAW_CAP, min(CONFLICT_RAW_CAP, c_raw))
    r = score_relationship(
        before,
        after,
        action,
        relationship_suppressed_for_email_to=relationship_suppressed_for_email_to,
    )
    t = score_task_completion(before, after, action, action_ok=action_ok)
    weighted_inner = W_CONFLICT * c + W_REL * r + W_TASK * t
    bd = aggregate_scores(
        c,
        r,
        t,
        conflict_raw=c_raw,
        critical_queue_bonus=crit_b,
        weighted_inner=weighted_inner,
        action_ok=action_ok,
        episode_done=episode_done,
        world_after=after,
    )
    return apply_do_nothing_penalty_floor(action, bd)
