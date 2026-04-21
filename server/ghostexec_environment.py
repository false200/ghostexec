# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GhostExec simulated world, agent step (Phases 2–3), and reward (Phase 4).

Scenario payloads load from scenarios/*.json. Observations are plain-text briefings.
Invalid actions return a structured error in observation metadata without raising.
Rewards aggregate conflict / relationship / task scores and log each step to outputs/logs/.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        Contact,
        Email,
        GhostexecAction,
        GhostexecObservation,
        Meeting,
        Mood,
        RewardBreakdown,
        Task,
        TaskStatus,
        WorldState,
    )
except ImportError:
    from models import (
        Contact,
        Email,
        GhostexecAction,
        GhostexecObservation,
        Meeting,
        Mood,
        RewardBreakdown,
        Task,
        TaskStatus,
        WorldState,
    )

try:
    from . import reward as _reward
except ImportError:
    try:
        from server import reward as _reward
    except ImportError:
        import reward as _reward  # type: ignore[no-redef]

_PRIORITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "normal": 2, "low": 3}
_REL_DISPLAY: dict[str, str] = {
    "board_member": "Board",
    "spouse": "Spouse",
    "investor": "Investor",
    "direct_report": "Direct report",
    "client": "Client",
    "friend": "Friend",
    "team_member": "Team",
}

_INVALID_ACTION_REWARD = -0.25
_DEFAULT_STEP_REWARD = 0.0


def _default_scenario_path() -> Path:
    return Path(__file__).resolve().parent.parent / "scenarios" / "phase2_core.json"


def _parse_dt(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _meeting_end(m: Meeting) -> datetime:
    start = _parse_dt(m.start)
    return start + timedelta(minutes=m.duration_minutes)


def _windows_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


class GhostexecEnvironment(Environment):
    """Inbox, calendar, contacts, tasks, actions, briefings, and Phase 4 rewards."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenario_path: str | Path | None = None,
        schema_drift_events_path: str | Path | None = None,
    ) -> None:
        self._scenario_path = Path(scenario_path) if scenario_path else _default_scenario_path()
        self._drift_events_path = (
            Path(schema_drift_events_path) if schema_drift_events_path is not None else None
        )
        self._drift_events: list[dict[str, Any]] = []
        if self._drift_events_path and self._drift_events_path.is_file():
            drift_raw = json.loads(self._drift_events_path.read_text(encoding="utf-8"))
            self._drift_events = list(drift_raw.get("events", []))
        self._reply_relationship_suppressed: set[str] = set()
        self._reward_log_path = (
            Path(__file__).resolve().parent.parent / "outputs" / "logs" / "episode_rewards.jsonl"
        )
        self._world: WorldState | None = None
        self._base_stress: int = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_step_ok: bool = True
        self._last_step_error: str | None = None
        self._last_step_detail: str = ""
        self._last_reward_breakdown: RewardBreakdown | None = None

    # --- lifecycle ---

    def reset(self) -> GhostexecObservation:  # type: ignore[override]
        self._world = self.load_world_from_json(self._scenario_path)
        self._base_stress = self._world.stress
        self._rebuild_conflict_list()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_step_ok = True
        self._last_step_error = None
        self._last_step_detail = "Episode started."
        self._reply_relationship_suppressed.clear()
        self._last_reward_breakdown = None
        self._ensure_reward_log_dir()
        briefing = self.build_briefing_text()
        return self._observation_from_briefing(
            briefing,
            reward=_DEFAULT_STEP_REWARD,
            done=False,
            reward_breakdown=None,
        )

    def step(self, action: GhostexecAction) -> GhostexecObservation:  # type: ignore[override]
        if self._world is None:
            # OpenEnv HTTP uses a new env per request; prime the world so this step still
            # runs the requested action (invalid actions get step_ok False, rewards apply).
            self.reset()

        assert self._world is not None
        if not self._world.episode_active:
            self._last_step_ok = False
            self._last_step_error = "Episode is already finished."
            bd = RewardBreakdown(
                final=_INVALID_ACTION_REWARD,
                invalid_step_adjustment=_INVALID_ACTION_REWARD,
            )
            self._last_reward_breakdown = bd
            return self._observation_from_briefing(
                self.build_briefing_text(),
                reward=bd.final,
                done=True,
                reward_breakdown=bd,
            )

        self._state.step_count += 1
        self._maybe_apply_schema_drift_events()

        if action.message.strip():
            self._world.action_log.append(f"note: {action.message.strip()}")

        before = self.world.model_copy(deep=True)
        action_ok = self._apply_action(action)
        self._rebuild_conflict_list()

        episode_done = False
        if self._state.step_count >= self._world.max_episode_steps:
            episode_done = True
            self._world.episode_active = False
            self._world.episode_end_reason = self._world.episode_end_reason or "step_limit"

        breakdown = _reward.compute_step_reward(
            before,
            self.world,
            action,
            action_ok=action_ok,
            episode_done=episode_done,
            relationship_suppressed_for_email_to=frozenset(self._reply_relationship_suppressed),
        )
        self._last_reward_breakdown = breakdown
        self._append_reward_log(breakdown, episode_done, action)

        briefing = self.build_briefing_text()
        return self._observation_from_briefing(
            briefing,
            reward=breakdown.final,
            done=episode_done,
            reward_breakdown=breakdown,
        )

    @property
    def state(self) -> State:
        return self._state

    @property
    def world(self) -> WorldState:
        if self._world is None:
            raise RuntimeError("World not initialised; call reset() first.")
        return self._world

    # --- Phase 3 briefing (plain text for LLM) ---

    def build_briefing_text(self) -> str:
        w = self.world
        now = _parse_dt(w.simulation_time)
        header = now.strftime("=== GHOSTEXEC BRIEFING — %a %d %b %Y %H:%M ===")

        unread = self.get_unread_emails_sorted()
        email_lines = [
            f"- [{e.priority.upper()}] From: {e.sender} ({_REL_DISPLAY.get(e.sender_relationship, e.sender_relationship)}) — "
            f'"{e.subject}"\n  Preview: {(e.body[:100] + ("…" if len(e.body) > 100 else "")).replace(chr(10), " ")}'
            for e in unread[:20]
        ]
        email_block = "\n".join(email_lines) if email_lines else "(none)"

        horizon = now + timedelta(hours=4)
        conflict_lines: list[str] = []
        for row in self.detect_meeting_conflicts():
            o0 = _parse_dt(row["overlap_start"])
            o1 = _parse_dt(row["overlap_end"])
            if o1 <= now or o0 >= horizon:
                continue
            ma = self._meeting_by_id(row["meeting_a"])
            mb = self._meeting_by_id(row["meeting_b"])
            if not ma or not mb or ma.cancelled or mb.cancelled:
                continue
            conflict_lines.append(
                f"- {_fmt_meeting_line(ma)} CLASHES WITH -> {_fmt_meeting_line(mb)}"
            )
        conflict_block = "\n".join(conflict_lines) if conflict_lines else "(none in next 4 hours)"

        top_contacts = sorted(w.contacts, key=lambda c: (-c.importance, c.name))[:5]
        contact_lines = [
            f"- {c.name}: {c.mood.upper()} — {_REL_DISPLAY.get(c.relationship_type, c.relationship_type)}; "
            f"prefers {c.communication_preference}"
            for c in top_contacts
        ]
        contact_block = "\n".join(contact_lines) if contact_lines else "(none)"

        soon = now + timedelta(hours=24)
        task_lines: list[str] = []
        for t in w.tasks:
            if t.status == "done":
                continue
            dl = _parse_dt(t.deadline)
            if dl < now or (now <= dl <= soon):
                flag = "OVERDUE" if dl < now else "due soon"
                task_lines.append(f"- [{flag}] {t.description} (deadline {t.deadline}, owner {t.owner})")
        task_block = "\n".join(task_lines[:15]) if task_lines else "(none)"

        remaining = max(0, w.max_episode_steps - self._state.step_count)

        parts = [
            header,
            "",
            f"UNREAD EMAILS ({len(unread)} unread):",
            email_block,
            "",
            "CALENDAR CONFLICTS IN NEXT 4 HOURS:",
            conflict_block,
            "",
            "CONTACTS TO WATCH (top 5 by importance):",
            contact_block,
            "",
            "OVERDUE OR DUE-SOON TASKS (next 24h window):",
            task_block,
            "",
            f"EXEC STRESS LEVEL: {w.stress}/100",
            f"STEPS REMAINING: {remaining}",
        ]
        if self._last_step_error:
            parts += ["", f"LAST ACTION: ERROR — {self._last_step_error}"]
        elif self._last_step_detail:
            parts += ["", f"LAST ACTION: OK — {self._last_step_detail}"]

        return "\n".join(parts)

    def _meeting_by_id(self, mid: str) -> Meeting | None:
        for m in self.world.meetings:
            if m.id == mid:
                return m
        return None

    # --- scenario IO ---

    @staticmethod
    def load_world_from_json(path: str | Path) -> WorldState:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return WorldState.model_validate(data)

    @staticmethod
    def world_to_json(world: WorldState) -> str:
        return world.model_dump_json()

    @staticmethod
    def world_from_json(blob: str) -> WorldState:
        return WorldState.model_validate_json(blob)

    # --- inbox ---

    def get_unread_emails_sorted(self) -> list[Email]:
        w = self.world
        unread = [e for e in w.emails if not e.read]
        return sorted(
            unread,
            key=lambda e: (_PRIORITY_RANK.get(e.priority, 99), e.id),
        )

    def mark_email_read(self, email_id: str) -> bool:
        for i, e in enumerate(self.world.emails):
            if e.id == email_id:
                self.world.emails[i] = e.model_copy(update={"read": True})
                return True
        return False

    def mark_email_replied(self, email_id: str) -> bool:
        for i, e in enumerate(self.world.emails):
            if e.id == email_id:
                self.world.emails[i] = e.model_copy(update={"read": True, "replied": True})
                return True
        return False

    # --- calendar ---

    def detect_meeting_conflicts(self) -> list[dict[str, Any]]:
        active = [m for m in self.world.meetings if not m.cancelled]
        out: list[dict[str, Any]] = []
        for i, a in enumerate(active):
            a_start = _parse_dt(a.start)
            a_end = _meeting_end(a)
            for b in active[i + 1 :]:
                b_start = _parse_dt(b.start)
                b_end = _meeting_end(b)
                if _windows_overlap(a_start, a_end, b_start, b_end):
                    overlap_start = max(a_start, b_start)
                    overlap_end = min(a_end, b_end)
                    out.append(
                        {
                            "meeting_a": a.id,
                            "meeting_b": b.id,
                            "overlap_start": overlap_start.isoformat(),
                            "overlap_end": overlap_end.isoformat(),
                        }
                    )
        return out

    def _reschedule_causes_overlap(self, meeting_id: str, new_start_iso: str) -> bool:
        idx = next((i for i, m in enumerate(self.world.meetings) if m.id == meeting_id), None)
        if idx is None:
            return True
        cand = self.world.meetings[idx].model_copy(update={"start": new_start_iso})
        c_start = _parse_dt(cand.start)
        c_end = _meeting_end(cand)
        for m in self.world.meetings:
            if m.cancelled or m.id == meeting_id:
                continue
            if _windows_overlap(c_start, c_end, _parse_dt(m.start), _meeting_end(m)):
                return True
        return False

    def reschedule_meeting(self, meeting_id: str, new_start_iso: str) -> bool:
        for i, m in enumerate(self.world.meetings):
            if m.id == meeting_id and not m.cancelled:
                self.world.meetings[i] = m.model_copy(update={"start": new_start_iso})
                self._rebuild_conflict_list()
                return True
        return False

    def cancel_meeting(self, meeting_id: str) -> bool:
        for i, m in enumerate(self.world.meetings):
            if m.id == meeting_id:
                self.world.meetings[i] = m.model_copy(update={"cancelled": True})
                self._rebuild_conflict_list()
                return True
        return False

    def add_meeting(self, meeting: Meeting) -> None:
        self.world.meetings.append(meeting)
        self._rebuild_conflict_list()

    # --- contacts ---

    def get_contact(self, name: str) -> Contact | None:
        for c in self.world.contacts:
            if c.name == name:
                return c
        return None

    def update_contact_mood(self, name: str, mood: Mood) -> bool:
        for i, c in enumerate(self.world.contacts):
            if c.name == name:
                self.world.contacts[i] = c.model_copy(update={"mood": mood})
                return True
        return False

    # --- tasks ---

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        for i, t in enumerate(self.world.tasks):
            if t.id == task_id:
                self.world.tasks[i] = t.model_copy(update={"status": status})
                return True
        return False

    def overdue_tasks_at(self, simulation_iso: str) -> list[Task]:
        now = _parse_dt(simulation_iso)
        out: list[Task] = []
        for t in self.world.tasks:
            if t.status in ("done",):
                continue
            if _parse_dt(t.deadline) < now:
                out.append(t)
        return out

    def set_simulation_time(self, simulation_iso: str) -> None:
        self.world.simulation_time = simulation_iso
        self._reapply_task_overdue_flags()
        self._rebuild_conflict_list()

    # --- Phase 3 action execution ---

    def _apply_action(self, action: GhostexecAction) -> bool:
        self._last_step_ok = True
        self._last_step_error = None
        self._last_step_detail = ""
        at = action.action_type

        if at == "do_nothing":
            self._last_step_detail = "No action taken."
            return True

        if at == "reply_email":
            if not action.email_id:
                return self._fail("reply_email requires email_id")
            if not any(e.id == action.email_id for e in self.world.emails):
                return self._fail(f"Unknown email_id {action.email_id!r}")
            if not action.message_body.strip():
                return self._fail("reply_email requires non-empty message_body")
            self.mark_email_replied(action.email_id)
            self._last_step_detail = f"Replied to email {action.email_id}."
            return True

        if at == "archive_email":
            if not action.email_id:
                return self._fail("archive_email requires email_id")
            if not self.mark_email_read(action.email_id):
                return self._fail(f"Unknown email_id {action.email_id!r}")
            self._last_step_detail = f"Archived (read) email {action.email_id}."
            return True

        if at == "reschedule_meeting":
            if not action.meeting_id or not action.new_time:
                return self._fail("reschedule_meeting requires meeting_id and new_time")
            if not any(m.id == action.meeting_id for m in self.world.meetings):
                return self._fail(f"Unknown meeting_id {action.meeting_id!r}")
            if self._reschedule_causes_overlap(action.meeting_id, action.new_time):
                return self._fail("Target time overlaps another active meeting.")
            if not self.reschedule_meeting(action.meeting_id, action.new_time):
                return self._fail("Could not reschedule meeting.")
            self._last_step_detail = f"Rescheduled {action.meeting_id} to {action.new_time}."
            return True

        if at == "cancel_meeting":
            if not action.meeting_id:
                return self._fail("cancel_meeting requires meeting_id")
            if not any(m.id == action.meeting_id for m in self.world.meetings):
                return self._fail(f"Unknown meeting_id {action.meeting_id!r}")
            if not self.cancel_meeting(action.meeting_id):
                return self._fail("Could not cancel meeting.")
            reason = action.reason.strip() or "(no reason given)"
            self._world.action_log.append(f"cancelled {action.meeting_id}: {reason}")
            self._last_step_detail = f"Cancelled meeting {action.meeting_id}."
            return True

        if at == "complete_task":
            if not action.task_id:
                return self._fail("complete_task requires task_id")
            t = next((x for x in self.world.tasks if x.id == action.task_id), None)
            if not t:
                return self._fail(f"Unknown task_id {action.task_id!r}")
            if t.status == "done":
                return self._fail("Task is already done.")
            self.update_task_status(action.task_id, "done")
            self._last_step_detail = f"Completed task {action.task_id}."
            return True

        if at == "delegate_task":
            if not action.task_id or not action.contact_name.strip():
                return self._fail("delegate_task requires task_id and contact_name")
            if not any(t.id == action.task_id for t in self.world.tasks):
                return self._fail(f"Unknown task_id {action.task_id!r}")
            if not self.get_contact(action.contact_name.strip()):
                return self._fail(f"Unknown contact {action.contact_name.strip()!r}")
            for i, t in enumerate(self.world.tasks):
                if t.id == action.task_id:
                    self.world.tasks[i] = t.model_copy(
                        update={
                            "delegated_to": action.contact_name.strip(),
                            "status": "in-progress",
                        }
                    )
                    break
            self._last_step_detail = f"Delegated {action.task_id} to {action.contact_name.strip()}."
            return True

        if at == "send_message":
            name = action.contact_name.strip()
            if not name:
                return self._fail("send_message requires contact_name")
            if not self.get_contact(name):
                return self._fail(f"Unknown contact {name!r}")
            if not action.message_body.strip():
                return self._fail("send_message requires non-empty message_body")
            self._world.action_log.append(f"message to {name}: {action.message_body.strip()[:500]}")
            self._last_step_detail = f"Message sent to {name}."
            return True

        return self._fail(f"Unsupported action_type {at!r}")

    def _fail(self, msg: str) -> bool:
        self._last_step_ok = False
        self._last_step_error = msg
        self._last_step_detail = ""
        self._world.action_log.append(f"error: {msg}")
        return False

    def _ensure_reward_log_dir(self) -> None:
        self._reward_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_reward_log(
        self,
        breakdown: RewardBreakdown,
        episode_done: bool,
        action: GhostexecAction,
    ) -> None:
        self._ensure_reward_log_dir()
        w = self.world
        crit_open = sum(1 for e in w.emails if e.priority == "critical" and not e.replied)
        overdue_n = len(self.overdue_tasks_at(w.simulation_time))
        line = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "action_type": action.action_type,
            "step_ok": self._last_step_ok,
            "reward": breakdown.final,
            "conflict_raw": breakdown.conflict_raw,
            "critical_queue_bonus": breakdown.critical_queue_bonus,
            "conflict": breakdown.conflict,
            "relationship": breakdown.relationship,
            "task": breakdown.task,
            "weighted_base": breakdown.weighted_base,
            "output_scale": breakdown.output_scale,
            "invalid_step_adjustment": breakdown.invalid_step_adjustment,
            "episode_completion_bonus": breakdown.episode_completion_bonus,
            "catastrophic_penalty": breakdown.catastrophic_penalty,
            "episode_done": episode_done,
            "calendar_overlap_pairs": len(self.detect_meeting_conflicts()),
            "critical_unreplied": crit_open,
            "overdue_tasks": overdue_n,
        }
        with self._reward_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def _maybe_apply_schema_drift_events(self) -> None:
        if not self._world or not self._drift_events:
            return
        step = self._state.step_count
        for ev in self._drift_events:
            if ev.get("after_step") != step:
                continue
            if "shift_all_meetings_hours" in ev:
                delta = int(ev["shift_all_meetings_hours"])
                for i, m in enumerate(self._world.meetings):
                    new_start = (_parse_dt(m.start) + timedelta(hours=delta)).replace(tzinfo=None)
                    self._world.meetings[i] = m.model_copy(
                        update={"start": new_start.isoformat(timespec="seconds")}
                    )
                self._world.action_log.append(
                    f"schema drift: shifted all meeting starts by {delta:+d} hour(s) (calendar TZ policy)."
                )
            pref = ev.get("set_contact_preference")
            if isinstance(pref, dict):
                name = str(pref.get("name", ""))
                comm = str(pref.get("communication_preference", "text"))
                for i, c in enumerate(self._world.contacts):
                    if c.name == name:
                        self._world.contacts[i] = c.model_copy(
                            update={"communication_preference": comm}  # type: ignore[arg-type]
                        )
                        break
                self._world.action_log.append(
                    f"schema drift: contact {name!r} now prefers {comm} only (relationship channel change)."
                )
            td = ev.get("set_task_deadline")
            if isinstance(td, dict):
                tid = str(td.get("task_id", ""))
                dl = str(td.get("deadline", ""))
                for i, t in enumerate(self._world.tasks):
                    if t.id == tid:
                        self._world.tasks[i] = t.model_copy(update={"deadline": dl})
                        break
                self._world.action_log.append(
                    f"schema drift: task {tid!r} deadline moved earlier to {dl!r}."
                )
            for name in ev.get("suppress_reply_relationship_for_senders", []) or []:
                self._reply_relationship_suppressed.add(str(name))
                self._world.action_log.append(
                    f"schema drift: replies to emails from {name!r} yield zero relationship score this episode."
                )
            scm = ev.get("set_contact_mood")
            if isinstance(scm, dict):
                cname = str(scm.get("name", ""))
                mood_raw = str(scm.get("mood", "neutral"))
                allowed: tuple[Mood, ...] = ("happy", "neutral", "annoyed", "angry", "furious")
                if cname and mood_raw in allowed and self.update_contact_mood(cname, mood_raw):
                    self._world.action_log.append(
                        f"schema drift: stakeholder {cname!r} mood is now {mood_raw} (external pressure)."
                    )
        if any(ev.get("after_step") == step for ev in self._drift_events):
            self._rebuild_conflict_list()

    # --- internals ---

    def _reapply_task_overdue_flags(self) -> None:
        now = _parse_dt(self.world.simulation_time)
        for i, t in enumerate(self.world.tasks):
            if t.status == "done":
                continue
            if _parse_dt(t.deadline) < now and t.status != "overdue":
                self.world.tasks[i] = t.model_copy(update={"status": "overdue"})

    def _rebuild_conflict_list(self) -> None:
        lines: list[str] = []
        for row in self.detect_meeting_conflicts():
            lines.append(
                f"Calendar overlap: {row['meeting_a']} vs {row['meeting_b']} "
                f"({row['overlap_start']} – {row['overlap_end']})"
            )
        for e in self.world.emails:
            if e.priority == "critical" and not e.replied:
                lines.append(f"Unanswered critical email {e.id}: {e.subject}")
        bump = min(35, len(lines) * 2)
        self.world.active_conflicts = lines
        self.world.stress = min(100, self._base_stress + bump)

    def _observation_from_briefing(
        self,
        briefing: str,
        reward: float,
        done: bool,
        reward_breakdown: RewardBreakdown | None = None,
    ) -> GhostexecObservation:
        w = self.world
        unread_sorted = self.get_unread_emails_sorted()
        meta: dict[str, Any] = {
            "simulation_time": w.simulation_time,
            "stress": w.stress,
            "unread_email_count": sum(1 for e in w.emails if not e.read),
            "calendar_conflict_pairs": len(self.detect_meeting_conflicts()),
            "episode_step": self._state.step_count,
            "step_ok": self._last_step_ok,
            "step_error": self._last_step_error,
            "step_detail": self._last_step_detail,
            # Compact ids for remote trainers / Colab (briefing stays plain text).
            "critical_unreplied_email_ids": [
                e.id for e in w.emails if e.priority == "critical" and not e.replied
            ][:12],
            "unread_email_ids": [e.id for e in unread_sorted[:15]],
            "overdue_task_ids": [t.id for t in self.overdue_tasks_at(w.simulation_time)][:12],
            "active_meeting_ids": [m.id for m in w.meetings if not m.cancelled][:20],
        }
        if reward_breakdown is not None:
            meta["reward_breakdown"] = reward_breakdown.model_dump()
        cap = 48_000
        text = briefing if len(briefing) <= cap else briefing[: cap - 1] + "…"
        return GhostexecObservation(
            echoed_message=text,
            message_length=len(text),
            done=done,
            reward=reward,
            metadata=meta,
        )


def _fmt_meeting_line(m: Meeting) -> str:
    st = _parse_dt(m.start)
    return f"{st.strftime('%H:%M')}: {m.title} ({m.duration_minutes}min)"
