# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Dead-test suite for Phase 4 step rewards: 100+ independent scenarios on
# phase2_core.json. Asserts penalization (do_nothing, invalid), priority
# ordering (VIP critical > normal), and legal-action signatures for GRPO-style
# post-training signal quality.

from __future__ import annotations

from pathlib import Path

import pytest

from ghostexec.models import GhostexecAction
from ghostexec.server import reward as reward_mod
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"

# All inbox ids from phase2_core (e01–e30).
REPLY_EMAIL_IDS = [f"e{i:02d}" for i in range(1, 31)]

# Unread or replyable ids suitable for archive (skip if unknown — all exist).
ARCHIVE_EMAIL_IDS = [f"e{i:02d}" for i in range(1, 16)]

# Pending / in-progress tasks only (t09 is done in fixture).
COMPLETE_TASK_IDS = [f"t{i:02d}" for i in range(1, 13) if i != 9]

# Known non-overlapping reschedules for 08:00 sim time (from phase4 tests).
_SAFE_RESCHEDULES: list[tuple[str, str]] = [
    ("m02", "2026-04-21T18:00:00"),
    ("m03", "2026-04-21T18:30:00"),
    ("m06", "2026-04-21T20:00:00"),
    ("m09", "2026-04-21T21:00:00"),
    ("m04", "2026-04-21T19:00:00"),
    ("m05", "2026-04-21T19:30:00"),
    ("m07", "2026-04-21T20:30:00"),
    ("m08", "2026-04-21T21:30:00"),
    ("m01", "2026-04-21T17:00:00"),
    ("m10", "2026-04-21T22:00:00"),
]

MEETING_IDS_CANCEL = [f"m{i:02d}" for i in range(1, 11)]

KNOWN_CONTACTS = ["Jordan Lee", "Jamie Liu", "Marcus Webb", "Sarah Chen", "Priya Sharma", "David Okonkwo"]

_BODY = "Thanks — triaging and will follow up shortly."


# --- 30 cases: reply every email id -------------------------------------------


@pytest.mark.parametrize("email_id", REPLY_EMAIL_IDS)
def test_dead_reply_email_each_id_positive_or_neutral(email_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="reply_email", email_id=email_id, message_body=_BODY))
    assert obs.metadata.get("step_ok") is True
    assert obs.reward is not None
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert bd.get("invalid_step_adjustment", 0) == pytest.approx(0.0)
    assert bd.get("do_nothing_floor", 0) == pytest.approx(0.0)
    # No snapshot -4 conflict tax: legal reply should not tank below -0.5
    assert float(obs.reward) > -0.5


@pytest.mark.parametrize("email_id", ("e01", "e03", "e12", "e21", "e27"))
def test_dead_reply_vip_critical_queue_bonus(email_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="reply_email", email_id=email_id, message_body=_BODY))
    assert obs.metadata.get("step_ok") is True
    # VIP+critical micro + critical_queue bonus; exact float varies slightly (0.48 scale).
    assert float(obs.reward or 0) > 0.06
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("critical_queue_bonus") or 0) > 0


@pytest.mark.parametrize("email_id", ("e02", "e04", "e06", "e14", "e23"))
def test_dead_reply_high_or_normal_small_positive(email_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="reply_email", email_id=email_id, message_body=_BODY))
    assert obs.metadata.get("step_ok") is True
    assert float(obs.reward or 0) > 0.0


# --- 20 cases: do_nothing always penalized ------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_dead_do_nothing_strict_penalty(seed: int) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="do_nothing"))
    assert obs.metadata.get("step_ok") is True
    assert float(obs.reward or 0) < 0
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("do_nothing_floor") or 0) == pytest.approx(reward_mod._DO_NOTHING_STRICT_PENALTY)


# --- 15 cases: archive --------------------------------------------------------


@pytest.mark.parametrize("email_id", ARCHIVE_EMAIL_IDS)
def test_dead_archive_email_step_ok(email_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="archive_email", email_id=email_id))
    assert obs.metadata.get("step_ok") is True
    assert obs.reward is not None


# --- 11 cases: complete pending task -----------------------------------------


@pytest.mark.parametrize("task_id", COMPLETE_TASK_IDS)
def test_dead_complete_task_step_ok(task_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(GhostexecAction(action_type="complete_task", task_id=task_id))
    assert obs.metadata.get("step_ok") is True
    assert obs.reward is not None
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("task") or 0) >= reward_mod._COMPLETE_TASK_VALID_MICRO_BONUS


# --- 10 cases: reschedule safe slots -----------------------------------------


@pytest.mark.parametrize("meeting_id,new_time", _SAFE_RESCHEDULES)
def test_dead_reschedule_meeting_resolves_or_micro(meeting_id: str, new_time: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(
        GhostexecAction(action_type="reschedule_meeting", meeting_id=meeting_id, new_time=new_time)
    )
    assert obs.metadata.get("step_ok") is True
    assert obs.reward is not None
    # Should beat idle do-nothing on same fresh env
    e2 = GhostexecEnvironment(SCENARIO)
    e2.reset()
    idle = e2.step(GhostexecAction(action_type="do_nothing"))
    assert float(obs.reward or 0) > float(idle.reward or 0)


# --- 10 cases: cancel meeting --------------------------------------------------


@pytest.mark.parametrize("meeting_id", MEETING_IDS_CANCEL)
def test_dead_cancel_meeting_step_ok(meeting_id: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(
        GhostexecAction(action_type="cancel_meeting", meeting_id=meeting_id, reason="dead test cancel")
    )
    assert obs.metadata.get("step_ok") is True
    assert obs.reward is not None


# --- 6 cases: send_message -----------------------------------------------------


@pytest.mark.parametrize("contact_name", KNOWN_CONTACTS)
def test_dead_send_message_known_contact(contact_name: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(
        GhostexecAction(
            action_type="send_message",
            contact_name=contact_name,
            message_body="Quick sync on priorities.",
        )
    )
    assert obs.metadata.get("step_ok") is True
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("relationship") or 0) >= reward_mod._SEND_MESSAGE_VALID_MICRO_BONUS - 0.01


# --- 5 cases: delegate_task ---------------------------------------------------


@pytest.mark.parametrize(
    "task_id,contact",
    [
        ("t08", "Jordan Lee"),
        ("t07", "Jamie Liu"),
        ("t01", "Marcus Webb"),
        ("t02", "Sarah Chen"),
        ("t11", "Casey Nguyen"),
    ],
)
def test_dead_delegate_task(task_id: str, contact: str) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(
        GhostexecAction(action_type="delegate_task", task_id=task_id, contact_name=contact)
    )
    assert obs.metadata.get("step_ok") is True
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("task") or 0) >= reward_mod._DELEGATE_TASK_VALID_MICRO_BONUS - 0.01


# --- 10 cases: invalid actions ------------------------------------------------


@pytest.mark.parametrize(
    "action,expect_ok",
    [
        (GhostexecAction(action_type="reply_email", email_id="nope", message_body="x"), False),
        (GhostexecAction(action_type="complete_task", task_id="t09"), False),
        (GhostexecAction(action_type="archive_email", email_id="nope"), False),
        (GhostexecAction(action_type="reschedule_meeting", meeting_id="m99", new_time="2026-04-21T18:00:00"), False),
        (GhostexecAction(action_type="cancel_meeting", meeting_id="m99", reason="x"), False),
        (GhostexecAction(action_type="delegate_task", task_id="t01", contact_name="Nobody"), False),
        (GhostexecAction(action_type="send_message", contact_name="Nobody", message_body="hi"), False),
        (GhostexecAction(action_type="reply_email", email_id="", message_body="hi"), False),
        (GhostexecAction(action_type="complete_task", task_id=""), False),
        (GhostexecAction(action_type="archive_email", email_id=""), False),
    ],
)
def test_dead_invalid_action_step_ok_false(action: GhostexecAction, expect_ok: bool) -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(action)
    assert obs.metadata.get("step_ok") is expect_ok
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("invalid_step_adjustment") or 0) == pytest.approx(-0.25)


# --- Ordering: VIP critical reply >> do_nothing --------------------------------


def test_dead_priority_ordering_vip_critical_over_normal_over_idle() -> None:
    r_vip: list[float] = []
    r_norm: list[float] = []
    r_idle: list[float] = []
    for _ in range(5):
        e1 = GhostexecEnvironment(SCENARIO)
        e1.reset()
        r_vip.append(float(e1.step(GhostexecAction(action_type="reply_email", email_id="e01", message_body=_BODY)).reward or 0))
        e2 = GhostexecEnvironment(SCENARIO)
        e2.reset()
        r_norm.append(float(e2.step(GhostexecAction(action_type="reply_email", email_id="e14", message_body=_BODY)).reward or 0))
        e3 = GhostexecEnvironment(SCENARIO)
        e3.reset()
        r_idle.append(float(e3.step(GhostexecAction(action_type="do_nothing")).reward or 0))
    assert min(r_vip) > max(r_idle)
    assert min(r_norm) > max(r_idle)
    assert sum(r_vip) / len(r_vip) > sum(r_norm) / len(r_norm)


# --- Tone penalty: casual to angry board contact ------------------------------


def test_dead_tone_penalty_casual_to_angry_board() -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    # Marcus Webb is board; ensure angry mood in scenario or pick contact - phase2 has Marcus ANGRY in briefing
    obs_bad = e.step(
        GhostexecAction(
            action_type="reply_email",
            email_id="e01",
            message_body="hey lol no worries",
        )
    )
    assert obs_bad.metadata.get("step_ok") is True
    e2 = GhostexecEnvironment(SCENARIO)
    e2.reset()
    obs_good = e2.step(
        GhostexecAction(
            action_type="reply_email",
            email_id="e01",
            message_body="Dear Marcus, sincerely addressing the board request now.",
        )
    )
    assert float(obs_good.reward or 0) > float(obs_bad.reward or 0)


# --- Reschedule adds conflict channel micro even if overlap unchanged ---------


def test_dead_reschedule_micro_in_breakdown() -> None:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    obs = e.step(
        GhostexecAction(action_type="reschedule_meeting", meeting_id="m07", new_time="2026-04-21T20:30:00")
    )
    assert obs.metadata.get("step_ok") is True
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("conflict_raw") or 0) >= reward_mod._RESCHEDULE_VALID_MICRO_BONUS - 0.01


# --- Unit: compute_step_reward invalid vs noop delta matches contract ---------


def test_dead_compute_reward_invalid_vs_noop_delta() -> None:
    w = GhostexecEnvironment.load_world_from_json(SCENARIO)
    noop = GhostexecAction(action_type="do_nothing")
    bad = GhostexecAction(action_type="reply_email", email_id="missing", message_body="x")
    bd_ok = reward_mod.compute_step_reward(w, w, noop, action_ok=True, episode_done=False)
    bd_bad = reward_mod.compute_step_reward(w, w, bad, action_ok=False, episode_done=False)
    assert bd_bad.final == pytest.approx(bd_ok.final - (0.25 - 0.15))


def test_dead_vip_critical_reply_outscores_professional_critical() -> None:
    """VIP x2 micro on critical senders should dominate professional critical."""
    e_vip = GhostexecEnvironment(SCENARIO)
    e_vip.reset()
    r_vip = float(
        e_vip.step(GhostexecAction(action_type="reply_email", email_id="e01", message_body=_BODY)).reward or 0
    )
    e_pro = GhostexecEnvironment(SCENARIO)
    e_pro.reset()
    r_pro = float(
        e_pro.step(GhostexecAction(action_type="reply_email", email_id="e21", message_body=_BODY)).reward or 0
    )
    assert r_vip > r_pro
