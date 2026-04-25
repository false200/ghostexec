"""Phase 3: plain-text briefing, eight legal actions, validation without crashes."""

from pathlib import Path

import pytest

from ghostexec.models import GhostexecAction
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"


def _env() -> GhostexecEnvironment:
    e = GhostexecEnvironment(SCENARIO)
    e.reset()
    return e


def test_briefing_is_plain_text_after_reset():
    env = _env()
    obs = env.reset()
    text = obs.echoed_message
    assert "=== GHOSTEXEC BRIEFING" in text
    assert "UNREAD EMAILS" in text
    assert "CALENDAR CONFLICTS IN NEXT 4 HOURS" in text
    assert "CONTACTS TO WATCH" in text
    assert "OVERDUE OR DUE-SOON TASKS" in text
    assert "EXEC STRESS LEVEL" in text
    assert "STEPS REMAINING" in text
    assert obs.message_length == len(text)


@pytest.mark.parametrize(
    "action,check",
    [
        (
            GhostexecAction(action_type="reply_email", email_id="e05", message_body="On it."),
            lambda env: next(e for e in env.world.emails if e.id == "e05").replied is True,
        ),
        (
            GhostexecAction(action_type="archive_email", email_id="e09"),
            lambda env: next(e for e in env.world.emails if e.id == "e09").read is True,
        ),
        (
            GhostexecAction(
                action_type="reschedule_meeting",
                meeting_id="m03",
                new_time="2026-04-21T18:00:00",
            ),
            lambda env: next(m for m in env.world.meetings if m.id == "m03").start
            == "2026-04-21T18:00:00",
        ),
        (
            GhostexecAction(
                action_type="cancel_meeting",
                meeting_id="m10",
                reason="Merged into ops review",
            ),
            lambda env: next(m for m in env.world.meetings if m.id == "m10").cancelled is True,
        ),
        (
            GhostexecAction(action_type="complete_task", task_id="t07"),
            lambda env: next(t for t in env.world.tasks if t.id == "t07").status == "done",
        ),
        (
            GhostexecAction(
                action_type="delegate_task",
                task_id="t08",
                contact_name="Jordan Lee",
            ),
            lambda env: next(t for t in env.world.tasks if t.id == "t08").delegated_to == "Jordan Lee",
        ),
        (
            GhostexecAction(
                action_type="send_message",
                contact_name="Jamie Liu",
                message_body="Thanks for the demo feedback.",
            ),
            lambda env: any("message to Jamie Liu" in line for line in env.world.action_log),
        ),
        (
            GhostexecAction(action_type="do_nothing"),
            lambda env: True,
        ),
    ],
)
def test_each_legal_action_runs_without_crash(action, check):
    env = _env()
    obs = env.step(action)
    assert obs.echoed_message
    assert check(env)


def test_reply_marks_email_handled():
    env = _env()
    e = next(x for x in env.world.emails if x.id == "e14")
    assert not e.read
    env.step(GhostexecAction(action_type="reply_email", email_id="e14", message_body="Noted."))
    e2 = next(x for x in env.world.emails if x.id == "e14")
    assert e2.read and e2.replied


def test_invalid_actions_return_error_metadata_not_exception():
    base = _env()
    r_do_nothing = base.step(GhostexecAction(action_type="do_nothing")).reward

    env = _env()
    obs = env.step(GhostexecAction(action_type="reply_email", email_id="nope", message_body="x"))
    assert obs.metadata.get("step_ok") is False
    assert obs.metadata.get("step_error")
    # Same before→after sub-scores as do_nothing, plus explicit invalid add-on.
    # do_nothing has an additional strict additive floor (-0.15), so the delta is -0.10 here.
    assert obs.reward == pytest.approx((r_do_nothing or 0) - (0.25 - 0.15))

    obs2 = env.step(GhostexecAction(action_type="complete_task", task_id="t09"))
    assert obs2.metadata.get("step_ok") is False
    assert "already done" in (obs2.metadata.get("step_error") or "").lower()

    obs3 = env.step(
        GhostexecAction(
            action_type="send_message",
            contact_name="Nobody By That Name",
            message_body="hello",
        )
    )
    assert obs3.metadata.get("step_ok") is False

    obs4 = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m03",
            new_time="2026-04-21T09:30:00",
        )
    )
    assert obs4.metadata.get("step_ok") is False
    assert "overlap" in (obs4.metadata.get("step_error") or "").lower()


def test_reschedule_resolves_prior_conflict_pair():
    env = _env()
    before = {frozenset((r["meeting_a"], r["meeting_b"])) for r in env.detect_meeting_conflicts()}
    assert frozenset(("m01", "m02")) in before
    obs = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        )
    )
    assert obs.metadata.get("step_ok") is True
    after = {frozenset((r["meeting_a"], r["meeting_b"])) for r in env.detect_meeting_conflicts()}
    assert frozenset(("m01", "m02")) not in after
