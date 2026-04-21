"""Phase 2: world state, inbox, calendar, contacts, tasks (scenario-driven)."""

from pathlib import Path

from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"


def test_scenario_file_exists():
    assert SCENARIO.is_file()


def test_world_json_roundtrip():
    world = GhostexecEnvironment.load_world_from_json(SCENARIO)
    blob = GhostexecEnvironment.world_to_json(world)
    again = GhostexecEnvironment.world_from_json(blob)
    assert again.simulation_time == world.simulation_time
    assert len(again.emails) == len(world.emails)
    assert len(again.meetings) == len(world.meetings)


def test_pool_sizes_from_scenario():
    w = GhostexecEnvironment.load_world_from_json(SCENARIO)
    assert len(w.emails) >= 30
    assert len(w.meetings) >= 8
    assert len(w.contacts) >= 15
    assert len(w.tasks) >= 10


def test_inbox_unread_priority_order():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    unread = env.get_unread_emails_sorted()
    priorities = [e.priority for e in unread]
    rank = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    assert priorities == sorted(priorities, key=lambda p: rank[p])
    assert unread[0].priority == "critical"


def test_calendar_detects_four_conflicts():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    conflicts = env.detect_meeting_conflicts()
    assert len(conflicts) >= 4


def test_contact_mood_update():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    c = env.get_contact("David Okonkwo")
    assert c is not None
    assert c.mood == "angry"
    assert env.update_contact_mood("David Okonkwo", "neutral")
    assert env.get_contact("David Okonkwo") is not None
    assert env.get_contact("David Okonkwo").mood == "neutral"


def test_overdue_tasks_after_time_advance():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    future = "2026-04-22T12:00:00"
    env.set_simulation_time(future)
    overdue = env.overdue_tasks_at(future)
    assert len(overdue) >= 2
    assert all(t.status == "overdue" for t in overdue)


def test_mark_email_read_and_reschedule_reduces_calendar_conflicts():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    before = len(env.detect_meeting_conflicts())
    assert env.reschedule_meeting("m02", "2026-04-21T18:00:00")
    after = len(env.detect_meeting_conflicts())
    assert after < before
    assert env.mark_email_read("e01")
