"""Phase 4: reward sub-scores, aggregation, logging, schema drift."""

import json
import random
import statistics
from pathlib import Path

import pytest

from ghostexec.models import GhostexecAction
from ghostexec.server import reward as reward_mod
from ghostexec.server.reward import aggregate_scores
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"
DRIFT = ROOT / "scenarios" / "schema_drift_test.json"


def test_reward_weights_and_aggregator_helpers():
    w = GhostexecEnvironment.load_world_from_json(SCENARIO)
    c, r, t = 1.0, -1.0, 2.5
    weighted_inner = reward_mod.W_CONFLICT * c + reward_mod.W_REL * r + reward_mod.W_TASK * t
    bd = aggregate_scores(
        c,
        r,
        t,
        conflict_raw=c,
        critical_queue_bonus=0.0,
        weighted_inner=weighted_inner,
        weighted_base_only=weighted_inner,
        shaping_synergy=0.0,
        shaping_tradeoff=0.0,
        shaping_potential=0.0,
        shaping_scaffold=0.0,
        shaping_quality=0.0,
        action_ok=True,
        episode_done=False,
        world_after=w,
    )
    assert bd.weighted_base == pytest.approx(reward_mod.WEIGHTED_OUTPUT_SCALE * weighted_inner)


def test_catastrophic_and_completion_bonuses_only_when_episode_done():
    w0 = GhostexecEnvironment.load_world_from_json(SCENARIO)
    w1 = w0.model_copy(deep=True)
    w1.stress = 30
    w2 = w1.model_copy(deep=True)
    action = GhostexecAction(action_type="do_nothing")
    mid = reward_mod.compute_step_reward(w1, w2, action, action_ok=True, episode_done=False)
    assert mid.episode_completion_bonus == 0.0
    assert mid.catastrophic_penalty == 0.0

    w_bad = w1.model_copy(deep=True)
    for i, c in enumerate(w_bad.contacts):
        if c.name == "Marcus Webb":
            w_bad.contacts[i] = c.model_copy(update={"mood": "furious"})
            break
    end = reward_mod.compute_step_reward(w1, w_bad, action, action_ok=True, episode_done=True)
    assert end.episode_completion_bonus == pytest.approx(10.0)
    assert end.catastrophic_penalty == pytest.approx(-15.0)


def test_invalid_step_matches_do_nothing_subscores_plus_invalid_addon():
    w = GhostexecEnvironment.load_world_from_json(SCENARIO)
    noop = GhostexecAction(action_type="do_nothing")
    bad = GhostexecAction(action_type="reply_email", email_id="missing", message_body="x")
    bd_ok = reward_mod.compute_step_reward(w, w, noop, action_ok=True, episode_done=False)
    bd_bad = reward_mod.compute_step_reward(w, w, bad, action_ok=False, episode_done=False)
    assert bd_bad.invalid_step_adjustment == pytest.approx(-0.25)
    # do_nothing carries an additional strict additive floor (-0.15) not applied to invalid non-idle actions.
    assert bd_bad.final == pytest.approx(bd_ok.final - (0.25 - 0.15))


def test_scripted_episode_reward_direction_and_log(tmp_path, monkeypatch):
    logf = tmp_path / "rewards.jsonl"
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    monkeypatch.setattr(env, "_reward_log_path", logf)

    r_resolve = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        )
    )
    r_bad = env.step(GhostexecAction(action_type="do_nothing"))

    assert r_resolve.metadata.get("step_ok") is True
    assert r_bad.metadata.get("step_ok") is True
    assert (r_resolve.reward or 0) > (r_bad.reward or 0)

    assert logf.is_file()
    lines = logf.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    row = json.loads(lines[0])
    assert "reward" in row and "episode_id" in row
    assert row.get("action_type") == "reschedule_meeting"
    assert "conflict_raw" in row and "step_ok" in row
    assert "shaping_total" in row and "shaping_to_base_ratio" in row
    assert "shaping_scaffold" in row
    assert row.get("reward_mode") == "full"


def test_reward_mode_base_turns_off_shaping_terms():
    env = GhostexecEnvironment(SCENARIO, reward_mode="base")
    env.reset()
    obs = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        )
    )
    bd = (obs.metadata or {}).get("reward_breakdown") or {}
    assert float(bd.get("shaping_synergy") or 0.0) == pytest.approx(0.0)
    assert float(bd.get("shaping_tradeoff") or 0.0) == pytest.approx(0.0)
    assert float(bd.get("shaping_potential") or 0.0) == pytest.approx(0.0)


def test_schema_drift_events_mutate_world():
    env = GhostexecEnvironment(SCENARIO, schema_drift_events_path=DRIFT)
    env.reset()
    assert env.step(GhostexecAction(action_type="do_nothing")).metadata.get("step_ok") is True
    assert any("schema drift: shifted" in x for x in env.world.action_log)
    env.step(GhostexecAction(action_type="do_nothing"))
    sarah = env.get_contact("Sarah Chen")
    assert sarah is not None
    assert sarah.communication_preference == "text"
    env.step(GhostexecAction(action_type="do_nothing"))
    t02 = next(t for t in env.world.tasks if t.id == "t02")
    assert t02.deadline == "2026-04-21T07:00:00"
    assert "Marcus Webb" in env._reply_relationship_suppressed  # noqa: SLF001


def test_rewards_differ_between_helpful_and_idle_steps():
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    r_help = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        )
    ).reward
    r_idle = env.step(GhostexecAction(action_type="do_nothing")).reward
    assert r_help is not None and r_idle is not None
    assert r_help != r_idle


# Whitelisted reschedules (known non-overlapping targets for phase2_core at 08:00).
_SAFE_RESCHEDULES: list[tuple[str, str]] = [
    ("m02", "2026-04-21T18:00:00"),
    ("m03", "2026-04-21T18:30:00"),
    ("m06", "2026-04-21T20:00:00"),
    ("m09", "2026-04-21T21:00:00"),
]


def test_seeded_stochastic_policy_reward_spread():
    random.seed(1234)
    K = 80
    archive_ids = [f"e{i:02d}" for i in range(1, 31)]
    contacts = ["Jordan Lee", "Jamie Liu", "Marcus Webb", "Sarah Chen"]
    env = GhostexecEnvironment(SCENARIO)
    env.reset()
    rewards: list[float] = []
    ai = ri = 0
    for _ in range(K):
        u = random.random()
        if u < 0.32:
            obs = env.step(GhostexecAction(action_type="do_nothing"))
        elif u < 0.58:
            eid = archive_ids[ai % len(archive_ids)]
            ai += 1
            obs = env.step(GhostexecAction(action_type="archive_email", email_id=eid))
        elif u < 0.78:
            mid, nt = _SAFE_RESCHEDULES[ri % len(_SAFE_RESCHEDULES)]
            ri += 1
            obs = env.step(
                GhostexecAction(action_type="reschedule_meeting", meeting_id=mid, new_time=nt)
            )
        else:
            cname = contacts[ai % len(contacts)]
            ai += 1
            obs = env.step(
                GhostexecAction(
                    action_type="send_message",
                    contact_name=cname,
                    message_body="Quick sync on priorities.",
                )
            )
        assert obs.reward is not None
        rewards.append(float(obs.reward))

    std = statistics.pstdev(rewards)
    sr = sorted(rewards)
    p5 = sr[max(0, int(0.05 * (len(sr) - 1)))]
    p95 = sr[min(len(sr) - 1, int(0.95 * (len(sr) - 1)))]
    assert std > 0.06
    assert (p95 - p5) > 0.09


def test_good_script_beats_do_nothing_spam_on_mean_reward():
    good = GhostexecEnvironment(SCENARIO)
    good.reset()
    good_actions = [
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        ),
        GhostexecAction(action_type="reply_email", email_id="e01", message_body="Drafting revised figures now."),
        GhostexecAction(action_type="archive_email", email_id="e09"),
        GhostexecAction(
            action_type="send_message",
            contact_name="Jordan Lee",
            message_body="Standup notes attached.",
        ),
        GhostexecAction(action_type="complete_task", task_id="t06"),
    ]
    g_rewards = [good.step(a).reward for a in good_actions]
    g_mean = sum(float(x) for x in g_rewards) / len(g_rewards)

    bad = GhostexecEnvironment(SCENARIO)
    bad.reset()
    b_rewards = [bad.step(GhostexecAction(action_type="do_nothing")).reward for _ in range(5)]
    b_mean = sum(float(x) for x in b_rewards) / len(b_rewards)

    assert g_mean > b_mean + 0.2
