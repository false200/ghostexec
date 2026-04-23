# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the shaping reward channels in training/grpo_ghostexec_reward.py."""

from __future__ import annotations

import os

import pytest

from training.grpo_ghostexec_reward import (
    REWARD_FUNCS_ALL,
    ghostexec_env_step_reward,
    reward_format_valid,
    reward_group_diversity,
    reward_id_relevance,
    reward_vip_critical_reply_bonus,
)


REPLY_VIP_CRIT_E01 = (
    '{"action_type": "reply_email", "email_id": "e01", '
    '"message_body": "On it — drafting now."}'
)
REPLY_INVALID_ID = (
    '{"action_type": "reply_email", "email_id": "NOPE_999", '
    '"message_body": "hello"}'
)
DO_NOTHING = '{"action_type": "do_nothing"}'
BROKEN = "not json at all"


def test_reward_funcs_all_is_list_of_callables() -> None:
    assert len(REWARD_FUNCS_ALL) >= 5
    assert all(callable(f) for f in REWARD_FUNCS_ALL)


def test_format_valid_rewards() -> None:
    out = reward_format_valid([""], [REPLY_VIP_CRIT_E01, BROKEN, DO_NOTHING])
    assert out == [1.0, -1.0, 1.0]


def test_group_diversity_penalizes_duplicates() -> None:
    out = reward_group_diversity([""], [DO_NOTHING, DO_NOTHING, REPLY_VIP_CRIT_E01])
    # The two duplicates each get (1/2)-1 = -0.5; the unique one gets 0.
    assert out[0] == pytest.approx(-0.5)
    assert out[1] == pytest.approx(-0.5)
    assert out[2] == 0.0


def test_id_relevance_penalizes_unknown_ids() -> None:
    out = reward_id_relevance([""], [REPLY_VIP_CRIT_E01, REPLY_INVALID_ID, DO_NOTHING])
    assert out[0] == 0.0
    assert out[1] == pytest.approx(-0.25)
    assert out[2] == 0.0


def test_vip_critical_reply_bonus_fires_only_for_vip_crit() -> None:
    out = reward_vip_critical_reply_bonus(
        [""],
        [REPLY_VIP_CRIT_E01, REPLY_INVALID_ID, DO_NOTHING],
    )
    assert out[0] == pytest.approx(0.5)
    assert out[1] == 0.0
    assert out[2] == 0.0


def test_env_step_reward_runs_and_returns_floats() -> None:
    out = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01, DO_NOTHING])
    assert len(out) == 2
    for r in out:
        assert isinstance(r, float)


def test_env_step_reward_honors_kstep_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    # k=0 (default) and k=2 should both return floats; k>0 exercises the
    # lazy smart_action import path.
    base = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01])[0]
    monkeypatch.setenv("GHOSTEXEC_REWARD_KSTEPS", "2")
    boosted = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01])[0]
    assert isinstance(base, float)
    assert isinstance(boosted, float)


def test_do_nothing_is_not_reward_preferred() -> None:
    good, nothing = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01, DO_NOTHING])
    assert good > nothing, "do_nothing must not out-score a valid VIP-critical reply"


# --- PAR-style boundedness + anti-dead-signal jitter tests --------------------
# The reward shaping paper (arXiv:2502.18770) argues an RL reward should be (1)
# bounded and (2) exhibit rapid initial growth then gradual convergence. Our
# ``ghostexec_env_step_reward`` wraps its raw output in ``tanh`` exactly for
# that reason — below we assert the property is actually in effect.


def test_env_step_reward_is_bounded_when_squash_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """With squash on (default), every reward must land strictly inside (-1, 1)."""
    monkeypatch.setenv("GHOSTEXEC_REWARD_SQUASH", "1")
    monkeypatch.setenv("GHOSTEXEC_REWARD_JITTER", "0")
    out = ghostexec_env_step_reward(
        [""],
        [REPLY_VIP_CRIT_E01, REPLY_INVALID_ID, DO_NOTHING, BROKEN],
    )
    for r in out:
        assert -1.0 < r < 1.0, f"squashed reward {r} escaped (-1, 1)"


def test_env_step_reward_squash_preserves_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tanh`` is monotonic: a VIP-critical reply must still out-rank do_nothing."""
    monkeypatch.setenv("GHOSTEXEC_REWARD_SQUASH", "1")
    monkeypatch.setenv("GHOSTEXEC_REWARD_JITTER", "0")
    good, nothing = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01, DO_NOTHING])
    assert good > nothing


def test_env_step_reward_jitter_breaks_zero_variance_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different-text but same-reward completions collapse to std=0; jitter breaks that."""
    monkeypatch.setenv("GHOSTEXEC_REWARD_SQUASH", "1")
    monkeypatch.setenv("GHOSTEXEC_REWARD_JITTER", "1")
    monkeypatch.setenv("GHOSTEXEC_REWARD_JITTER_MAG", "1e-3")
    # Three unparsable strings — all fall through to do_nothing and score
    # identically, but the text hashes differ so jitter must separate them.
    distinct = ["not json at all", "also not json", "still not parseable"]
    out = ghostexec_env_step_reward([""], distinct)
    assert len(set(out)) > 1, "jitter did not break the tie on zero-variance group"
    spread = max(out) - min(out)
    assert spread < 1e-2, f"jitter magnitude too large ({spread}); would teach noise"


def test_env_step_reward_jitter_off_leaves_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabling jitter keeps same-reward completions at identical rewards."""
    monkeypatch.setenv("GHOSTEXEC_REWARD_SQUASH", "1")
    monkeypatch.setenv("GHOSTEXEC_REWARD_JITTER", "0")
    distinct = ["not json at all", "also not json", "still not parseable"]
    out = ghostexec_env_step_reward([""], distinct)
    assert len(set(out)) == 1


def test_env_step_reward_extreme_ksteps_env_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Participant Help Guide §5–6: absurd GHOSTEXEC_REWARD_KSTEPS must not hang reward."""
    monkeypatch.setenv("GHOSTEXEC_REWARD_KSTEPS", "999999")
    monkeypatch.delenv("GHOSTEXEC_GRPO_SCENARIO", raising=False)
    out = ghostexec_env_step_reward([""], [REPLY_VIP_CRIT_E01])
    assert len(out) == 1
    assert isinstance(out[0], float)
