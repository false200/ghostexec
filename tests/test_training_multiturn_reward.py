# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/multiturn_reward.py.

We stub out the model/tokenizer with a tiny fake that has ``apply_chat_template``
and route generation through ``generate_fn`` so no HF / torch is required.
"""

from __future__ import annotations

import pytest

from training.multiturn_reward import make_multiturn_reward


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        **kwargs: object,
    ) -> str:
        return "\n".join(m["content"] for m in messages)


REPLY_VIP_CRIT_E01 = (
    '{"action_type": "reply_email", "email_id": "e01", '
    '"message_body": "On it — drafting now."}'
)
DO_NOTHING = '{"action_type": "do_nothing"}'


def test_multiturn_reward_runs_with_generate_fn_stub() -> None:
    call_count = {"n": 0}

    def fake_generate(prompt: str) -> str:
        call_count["n"] += 1
        # Always produce a valid do_nothing so follow-up turns parse cleanly.
        return DO_NOTHING

    reward_fn = make_multiturn_reward(
        model=object(),
        tokenizer=_FakeTokenizer(),
        num_turns=3,
        gamma=0.9,
        generate_fn=fake_generate,
    )
    rewards = reward_fn([""], [REPLY_VIP_CRIT_E01, DO_NOTHING])
    assert len(rewards) == 2
    for r in rewards:
        assert isinstance(r, float)
    # 2 samples * (num_turns - 1) = 4 follow-up generations expected at most
    # (fewer if episode ended early via done=True).
    assert call_count["n"] <= 4


def test_multiturn_reward_gracefully_handles_broken_generation() -> None:
    def broken_generate(prompt: str) -> str:
        raise RuntimeError("simulated generation failure")

    reward_fn = make_multiturn_reward(
        model=object(),
        tokenizer=_FakeTokenizer(),
        num_turns=4,
        gamma=0.9,
        generate_fn=broken_generate,
    )
    # Should NOT raise — the first turn still runs from the GRPO completion,
    # follow-ups are skipped on exception.
    rewards = reward_fn([""], [REPLY_VIP_CRIT_E01])
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)


def test_multiturn_num_turns_clamped_to_sane_max() -> None:
    """Participant Help Guide §5–6: extreme num_turns cannot spawn unbounded follow-ups."""
    call_count = {"n": 0}

    def fake_generate(prompt: str) -> str:
        call_count["n"] += 1
        return DO_NOTHING

    reward_fn = make_multiturn_reward(
        model=object(),
        tokenizer=_FakeTokenizer(),
        num_turns=10_000,
        gamma=0.9,
        generate_fn=fake_generate,
    )
    reward_fn([""], [REPLY_VIP_CRIT_E01])
    assert call_count["n"] <= 31


def test_single_turn_matches_first_step_scale() -> None:
    """num_turns=1 should behave like a plain first-step reward."""

    def never_called(prompt: str) -> str:
        raise AssertionError("follow-up generator must not run when num_turns=1")

    reward_fn = make_multiturn_reward(
        model=object(),
        tokenizer=_FakeTokenizer(),
        num_turns=1,
        generate_fn=never_called,
    )
    rewards = reward_fn([""], [REPLY_VIP_CRIT_E01])
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)


def test_multiturn_rewards_dominated_by_first_turn_when_gamma_small() -> None:
    reward_fn = make_multiturn_reward(
        model=object(),
        tokenizer=_FakeTokenizer(),
        num_turns=4,
        gamma=0.01,
        generate_fn=lambda p: DO_NOTHING,
    )
    r_good = reward_fn([""], [REPLY_VIP_CRIT_E01])[0]
    r_bad = reward_fn([""], [DO_NOTHING])[0]
    assert r_good > r_bad, "first-turn VIP reply must out-score first-turn do_nothing"
