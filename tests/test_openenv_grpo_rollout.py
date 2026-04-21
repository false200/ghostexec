# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# OpenEnv tutorial–aligned GRPO helpers (kwargs rewards, optional rollout stack).

from __future__ import annotations

import os

import pytest

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def test_reward_funcs_read_rollout_kwargs() -> None:
    from training.openenv_grpo_rollout import reward_ghostexec_env_step, reward_ghostexec_parse_ok

    completions = ["{}"]
    assert reward_ghostexec_parse_ok(completions, parse_ok=[1.0, 0.0]) == [1.0, 0.0]
    assert reward_ghostexec_env_step(completions, env_step_reward=[0.25, -0.1]) == [0.25, -0.1]
    assert reward_ghostexec_parse_ok(completions) == [0.0]
    assert reward_ghostexec_env_step(completions) == [0.0]


def test_openenv_rollout_stack_probe_does_not_crash() -> None:
    from training.openenv_grpo_rollout import openenv_rollout_stack_available

    # May be False on minimal installs (no datasets / old TRL); must not raise.
    assert isinstance(openenv_rollout_stack_available(), bool)
