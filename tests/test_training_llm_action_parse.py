# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/llm_action_parse.py (no Unsloth / GPU required)."""

from __future__ import annotations

from training.grpo_ghostexec_reward import ghostexec_env_step_reward
from training.llm_action_parse import parse_completion_to_action


def test_parse_raw_json_action() -> None:
    act = parse_completion_to_action('{"action_type": "do_nothing"}')
    assert act is not None
    assert act.action_type == "do_nothing"


def test_parse_fenced_json() -> None:
    text = """Here is the action:
```json
{"action_type": "archive_email", "email_id": "e09"}
```
"""
    act = parse_completion_to_action(text)
    assert act is not None
    assert act.action_type == "archive_email"
    assert act.email_id == "e09"


def test_parse_invalid_returns_none() -> None:
    assert parse_completion_to_action("not json") is None
    assert parse_completion_to_action('{"action_type": "nope"}') is None


def test_grpo_reward_returns_floats() -> None:
    prompts = ["p1", "p2"]
    completions = [
        '{"action_type": "do_nothing"}',
        '{"action_type": "do_nothing"}',
    ]
    out = ghostexec_env_step_reward(prompts, completions)
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)
