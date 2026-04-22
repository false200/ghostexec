# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Local smoke test: simulate 20 GRPO steps worth of reward computation.

Confirms the patched reward pipeline (bounded squash + anti-tie jitter +
multi-channel combination) produces non-zero within-group variance on
realistic-looking completion batches, BEFORE the user burns a GPU hour
on a full run.

The "dead signal" failure pattern we debugged earlier was: mean reward
flat for 40 steps, ``reward_std=0.0`` every step, zero gradient. This
script reproduces the GRPO batching shape (num_generations completions
per step) and asserts ``std > 1e-6`` for the combined reward across
channels for nearly every step.
"""

from __future__ import annotations

import os
import random
import statistics

os.environ["GHOSTEXEC_REWARD_SQUASH"] = "1"
os.environ["GHOSTEXEC_REWARD_JITTER"] = "1"
os.environ["GHOSTEXEC_CURRICULUM"] = "mixed"
os.environ["GHOSTEXEC_PERTURB"] = "1"

from training.grpo_ghostexec_reward import (  # noqa: E402
    ghostexec_env_step_reward,
    reward_format_valid,
    reward_group_diversity,
    reward_id_relevance,
    reward_vip_critical_reply_bonus,
)

TEMPLATES = [
    '{"action_type": "reply_email", "email_id": "e01", "message_body": "On it."}',
    '{"action_type": "reply_email", "email_id": "e02", "message_body": "Working on it now."}',
    '{"action_type": "reschedule_meeting", "meeting_id": "m01", '
    '"new_time": "2025-01-10T14:00:00Z", "reason": "conflict"}',
    '{"action_type": "complete_task", "task_id": "t01"}',
    '{"action_type": "delegate_task", "task_id": "t01", "contact_name": "Alice"}',
    '{"action_type": "do_nothing"}',
    "not json at all",
    '{"action_type": "reply_email", "email_id": "NOPE", "message_body": "hi"}',
]


def main() -> None:
    dead = 0
    steps = 20
    num_generations = 8
    for step in range(steps):
        rng = random.Random(step)
        completions = [rng.choice(TEMPLATES) for _ in range(num_generations)]
        env_r = ghostexec_env_step_reward([""], completions)
        fmt_r = reward_format_valid([""], completions)
        div_r = reward_group_diversity([""], completions)
        idr_r = reward_id_relevance([""], completions)
        vip_r = reward_vip_critical_reply_bonus([""], completions)
        combined = [sum(x) / 5.0 for x in zip(env_r, fmt_r, div_r, idr_r, vip_r)]
        sigma = statistics.pstdev(combined)
        mean = statistics.mean(combined)
        status = "DEAD" if sigma < 1e-6 else "OK"
        if sigma < 1e-6:
            dead += 1
        print(f"step={step:2d} mean={mean:+.4f} std={sigma:.4f} {status}")

    print(f"\n-- dead-signal steps: {dead}/{steps} --")
    assert dead <= 2, f"Too many zero-variance steps: {dead}/{steps}"
    print("SMOKE OK: within-group variance is alive.")


if __name__ == "__main__":
    main()
