# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# OpenEnv-style GRPO integration (Meta tutorial 04-training.md pattern):
# rollout_func + generate_rollout_completions + reward_funcs that read kwargs.
#
# Tutorial: https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md
# TRL OpenEnv helpers live under trl.experimental (requires a TRL build that includes them).

from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path
from typing import Any

from .llm_action_parse import parse_completion_to_action

try:
    from ghostexec.models import GhostexecAction
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction
    from server.ghostexec_environment import GhostexecEnvironment

_HAS_OPENENV_ROLLOUT: bool | None = None


def openenv_rollout_stack_available() -> bool:
    """
    True if this TRL build ships ``trl.experimental.openenv`` (lightweight ``find_spec`` only).

    The first real call to ``_get_generate_rollout_completions()`` still imports
    ``datasets`` and TRL internals; that can fail if optional deps are missing.
    """
    global _HAS_OPENENV_ROLLOUT
    if _HAS_OPENENV_ROLLOUT is not None:
        return _HAS_OPENENV_ROLLOUT
    try:
        if importlib.util.find_spec("trl") is None:
            _HAS_OPENENV_ROLLOUT = False
            return _HAS_OPENENV_ROLLOUT
        _HAS_OPENENV_ROLLOUT = (
            importlib.util.find_spec("trl.experimental.openenv") is not None
        )
    except (ImportError, ValueError, ModuleNotFoundError):
        _HAS_OPENENV_ROLLOUT = False
    return _HAS_OPENENV_ROLLOUT


def _scenario_path() -> Path:
    raw = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parents[1] / "scenarios" / "phase2_core.json"


GHOSTEXEC_SYSTEM_PROMPT = """You are Ghostexec, an executive chief-of-staff agent.

You receive a plain-text briefing (inbox, calendar, contacts, tasks). You must respond with
EXACTLY one JSON object (no markdown fences) matching GhostexecAction:
- action_type (required string): one of
  reply_email, archive_email, reschedule_meeting, cancel_meeting, complete_task,
  delegate_task, send_message, do_nothing
- Optional fields as needed: email_id, message_body, meeting_id, new_time, reason,
  task_id, contact_name, message

Do not repeat boilerplate; pick the single best legal next action for the briefing."""


DEFAULT_TASK_PROMPT = (
    "Read the briefing and output one JSON object for the best immediate executive action."
)


def _get_generate_rollout_completions():
    if not openenv_rollout_stack_available():
        raise RuntimeError(
            "OpenEnv-style GRPO rollout requires ``trl.experimental.openenv`` (see Meta "
            "https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md ). "
            "Use a recent TRL, or the simpler scalar reward in ``grpo_ghostexec_reward.ghostexec_env_step_reward``."
        )
    try:
        from trl.experimental.openenv import generate_rollout_completions
    except ImportError as e:
        raise RuntimeError(
            "Could not import ``generate_rollout_completions``. Install optional training deps "
            "(``uv sync --extra training`` includes ``datasets``) and a TRL build with OpenEnv helpers."
        ) from e
    return generate_rollout_completions


def rollout_once_ghostexec(
    trainer: Any,
    env: GhostexecEnvironment,
    tokenizer: Any,
    dataset_prompt: str,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    One Ghostexec transition: ``reset()`` → model generation → ``step(action)``.

    Returns token lists and scalar signals for ``reward_funcs`` (OpenEnv Wordle tutorial shape).
    """
    generate_rollout_completions = _get_generate_rollout_completions()
    sys_prompt = system_prompt or GHOSTEXEC_SYSTEM_PROMPT

    obs = env.reset()
    briefing = obs.echoed_message or ""
    task = dataset_prompt.strip() if dataset_prompt.strip() else DEFAULT_TASK_PROMPT
    user_content = f"{task}\n\n=== BRIEFING ===\n{briefing}"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
    prompt_ids = list(rollout_outputs["prompt_ids"])
    completion_ids = list(rollout_outputs["completion_ids"])
    logprobs = list(rollout_outputs["logprobs"])
    completion_text = rollout_outputs.get("text") or tokenizer.decode(
        rollout_outputs["completion_ids"],
        skip_special_tokens=True,
    )

    parsed = parse_completion_to_action(completion_text)
    parse_ok = 1.0 if parsed is not None else 0.0
    act = parsed if parsed is not None else GhostexecAction(action_type="do_nothing")
    obs2 = env.step(act)
    step_reward = float(obs2.reward or 0.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "parse_ok": parse_ok,
        "env_step_reward": step_reward,
    }


def rollout_multiturn_ghostexec(
    trainer: Any,
    env: GhostexecEnvironment,
    tokenizer: Any,
    dataset_prompt: str,
    *,
    num_turns: int = 3,
    gamma: float = 0.9,
    system_prompt: str | None = None,
    follow_up_instruction: str = (
        "The previous action was executed. Here is the updated briefing. "
        "Respond with exactly one JSON object for the next executive action."
    ),
) -> dict[str, Any]:
    """Multi-turn Ghostexec episode using the OpenEnv ``generate_rollout_completions`` path.

    Each turn: build a chat-template prompt from the current briefing, let the
    trainer's generator sample a completion, parse + step the env, continue
    until ``done`` or ``num_turns`` turns. Returns the *last turn's* token
    stream plus aggregated per-episode signals; TRL reward funcs read the
    aggregates from kwargs.

    Attribution note: we return the last turn's ``prompt_ids`` / ``completion_ids``
    / ``logprobs`` because TRL's OpenEnv path expects a single
    (prompt, completion, logprobs) triple per sample. The whole-episode
    discounted reward still gets credited to that final action — consistent
    with how the 2048 Wordle tutorial attributes rollout returns.
    """
    generate_rollout_completions = _get_generate_rollout_completions()
    sys_prompt = system_prompt or GHOSTEXEC_SYSTEM_PROMPT
    task = dataset_prompt.strip() if dataset_prompt.strip() else DEFAULT_TASK_PROMPT

    obs = env.reset()
    per_turn_rewards: list[float] = []
    per_turn_parse_ok: list[float] = []
    last_prompt_ids: list[int] = []
    last_completion_ids: list[int] = []
    last_logprobs: list[float] = []

    for turn in range(max(1, num_turns)):
        briefing = obs.echoed_message or ""
        user_header = task if turn == 0 else follow_up_instruction
        user_content = f"{user_header}\n\n=== BRIEFING ===\n{briefing}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        rollout_out = generate_rollout_completions(trainer, [prompt_text])[0]
        last_prompt_ids = list(rollout_out["prompt_ids"])
        last_completion_ids = list(rollout_out["completion_ids"])
        last_logprobs = list(rollout_out["logprobs"])
        completion_text = rollout_out.get("text") or tokenizer.decode(
            rollout_out["completion_ids"], skip_special_tokens=True
        )

        parsed = parse_completion_to_action(completion_text)
        per_turn_parse_ok.append(1.0 if parsed is not None else 0.0)
        act = parsed if parsed is not None else GhostexecAction(action_type="do_nothing")
        obs = env.step(act)
        per_turn_rewards.append(float(obs.reward or 0.0))
        if obs.done:
            break

    total = 0.0
    discount = 1.0
    for r in per_turn_rewards:
        total += discount * r
        discount *= gamma

    return {
        "prompt_ids": last_prompt_ids,
        "completion_ids": last_completion_ids,
        "logprobs": last_logprobs,
        "parse_ok": sum(per_turn_parse_ok) / max(1, len(per_turn_parse_ok)),
        "env_step_reward": total,
        "num_turns_run": float(len(per_turn_rewards)),
    }


def ghostexec_rollout_func(prompts: list[str], trainer: Any = None) -> dict[str, Any]:
    """
    TRL ``rollout_func`` entrypoint (same contract as OpenEnv Wordle tutorial).

    Uses a **fresh** in-process :class:`GhostexecEnvironment` per prompt in
    ``prompts`` (scenario from ``GHOSTEXEC_GRPO_SCENARIO`` or ``phase2_core.json``)
    so GRPO samples in one batch do not leak world state across completions.

    Multi-turn mode: set ``GHOSTEXEC_ROLLOUT_TURNS`` (default ``3``) to run
    multiple model turns per episode via ``rollout_multiturn_ghostexec``.
    Set ``GHOSTEXEC_ROLLOUT_TURNS=1`` for single-step-only rollouts.
    Discount controlled by ``GHOSTEXEC_ROLLOUT_GAMMA`` (default ``0.9``).
    """
    if trainer is None:
        raise ValueError("ghostexec_rollout_func requires trainer= from GRPOTrainer.")

    _get_generate_rollout_completions()
    if os.environ.get("TRL_EXPERIMENTAL_SILENCE", "").strip() != "1":
        warnings.warn(
            "Using trl.experimental.openenv (unstable API). Set TRL_EXPERIMENTAL_SILENCE=1 to hide this.",
            stacklevel=2,
        )

    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Trainer must expose processing_class or tokenizer.")

    episode_prompt_ids: list[Any] = []
    episode_completion_ids: list[Any] = []
    episode_logprobs: list[Any] = []
    parse_oks: list[float] = []
    step_rewards: list[float] = []
    turns_run: list[float] = []

    try:
        num_turns = max(1, int(os.environ.get("GHOSTEXEC_ROLLOUT_TURNS", "3")))
    except ValueError:
        num_turns = 3
    try:
        gamma = float(os.environ.get("GHOSTEXEC_ROLLOUT_GAMMA", "0.9"))
    except ValueError:
        gamma = 0.9

    for prompt_text in prompts:
        # Fresh env per sample so parallel GRPO completions never share mutated world.
        env = GhostexecEnvironment(_scenario_path())
        if num_turns > 1:
            episode = rollout_multiturn_ghostexec(
                trainer,
                env,
                tokenizer,
                prompt_text,
                num_turns=num_turns,
                gamma=gamma,
            )
            turns_run.append(float(episode.get("num_turns_run", 1.0)))
        else:
            episode = rollout_once_ghostexec(
                trainer,
                env,
                tokenizer,
                prompt_text,
            )
            turns_run.append(1.0)
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        parse_oks.append(episode["parse_ok"])
        step_rewards.append(episode["env_step_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "parse_ok": parse_oks,
        "env_step_reward": step_rewards,
        "num_turns_run": turns_run,
    }


def reward_ghostexec_parse_ok(completions: list[str], **kwargs: Any) -> list[float]:
    """Shaped parse signal (0/1) from ``ghostexec_rollout_func`` kwargs."""
    rewards = kwargs.get("parse_ok") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_ghostexec_env_step(completions: list[str], **kwargs: Any) -> list[float]:
    """Environment step reward from ``ghostexec_rollout_func`` kwargs."""
    rewards = kwargs.get("env_step_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]
