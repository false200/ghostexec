# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Multi-turn reward factory for TRL GRPOTrainer.
#
# Each group completion becomes the *first* action in an episode; the factory
# then uses the model itself to generate k-1 more JSON actions, stepping the
# environment after every one. The returned reward is the discounted sum of
# per-step env rewards across the full episode.
#
# Shape contract matches TRL ``reward_funcs`` entries:
#   reward_fn(prompts, completions, **kwargs) -> list[float]
#
# This complements the scripted k-step lookahead in
# ``grpo_ghostexec_reward`` (which uses ``smart_action`` as a proxy): here the
# policy itself closes the loop, so the reward credits "first action that
# enables good follow-ups" rather than "first action a heuristic can extend."

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable

from .llm_action_parse import parse_completion_to_action
from .scenarios_sampler import pick_scenario, scenario_path

try:
    from ghostexec.models import GhostexecAction, GhostexecObservation
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction, GhostexecObservation
    from server.ghostexec_environment import GhostexecEnvironment


RewardFn = Callable[..., list[float]]


def _default_scenario_path() -> Path:
    env_path = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[1] / "scenarios" / "phase2_core.json"


def _pick_scenario(rng: random.Random) -> Path:
    env_path = os.environ.get("GHOSTEXEC_GRPO_SCENARIO", "").strip()
    if env_path:
        return Path(env_path)
    level = os.environ.get("GHOSTEXEC_CURRICULUM", "").strip()
    if level:
        return pick_scenario(rng, level)
    return _default_scenario_path()


def _messages_for_turn(system_prompt: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Best-effort chat template rendering that tolerates older tokenizers."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def _model_generate_text(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Single-shot HF ``generate`` -> decoded string.

    Isolated helper so tests can swap in a deterministic generator.
    """
    import torch  # lazy (training-only dep)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(getattr(model, "device", "cpu"))
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(1e-5, temperature),
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        )
    # Strip the prompt prefix — HF concatenates prompt + completion.
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def make_multiturn_reward(
    model: Any,
    tokenizer: Any,
    *,
    num_turns: int = 3,
    gamma: float = 0.9,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    follow_up_instruction: str = (
        "The previous action was executed. Here is the updated briefing. "
        "Respond with exactly one JSON object for the next executive action."
    ),
    generate_fn: Callable[[str], str] | None = None,
) -> RewardFn:
    """Build a TRL reward function that rolls out multi-turn episodes.

    Parameters
    ----------
    model, tokenizer:
        HF model + tokenizer used for turn-2..N generation. Turn 1 reuses the
        completion that GRPO sampled. If you also want turn 1 to be schema-
        constrained, call ``patch_model_for_json_generation`` first.
    num_turns:
        Total turns per episode (including the first one from the GRPO sample).
    gamma:
        Per-turn discount applied to rewards after turn 1.
    max_new_tokens, temperature:
        Generation knobs for follow-up turns.
    system_prompt:
        Optional override for the system message used on follow-up turns.
    generate_fn:
        Optional hook that takes a prompt string and returns completion text.
        Intended for tests — skips the HF/torch path entirely.

    Returns
    -------
    reward_fn : callable
        ``reward_fn(prompts, completions, **kwargs) -> list[float]``, usable
        directly inside ``GRPOConfig.reward_funcs``.
    """
    # Participant Help Guide §5–6 / §12: hard bounds so reward rollouts cannot
    # exhaust GPU time or tokens if callers pass extreme hyperparameters.
    num_turns = max(1, min(int(num_turns), 32))
    max_new_tokens = max(8, min(int(max_new_tokens), 512))
    gamma = max(0.0, min(float(gamma), 1.0))

    # Lazy import of the default system prompt keeps this module import-light.
    from .openenv_grpo_rollout import GHOSTEXEC_SYSTEM_PROMPT

    sys_prompt = system_prompt or GHOSTEXEC_SYSTEM_PROMPT

    def _generate(prompt_text: str) -> str:
        if generate_fn is not None:
            return generate_fn(prompt_text)
        return _model_generate_text(
            model,
            tokenizer,
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _episode_reward(completion: Any, seed: int) -> float:
        rng = random.Random(seed)
        scen = _pick_scenario(rng)
        env = GhostexecEnvironment(scen)
        env.reset()

        act = parse_completion_to_action(completion) or GhostexecAction(
            action_type="do_nothing"
        )
        obs: GhostexecObservation = env.step(act)
        total = float(obs.reward or 0.0)

        discount = gamma
        for _ in range(max(0, num_turns - 1)):
            if obs.done:
                break
            user_text = (
                f"{follow_up_instruction}\n\n=== BRIEFING ===\n"
                f"{obs.echoed_message or ''}"
            )
            prompt_text = _apply_chat_template(
                tokenizer, _messages_for_turn(sys_prompt, user_text)
            )
            try:
                completion_text = _generate(prompt_text)
            except Exception:
                break
            next_act = parse_completion_to_action(completion_text) or GhostexecAction(
                action_type="do_nothing"
            )
            obs = env.step(next_act)
            total += discount * float(obs.reward or 0.0)
            discount *= gamma
        return total

    def reward_fn(
        prompts: list[Any], completions: list[Any], **kwargs: object
    ) -> list[float]:
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            rewards.append(_episode_reward(completion, seed=1_000_003 * (i + 1)))
        return rewards

    reward_fn.__name__ = "ghostexec_multiturn_reward"
    return reward_fn


__all__ = ["make_multiturn_reward"]
