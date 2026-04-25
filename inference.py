"""
Baseline runner for the Ghostexec submission.

This script queries a chat model through the OpenAI client, sends its decision
to the environment server, and prints machine-readable lines expected by simple
evaluators/log parsers.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable

import requests
from pydantic import ValidationError

try:
    from .graders import dinner_disaster_grader, monday_morning_grader, phase2_core_grader
    from .models import GhostexecAction
except ImportError:
    from graders import dinner_disaster_grader, monday_morning_grader, phase2_core_grader
    from models import GhostexecAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
TASK_OVERRIDE = os.getenv("TASK_NAME", "").strip()
BENCHMARK = "ghostexec"

TASK_SETS: dict[str, tuple[str, ...]] = {
    "easy": ("phase2_core",),
    "medium": ("monday_morning",),
    "hard": ("dinner_disaster",),
    "all": ("phase2_core", "monday_morning", "dinner_disaster"),
}

TASK_TO_GRADER = {
    "phase2_core": phase2_core_grader,
    "monday_morning": monday_morning_grader,
    "dinner_disaster": dinner_disaster_grader,
}

SYSTEM_MESSAGE = """
You are acting as an AI Chief-of-Staff assistant in Ghostexec.

You must output exactly one JSON object that matches GhostexecAction.

Allowed action_type values:
- reply_email
- archive_email
- reschedule_meeting
- cancel_meeting
- complete_task
- delegate_task
- send_message
- do_nothing

Allowed keys:
- action_type
- email_id
- message_body
- meeting_id
- new_time
- reason
- task_id
- contact_name
- message

Rules:
- Output valid JSON only (no markdown, no prose).
- Prefer high-impact conflict-reducing actions over do_nothing.
- Only reference ids/entities that appear in the briefing.
- If unsure, output {"action_type":"do_nothing"}.
""".strip()


def emit_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def emit_step(step_no: int, action_text: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    print(
        f"[STEP] step={step_no} action={action_text} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.6f} rewards={reward_text}",
        flush=True,
    )


def choose_tasks(selection: str) -> Iterable[str]:
    if TASK_OVERRIDE:
        return (TASK_OVERRIDE,)
    return TASK_SETS[selection]


def client() -> Any:
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN or API_KEY must be set before running inference.py")
    from openai import OpenAI

    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def fetch_reset(task_name: str) -> dict[str, Any]:
    response = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_name},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def submit_action(action: GhostexecAction) -> dict[str, Any]:
    response = requests.post(
        f"{ENV_URL}/step",
        json={"action": action.model_dump()},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _extract_json_object(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        # tolerate fenced output from weak model instruction following
        s = s.strip("`")
        if "\n" in s:
            s = s.split("\n", 1)[1]
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", s, 0)
    return s[start : end + 1]


def prompt_for_case(observation: dict[str, Any]) -> str:
    return (
        "Take one best next action for the Ghostexec environment.\n\n"
        "Return one final structured GhostexecAction JSON object.\n\n"
        f"{json.dumps(observation, ensure_ascii=True, indent=2)}\n\n"
        "Choose the action that most reduces conflicts, protects relationships, "
        "and advances urgent tasks."
    )


def ask_model(llm: Any, observation: dict[str, Any]) -> GhostexecAction:
    completion = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_for_case(observation)},
        ],
        temperature=0.0,
        max_tokens=260,
        stream=False,
    )
    text = (completion.choices[0].message.content or "").strip()
    payload = json.loads(_extract_json_object(text))
    return GhostexecAction(**payload)


def compact_action(action: GhostexecAction) -> str:
    label = action.action_type
    for candidate in (action.email_id, action.meeting_id, action.task_id, action.contact_name):
        if candidate:
            return f"{label}/{candidate}"
    return label


def _extract_reward(payload: dict[str, Any]) -> float:
    reward_payload = payload.get("reward")
    if isinstance(reward_payload, dict):
        return float(reward_payload.get("total", 0.0))
    if reward_payload is not None:
        return float(reward_payload)
    obs = payload.get("observation")
    if isinstance(obs, dict) and obs.get("reward") is not None:
        return float(obs["reward"])
    return 0.0


def final_score(task_name: str, rewards: list[float]) -> float:
    grader = TASK_TO_GRADER.get(task_name)
    if grader is None:
        score = sum(rewards) / len(rewards) if rewards else 0.0
        return min(max(round(score, 4), 0.01), 0.99)
    return float(grader({"rewards": rewards}))


def run_one_task(llm: Any, task_name: str) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    emit_start(task_name)

    try:
        result = fetch_reset(task_name)
        done = bool(result.get("done", False))

        while not done:
            observation = result.get("observation", result)
            action = ask_model(llm, observation if isinstance(observation, dict) else result)
            action_text = compact_action(action)

            result = submit_action(action)
            reward = _extract_reward(result)
            done = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken += 1
            emit_step(steps_taken, action_text, reward, done, None)

        score = final_score(task_name, rewards)
        success = score >= 0.60

    except json.JSONDecodeError:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "parse_error", 0.0, True, "parse_error")
    except ValidationError:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "schema_error", 0.0, True, "schema_error")
    except Exception as exc:
        rewards = [0.0]
        steps_taken = 1
        emit_step(1, "error", 0.0, True, str(exc))
    finally:
        emit_end(success, steps_taken, score, rewards or [0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ghostexec baseline agent")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task subset to run",
    )
    args = parser.parse_args()

    llm = client()
    for task_name in choose_tasks(args.difficulty):
        run_one_task(llm, task_name)


if __name__ == "__main__":
    main()
