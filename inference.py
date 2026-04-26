"""
Baseline runner for the Ghostexec OpenEnv submission.

Links (keep these in sync when you change the env):
  - **openenv.yaml** — `name`, `port`, `tasks[].id`, `tasks[].grader`, `max_steps`, `difficulties`
  - **graders.py** — episode-level scores in (0.01, 0.99); symbols referenced by `tasks[].grader`
  - **scenarios/*.json** — fixtures named in each task description in `openenv.yaml`
  - **server/** — FastAPI app from `openenv.yaml` `app:` (`server.app:app`)

This script calls the deployed/local env over HTTP (`/reset`, `/step`), queries an LLM via the
OpenAI-compatible HF router, then aggregates step rewards with the **same** grader functions
used for OpenEnv validation (must match `openenv.yaml` task table).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import requests
from pydantic import ValidationError

try:
    from .graders import dinner_disaster_grader, monday_morning_grader, phase2_core_grader
    from .models import GhostexecAction
except ImportError:
    from graders import dinner_disaster_grader, monday_morning_grader, phase2_core_grader
    from models import GhostexecAction

REPO_ROOT = Path(__file__).resolve().parent
OPENENV_SPEC = REPO_ROOT / "openenv.yaml"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
# Default matches openenv.yaml `port: 8000` and `uv run server` / Spaces proxy.
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000").rstrip("/")
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

_GRADER_TO_SYMBOL = {
    phase2_core_grader: "graders.phase2_core_grader",
    monday_morning_grader: "graders.monday_morning_grader",
    dinner_disaster_grader: "graders.dinner_disaster_grader",
}


def load_openenv_task_rows(spec_path: Path) -> list[dict[str, str]]:
    """Parse task `id` + `grader` from openenv.yaml without requiring PyYAML."""
    if not spec_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for raw in spec_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        m_id = re.match(r"^\s*-\s+id:\s*(\S+)\s*$", line)
        if m_id:
            if cur and cur.get("id"):
                rows.append(cur)
            cur = {"id": m_id.group(1).strip()}
            continue
        if cur is not None:
            m_gr = re.match(r"^\s+grader:\s*(\S+)\s*$", line)
            if m_gr:
                cur["grader"] = m_gr.group(1).strip()
    if cur and cur.get("id"):
        rows.append(cur)
    return rows


def openenv_max_steps(spec_path: Path) -> int | None:
    if not spec_path.is_file():
        return None
    m = re.search(r"(?m)^max_steps:\s*(\d+)\s*$", spec_path.read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def verify_openenv_alignment(spec_path: Path = OPENENV_SPEC) -> list[str]:
    """Return human-readable warnings if inference tables drift from openenv.yaml."""
    warnings: list[str] = []
    rows = load_openenv_task_rows(spec_path)
    if not rows:
        warnings.append(f"Could not read tasks from {spec_path} — skipping alignment check.")
        return warnings

    yaml_ids = [r["id"] for r in rows]
    if tuple(yaml_ids) != TASK_SETS["all"]:
        warnings.append(
            f"openenv.yaml task order/ids {yaml_ids!r} != inference TASK_SETS['all'] {list(TASK_SETS['all'])!r}"
        )

    for row in rows:
        tid = row["id"]
        gref = row.get("grader", "")
        fn = TASK_TO_GRADER.get(tid)
        if fn is None:
            warnings.append(f"openenv.yaml task {tid!r} has no TASK_TO_GRADER entry in inference.py")
            continue
        expected = _GRADER_TO_SYMBOL.get(fn)
        if expected and gref and gref != expected:
            warnings.append(
                f"Task {tid!r}: openenv.yaml grader {gref!r} != inference mapping {expected!r}"
            )

    for tid in TASK_SETS["all"]:
        if tid not in yaml_ids:
            warnings.append(f"inference TASK_SETS includes {tid!r} but openenv.yaml has no such task id")

    return warnings


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


def emit_start(task_name: str, max_steps_hint: int | None) -> None:
    ms = f" max_steps={max_steps_hint}" if max_steps_hint is not None else ""
    print(
        f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME} env_url={ENV_URL}{ms}",
        flush=True,
    )


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


def run_one_task(llm: Any, task_name: str, *, max_steps_hint: int | None) -> None:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    emit_start(task_name, max_steps_hint)

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
    parser = argparse.ArgumentParser(
        description="Run the Ghostexec baseline agent (HTTP env + HF OpenAI-compatible router)."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task subset to run (mirrors openenv.yaml difficulties / tasks).",
    )
    parser.add_argument(
        "--env-url",
        default="",
        help="Override Ghostexec HTTP base URL (else ENV_URL env or default 127.0.0.1:8000).",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print tasks parsed from openenv.yaml and exit.",
    )
    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help="Verify inference.py TASK_TO_GRADER matches openenv.yaml; print warnings and exit 1 if drift.",
    )
    args = parser.parse_args()

    global ENV_URL
    if args.env_url.strip():
        ENV_URL = args.env_url.strip().rstrip("/")

    if args.list_tasks:
        for row in load_openenv_task_rows(OPENENV_SPEC):
            print(row.get("id", ""), "->", row.get("grader", "?"))
        return

    drift = verify_openenv_alignment(OPENENV_SPEC)
    for w in drift:
        print(f"[openenv] {w}", flush=True)

    if args.check_alignment:
        hard = [x for x in drift if not x.startswith("Could not read")]
        if hard:
            for x in hard:
                print(f"[ALIGNMENT ERROR] {x}", flush=True)
            raise SystemExit(1)
        return

    max_steps_hint = openenv_max_steps(OPENENV_SPEC)
    llm = client()
    for task_name in choose_tasks(args.difficulty):
        run_one_task(llm, task_name, max_steps_hint=max_steps_hint)


if __name__ == "__main__":
    main()
