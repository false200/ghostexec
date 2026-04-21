---
title: Ghostexec Environment Server
emoji: 📢
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ghostexec

**Ghostexec** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment that simulates a busy executive’s world: inbox, calendar, contacts, tasks, and stakeholder moods. The agent chooses **structured actions** (reply, reschedule, delegate, …); the server returns a **plain-text briefing** as the main observation and a **scalar reward** shaped around conflict, relationships, and task progress. Scenario data lives in `scenarios/*.json` — nothing is hardcoded in Python for world content.

**Manifest:** `openenv.yaml` (name **`ghostexec`**, HF Space identifier).  
**Package:** `openenv-ghostexec` in `pyproject.toml` (import as `ghostexec`).

---

## Features

- **Legal action set** — `reply_email`, `archive_email`, `reschedule_meeting`, `cancel_meeting`, `complete_task`, `delegate_task`, `send_message`, `do_nothing` (see `models.py`).
- **Human-readable observations** — `GhostexecObservation.echoed_message` is the full briefing text for the model (not raw JSON).
- **Invalid actions** — Handled in-process: structured metadata (e.g. `step_ok`), no server crash.
- **Reward** — Weighted blend of conflict, relationship, and task signals (see [Reward](#reward)); per-step logging under `outputs/logs/` (gitignored).
- **HTTP + WebSocket** — FastAPI app in `server/app.py`; `GhostexecEnv` uses WebSockets for persistent episodes.

---

## Quick start (Python client)

From the repo root (`ghostexec/` — where `pyproject.toml` lives):

```bash
uv sync
uv run server --port 8000
```

In another terminal or notebook:

```python
from ghostexec import GhostexecAction, GhostexecEnv

with GhostexecEnv(base_url="http://127.0.0.1:8000") as env:
    out = env.reset()
    print(out.observation.echoed_message[:500], "…")  # plain-text briefing

    step = env.step(
        GhostexecAction(
            action_type="reply_email",
            email_id="e01",
            message_body=(
                "Marcus — acknowledged. Revised figures and short rationale "
                "before noon. — Exec"
            ),
        )
    )
    print("reward:", step.reward)
    print("metadata keys:", sorted((step.observation.metadata or {}).keys()))
```

**Docker image** (optional): if your OpenEnv client supports it, you can point `GhostexecEnv` at a container built from `server/Dockerfile`. Build from repo root:

```bash
docker build -t ghostexec-env:latest -f server/Dockerfile .
```

---

## Actions and fields

`GhostexecAction` (`models.py`) includes:

| `action_type`          | Typical fields used |
|------------------------|----------------------|
| `reply_email`          | `email_id`, `message_body` |
| `archive_email`      | `email_id` |
| `reschedule_meeting` | `meeting_id`, `new_time`, `reason` |
| `cancel_meeting`     | `meeting_id`, `reason` |
| `complete_task`      | `task_id` |
| `delegate_task`      | `task_id`, `contact_name` |
| `send_message`       | `contact_name`, `message` (channel text) |
| `do_nothing`         | — (intentionally weak / penalised path) |

Unknown or malformed HTTP payloads deserialize safely to `do_nothing`-style defaults where applicable so older clients do not crash.

---

## Observation

`GhostexecObservation`:

- **`echoed_message`** — Full briefing (emails, conflicts, contacts, tasks, stress, steps remaining).
- **`message_length`** — Length of `echoed_message` for quick checks.
- **`reward`**, **`done`**, **`metadata`** — Step outcome; metadata carries flags such as `step_ok`, reward breakdown fields, and ids for debugging.

---

## Reward

Phase-4 scoring (`server/reward.py`) combines three channels with **fixed weights**:

\[
\text{weighted base} = 0.35 \cdot \text{conflict} + 0.35 \cdot \text{relationship} + 0.30 \cdot \text{task}
\]

Then applies output scaling, invalid-step adjustments, bonuses/penalties, and a floor for `do_nothing`. Full component values are available on `RewardBreakdown` and are mirrored into observation metadata where configured. **Episode reward traces** append to `outputs/logs/episode_rewards.jsonl` (directory gitignored).

---

## HTTP vs WebSocket (episode state)

- **HTTP** `POST /reset` and `POST /step` often bind to **short-lived** environment instances depending on deployment; consecutive HTTP calls may not share one in-memory episode.
- **Ghostexec** still applies your action against a scenario-primed instance so a lone `POST /step` can return a meaningful reward and metadata.
- **WebSocket `/ws`** — Use this (or `GhostexecEnv(base_url=...)`, which speaks WebSocket) for **multi-step episodes** on the same session.

Endpoints (typical OpenEnv layout): **`/web`**, **`/docs`**, **`/health`**, **`/ws`**.

---

## Running and testing locally

```bash
# Dev server (package layout)
uv run uvicorn ghostexec.server.app:app --reload --host 0.0.0.0 --port 8000

# Or console entrypoint (matches Dockerfile)
uv run server --port 8000
```

**Smoke script** (HTTP):

```bash
uv run python scripts/http_endpoint_smoke.py --local
uv run python scripts/http_endpoint_smoke.py --url http://127.0.0.1:8000
uv run python scripts/http_endpoint_smoke.py --print-curl
```

**Tests:**

```bash
uv run pytest tests/ -q
```

With the server already on port 8000:

```bash
uv run pytest tests/test_live_server_exhaustive.py -v --tb=short
```

Override live URL (Windows PowerShell example):

```powershell
$env:GHOSTEXEC_LIVE_BASE_URL = "http://127.0.0.1:9000"
uv run pytest tests/test_live_server_exhaustive.py -q
```

Optional real WebSocket client check:

```bash
# Terminal 1
uv run server --port 8000
# Terminal 2
set GHOSTEXEC_WS_BASE_URL=http://127.0.0.1:8000
uv run pytest tests/test_complete_integration.py::test_ghostexec_env_client_against_live_url_if_set -q
```

---

## Hugging Face Spaces

Deploy with the OpenEnv CLI from this directory:

```bash
openenv push
# openenv push --repo-id your-username/ghostexec
```

Use a **public** Space for the default hackathon flow unless you intentionally need a private Space. Authenticate with Hugging Face first (`huggingface-cli login` or equivalent).

---

## Training

**Local scripted RL** (writes under `outputs/` — gitignored):

```bash
uv run python training/train.py --backend local --agent reinforce --episodes 30 --max-steps 14
uv run python training/train.py --backend local --agent smart --episodes 10 --max-steps 14
```

Optional LM stack: `uv sync --extra training`.

**Colab — reward demo:** `training/ghostexec_colab.ipynb` (set the notebook working directory to this repo so `pyproject.toml` is visible), then **Run All**.

**Colab — Unsloth + TRL + GRPO:** `training/ghostexec_unsloth_grpo_colab.ipynb`  
Uses the same **`uv` + Unsloth-from-git + pinned `transformers==4.56.2` / `trl==0.22.2`** pattern as Unsloth’s [OpenEnv 2048 GRPO notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb) so `GRPOTrainer` imports reliably on Colab (avoids newer TRL + `mergekit` / Pydantic issues). Use a **GPU** runtime (e.g. T4+). Tune knobs via environment variables in the knobs cell, for example:

- `GHOSTEXEC_REPO_URL` — Public git URL to `git clone` when the repo is not already on the VM (e.g. `https://github.com/false200/ghostexec.git`).
- `GHOSTEXEC_RUN_SFT` — Set to `0` to skip optional SFT.
- `GHOSTEXEC_SFT_SAMPLES`, `GHOSTEXEC_SFT_MAX_STEPS`, `GHOSTEXEC_GRPO_ROWS`, `GHOSTEXEC_GRPO_MAX_STEPS`, `GHOSTEXEC_NUM_GENERATIONS`, `GHOSTEXEC_MODEL`, `GHOSTEXEC_MAX_SEQ`.

Helpers: `training/llm_action_parse.py`, `training/grpo_ghostexec_reward.py`.  
HF-facing write-up draft: `training/HF_BLOG_DRAFT.md`.

---

## Scenarios

| File | Role |
|------|------|
| `scenarios/phase2_core.json` | Default dense inbox/calendar/tasks fixture |
| `scenarios/monday_morning.json`, `dinner_disaster.json`, `vip_meltdown.json` | Narrative demos |
| `scenarios/vip_meltdown_drift.json` | Mood / escalation drift |
| `scenarios/schema_drift_test.json` | Drift-event harness |

---

## Concurrent WebSocket sessions

`server/app.py` passes **`GhostexecEnvironment`** (the class) into `create_app` with `max_concurrent_envs=1` by default. Increase `max_concurrent_envs` if you need multiple simultaneous WebSocket clients.

---

## Project layout

```
ghostexec/
├── openenv.yaml           # OpenEnv name, version, description
├── pyproject.toml         # Package metadata + optional [training]
├── uv.lock
├── models.py              # World + GhostexecAction / GhostexecObservation
├── client.py              # GhostexecEnv (WebSocket client)
├── scenarios/             # World JSON (source of truth for episodes)
├── training/              # train.py, Colab notebooks, GRPO reward
├── scripts/               # http_endpoint_smoke.py
├── tests/
└── server/
    ├── app.py             # FastAPI + create_app
    ├── ghostexec_environment.py
    ├── reward.py
    └── Dockerfile
```

---

## License

BSD-style — see the license notice at the top of each source file (Meta / OpenEnv lineage).
