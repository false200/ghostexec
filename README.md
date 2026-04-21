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
- `GHOSTEXEC_SFT_SAMPLES`, `GHOSTEXEC_SFT_MAX_STEPS`, `GHOSTEXEC_GRPO_ROWS`, `GHOSTEXEC_GRPO_MAX_STEPS`, `GHOSTEXEC_NUM_GENERATIONS`, `GHOSTEXEC_GRPO_TEMPERATURE`, `GHOSTEXEC_GRPO_MAX_COMPLETION_LENGTH`, `GHOSTEXEC_GRPO_LR`, `GHOSTEXEC_MODEL`, `GHOSTEXEC_MAX_SEQ`.

The **GRPO** notebook cell **re-reads** `GHOSTEXEC_GRPO_MAX_STEPS`, `GHOSTEXEC_GRPO_ROWS`, `GHOSTEXEC_NUM_GENERATIONS`, `GHOSTEXEC_GRPO_TEMPERATURE`, `GHOSTEXEC_GRPO_MAX_COMPLETION_LENGTH`, and `GHOSTEXEC_GRPO_LR` from `os.environ` right before building `GRPOConfig`, so setting those env vars in an earlier cell still applies even if you did not re-run the knobs cell (look for the printed `GRPO (env re-sync): ...` line).

Helpers: `training/llm_action_parse.py`, `training/grpo_ghostexec_reward.py`.

**Kaggle:** enable **Internet** in notebook settings if you use `git clone` from GitHub. If you use a **Dataset** instead (repo under `/kaggle/input/...`), run the **Notebook bootstrap** cell in `ghostexec_unsloth_grpo_colab.ipynb` right after the repository cell so `from ghostexec...` resolves (`notebook_setup.py` at repo root runs `pip install -e .` from the detected root). Otherwise you may see `ModuleNotFoundError: No module named 'ghostexec'` while `cwd` stays `/kaggle/working`.

**Reward channels + anti-hacking knobs.** The Colab GRPO cell now passes a **list** of reward functions so GRPO averages across them while the core 0.35 / 0.35 / 0.30 conflict/relationship/task blend stays intact inside the env step reward:

- `ghostexec_env_step_reward` — parse JSON → fresh env reset → one `step()` (plus optional k-step scripted lookahead).
- `reward_format_valid` — ±1 if the completion parses to a `GhostexecAction`.
- `reward_group_diversity` — penalizes duplicate completions inside a GRPO group (kills collapse to one safe JSON).
- `reward_id_relevance` — small penalty when `email_id` / `task_id` / `meeting_id` don’t exist in the scenario.
- `reward_vip_critical_reply_bonus` — bonus for replying to a VIP + critical + unreplied email.

Environment variables (set before running the GRPO cell):

| Var | Effect |
|-----|--------|
| `GHOSTEXEC_GRPO_SCENARIO` | Pin the reward scenario to one file. |
| `GHOSTEXEC_CURRICULUM` | Rotate over `easy` \| `mid` \| `hard` \| `all` (`training/scenarios_sampler.py`). |
| `GHOSTEXEC_PERTURB=1` | Shuffle list order (+ optional time shift) per call. |
| `GHOSTEXEC_REWARD_KSTEPS` | k scripted follow-up steps (`smart_action`) to score long-term consequences. |
| `GHOSTEXEC_REWARD_GAMMA` | Discount for those follow-up steps (default 0.9). |

**Rejection-sampled SFT** (`training/rejection_sft.py`): generate `(briefing, smart_action JSON)` pairs across scenarios, filter to the top quantile by env reward, feed that into `SFTTrainer`. Faster convergence than raw demos.

**Held-out evaluation** (`training/eval_harness.py`): run N episodes on `EVAL_SCENARIOS` (`vip_meltdown.json`) after training, report mean return, format-valid rate, VIP-critical-reply rate, conflicts-resolved rate, and per-channel averages (conflict / relationship / task). Reward-hackers show up as “task up, relationship down”.

**Constrained decoding** (`training/constrained_decode.py`): `patch_model_for_json_generation(model, tokenizer)` monkey-patches `model.generate` so every call (including the ones TRL's `GRPOTrainer` samples) is constrained to the `GhostexecAction` JSON schema. Backends are optional: install `lm-format-enforcer` (preferred — works with the Unsloth tokenizer as-is via `prefix_allowed_tokens_fn`) or `outlines` (HF `LogitsProcessorList`). In the Colab notebook set `GHOSTEXEC_CONSTRAIN_JSON=1` before the constrained-decode cell. Every sampled completion then already parses — the reward reflects policy quality, not syntax luck.

**Multi-turn reward** (`training/multiturn_reward.py`): `make_multiturn_reward(model, tokenizer, num_turns=3, gamma=0.9)` returns a TRL-compatible reward function that rolls out an entire episode for each GRPO sample: the completion is turn 1, then the model itself generates k-1 follow-up JSON actions, each stepping the env. Reward is the discounted sum. Credits "first actions that enable good follow-ups" instead of one-shot tricks. Toggle via `GHOSTEXEC_MULTITURN=1` (with `GHOSTEXEC_MULTITURN_TURNS`, `GHOSTEXEC_MULTITURN_GAMMA`). For the full OpenEnv `rollout_func` path, `training/openenv_grpo_rollout.py::rollout_multiturn_ghostexec` runs the same pattern through `generate_rollout_completions` when `GHOSTEXEC_ROLLOUT_TURNS>1`.

**Observable evidence of training progress.** The Colab notebook produces a self-contained artifact bundle under `outputs/` so reviewers can see that GRPO actually moved the policy:

| Artifact | What it shows |
|----------|---------------|
| `outputs/plots/grpo_reward_curve.png` | Mean reward over GRPO log steps. |
| `outputs/plots/grpo_reward_channels.png` | Per-channel reward (env step / format / diversity / id-relevance / VIP-critical) — detects hacking. |
| `outputs/eval/llm_before.json`, `outputs/eval/llm_after.json` | Same-schema held-out eval on `vip_meltdown.json` before and after training. |
| `outputs/eval/scripted_baseline.json` | Reference baseline from the hand-written `smart_action` policy. |
| `outputs/eval/before_sample.txt`, `outputs/eval/after_sample.txt` | Model's action on a fixed `phase2_core.json` briefing, before vs after. |
| `outputs/eval/before_after.csv` | 7-row delta table: `mean_return`, `format_valid_rate`, `vip_critical_first_reply_rate`, `conflicts_resolved_rate`, and per-channel means. |

Everything is produced by **Run All** on the notebook once `GHOSTEXEC_CONSTRAIN_JSON=1` (+ `pip install lm-format-enforcer`) and a sensible `GHOSTEXEC_GRPO_MAX_STEPS` are set **before the GRPO cell** (knobs cell or any earlier cell — GRPO re-reads env).

**OpenEnv + TRL (advanced, matches Meta tutorial):** [OpenEnv 04-training — Wordle GRPO](https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md) describes `rollout_func`, `generate_rollout_completions`, and split `reward_funcs` that read kwargs from the rollout. Ghostexec mirrors that in `training/openenv_grpo_rollout.py` (`ghostexec_rollout_func`, `reward_ghostexec_parse_ok`, `reward_ghostexec_env_step`) when your TRL build includes `trl.experimental.openenv` (recent `trl` + `datasets`; not the same as the Colab pin `trl==0.22.2`, which keeps the simpler scalar reward in `ghostexec_unsloth_grpo_colab.ipynb`).

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
