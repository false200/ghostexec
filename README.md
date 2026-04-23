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

## OpenEnv Hackathon alignment (themes + submission checklist)

**Theme fit (examples, not exhaustive):** Ghostexec targets **Theme 3.2 — Personalized tasks** (executive-style inbox, calendar, conflicts, delegation via structured actions) and supports **Theme 2 — Long-horizon planning** through multi-step scoring (`GHOSTEXEC_REWARD_KSTEPS`, `GHOSTEXEC_MULTITURN=1`, `training/multiturn_reward.py`, `training/openenv_grpo_rollout.py`). **Theme 4** is partially supported via curriculum + perturb (`GHOSTEXEC_CURRICULUM`, `GHOSTEXEC_PERTURB`) and diverse scenarios under `scenarios/`.

**Minimum submission checklist (fill before freeze):**

| Item | Status |
|------|--------|
| OpenEnv-based env + `openenv.yaml` | Done in-repo (`openenv-core[core]>=0.2.3` in `pyproject.toml`; aligns with current PyPI release line). |
| Training notebook (Unsloth + TRL GRPO) | `training/ghostexec_unsloth_grpo_colab.ipynb` — installs pinned `transformers` / `trl`, Hub caps, `bitsandbytes`, `lm-format-enforcer` (does **not** pip-install `torch` / `xformers`; bring your own CUDA stack); GRPO calls into `GhostexecEnvironment`; before/after eval + plots. |
| Evidence of a real run (loss/reward plots) | **You:** run notebook → copy key PNGs into `docs/submission_results/` (see that folder) and **embed or link** them from this README. Do not rely only on gitignored `outputs/`. |
| Short write-up or &lt;2 min video | **You:** publish HF post or YouTube, then add the **URL here** (placeholder: replace `YOUR_HF_BLOG_URL`, `YOUR_YOUTUBE_URL`). |
| Public HF Space URL | **You:** `openenv push` → add **one line** near the top of this README: `**Live Space:** https://huggingface.co/spaces/<org>/<name>` |

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

**Reward-engineering provenance.** The design follows the reward-shaping playbook surveyed in *Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications* ([arXiv:2408.10215](https://arxiv.org/abs/2408.10215)): dense per-step shaping around proxy signals (conflict / relationship / task) instead of a single sparse end-of-episode reward, fixed weights to keep channel trade-offs inspectable, and bounded per-step magnitudes to resist hacking. The training-side add-ons in `training/grpo_ghostexec_reward.py` (format-valid, id-relevance, group-diversity, VIP-critical reply bonus, tanh-squash + jitter) implement the proxy-reward / anti-hacking patterns catalogued in *Reward Engineering for Reinforcement Learning in Software Tasks* ([arXiv:2601.19100](https://arxiv.org/abs/2601.19100)) — particularly structure-valid checks, semantic-relevance checks, and bounded + tie-broken aggregates. Both papers are cited in full in [Resources & references](#resources--references).

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

Full OpenEnv CLI flow from this directory (matches steps 5–8 of the [Packaging & Deploying guide](https://meta-pytorch.org/OpenEnv/auto_getting_started/environment-builder.html)):

```bash
openenv serve                       # local dev server on :8000
openenv build                       # build the Docker image
openenv validate --verbose          # structure + Dockerfile + entrypoint checks
openenv push                        # deploy to HF Spaces
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
| `GHOSTEXEC_MULTITURN=1` | Swap the core reward channel for a k-turn **process-aware** rollout (see "Process supervision visibility"). |
| `GHOSTEXEC_MULTITURN_TURNS`, `GHOSTEXEC_MULTITURN_GAMMA` | Turns + discount for the multi-turn reward (defaults 3, 0.9). |
| `GHOSTEXEC_SAVE_MERGED=1` | After training, also write a merged 16-bit checkpoint via Unsloth's `save_pretrained_merged(..., save_method="merged_16bit")` alongside the default adapter-only save. |

**Rejection-sampled SFT** (`training/rejection_sft.py`): generate `(briefing, smart_action JSON)` pairs across scenarios, filter to the top quantile by env reward, feed that into `SFTTrainer`. Faster convergence than raw demos.

**Held-out evaluation** (`training/eval_harness.py`): run N episodes on `EVAL_SCENARIOS` (`vip_meltdown.json`) after training, report mean return, format-valid rate, VIP-critical-reply rate, conflicts-resolved rate, and per-channel averages (conflict / relationship / task). Reward-hackers show up as “task up, relationship down”.

**Constrained decoding** (`training/constrained_decode.py`): `patch_model_for_json_generation(model, tokenizer)` monkey-patches `model.generate` so every call (including the ones TRL's `GRPOTrainer` samples) is constrained to the `GhostexecAction` JSON schema. Backends are optional: install `lm-format-enforcer` (preferred — works with the Unsloth tokenizer as-is via `prefix_allowed_tokens_fn`) or `outlines` (HF `LogitsProcessorList`). In the Colab notebook set `GHOSTEXEC_CONSTRAIN_JSON=1` before the constrained-decode cell. Every sampled completion then already parses — the reward reflects policy quality, not syntax luck.

**Multi-turn reward** (`training/multiturn_reward.py`): `make_multiturn_reward(model, tokenizer, num_turns=3, gamma=0.9)` returns a TRL-compatible reward function that rolls out an entire episode for each GRPO sample: the completion is turn 1, then the model itself generates k-1 follow-up JSON actions, each stepping the env. Reward is the discounted sum. Credits "first actions that enable good follow-ups" instead of one-shot tricks. Toggle via `GHOSTEXEC_MULTITURN=1` (with `GHOSTEXEC_MULTITURN_TURNS`, `GHOSTEXEC_MULTITURN_GAMMA`). For the full OpenEnv `rollout_func` path, `training/openenv_grpo_rollout.py::rollout_multiturn_ghostexec` runs the same pattern through `generate_rollout_completions` when `GHOSTEXEC_ROLLOUT_TURNS>1`.

**Process supervision visibility.** When `GHOSTEXEC_MULTITURN=1`, each GRPO sample is scored by a k-turn rollout instead of a one-shot env step, and the GRPO config cell prints `PROCESS SUPERVISION: ON` plus the turn count and gamma. The plot cell then prints a dedicated `process_reward_mean: ...` line (also saved to `outputs/logs/process_reward_summary.json`) and the `ghostexec_multiturn_reward` channel appears in `outputs/plots/grpo_reward_channels.png` — so a reviewer can tell at a glance whether the run used process-aware rewards or final-only rewards.

**Observable evidence of training progress.** The Colab notebook produces a self-contained artifact bundle under `outputs/` so reviewers can see that GRPO actually moved the policy:

| Artifact | What it shows |
|----------|---------------|
| `outputs/plots/grpo_reward_curve.png` | Mean reward over GRPO log steps. |
| `outputs/plots/grpo_reward_channels.png` | Per-channel reward (env step / format / diversity / id-relevance / VIP-critical / multi-turn) — detects hacking, surfaces process rewards. |
| `outputs/plots/grpo_diagnostics.png` | 2x2 diagnostics: mean reward + EMA, within-group reward std, mean completion length, per-channel curves. |
| `outputs/plots/before_after_eval.png` | **Judge-facing BEFORE vs AFTER bar chart**: `mean_return`, `format_valid`, `vip_critical_reply`, `conflicts_resolved` on the same held-out scenarios. |
| `outputs/plots/before_after_channels.png` | BEFORE vs AFTER per-channel reward means (conflict / relationship / task). |
| `outputs/eval/llm_before.json`, `outputs/eval/llm_after.json` | Same-schema held-out eval on `vip_meltdown.json` before and after training. |
| `outputs/eval/scripted_baseline.json` | Reference baseline from the hand-written `smart_action` policy. |
| `outputs/eval/before_sample.txt`, `outputs/eval/after_sample.txt` | Model's action on a fixed `phase2_core.json` briefing, before vs after. |
| `outputs/eval/before_after.csv` | 7-row delta table: `mean_return`, `format_valid_rate`, `vip_critical_first_reply_rate`, `conflicts_resolved_rate`, and per-channel means. |
| `outputs/logs/process_reward_summary.json` | `process_reward_mean` / last / step count for the multi-turn reward channel (when `GHOSTEXEC_MULTITURN=1`). |
| `outputs/training/grpo_adapter/` | LoRA adapter + tokenizer saved via `model.save_pretrained(...)` (correct path for 4-bit Unsloth base). |
| `outputs/training/grpo_merged_16bit/` | Optional merged 16-bit checkpoint from `model.save_pretrained_merged(..., save_method="merged_16bit")` (set `GHOSTEXEC_SAVE_MERGED=1`). |
| `outputs/training/save_report.json` | Adapter / merged / smoke-inference status for the saved policy. |

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

## Resources & references

Ghostexec is built against the official Meta PyTorch OpenEnv stack. Every design choice below is traceable to one of these sources.

**OpenEnv core.** The Gymnasium-style `reset()` / `step()` / `state` interface in `server/ghostexec_environment.py`, the `EnvClient` subclass in `client.py`, and the `create_app(...)` wiring in `server/app.py` follow the [Packaging & Deploying guide](https://meta-pytorch.org/OpenEnv/auto_getting_started/environment-builder.html) exactly.

- Core repo: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- Docs: [meta-pytorch.org/OpenEnv](https://meta-pytorch.org/OpenEnv/)

**OpenEnv Hub (Hugging Face).** Target deployment for `openenv push`. The Space metadata at the top of this README + `openenv.yaml` are the knobs HF Spaces reads.

- Environments: [huggingface.co/openenv](https://huggingface.co/openenv)
- Spaces: [huggingface.co/openenv/spaces](https://huggingface.co/openenv/spaces)

**Tutorials.** The GRPO training path in `training/openenv_grpo_rollout.py` (`rollout_func`, `generate_rollout_completions`, split `reward_funcs`) is the pattern from the [OpenEnv 04-training — Wordle GRPO tutorial](https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md). The simpler scalar-reward path in `training/ghostexec_unsloth_grpo_colab.ipynb` mirrors the [OpenEnv 2048 GRPO Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb).

- All tutorials: [OpenEnv/tutorial](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)
- Training examples: [OpenEnv/tutorial/examples](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples)
- Environment examples: [OpenEnv/envs](https://github.com/meta-pytorch/OpenEnv/tree/main/envs)

**YouTube — Building RL environments.** Talks from Meta / OpenEnv contributors that informed the scenario-driven reset, WebSocket session model, and reward breakdown used here:

- [Building RL Environments with OpenEnv](https://www.youtube.com/watch?v=0airz7BhBiA)
- [OpenEnv Deep Dive](https://www.youtube.com/watch?v=ap4q4sAK4OY)
- [Agentic RL Environments](https://www.youtube.com/watch?v=Jew4lhAiqnw)
- [OpenEnv Livestream (4-hour walkthrough)](https://www.youtube.com/live/kkCNMz0Ptd8)

**Reward-engineering papers.** See [Reward](#reward) for how each paper maps to specific components of `server/reward.py` and `training/grpo_ghostexec_reward.py`.

- Jnadi, A. (2024). *Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications*. [arXiv:2408.10215](https://arxiv.org/abs/2408.10215). Informs the dense per-step conflict / relationship / task shaping and the bounded-magnitude design.
- Masud, M.R., Wasi, A.T., Rahman, S., Parvez, M.R. (2026). *Reward Engineering for Reinforcement Learning in Software Tasks*. [arXiv:2601.19100](https://arxiv.org/abs/2601.19100). Informs the training-side proxy channels (`reward_format_valid`, `reward_group_diversity`, `reward_id_relevance`, `reward_vip_critical_reply_bonus`) and the tanh-squash + tie-break anti-hacking aggregate.

---

## License

BSD-style — see the license notice at the top of each source file (Meta / OpenEnv lineage).
