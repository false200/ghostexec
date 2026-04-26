---
title: Ghostexec Environment Server
emoji: 📢
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Ghostexec

**Ghostexec** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment that simulates a busy executive’s world: inbox, calendar, contacts, tasks, and stakeholder moods. The agent chooses **structured actions** (reply, reschedule, delegate, …); the server returns a **plain-text briefing** as the main observation and a **scalar reward** shaped around conflict, relationships, and task progress. Scenario data lives in `scenarios/*.json` — nothing is hardcoded in Python for world content.

**Manifest:** `openenv.yaml` (name **`ghostexec`**, HF Space identifier).  
**Package:** `openenv-ghostexec` in `pyproject.toml` (import as `ghostexec`).

---

## Deliverables

| Deliverable | URL |
|-------------|-----|
| Public HF Space (required) | [https://huggingface.co/spaces/modelbuilderhq/ghostexec](https://huggingface.co/spaces/modelbuilderhq/ghostexec) |
| Write-up / blog (HF post preferred) | `TODO: https://huggingface.co/blog/...` |
| Short demo video (&lt;2 min) | `TODO: https://youtube.com/...` |

Fill these URLs before submission freeze so reviewers can verify everything from one place.

---

## OpenEnv Hackathon alignment (themes + submission checklist)

**Theme fit (examples, not exhaustive):** Ghostexec targets **Theme 3.2 — Personalized tasks** (executive-style inbox, calendar, conflicts, delegation via structured actions). **Theme 4** is partially supported via curriculum + perturb (`GHOSTEXEC_CURRICULUM`, `GHOSTEXEC_PERTURB`) and diverse scenarios under `scenarios/`.

**Minimum submission checklist (fill before freeze):**

| Item | Status |
|------|--------|
| OpenEnv-based env + `openenv.yaml` | Done in-repo (`openenv-core[core]>=0.2.3` in `pyproject.toml`; aligns with current PyPI release line). |
| Short write-up or &lt;2 min video | **You:** publish and paste links in [Deliverables](#deliverables). |
| Public HF Space URL | [Space in Deliverables](#deliverables); deploy with `openenv push --repo-id <your>/ghostexec`. |

---

## Why Ghostexec is a strong environment submission

This section is about **what makes the world worth grading**: how hard the situation is for an agent, and whether success reflects real capabilities (parsing, planning, tool use) rather than a single canned answer. A **standalone** brief for reviewers (TOC, Space URL, paths from repo root) lives at **[environment-innovation/README.md](environment-innovation/README.md)**.

### How this README is organized

| Focus | Where we cover it |
|-------|-------------------|
| What is **hard or distinctive** about the env itself | **Short answers** + **§§1–4** (what Ghostexec is, observations, actions, dynamics / drift). |
| What **behaviors** are actually exercised, and why this isn’t a toy prompt loop | **Short answers** + **§§3–6** (legality, scoring, tasks) and **§8** (honest claim: the *composition* is what’s uncommon). |

### Short answers (so nothing is buried)

**Is it genuinely challenging?** Yes. The agent must survive **dense natural-language state**, emit **strict structured actions** that **mutate** a multi-entity world, and accept **time pressure**, **social consequences**, and **invalid-action economics** without crashing the server. “Easy” wins are rare because channels **compete**: mail, calendar, tasks, and relationships all pull in different directions.

**Is it a meaningful test of behavior?** Yes. Success requires **grounded parsing** (real ids from the briefing), **tool discipline** (legal JSON schema), **sequencing** over multiple steps (WebSocket sessions for real episodes; HTTP for resets and single steps), and **tradeoffs** reflected in a **multi-channel** reward—not a single template answer.

**Is every ingredient globally novel?** No—and we do not claim otherwise. Inboxes and calendars are familiar. What *is* uncommon is the **composition**: OpenEnv-first packaging, **plain-text-only** observations, **data-driven** scenarios, **live dynamics** and **timed drift**, **dual** evaluation (**dense step rewards** + **trajectory graders** in strict `(0.01, 0.99)`), and a **production-shaped** action API—together—in one environment you can train and ship.

---

### 1. What Ghostexec is (one paragraph)

Ghostexec is an **executive chief-of-staff simulator**. Each episode starts from JSON scenario data under `scenarios/`, selected by **task id** in `openenv.yaml`. The **engine** lives in `server/ghostexec_environment.py` and `server/reward.py`; the **contract** for Hugging Face / OpenEnv is `openenv.yaml` (name **`ghostexec`**, FastAPI `server.app:app`, port **8000**). The model never sees raw scenario JSON as its primary observation: it sees a **rendered briefing**—the same class of messy, overlapping information a human would scan under time pressure.

---

### 2. What the agent observes (and why that matters)

After `reset` (or the WebSocket equivalent), the policy receives `GhostexecObservation.echoed_message`: a **single plain-text** block that includes, at minimum:

- A **timestamped header** (simulated “now”).
- **Unread emails** with priority, sender, relationship, subject, and a short preview.
- **Calendar conflicts** in a rolling horizon (overlaps the agent could resolve or worsen).
- **Top contacts** with **mood**, relationship type, and communication preference.
- **Tasks** that are overdue or due soon.
- **Executive stress** and **steps remaining** toward `max_steps` (see `openenv.yaml`, default **20**).

**Why this matters for “challenging”:** many demos hide structure in JSON observations or tool schemas. Here, the **only** narrative state the model is supposed to “read” like a user is **natural language**, while the **law** of the world is still **typed actions**. That forces **comprehension + compliance** together—hallucinated ids and “vibes-only” plans fail in ways you can measure.

---

### 3. What the agent can do (actions and legality)

Each step the agent returns **exactly one** `GhostexecAction` (`models.py`): `reply_email`, `archive_email`, `reschedule_meeting`, `cancel_meeting`, `complete_task`, `delegate_task`, `send_message`, or `do_nothing`.

**Validity is enforced against the live world:** wrong `email_id` / `meeting_id` / `task_id`, missing required fields, or impossible combinations produce an **invalid step**. The server **does not throw**; it returns structured metadata (`step_ok`, error text) so RL and HTTP clients can learn from mistakes instead of dying.

**Valid actions mutate state:** mail can be replied or archived; meetings moved or cancelled; tasks completed or delegated; direct messages sent. The episode is therefore a **small transactional simulation**, not a static Q&A.

---

### 4. What changes between steps (dynamics and drift)

Ghostexec is **not** a static paragraph with a hidden answer key. After actions, the environment runs **post-step dynamics** (see `server/ghostexec_environment.py`):

- **Clock:** simulation time advances (default **20 minutes** per step), which can flip tasks into overdue and change what “urgent” means.
- **Mood:** stakeholders move along a mood ladder after real actions (e.g. a thoughtful reply can improve a sender; cancelling a meeting can upset attendees).
- **Pressure on idle / invalid behavior:** if the agent **`do_nothing`**s or **fails** while **critical** mail is still unanswered, mood pressure can concentrate on the sender who is actually waiting—so “safe” inaction is not safe in the social graph.
- **Stress and conflicts:** the world rebuilds an **active conflict list** (overlaps, unanswered critical mail) and maps that into the **stress** value surfaced in the briefing—so calendar debt is not cosmetic.

**Scenario-driven schema drift:** harder JSON can schedule **`after_step`** events that reshuffle the world mid-episode: shift meetings, move deadlines, change communication preferences, **suppress relationship credit** for certain reply paths, or force moods. That tests **adaptation**, not memorization of the first screen.

---

### 5. How success is scored (two layers, on purpose)

**A. Dense step reward (training and fine-grained analysis)** — `server/reward.py`  
A **fixed** weighted core (**0.35 conflict + 0.35 relationship + 0.30 task**) plus **bounded** shaping terms (synergy, tradeoffs, progress-style shaping, scaffold, quality separation). Invalid steps and **`do_nothing`** are handled explicitly (idle is **penalised**, not neutral). Rich `RewardBreakdown` fields can be logged to `outputs/logs/episode_rewards.jsonl` (gitignored) for auditing *why* a step moved.

**B. Trajectory graders (OpenEnv / hackathon validation)** — `graders.py`  
Each public task in `openenv.yaml` binds a grader (`graders.phase2_core_grader`, etc.). Graders read **trajectory-shaped** payloads (e.g. lists of rewards) and return scores **strictly inside `(0.01, 0.99)`**—the validator-facing layer—while the step engine remains the **dense teaching signal**.

That split is deliberate: **agents learn from fine structure**, **judges certify** with stable bounded scores.

---

### 6. The public task ladder (difficulty in *data*, not vibes)

| Task id | Difficulty | Scenario file | What gets harder |
|---------|------------|----------------|------------------|
| `phase2_core` | easy | `scenarios/phase2_core.json` | Dense default triage: VIP mail, calendar relief, overlapping obligations. |
| `monday_morning` | medium | `scenarios/monday_morning.json` | Stacked Monday rush: more concurrent fires, less slack. |
| `dinner_disaster` | hard | `scenarios/dinner_disaster.json` | Personal vs professional collision with **escalation risk**. |

All of this is declared in **`openenv.yaml`** so the Space, CLI, and notebooks agree on **names**, **ports**, and **grader wiring** without a second source of truth.

---

### 7. How a reviewer can verify (5-minute checklist)

1. Open **`openenv.yaml`** — confirm three tasks, `max_steps`, `app: server.app:app`, **`name: ghostexec`**.  
2. Open **`scenarios/*.json`** — confirm episodes are **data**, not hardcoded Python lore.  
3. Skim **`server/ghostexec_environment.py`** — `build_briefing_text`, `_apply_action`, `_apply_post_action_dynamics`, `_maybe_apply_schema_drift_events`.  
4. Skim **`server/reward.py`** — fixed weights, invalid / idle handling, shaping caps.  
5. Open **`graders.py`** — strict output bounds and trajectory consumption.  
6. Hit the **public Space** (see [Deliverables](#deliverables)) — `/docs` or `/reset` + `/step` smoke: legal actions change state; illegal actions return errors, not stack traces.

---

### 8. Closing

**World quality.** The challenge is **interactional and operational**: overlapping human-style goals, strict tool use, evolving social signals, and mid-episode drift—**not** a single binary “did you answer correctly.”

**What this stack proves.** If you strip Ghostexec to one bullet, it is: **plain-text situational awareness + legal structured world edits + multi-channel rewards + timed scenario pressure + OpenEnv-native deployment and graders**—in one coherent package you can train, log, and host.

That is the **innovation case** this repository is built to defend.

### Other hackathon criteria (at a glance)

- **Storytelling & Presentation (30%)** — scenarios encode narrative arcs (VIP escalations, family/professional collisions, deadline cascades) so decisions read like real staff work, not abstract moves.  
- **Showing Improvement in Rewards (20%)** — rewards are deterministic, inspectable, and traceable via metadata + `outputs/logs/` (gitignored).  
- **Reward Quality (10%)** — fixed core weights (0.35 / 0.35 / 0.30), bounded shaping, explicit invalid handling, **`do_nothing` penalised**.

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

**Docker image** (optional): if your OpenEnv client supports it, you can point `GhostexecEnv` at a container built from the root `Dockerfile`. Build from repo root:

```bash
docker build -t ghostexec-env:latest .
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

**Reward-engineering provenance.** The design follows the reward-shaping playbook surveyed in *Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications* ([arXiv:2408.10215](https://arxiv.org/abs/2408.10215)): dense per-step shaping around proxy signals (conflict / relationship / task) instead of a single sparse end-of-episode reward, fixed weights to keep channel trade-offs inspectable, and bounded per-step magnitudes to resist hacking.

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

Opt-in Docker build smoke (Phase 1 gate):

```bash
GHOSTEXEC_RUN_DOCKER_BUILD=1 uv run pytest tests/test_docker_build.py -q
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

Post-training plot pack (loss + reward + components + baseline bar):

```bash
uv run python scripts/plot_training_report.py \
  --trainer-history outputs/trainer_state.json \
  --reward-csv outputs/reward_log.csv \
  --baselines-json outputs/compliance_manifest.json \
  --out-dir outputs/plots
```

The script writes:
- `outputs/plots/loss_curve.png`
- `outputs/plots/reward_curve.png`
- `outputs/plots/components_curve.png`
- `outputs/plots/baseline_comparison.png`

SFT before GRPO (with partial live-env usage during SFT data generation and GRPO rewards):

```bash
uv run python scripts/train_sft_then_grpo.py \
  --model-preset small_iter_fast \
  --training-preset hackathon_turbo \
  --env-url http://127.0.0.1:8000 \
  --generate-sft-from-env \
  --sft-samples 120 \
  --max-sft-steps 60 \
  --max-grpo-steps 120 \
  --env-reward-scale 1.0 \
  --local-reward-scale 0.35 \
  --complexity-curriculum easy_to_full \
  --curriculum-ramp-ratio 0.60
```

This performs:
- SFT warm-start on JSONL (`prompt` + `completion`) generated from live `/reset` briefings.
- GRPO continuation from the SFT adapter.
- Mixed reward shaping where env-derived reward remains active and local shaping can be down-weighted/up-weighted via scales.
- Optional complexity curriculum (`easy_to_full`) that starts with stronger scaffold/local signals and anneals to env-dominant reward later.
- Stability-first optimization defaults (cosine schedule + warmup + grad clipping + higher GRPO KL beta). Optional `--reward-ema-decay 0..1` smooths the *env* reward channel (defaults come from `--training-preset`). Training always runs the full `max_*_steps` (no early-stop callbacks).

Recommended model strategy for hackathon iteration speed:
- Start with `--model-preset small_iter_fast` (`unsloth/Qwen2.5-3B-Instruct`) + QLoRA.
- Run many short SFT->GRPO loops, improve reward signals, then scale model size only after curves stabilize.
- Use larger presets only when memory + runtime are consistently stable.
- Use `--training-preset hackathon_turbo` to apply stable aggressive defaults for iterative win-rate.
- Script prints SFT/GRPO LoRA delta checks; if deltas are near zero it stops, so you never mistake a no-op run for real finetuning.

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
├── pyproject.toml         # Package metadata + optional extras
├── uv.lock
├── models.py              # World + GhostexecAction / GhostexecObservation
├── client.py              # GhostexecEnv (WebSocket client)
├── scenarios/             # World JSON (source of truth for episodes)
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

**Tutorials.** General OpenEnv environment patterns are documented in the official tutorial pages and examples.

- All tutorials: [OpenEnv/tutorial](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)
- Environment examples: [OpenEnv/envs](https://github.com/meta-pytorch/OpenEnv/tree/main/envs)

**YouTube — Building RL environments.** Talks from Meta / OpenEnv contributors that informed the scenario-driven reset, WebSocket session model, and reward breakdown used here:

- [Building RL Environments with OpenEnv](https://www.youtube.com/watch?v=0airz7BhBiA)
- [OpenEnv Deep Dive](https://www.youtube.com/watch?v=ap4q4sAK4OY)
- [Agentic RL Environments](https://www.youtube.com/watch?v=Jew4lhAiqnw)
- [OpenEnv Livestream (4-hour walkthrough)](https://www.youtube.com/live/kkCNMz0Ptd8)

**Reward-engineering papers.** See [Reward](#reward) for how each paper maps to specific components of `server/reward.py`.

- Jnadi, A. (2024). *Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications*. [arXiv:2408.10215](https://arxiv.org/abs/2408.10215). Informs the dense per-step conflict / relationship / task shaping and the bounded-magnitude design.

---

## License

BSD-style — see the license notice at the top of each source file (Meta / OpenEnv lineage).
