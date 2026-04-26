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

**Ghostexec** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment: a busy **executive chief-of-staff** simulator with inbox, calendar, contacts, tasks, and stakeholder moods. The agent must read a **plain-text briefing**, then emit **one structured action per step** (`reply_email`, `reschedule_meeting`, …). The server returns rewards shaped around **conflict**, **relationships**, and **tasks**—plus trajectory **graders** for hackathon validation. All episode **content** lives in `scenarios/*.json`; the engine is in `server/ghostexec_environment.py` and `server/reward.py`.

| Item | Value |
|------|--------|
| **HF Space name / manifest** | `ghostexec` in [`openenv.yaml`](openenv.yaml) |
| **Python package** | `openenv-ghostexec` in [`pyproject.toml`](pyproject.toml) (import `ghostexec`) |
| **Public Space** | [modelbuilderhq/ghostexec](https://huggingface.co/spaces/modelbuilderhq/ghostexec) |
| **Deeper innovation-only brief** | [`environment-innovation/README.md`](environment-innovation/README.md) |

---

## Deliverables (fill before freeze)

| Deliverable | URL |
|-------------|-----|
| Public HF Space (required) | [https://huggingface.co/spaces/modelbuilderhq/ghostexec](https://huggingface.co/spaces/modelbuilderhq/ghostexec) |
| Write-up / blog (HF post preferred) | `TODO: paste your post URL` |
| Short demo video (&lt;2 min) | `TODO: paste your video URL` |

---

## Contents

**Judging criteria (this README is organized around them)**

1. [Criterion: Environment Innovation (40%)](#ghostexec-env-innovation)
2. [Criterion: Storytelling & Presentation (30%)](#ghostexec-storytelling)
3. [Criterion: Showing Improvement in Rewards (20%)](#ghostexec-reward-improvement)
4. [Criterion: Reward & Training Pipeline (10%)](#ghostexec-reward-pipeline)

**Reference**

5. [Hackathon themes & checklist](#openenv-hackathon-themes--checklist)
6. [Quick start](#quick-start-python-client)
7. [Actions](#actions-and-fields)
8. [Observation](#observation)
9. [Reward (formula summary)](#reward-formula-summary)
10. [HTTP vs WebSocket](#http-vs-websocket-episode-state)
11. [Running and testing locally](#running-and-testing-locally)
12. [Hugging Face Spaces](#hugging-face-spaces)
13. [Scenarios](#scenarios)
14. [Project layout](#project-layout)
15. [Resources & references](#resources--references)
16. [License](#license)

---

## Criterion: Environment Innovation (40%)

<a id="ghostexec-env-innovation"></a>

**Weight:** 40%

**What it means:**

- Is the environment novel, creative, or genuinely challenging?
- Does it meaningfully test agent behavior in a way that hasn't been done before?

### How Ghostexec answers this

**Challenging world.** The policy sees **one dense natural-language briefing** per step (emails, calendar overlaps, contacts with mood, overdue tasks, stress, steps remaining)—not a JSON dump of the world. It must **ground** decisions in real ids from that text, return **valid typed actions**, and accept **time pressure** and **social fallout** when meetings move or mail goes unanswered. Invalid actions **do not crash** the server; they return structured errors so learning signals stay intact.

**Meaningful behavior, not a toy Q&A.** Success needs **comprehension + tool discipline**: legal JSON schema, multi-step **sequences** (WebSocket sessions for real episodes), and **tradeoffs** across channels (mail vs calendar vs tasks vs relationships). **`do_nothing` is penalised** so “safe” idleness is costly when fires are burning.

**Dynamics, not a static paragraph.** After each valid action, the simulation **advances the clock**, updates **moods**, rebuilds **conflicts**, and can apply **scenario-driven drift** (`after_step` events in JSON): shifted meetings, new deadlines, preference changes—so the agent is tested on **adaptation**, not memorizing the first screen.

**Dual evaluation.** **Dense step rewards** in `server/reward.py` teach fine structure; **trajectory graders** in `graders.py` return scores strictly in **`(0.01, 0.99)`** per OpenEnv task wiring in `openenv.yaml`. Agents learn from the dense signal; judges get bounded certification scores.

**Honest novelty claim.** Inboxes and calendars are familiar **ingredients**. What is less common is the **composition**: OpenEnv-native packaging, **plain-text-only** observations, **data-defined** scenarios, live dynamics + drift, dual reward/grader stack, and a **transactional** action API in one trainable, hostable environment.

### Task ladder (difficulty in data)

| Task id | Difficulty | Scenario | What gets harder |
|---------|------------|----------|------------------|
| `phase2_core` | easy | `scenarios/phase2_core.json` | Dense triage: VIP mail, calendar relief, overlapping work. |
| `monday_morning` | medium | `scenarios/monday_morning.json` | Stacked Monday rush, less slack. |
| `dinner_disaster` | hard | `scenarios/dinner_disaster.json` | Personal vs professional collision, escalation risk. |

### 5-minute verification checklist

1. **`openenv.yaml`** — three tasks, `max_steps`, `app: server.app:app`, `name: ghostexec`, grader paths.
2. **`scenarios/*.json`** — world content is **data**, not hardcoded lore in Python.
3. **`server/ghostexec_environment.py`** — `build_briefing_text`, `_apply_action`, post-step dynamics, schema drift hooks.
4. **`server/reward.py`** — fixed 0.35 / 0.35 / 0.30 core, invalid / idle handling, shaping caps.
5. **`graders.py`** — bounded grader outputs, trajectory consumption.
6. **Live Space** — `/docs` or `POST /reset` + `POST /step`: legal steps change state; illegal steps return errors, not stack traces.

For a **standalone** walkthrough of the innovation angle only, see **[environment-innovation/README.md](environment-innovation/README.md)**.

---

## Criterion: Storytelling & Presentation (30%)

<a id="ghostexec-storytelling"></a>

**Weight:** 30%

**What it means:**

- Can you clearly explain the problem, the environment, and what the agent learned?
- Is the demo engaging and easy to follow for a non-technical audience?

### The problem (plain language)

An executive’s day is **messy**: urgent email from a board member, a double-booked calendar, a spouse texting about dinner, a report due at noon, and every choice **ripples**—someone feels heard or ignored, a conflict gets better or worse, a task slips or gets done. Ghostexec turns that into a **small simulator** the model must **run**, not a single paragraph to summarize.

### The environment (one sentence)

**You read a realistic staff briefing; you pick one legal “move” (reply, reschedule, delegate, …); the world updates; you get a score that reflects tension across work, people, and tasks.**

### What the agent is supposed to learn

- **Read carefully** — wrong `email_id` / `meeting_id` / `task_id` fails cleanly with feedback.
- **Act under pressure** — clock, `max_steps`, and stress push toward decisions, not endless analysis.
- **Balance competing goals** — improving relationships can conflict with clearing the calendar or finishing tasks; rewards encode that tradeoff.
- **Recover from change** — drift events mean the “right” plan from step 1 may not stay right at step 8.

### Demo tips for a non-technical audience

1. **Show the briefing first** — let viewers see the same wall of text the model sees (relatable chaos).
2. **Show one good step vs one bad step** — e.g. thoughtful reply vs invalid id or `do_nothing` while critical mail waits (mood / reward visibly differ).
3. **Name the three “channels”** — calmer calendar, happier stakeholders, tasks moving forward—without math jargon.
4. **End on “what improved”** — after training, pick the same scenario and show fewer invalid steps, higher rewards, or a grader curve (ties to the 20% section below).

### Hackathon alignment (themes)

**Theme fit (examples):** Ghostexec fits **Theme 3.2 — Personalized tasks** (executive-style inbox, calendar, delegation). **Theme 4** is partially supported via `GHOSTEXEC_CURRICULUM`, `GHOSTEXEC_PERTURB`, and diverse `scenarios/`.

---

## Criterion: Showing Improvement in Rewards (20%)

<a id="ghostexec-reward-improvement"></a>

**Weight:** 20%

**What it means:**

- Is there observable evidence of training progress? Reward curves, before/after behavior, comparison against a baseline—anything that proves the agent learned something.

### Where evidence lives in this repo

| Artifact | Role |
|----------|------|
| `outputs/logs/episode_rewards.jsonl` | Per-step reward trace (gitignored); use for **reward curves** and component debugging. |
| `outputs/trainer_state.json` / training logs | Produced by training scripts when configured; feed into plotting. |
| `outputs/reward_log.csv` | Optional CSV companion for plotting pipelines. |
| `outputs/compliance_manifest.json` | Baseline / compliance metadata for **comparison** charts. |
| `outputs/plots/*.png` | Generated report figures (see command below). |

**Plot pack (loss + reward + components + baseline bar):**

```bash
uv run python scripts/plot_training_report.py \
  --trainer-history outputs/trainer_state.json \
  --reward-csv outputs/reward_log.csv \
  --baselines-json outputs/compliance_manifest.json \
  --out-dir outputs/plots
```

Writes `loss_curve.png`, `reward_curve.png`, `components_curve.png`, `baseline_comparison.png` under `outputs/plots/`.

**End-to-end notebook:** [`notebooks/ghostexec_unsloth_grpo_hf_api.ipynb`](notebooks/ghostexec_unsloth_grpo_hf_api.ipynb) is intended to **Run All** without manual steps (per project convention).

**Before / after narrative for judges:** same `task_id` and seed—show **lower invalid rate**, **higher mean step reward**, or **clearer grader trajectory** after finetuning. Pair numbers with **one short clip** of two runs side by side on the Space or local server.

---

## Criterion: Reward & Training Pipeline (10%)

<a id="ghostexec-reward-pipeline"></a>

**Weight:** 10%

**What it means:**

- Is the reward logic coherent?
- Does the pipeline produce meaningful improvement in the trained agent's behavior?

### Reward logic (coherent and inspectable)

Phase-4 scoring in `server/reward.py` uses a **fixed** core blend:

\[
\text{weighted base} = 0.35 \cdot \text{conflict} + 0.35 \cdot \text{relationship} + 0.30 \cdot \text{task}
\]

Then bounded shaping, invalid-step handling, and explicit penalties (including **`do_nothing`**). Components surface on `RewardBreakdown` and in observation **metadata** where configured—so “why did this step score X?” is **auditable**, not a black box.

Design rationale is aligned with dense reward-shaping practice (see [arXiv:2408.10215](https://arxiv.org/abs/2408.10215))—fixed channel weights, bounded magnitudes, sparse end-of-episode avoided for training.

### Training pipeline (entrypoints)

| Step | Command / artifact |
|------|---------------------|
| Install | `uv sync` (from repo root) |
| Server (matches Dockerfile) | `uv run server --port 8000` |
| SFT → GRPO script | `uv run python scripts/train_sft_then_grpo.py` (see [Running and testing locally](#running-and-testing-locally) for a full example invocation) |
| Tests | `uv run pytest tests/ -q` |
| Docker build gate | `GHOSTEXEC_RUN_DOCKER_BUILD=1 uv run pytest tests/test_docker_build.py -q` |

The pipeline is **meaningful** when tied to the **20% evidence** above: same env URL, logged rewards, and plots that move in the right direction over training—not when loss alone decreases.

---

## OpenEnv Hackathon themes & checklist

| Item | Status |
|------|--------|
| OpenEnv-based env + `openenv.yaml` | In-repo (`openenv-core[core]>=0.2.3`). |
| Short write-up or &lt;2 min video | **You:** publish and paste URLs in [Deliverables](#deliverables-fill-before-freeze). |
| Public HF Space | [Deliverables](#deliverables-fill-before-freeze); deploy with `openenv push --repo-id <your>/ghostexec`. |

---

## Quick start (Python client)

From the repo root (where `pyproject.toml` lives):

```bash
uv sync
uv run server --port 8000
```

```python
from ghostexec import GhostexecAction, GhostexecEnv

with GhostexecEnv(base_url="http://127.0.0.1:8000") as env:
    out = env.reset()
    print(out.observation.echoed_message[:500], "…")

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

**Docker (optional):**

```bash
docker build -t ghostexec-env:latest .
```

---

## Actions and fields

`GhostexecAction` (`models.py`):

| `action_type` | Typical fields |
|---------------|----------------|
| `reply_email` | `email_id`, `message_body` |
| `archive_email` | `email_id` |
| `reschedule_meeting` | `meeting_id`, `new_time`, `reason` |
| `cancel_meeting` | `meeting_id`, `reason` |
| `complete_task` | `task_id` |
| `delegate_task` | `task_id`, `contact_name` |
| `send_message` | `contact_name`, `message` |
| `do_nothing` | — (penalised path) |

Malformed HTTP payloads are handled safely so clients do not crash the server.

---

## Observation

- **`echoed_message`** — Full plain-text briefing.
- **`message_length`** — Length of briefing.
- **`reward`**, **`done`**, **`metadata`** — Step outcome; metadata includes `step_ok`, reward breakdown fields, and debug ids.

---

## Reward (formula summary)

Full detail is under [Criterion: Reward & Training Pipeline (10%)](#criterion-reward--training-pipeline-10). Episode logs: `outputs/logs/episode_rewards.jsonl` (gitignored).

---

## HTTP vs WebSocket (episode state)

- **HTTP** `POST /reset` and `POST /step` may use **short-lived** instances; consecutive HTTP calls might not share one in-memory episode.
- **WebSocket `/ws`** (or `GhostexecEnv`) — use for **multi-step episodes** on one session.

Endpoints: **`/web`**, **`/docs`**, **`/health`**, **`/ws`**.

---

## Running and testing locally

```bash
uv run uvicorn ghostexec.server.app:app --reload --host 0.0.0.0 --port 8000
# or
uv run server --port 8000
```

**HTTP smoke:**

```bash
uv run python scripts/http_endpoint_smoke.py --local
```

**Tests:**

```bash
uv run pytest tests/ -q
GHOSTEXEC_RUN_DOCKER_BUILD=1 uv run pytest tests/test_docker_build.py -q
uv run pytest tests/test_live_server_exhaustive.py -v --tb=short   # server on :8000
```

**SFT → GRPO (example):**

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

---

## Hugging Face Spaces

```bash
openenv serve
openenv build
openenv validate --verbose
openenv push
# openenv push --repo-id your-username/ghostexec
```

Use a **public** Space for the default hackathon flow. `openenv.yaml` carries **name**, **version**, and **description** for metadata—keep them in sync with submission needs.

---

## Scenarios

| File | Role |
|------|------|
| `scenarios/phase2_core.json` | Default dense fixture |
| `scenarios/monday_morning.json`, `dinner_disaster.json`, `vip_meltdown.json` | Narrative pressure |
| `scenarios/vip_meltdown_drift.json` | Mood / escalation drift |
| `scenarios/schema_drift_test.json` | Drift-event harness |

---

## Project layout

```
ghostexec/
├── openenv.yaml
├── pyproject.toml
├── models.py
├── client.py
├── graders.py
├── scenarios/
├── scripts/
├── notebooks/
├── tests/
└── server/
    ├── app.py
    ├── ghostexec_environment.py
    ├── reward.py
    └── Dockerfile
```

---

## Resources & references

- [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) — core stack  
- [Packaging & Deploying](https://meta-pytorch.org/OpenEnv/auto_getting_started/environment-builder.html)  
- [OpenEnv Hub](https://huggingface.co/openenv)  
- [Building RL Environments with OpenEnv](https://www.youtube.com/watch?v=0airz7BhBiA) (and related talks linked in prior README iterations)

---

## License

BSD-style — see license notices in source files (Meta / OpenEnv lineage).
