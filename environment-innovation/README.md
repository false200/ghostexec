# Ghostexec — innovation brief (for reviewers)

**Repository:** [Ghostexec (OpenEnv)](../README.md)  
**Public Space:** https://huggingface.co/spaces/modelbuilderhq/ghostexec  

This README is a **standalone** walkthrough for reviewers: why the environment is hard, what agent capabilities it stresses, how to verify claims in code and on the live Space. You can read it **without** opening the rest of the repo narrative.

---

## Contents

1. [How to read this document](#how-to-read-this-document)  
2. [Short answers](#short-answers-so-nothing-is-buried)  
3. [What Ghostexec is](#1-what-ghostexec-is-one-paragraph)  
4. [What the agent observes](#2-what-the-agent-observes-and-why-that-matters)  
5. [What the agent can do](#3-what-the-agent-can-do-actions-and-legality)  
6. [What changes between steps](#4-what-changes-between-steps-dynamics-and-drift)  
7. [How success is scored](#5-how-success-is-scored-two-layers-on-purpose)  
8. [Task ladder](#6-the-public-task-ladder-difficulty-in-data-not-vibes)  
9. [Reviewer checklist](#7-how-a-reviewer-can-verify-5-minute-checklist)  
10. [Closing](#8-closing)  
11. [Key files (from repo root)](#key-files-from-repo-root)

---

## How to read this document

We group the argument under **two angles** reviewers typically care about. Everything below maps to one or both:

| Angle | Sections that answer it |
|-------|-------------------------|
| Is the **world** itself interesting and genuinely hard? | [Short answers](#short-answers-so-nothing-is-buried), [§1–§4](#1-what-ghostexec-is-one-paragraph) |
| Does it **stress-test agents** in a way a toy demo would not? | [Short answers](#short-answers-so-nothing-is-buried), [§3–§6](#3-what-the-agent-can-do-actions-and-legality), [§8](#8-closing) |

---

## Short answers (so nothing is buried)

**Is it genuinely challenging?** Yes. The agent must survive **dense natural-language state**, emit **strict structured actions** that **mutate** a multi-entity world, and accept **time pressure**, **social consequences**, and **invalid-action economics** without crashing the server. “Easy” wins are rare because channels **compete**: mail, calendar, tasks, and relationships all pull in different directions.

**Is it a meaningful test of behavior?** Yes. Success requires **grounded parsing** (real ids from the briefing), **tool discipline** (legal JSON schema), **sequencing** over multiple steps (WebSocket sessions for real episodes; HTTP for resets and single steps), and **tradeoffs** reflected in a **multi-channel** reward—not a single template answer.

**Is every ingredient globally novel?** No—and we do not claim otherwise. Inboxes and calendars are familiar. What *is* uncommon is the **composition**: OpenEnv-first packaging, **plain-text-only** observations, **data-driven** scenarios, **live dynamics** and **timed drift**, **dual** evaluation (**dense step rewards** + **trajectory graders** in strict `(0.01, 0.99)`), and a **production-shaped** action API—together—in one environment you can train and ship.

---

### 1. What Ghostexec is (one paragraph)

Ghostexec is an **executive chief-of-staff simulator**. Each episode starts from JSON scenario data under `../scenarios/`, selected by **task id** in `../openenv.yaml`. The **engine** lives in `../server/ghostexec_environment.py` and `../server/reward.py`; the **deployment contract** for Hugging Face / OpenEnv is `../openenv.yaml` (name **`ghostexec`**, FastAPI `server.app:app`, port **8000**). The model never sees raw scenario JSON as its primary observation: it sees a **rendered briefing**—the same class of messy, overlapping information a human would scan under time pressure.

---

### 2. What the agent observes (and why that matters)

After `reset` (or the WebSocket equivalent), the policy receives `GhostexecObservation.echoed_message`: a **single plain-text** block that includes, at minimum:

- A **timestamped header** (simulated “now”).
- **Unread emails** with priority, sender, relationship, subject, and a short preview.
- **Calendar conflicts** in a rolling horizon (overlaps the agent could resolve or worsen).
- **Top contacts** with **mood**, relationship type, and communication preference.
- **Tasks** that are overdue or due soon.
- **Executive stress** and **steps remaining** toward `max_steps` (see `../openenv.yaml`, default **20**).

**Why this matters for “challenging”:** many demos hide structure in JSON observations or tool schemas. Here, the **only** narrative state the model is supposed to “read” like a user is **natural language**, while the **law** of the world is still **typed actions**. That forces **comprehension + compliance** together—hallucinated ids and “vibes-only” plans fail in ways you can measure.

---

### 3. What the agent can do (actions and legality)

Each step the agent returns **exactly one** `GhostexecAction` (`../models.py`): `reply_email`, `archive_email`, `reschedule_meeting`, `cancel_meeting`, `complete_task`, `delegate_task`, `send_message`, or `do_nothing`.

**Validity is enforced against the live world:** wrong `email_id` / `meeting_id` / `task_id`, missing required fields, or impossible combinations produce an **invalid step**. The server **does not throw**; it returns structured metadata (`step_ok`, error text) so RL and HTTP clients can learn from mistakes instead of dying.

**Valid actions mutate state:** mail can be replied or archived; meetings moved or cancelled; tasks completed or delegated; direct messages sent. The episode is therefore a **small transactional simulation**, not a static Q&A.

---

### 4. What changes between steps (dynamics and drift)

Ghostexec is **not** a static paragraph with a hidden answer key. After actions, the environment runs **post-step dynamics** (see `../server/ghostexec_environment.py`):

- **Clock:** simulation time advances (default **20 minutes** per step), which can flip tasks into overdue and change what “urgent” means.
- **Mood:** stakeholders move along a mood ladder after real actions (e.g. a thoughtful reply can improve a sender; cancelling a meeting can upset attendees).
- **Pressure on idle / invalid behavior:** if the agent **`do_nothing`**s or **fails** while **critical** mail is still unanswered, mood pressure can concentrate on the sender who is actually waiting—so “safe” inaction is not safe in the social graph.
- **Stress and conflicts:** the world rebuilds an **active conflict list** (overlaps, unanswered critical mail) and maps that into the **stress** value surfaced in the briefing—so calendar debt is not cosmetic.

**Scenario-driven schema drift:** harder JSON can schedule **`after_step`** events that reshuffle the world mid-episode: shift meetings, move deadlines, change communication preferences, **suppress relationship credit** for certain reply paths, or force moods. That tests **adaptation**, not memorization of the first screen.

---

### 5. How success is scored (two layers, on purpose)

**A. Dense step reward (training and fine-grained analysis)** — `../server/reward.py`  
A **fixed** weighted core (**0.35 conflict + 0.35 relationship + 0.30 task**) plus **bounded** shaping terms (synergy, tradeoffs, progress-style shaping, scaffold, quality separation). Invalid steps and **`do_nothing`** are handled explicitly (idle is **penalised**, not neutral). Rich `RewardBreakdown` fields can be logged to `outputs/logs/episode_rewards.jsonl` (gitignored) for auditing *why* a step moved.

**B. Trajectory graders (OpenEnv / hackathon validation)** — `../graders.py`  
Each public task in `../openenv.yaml` binds a grader (`graders.phase2_core_grader`, etc.). Graders read **trajectory-shaped** payloads (e.g. lists of rewards) and return scores **strictly inside `(0.01, 0.99)`**—the validator-facing layer—while the step engine remains the **dense teaching signal**.

That split is deliberate: **agents learn from fine structure**, **judges certify** with stable bounded scores.

---

### 6. The public task ladder (difficulty in *data*, not vibes)

| Task id | Difficulty | Scenario file | What gets harder |
|---------|------------|----------------|------------------|
| `phase2_core` | easy | `../scenarios/phase2_core.json` | Dense default triage: VIP mail, calendar relief, overlapping obligations. |
| `monday_morning` | medium | `../scenarios/monday_morning.json` | Stacked Monday rush: more concurrent fires, less slack. |
| `dinner_disaster` | hard | `../scenarios/dinner_disaster.json` | Personal vs professional collision with **escalation risk**. |

All of this is declared in **`../openenv.yaml`** so the Space, CLI, and notebooks agree on **names**, **ports**, and **grader wiring** without a second source of truth.

---

### 7. How a reviewer can verify (5-minute checklist)

1. Open **`../openenv.yaml`** — confirm three tasks, `max_steps`, `app: server.app:app`, **`name: ghostexec`**.  
2. Open **`../scenarios/*.json`** — confirm episodes are **data**, not hardcoded Python lore.  
3. Skim **`../server/ghostexec_environment.py`** — `build_briefing_text`, `_apply_action`, `_apply_post_action_dynamics`, `_maybe_apply_schema_drift_events`.  
4. Skim **`../server/reward.py`** — fixed weights, invalid / idle handling, shaping caps.  
5. Open **`../graders.py`** — strict output bounds and trajectory consumption.  
6. Open the **public Space**: https://huggingface.co/spaces/modelbuilderhq/ghostexec — use `/docs` or `POST /reset` + `POST /step`: legal actions change state; illegal actions return errors, **not** stack traces.

---

### 8. Closing

**World quality.** The challenge is **interactional and operational**: overlapping human-style goals, strict tool use, evolving social signals, and mid-episode drift—**not** a single binary “did you answer correctly.”

**What this stack proves.** If you strip Ghostexec to one bullet, it is: **plain-text situational awareness + legal structured world edits + multi-channel rewards + timed scenario pressure + OpenEnv-native deployment and graders**—in one coherent package you can train, log, and host.

That is the **innovation case** this repository is built to defend.

---

## Key files (from repo root)

| Path | Role |
|------|------|
| `openenv.yaml` | Space name, port, tasks, graders, `max_steps` |
| `scenarios/*.json` | Episode **data** (world content, drift hooks) |
| `server/ghostexec_environment.py` | Briefing text, actions, dynamics, drift |
| `server/reward.py` | Step reward, fixed 0.35 / 0.35 / 0.30 core + shaping |
| `graders.py` | Trajectory scores in `(0.01, 0.99)` per task |
| `models.py` | `GhostexecAction`, `GhostexecObservation`, `RewardBreakdown` |

For install, tests, training scripts, and the rest of the hackathon submission, see the [main project README](../README.md).
