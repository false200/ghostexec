# Ghostexec: A tiny chief-of-staff simulator for agents

**Demo (&lt;2 min):** [Ghostexec on YouTube](https://youtu.be/g4IFZMEzfO8)

---

## The messy day, in one paragraph

Picture a morning where everything arrives at once: a board member’s email, a double-booked calendar, a message from home about dinner, a report due at noon, and a teammate who is already annoyed. You cannot “solve” that with a single summary. You **sequence** decisions—who gets a reply, what gets rescheduled, what gets delegated—and each move changes how stressed you are, how people feel, and whether real work moves forward.

**Ghostexec** is a small, trainable environment that captures that feeling. It is built on **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)** so agents, researchers, and judges can talk to it through a **standard HTTP and WebSocket API**, run it on a **public Hugging Face Space**, and plug it into a **real RL or preference-optimization loop**.

---

## What Ghostexec actually is

Ghostexec is an **OpenEnv-compatible “AI chief of staff” simulator**:

- **Inbox, calendar, contacts, tasks**, and **stakeholder moods** live in **JSON scenarios** under `scenarios/` — the story is data, not hardcoded prose in Python.
- Each step, the policy sees a **plain-text briefing** (the same kind of wall-of-text a human assistant might scan), not a raw dump of the whole world object.
- The agent returns **one structured action per step** — for example `reply_email`, `reschedule_meeting`, `complete_task`, or `delegate_task` — with fields validated against a schema.
- **Invalid actions do not crash the server.** They return a controlled signal so learning (or evaluation) can continue.
- **`do_nothing` is penalised** so “freeze and hope it goes away” is not a free winning strategy when fires are burning.

Under the hood, the simulation advances **time**, **moods**, and **conflicts**, and optional **drift events** in the JSON can reshuffle the situation mid-episode so the agent is tested on **adaptation**, not memorizing the first screen.

---

## Why OpenEnv?

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) gives us a **shared contract**: reset, step, schema, health, WebSocket sessions, and tooling to **validate** and **ship** environments. Our manifest is in **`openenv.yaml`** (environment name `ghostexec`, three tasks with graders, FastAPI app entrypoint). That keeps the submission **inspectable** and **reproducible** — judges can open the Space, read the repo, and run tests locally with the same entrypoints we use in Docker.

---

## Rewards: teach the model, certify the run

We use **two layers** of feedback, on purpose:

1. **Dense step reward** (in `server/reward.py`) blends **conflict**, **relationship**, and **task** progress with fixed weights **0.35 / 0.35 / 0.30**, plus bounded shaping and explicit handling of invalid or idle steps. That signal is what you want when **training** with modern RL or GRPO-style methods.
2. **Trajectory graders** (in `graders.py`, wired in `openenv.yaml`) produce **bounded** scores for three **tasks** — easy, medium, and hard scenarios — so hackathon **certification** stays in a well-defined range.

Together: the model can **learn** from rich per-step feedback, while organizers can **score** full trajectories against clear tasks.

---

## Try it in sixty seconds

**Short demo video:** [https://youtu.be/g4IFZMEzfO8](https://youtu.be/g4IFZMEzfO8)

**Live Space (public):** [https://huggingface.co/spaces/modelbuilderhq/ghostexec](https://huggingface.co/spaces/modelbuilderhq/ghostexec)

- Open **`/docs`** on the Space for the interactive API, or **`/web`** for the OpenEnv playground.
- Full README (formatted, with tables and deep links):  
  [https://huggingface.co/spaces/modelbuilderhq/ghostexec/blob/main/README.md](https://huggingface.co/spaces/modelbuilderhq/ghostexec/blob/main/README.md)

**Local quick start** (from repo root):

```bash
uv sync
uv run server --port 8000
```

Then use **`GhostexecEnv`** from `client.py` (WebSocket session) for **multi-step episodes**, or raw HTTP if you only need a smoke test. The README’s “Quick start” section has a copy-paste Python snippet.

**Source:** mirror on GitHub as you prefer; the canonical hackathon artifact is the **Space + repo** layout described in the README.

---

## If you only have two minutes on video

Published walkthrough: [**youtu.be/g4IFZMEzfO8**](https://youtu.be/g4IFZMEzfO8)

A tight arc that works for non-technical viewers:

1. **Show the briefing** — scroll through the same text the model sees. Say: “This is not a quiz; it’s a shift at work.”
2. **One good action** — e.g. a thoughtful `reply_email` with a real `email_id` from the text; show reward or mood metadata if the UI exposes it.
3. **One bad action** — wrong id or `do_nothing` while something urgent waits; show that the world **does not crash**, but the score **hurts**.
4. **One sentence on training** — “We can optimize policies against this API with GRPO / TRL-style loops; graders score whole episodes for the hackathon.”

End on: **Ghostexec is the busy day, compressed — so models can practice being calm, fast, and fair before anyone trusts them near a real calendar.**

---

## Theme fit (hackathon)

Ghostexec aligns naturally with **personalized, high-stakes tasks**: executive triage, delegation, and tradeoffs between **people**, **calendar**, and **deadlines**. Diverse **`scenarios/*.json`** and optional curriculum / perturbation hooks (see README) make it easy to **stress-test** policies without rewriting core engine code.

---

## Where to read more

| Resource | Link |
|----------|------|
| Demo video (&lt;2 min) | [YouTube](https://youtu.be/g4IFZMEzfO8) |
| Full project README (judging sections, layout, commands) | [README on the Hub](https://huggingface.co/spaces/modelbuilderhq/ghostexec/blob/main/README.md) |
| Innovation-only deep dive | [`environment-innovation/README.md`](environment-innovation/README.md) in the repo |
| OpenEnv upstream | [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) |

---

## Closing

Ghostexec is not trying to replace human assistants. It is trying to give **models and researchers** a **credible, stressful, and kind** miniature office: text that reads like work, actions that look like tools, and scores that admit **tradeoffs**. If that sounds useful, spin up the Space, break something on purpose, and watch the environment **keep running** — that resilience is part of the point.

*— Ghostexec / OpenEnv submission*
