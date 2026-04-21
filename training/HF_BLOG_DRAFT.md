# GhostExec — teaching an AI chief of staff to survive Monday (HF blog draft)

**GhostExec** is an OpenEnv environment where an agent reads a plain-text executive briefing (inbox, calendar clashes, stakeholder moods, overdue tasks) and takes structured actions: reply, archive, reschedule, delegate, and more. The goal is not trivia — it is **conflict resolution under pressure**, the same trade-offs a real chief of staff navigates.

## Reward signal

Training uses a fixed weighted blend of **calendar conflict**, **relationship**, and **task** scores, plus episode bonuses and penalties for catastrophic stakeholder failure. Episode returns are logged to JSONL (`outputs/training/episode_returns.jsonl`) so you can plot improvement over time.

## What judges should see

- **Screenshot:** a matplotlib curve of episode return vs episode index (moving average optional). Even a modest upward slope shows the signal is learnable.
- **Before / after:** episode 0 first action vs a later episode (from `run_summary.json`) — e.g. shifting from idle `do_nothing` toward critical replies or conflict-breaking reschedules after `training/train.py` with the `reinforce` agent.

## Links

- **Space:** replace with your public Hugging Face Space URL (`uv run server` entrypoint).
- **Code:** replace with your GitHub repository URL.

_Paste this post into the Hugging Face community blog UI; add your real Space and repo links and attach the reward-curve screenshot._
