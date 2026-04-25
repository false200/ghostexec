#!/usr/bin/env python3
"""Export committable submission plots from gitignored outputs/.

Produces:
  - docs/submission_results/reward_curve.png
  - docs/submission_results/loss_curve.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DOCS = ROOT / "docs" / "submission_results"


def _read_episode_rewards() -> list[float]:
    """Episode totals: prefer legacy ``episode_reward`` lines; else sum ``reward`` by episode."""
    path = OUT / "logs" / "episode_rewards.jsonl"
    if not path.is_file():
        return []
    legacy: list[float] = []
    aggregated: list[float] = []
    cur_eid: str | None = None
    acc = 0.0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = row.get("episode_reward")
            if isinstance(v, (int, float)):
                legacy.append(float(v))
                continue
            eid = row.get("episode_id")
            r = row.get("reward")
            if eid is None or not isinstance(r, (int, float)):
                continue
            eid_s = str(eid)
            episode_done = bool(row.get("episode_done"))
            if cur_eid is None:
                cur_eid = eid_s
                acc = float(r)
            elif eid_s != cur_eid:
                aggregated.append(acc)
                cur_eid = eid_s
                acc = float(r)
            else:
                acc += float(r)
            if episode_done:
                aggregated.append(acc)
                cur_eid = None
                acc = 0.0
        if cur_eid is not None:
            aggregated.append(acc)
    return legacy if legacy else aggregated


def _read_training_losses() -> tuple[list[tuple[float, float]], str]:
    """Return (step, loss_or_objective) pairs and source id for axis labelling."""
    pairs: list[tuple[float, float]] = []
    for path in sorted((OUT / "training").glob("**/trainer_state.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        log_hist = payload.get("log_history")
        if not isinstance(log_hist, list):
            continue
        for row in log_hist:
            if not isinstance(row, dict):
                continue
            loss = row.get("loss")
            step = row.get("step")
            if isinstance(loss, (int, float)) and isinstance(step, (int, float)):
                pairs.append((float(step), float(loss)))
        if pairs:
            return sorted(pairs, key=lambda x: x[0]), "trainer_state"
    # REINFORCE / train.py episode log: one JSON object per line with ``return``
    returns_path = OUT / "training" / "smoke" / "reinforce_returns.jsonl"
    if not returns_path.is_file():
        return [], "none"
    idx = 0
    with returns_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ret = row.get("return")
            if not isinstance(ret, (int, float)):
                continue
            idx += 1
            pairs.append((float(idx), -float(ret)))
    return pairs, "reinforce_returns"


def _write_reward_curve(rewards: list[float]) -> Path:
    out = DOCS / "reward_curve.png"
    plt.figure(figsize=(8, 3.5))
    plt.plot(rewards, color="tab:blue", alpha=0.9, linewidth=1.8)
    plt.title("Episode reward curve")
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    return out


def _write_loss_curve(loss_pairs: list[tuple[float, float]], *, ylabel: str = "loss") -> Path:
    out = DOCS / "loss_curve.png"
    steps = [x[0] for x in loss_pairs]
    losses = [x[1] for x in loss_pairs]
    plt.figure(figsize=(8, 3.5))
    plt.plot(steps, losses, color="tab:orange", alpha=0.9, linewidth=1.8)
    plt.title("Training loss curve")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    return out


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)

    rewards = _read_episode_rewards()
    losses, loss_src = _read_training_losses()
    loss_ylabel = (
        "loss"
        if loss_src == "trainer_state"
        else "negative episode return" if loss_src == "reinforce_returns" else "loss"
    )

    wrote: list[Path] = []
    if rewards:
        wrote.append(_write_reward_curve(rewards))
    if losses:
        wrote.append(_write_loss_curve(losses, ylabel=loss_ylabel))

    if not wrote:
        print(
            "No plot inputs found. Expected at least one of:\n"
            " - outputs/logs/episode_rewards.jsonl\n"
            " - outputs/training/**/trainer_state.json\n"
            " - outputs/training/smoke/reinforce_returns.jsonl (REINFORCE episode returns)"
        )
        return 1

    print("Exported submission plots:")
    for p in wrote:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
