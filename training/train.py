# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Phase 5A: episode training loop with reward logging and optional checkpoints.
# Default path uses the in-process GhostexecEnvironment (no GPU). For LLM + GRPO
# on Hugging Face TRL, see Meta OpenEnv tutorial 04-training (Wordle GRPO pattern)
# https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md
# and this repo's ``training/openenv_grpo_rollout.py`` (rollout_func) +
# ``training/grpo_ghostexec_reward.py`` (simple scalar reward). Install:
# ``uv sync --extra training``.

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghostexec.client import GhostexecEnv  # noqa: E402
from ghostexec.models import GhostexecAction, GhostexecObservation  # noqa: E402
from ghostexec.server.ghostexec_environment import GhostexecEnvironment  # noqa: E402


def _parse_dt(value: str) -> datetime:
    if value.endswith("Z"):
        return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def bump_simulation_hours(iso: str, hours: int) -> str:
    dt = _parse_dt(iso.replace("Z", "+00:00"))
    return (dt + timedelta(hours=hours)).isoformat(timespec="seconds")


def smart_action(
    obs: GhostexecObservation,
    rng: random.Random,
) -> GhostexecAction:
    """Hand-crafted policy: prioritise critical replies, calendar relief, overdue tasks, inbox hygiene."""
    meta: dict[str, Any] = obs.metadata or {}
    text = obs.echoed_message or ""
    crit = list(meta.get("critical_unreplied_email_ids") or [])
    if crit:
        return GhostexecAction(
            action_type="reply_email",
            email_id=crit[0],
            message_body="On it — drafting a response and owners now.",
        )
    mids = list(meta.get("active_meeting_ids") or [])
    if "CLASHES WITH" in text and mids:
        nt = bump_simulation_hours(str(meta.get("simulation_time", "2026-04-21T08:00:00")), 10)
        for mid in mids[:3]:
            act = GhostexecAction(
                action_type="reschedule_meeting",
                meeting_id=mid,
                new_time=nt,
            )
            return act
    oids = list(meta.get("overdue_task_ids") or [])
    if oids:
        return GhostexecAction(action_type="complete_task", task_id=oids[0])
    uids = list(meta.get("unread_email_ids") or [])
    if uids:
        return GhostexecAction(action_type="archive_email", email_id=uids[rng.randrange(len(uids))])
    names = ("Jordan Lee", "Sarah Chen", "Marcus Webb")
    return GhostexecAction(
        action_type="send_message",
        contact_name=names[rng.randrange(len(names))],
        message_body="Quick update: triaging calendar and inbox this block.",
    )


def random_valid_action(
    obs: GhostexecObservation,
    rng: random.Random,
) -> GhostexecAction:
    """Mostly-valid random policy for variance / REINFORCE exploration."""
    meta: dict[str, Any] = obs.metadata or {}
    choices: list[GhostexecAction] = [GhostexecAction(action_type="do_nothing")]
    for eid in (meta.get("unread_email_ids") or [])[:5]:
        choices.append(GhostexecAction(action_type="archive_email", email_id=eid))
    for eid in (meta.get("critical_unreplied_email_ids") or [])[:2]:
        choices.append(
            GhostexecAction(
                action_type="reply_email",
                email_id=eid,
                message_body="Thanks — reviewing and will follow up shortly.",
            )
        )
    mids = list(meta.get("active_meeting_ids") or [])[:3]
    if mids:
        nt = bump_simulation_hours(str(meta.get("simulation_time", "2026-04-21T08:00:00")), 6 + rng.randint(0, 6))
        choices.append(
            GhostexecAction(action_type="reschedule_meeting", meeting_id=mids[0], new_time=nt)
        )
    return choices[rng.randrange(len(choices))]


def featurize(obs: GhostexecObservation, dim: int = 96) -> list[float]:
    text = (obs.echoed_message or "").lower()
    vec = [0.0] * dim
    for tok in text.split():
        h = hash(tok) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@dataclass
class LinearBanditAgent:
    """Lightweight diagonal linear preferences over discretised actions (REINFORCE-style updates)."""

    dim: int = 96
    n_actions: int = 6
    lr: float = 0.12
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.rng = self.rng or random.Random(0)
        self.theta = [
            [self.rng.gauss(0, 0.02) for _ in range(self.dim)] for _ in range(self.n_actions)
        ]

    def scores(self, phi: list[float]) -> list[float]:
        return [sum(t * p for t, p in zip(self.theta[a], phi)) for a in range(self.n_actions)]

    def act_index(self, phi: list[float]) -> int:
        logits = self.scores(phi)
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        s = sum(exps) or 1.0
        probs = [e / s for e in exps]
        u = self.rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if u <= c:
                return i
        return len(probs) - 1

    def update(self, phi: list[float], a: int, advantage: float) -> None:
        for j in range(self.dim):
            self.theta[a][j] += self.lr * advantage * phi[j]


def reinforce_action(
    obs: GhostexecObservation,
    agent: LinearBanditAgent,
    rng: random.Random,
) -> tuple[GhostexecAction, int]:
    phi = featurize(obs, agent.dim)
    idx = agent.act_index(phi)
    meta: dict[str, Any] = obs.metadata or {}
    if idx == 0:
        return GhostexecAction(action_type="do_nothing"), idx
    if idx == 1:
        crit = list(meta.get("critical_unreplied_email_ids") or [])
        if crit:
            return (
                GhostexecAction(
                    action_type="reply_email",
                    email_id=crit[0],
                    message_body="Acknowledged — working the thread now.",
                ),
                idx,
            )
    if idx == 2:
        u = list(meta.get("unread_email_ids") or [])
        if u:
            return GhostexecAction(action_type="archive_email", email_id=u[rng.randrange(len(u))]), idx
    if idx == 3:
        o = list(meta.get("overdue_task_ids") or [])
        if o:
            return GhostexecAction(action_type="complete_task", task_id=o[0]), idx
    if idx == 4:
        mids = list(meta.get("active_meeting_ids") or [])
        if mids:
            nt = bump_simulation_hours(str(meta.get("simulation_time", "2026-04-21T08:00:00")), 9)
            return (
                GhostexecAction(
                    action_type="reschedule_meeting",
                    meeting_id=mids[0],
                    new_time=nt,
                ),
                idx,
            )
    return (
        GhostexecAction(
            action_type="send_message",
            contact_name="Jordan Lee",
            message_body="Quick ping from training run.",
        ),
        idx,
    )


def run_episode_local(
    env: GhostexecEnvironment,
    policy: Callable[[GhostexecObservation], GhostexecAction],
    max_steps: int,
) -> dict[str, Any]:
    obs = env.reset()
    total = 0.0
    steps = 0
    first_action: dict[str, Any] | None = None
    for _ in range(max_steps):
        act = policy(obs)
        if first_action is None:
            first_action = act.model_dump(mode="json")
        obs = env.step(act)
        total += float(obs.reward or 0.0)
        steps += 1
        if obs.done:
            break
    return {
        "return": total,
        "length": steps,
        "mean_step_reward": total / max(1, steps),
        "first_action": first_action,
    }


def run_episode_remote(
    base_url: str,
    policy: Callable[[GhostexecObservation], GhostexecAction],
    max_steps: int,
) -> dict[str, Any]:
    client = GhostexecEnv(base_url=base_url).sync()
    with client:
        res = client.reset()
        obs = res.observation
        total = 0.0
        steps = 0
        first_action: dict[str, Any] | None = None
        for _ in range(max_steps):
            act = policy(obs)
            if first_action is None:
                first_action = act.model_dump(mode="json")
            res = client.step(act)
            obs = res.observation
            total += float(res.reward or 0.0)
            steps += 1
            if res.done:
                break
    return {
        "return": total,
        "length": steps,
        "mean_step_reward": total / max(1, steps),
        "first_action": first_action,
    }


def save_checkpoint(path: Path, episode: int, agent: LinearBanditAgent | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"episode": episode}
    if agent is not None:
        payload["theta"] = agent.theta
        payload["dim"] = agent.dim
        payload["n_actions"] = agent.n_actions
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="GhostExec Phase 5 training / logging driver.")
    p.add_argument("--backend", choices=("local", "remote"), default="local")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="OpenEnv server URL for remote backend.")
    p.add_argument(
        "--scenario",
        type=str,
        default=str(ROOT / "scenarios" / "phase2_core.json"),
        help="Path to scenario JSON (local backend only).",
    )
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument(
        "--agent",
        choices=("smart", "random", "reinforce"),
        default="reinforce",
        help="smart = scripted executive policy; random = stochastic valid-ish actions; reinforce = linear bandit REINFORCE.",
    )
    p.add_argument(
        "--log-path",
        type=str,
        default=str(ROOT / "outputs" / "training" / "episode_returns.jsonl"),
    )
    p.add_argument("--checkpoint-dir", type=str, default=str(ROOT / "outputs" / "training" / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_path = Path(args.scenario)
    if args.backend == "local" and not scenario_path.is_file():
        raise SystemExit(f"Scenario not found: {scenario_path}")

    agent = LinearBanditAgent(rng=random.Random(args.seed + 7)) if args.agent == "reinforce" else None
    moving = 0.0
    beta = 0.15

    def policy_smart(obs: GhostexecObservation) -> GhostexecAction:
        return smart_action(obs, rng)

    def policy_random(obs: GhostexecObservation) -> GhostexecAction:
        return random_valid_action(obs, rng)

    def policy_reinforce(obs: GhostexecObservation) -> GhostexecAction:
        assert agent is not None
        act, _idx = reinforce_action(obs, agent, rng)
        return act

    policies: dict[str, Callable[[GhostexecObservation], GhostexecAction]] = {
        "smart": policy_smart,
        "random": policy_random,
        "reinforce": policy_reinforce,
    }
    policy_fn = policies[args.agent]

    env_local: GhostexecEnvironment | None = None
    if args.backend == "local":
        env_local = GhostexecEnvironment(scenario_path)

    last_first: dict[str, Any] | None = None
    last_last: dict[str, Any] | None = None

    for ep in range(args.episodes):
        if args.backend == "local":
            assert env_local is not None
            if args.agent == "reinforce":
                obs0 = env_local.reset()
                ep_ret = 0.0
                ep_steps = 0
                first_action = None
                for _ in range(args.max_steps):
                    phi = featurize(obs0, agent.dim)  # type: ignore[union-attr]
                    act, aidx = reinforce_action(obs0, agent, rng)  # type: ignore[arg-type]
                    if first_action is None:
                        first_action = act.model_dump(mode="json")
                    obs0 = env_local.step(act)
                    r = float(obs0.reward or 0.0)
                    ep_ret += r
                    ep_steps += 1
                    adv = r - moving
                    moving = (1 - beta) * moving + beta * r
                    agent.update(phi, aidx, adv)  # type: ignore[union-attr]
                    if obs0.done:
                        break
                stats = {
                    "return": ep_ret,
                    "length": ep_steps,
                    "mean_step_reward": ep_ret / max(1, ep_steps),
                    "first_action": first_action,
                }
            else:
                stats = run_episode_local(env_local, policy_fn, args.max_steps)
        else:
            stats = run_episode_remote(args.base_url, policy_fn, args.max_steps)

        if ep == 0:
            last_first = stats.get("first_action")
        if ep == args.episodes - 1:
            last_last = stats.get("first_action")

        row = {
            "episode": ep,
            "scenario": str(scenario_path.name),
            "backend": args.backend,
            "agent": args.agent,
            "return": stats["return"],
            "length": stats["length"],
            "mean_step_reward": stats["mean_step_reward"],
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        if agent is not None and (ep + 1) % 50 == 0:
            save_checkpoint(Path(args.checkpoint_dir) / f"checkpoint_ep_{ep+1:05d}.json", ep + 1, agent)

    summary = {
        "episodes": args.episodes,
        "log_path": str(log_path),
        "first_episode_first_action": last_first,
        "last_episode_first_action": last_last,
    }
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.checkpoint_dir) / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import importlib.util

        if importlib.util.find_spec("trl") is not None:
            print(
                "Optional TRL is installed. For GRPO: OpenEnv tutorial 04-training + "
                "https://huggingface.co/docs/trl — use training/openenv_grpo_rollout.py "
                "(rollout_func) or training/grpo_ghostexec_reward.py (scalar reward).",
                file=sys.stderr,
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
