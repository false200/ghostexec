from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from ghostexec.models import GhostexecAction
from ghostexec.server.ghostexec_environment import GhostexecEnvironment


def _run_episode(mode: str, scenario: Path) -> float:
    env = GhostexecEnvironment(scenario_path=scenario, reward_mode=mode)
    env.reset()
    actions = [
        GhostexecAction(action_type="reschedule_meeting", meeting_id="m02", new_time="2026-04-21T18:00:00"),
        GhostexecAction(action_type="reply_email", email_id="e01", message_body="Sharing revised numbers now."),
        GhostexecAction(action_type="archive_email", email_id="e09"),
        GhostexecAction(action_type="send_message", contact_name="Jordan Lee", message_body="Quick status sync."),
        GhostexecAction(action_type="complete_task", task_id="t06"),
    ]
    rewards = [float(env.step(a).reward or 0.0) for a in actions]
    return statistics.fmean(rewards)


def _run(mode: str, scenario: Path, episodes: int) -> dict[str, float]:
    vals = [_run_episode(mode, scenario) for _ in range(episodes)]
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward-mode ablation for Ghostexec.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=ROOT / "scenarios" / "phase2_core.json",
    )
    args = parser.parse_args()

    modes = ("base", "full")
    results = {m: _run(m, args.scenario, args.episodes) for m in modes}
    print("Ghostexec reward ablation")
    print(f"scenario={args.scenario} episodes={args.episodes}")
    for m in modes:
        r = results[m]
        print(
            f"{m:>5}: mean={r['mean']:.4f} std={r['std']:.4f} "
            f"min={r['min']:.4f} max={r['max']:.4f}"
        )
    delta = results["full"]["mean"] - results["base"]["mean"]
    print(f"delta(full-base)={delta:.4f}")


if __name__ == "__main__":
    main()
