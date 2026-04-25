"""Hackathon automated gate: loss + reward PNGs committed under docs/."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REWARD_PNG = ROOT / "docs" / "submission_results" / "reward_curve.png"
LOSS_PNG = ROOT / "docs" / "submission_results" / "loss_curve.png"


@pytest.mark.parametrize("path", [REWARD_PNG, LOSS_PNG])
def test_committed_submission_plot_exists(path: Path) -> None:
    assert path.is_file(), f"missing {path.relative_to(ROOT)} — run scripts/generate_committed_submission_plots.py"
    assert path.stat().st_size > 2048, f"{path.name} looks empty or corrupt"


def test_export_reads_stepwise_reward_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import scripts.export_submission_plots as exp

    out = tmp_path / "outputs"
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logf = out / "logs" / "episode_rewards.jsonl"
    rows = [
        {"episode_id": "a", "reward": 0.1, "episode_done": False},
        {"episode_id": "a", "reward": 0.2, "episode_done": True},
        {"episode_id": "b", "reward": -0.05, "episode_done": True},
    ]
    logf.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "OUT", out)
    got = exp._read_episode_rewards()
    assert got == pytest.approx([0.3, -0.05])
