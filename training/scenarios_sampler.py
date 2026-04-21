# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Scenario sampling + curriculum + order-only perturbations for GRPO / eval.
# Used by ``grpo_ghostexec_reward`` and ``eval_harness`` to fight memorization
# of a single deterministic briefing.

from __future__ import annotations

import json
import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCENARIOS_ROOT = Path(__file__).resolve().parents[1] / "scenarios"

# Keep ``vip_meltdown*`` out of train pool so eval is truly held-out.
TRAIN_SCENARIOS: tuple[str, ...] = (
    "phase2_core.json",
    "monday_morning.json",
    "dinner_disaster.json",
)
EVAL_SCENARIOS: tuple[str, ...] = (
    "vip_meltdown.json",
)

CURRICULUM: dict[str, tuple[str, ...]] = {
    "easy": ("monday_morning.json",),
    "mid": ("monday_morning.json", "phase2_core.json"),
    "hard": ("phase2_core.json", "dinner_disaster.json"),
    "all": TRAIN_SCENARIOS,
}


def scenario_path(name: str) -> Path:
    """Resolve a scenario filename to its absolute path under ``scenarios/``."""
    return SCENARIOS_ROOT / name


def pick_scenario(rng: random.Random, level: str = "all") -> Path:
    """Pick a scenario from the curriculum pool for ``level`` (defaults to all train)."""
    pool = CURRICULUM.get(level) or TRAIN_SCENARIOS
    return scenario_path(rng.choice(pool))


def _shift_iso(ts: str, hours: int) -> str:
    raw = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt + timedelta(hours=hours)).isoformat(timespec="seconds")


def load_perturbed_scenario(
    path: Path,
    rng: random.Random,
    *,
    shuffle_emails: bool = True,
    shuffle_meetings: bool = False,
    shuffle_contacts: bool = True,
    shuffle_tasks: bool = True,
    hours_shift: int = 0,
) -> Path:
    """Write a temp scenario JSON with order-only perturbations (+ optional time shift).

    Only safe permutations are applied by default (order of lists, not IDs/relationships),
    so invariants like ``sender_relationship`` references and meeting overlaps stay valid.
    Caller is responsible for ``unlink()`` once the file has been consumed.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if shuffle_emails and isinstance(data.get("emails"), list):
        rng.shuffle(data["emails"])
    if shuffle_meetings and isinstance(data.get("meetings"), list):
        rng.shuffle(data["meetings"])
    if shuffle_contacts and isinstance(data.get("contacts"), list):
        rng.shuffle(data["contacts"])
    if shuffle_tasks and isinstance(data.get("tasks"), list):
        rng.shuffle(data["tasks"])
    if hours_shift and isinstance(data.get("simulation_time"), str):
        data["simulation_time"] = _shift_iso(data["simulation_time"], hours_shift)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="ghostexec_scn_",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(data, tmp)
    finally:
        tmp.close()
    return Path(tmp.name)
