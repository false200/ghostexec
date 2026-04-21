# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Rejection-sampled SFT: build (briefing -> JSON action) pairs via ``smart_action``
# and keep only the high-reward half (or any quantile). Feeds Unsloth/TRL
# ``SFTTrainer`` a cleaner target distribution than naive demos.

from __future__ import annotations

import json
import random
from pathlib import Path

from .scenarios_sampler import CURRICULUM, TRAIN_SCENARIOS, scenario_path

try:
    from ghostexec.models import GhostexecAction
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    from models import GhostexecAction
    from server.ghostexec_environment import GhostexecEnvironment


def _action_json(act: GhostexecAction) -> str:
    return json.dumps(
        act.model_dump(mode="json", exclude_defaults=False),
        ensure_ascii=False,
    )


def generate_samples(
    n: int,
    *,
    scenarios: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Produce candidate ``{"prompt","completion","reward","scenario"}`` rows.

    Uses the scripted ``smart_action`` policy so each sample comes with a
    legitimate env reward for rejection filtering.
    """
    from .train import smart_action  # lazy (avoids argparse-heavy module at import)

    pool = scenarios or TRAIN_SCENARIOS
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for i in range(n):
        name = pool[i % len(pool)]
        path = scenario_path(name)
        env = GhostexecEnvironment(path)
        obs = env.reset()
        act = smart_action(obs, rng)
        res = env.step(act)
        rows.append(
            {
                "prompt": obs.echoed_message or "",
                "completion": _action_json(act),
                "reward": float(res.reward or 0.0),
                "scenario": name,
            }
        )
    return rows


def filter_top(
    samples: list[dict[str, object]],
    *,
    quantile: float = 0.5,
    min_reward: float | None = None,
) -> list[dict[str, object]]:
    """Keep the top ``quantile`` fraction of samples (e.g. ``0.3`` -> top 30%).

    Also drops anything below ``min_reward`` if provided.
    """
    if not samples:
        return []
    ordered = sorted(samples, key=lambda s: float(s.get("reward", 0.0)))
    frac = max(0.0, min(1.0, quantile))
    k = max(1, int(round(len(ordered) * frac)))
    kept = ordered[-k:]
    if min_reward is not None:
        kept = [s for s in kept if float(s.get("reward", 0.0)) >= min_reward]
    return kept


def write_jsonl(samples: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_dataset(
    n: int,
    *,
    quantile: float = 0.5,
    min_reward: float | None = 0.0,
    scenarios: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[dict[str, object]]:
    """One-shot: generate then filter."""
    return filter_top(
        generate_samples(n, scenarios=scenarios, seed=seed),
        quantile=quantile,
        min_reward=min_reward,
    )


__all__ = [
    "CURRICULUM",
    "TRAIN_SCENARIOS",
    "build_dataset",
    "filter_top",
    "generate_samples",
    "write_jsonl",
]
