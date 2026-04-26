from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _load_trainer_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("log_history"), list):
        return [x for x in data["log_history"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _load_baselines(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "rewards" in data and isinstance(data["rewards"], dict):
        data = data["rewards"]
    out: dict[str, float] = {}
    for k in ("random", "frozen", "trained", "random_mean", "frozen_mean", "trained_mean"):
        if k in data:
            v = data[k]
            name = k.replace("_mean", "")
            out[name] = float(v)
    return out


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_loss(history: list[dict[str, Any]], out_dir: Path) -> bool:
    rows = []
    for i, h in enumerate(history):
        step = h.get("step", h.get("global_step", i))
        if "loss" in h:
            rows.append((float(step), float(h["loss"])))
    if not rows:
        return False
    df = pd.DataFrame(rows, columns=["step", "loss"]).sort_values("step")
    plt.figure(figsize=(9, 4.8))
    plt.plot(df["step"], df["loss"], label="train_loss")
    plt.xlabel("global step")
    plt.ylabel("loss")
    plt.title("Ghostexec training loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()
    return True


def _plot_reward_components(reward_csv: Path, out_dir: Path) -> tuple[bool, bool]:
    if not reward_csv.exists():
        return False, False
    df = pd.read_csv(reward_csv)
    if "global_step" not in df.columns:
        return False, False

    made_reward_curve = False
    for col in ("env", "reward", "mean_reward"):
        if col in df.columns:
            plt.figure(figsize=(9, 4.8))
            plt.plot(df["global_step"], df[col], label=col)
            plt.xlabel("global step")
            plt.ylabel("reward")
            plt.title("Ghostexec reward vs step")
            plt.grid(alpha=0.2)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "reward_curve.png", dpi=150)
            plt.close()
            made_reward_curve = True
            break

    component_cols = [c for c in ("env", "fmt", "semantic", "idle") if c in df.columns]
    if len(component_cols) >= 2:
        plt.figure(figsize=(9, 4.8))
        for c in component_cols:
            plt.plot(df["global_step"], df[c], label=c)
        plt.xlabel("global step")
        plt.ylabel("mean component reward")
        plt.title("Reward components vs step")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "components_curve.png", dpi=150)
        plt.close()
        return made_reward_curve, True
    return made_reward_curve, False


def _plot_baseline_bars(baselines: dict[str, float], out_dir: Path) -> bool:
    needed = ("random", "frozen", "trained")
    if not all(k in baselines for k in needed):
        return False
    names = list(needed)
    vals = [baselines[n] for n in names]
    colors = ["#888888", "#1f77b4", "#2ca02c"]
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(names, vals, color=colors)
    plt.ylabel("mean episode reward (higher is better)")
    plt.title("Ghostexec: random vs frozen vs trained")
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_comparison.png", dpi=150)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate post-training Ghostexec plots.")
    parser.add_argument(
        "--trainer-history",
        type=Path,
        default=Path("outputs/trainer_state.json"),
        help="JSON with HF/Unsloth log history (trainer_state.json or list of logs).",
    )
    parser.add_argument(
        "--reward-csv",
        type=Path,
        default=Path("outputs/reward_log.csv"),
        help="CSV containing global_step and reward columns.",
    )
    parser.add_argument(
        "--baselines-json",
        type=Path,
        default=Path("outputs/compliance_manifest.json"),
        help="JSON containing random/frozen/trained means (or rewards object).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory to save plot PNGs.",
    )
    args = parser.parse_args()

    _ensure_dir(args.out_dir)
    history = _load_trainer_history(args.trainer_history)
    baselines = _load_baselines(args.baselines_json)

    made_loss = _plot_loss(history, args.out_dir)
    made_reward, made_components = _plot_reward_components(args.reward_csv, args.out_dir)
    made_bars = _plot_baseline_bars(baselines, args.out_dir)

    print("Generated plots:")
    print(f"- loss_curve.png: {'yes' if made_loss else 'no (missing loss history)'}")
    print(f"- reward_curve.png: {'yes' if made_reward else 'no (missing reward csv columns)'}")
    print(
        f"- components_curve.png: {'yes' if made_components else 'no (missing component columns)'}"
    )
    print(
        f"- baseline_comparison.png: {'yes' if made_bars else 'no (missing random/frozen/trained means)'}"
    )
    print(f"Output directory: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
