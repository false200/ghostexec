from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import requests
from transformers import TrainerCallback


LEGAL_ACTION_TYPES = [
    "reply_email",
    "archive_email",
    "reschedule_meeting",
    "cancel_meeting",
    "complete_task",
    "delegate_task",
    "send_message",
    "do_nothing",
]

MODEL_PRESETS: dict[str, str] = {
    # Fast iteration winner preset: small, strong instruction following, QLoRA-friendly.
    "small_iter_fast": "unsloth/Qwen2.5-3B-Instruct",
    # Existing baseline used in this repo.
    "balanced_3b": "unsloth/Llama-3.2-3B-Instruct",
    # Larger option when compute budget is stable.
    "bigger_4b": "unsloth/Qwen3-4B-Instruct-2507",
}

TRAINING_PRESETS: dict[str, dict[str, float | int | str]] = {
    "hackathon_turbo": {
        "max_sft_steps": 80,
        "max_grpo_steps": 180,
        "env_reward_scale": 1.00,
        "local_reward_scale": 0.45,
        "complexity_curriculum": "easy_to_full",
        "curriculum_ramp_ratio": 0.65,
        "sft_samples": 180,
        # Optimizer / schedule knobs (stability-first for iterative winning runs)
        "sft_lr": 1.2e-5,
        "sft_grad_accum": 8,
        "grpo_lr": 3.0e-6,
        "grpo_grad_accum": 8,
        "grpo_beta": 0.08,
        "reward_ema_decay": 0.35,
    },
    # Quicker loop for smoke iterations on weaker hardware.
    "quick_smoke": {
        "max_sft_steps": 30,
        "max_grpo_steps": 80,
        "env_reward_scale": 0.95,
        "local_reward_scale": 0.35,
        "complexity_curriculum": "easy_to_full",
        "curriculum_ramp_ratio": 0.50,
        "sft_samples": 90,
        "sft_lr": 1.5e-5,
        "sft_grad_accum": 4,
        "grpo_lr": 4.0e-6,
        "grpo_grad_accum": 4,
        "grpo_beta": 0.06,
        "reward_ema_decay": 0.25,
    },
}


def _extract_briefing(reset_payload: dict[str, Any]) -> str:
    obs = reset_payload.get("observation", reset_payload)
    if isinstance(obs, dict):
        return str(obs.get("echoed_message", "")).strip()
    return ""


def _legal_action_heuristic(briefing: str) -> dict[str, Any]:
    # Minimal heuristic used only for SFT warm-start data generation.
    # Keeps the action schema valid and non-idle-biased.
    lower = briefing.lower()
    if "e01" in lower:
        return {
            "action_type": "reply_email",
            "email_id": "e01",
            "message_body": "Acknowledged. Sharing a concise update shortly.",
        }
    if "m02" in lower:
        return {
            "action_type": "reschedule_meeting",
            "meeting_id": "m02",
            "new_time": "2026-04-21T18:00:00",
            "reason": "Resolve overlap with higher priority commitments.",
        }
    if "t06" in lower:
        return {"action_type": "complete_task", "task_id": "t06"}
    return {"action_type": random.choice(LEGAL_ACTION_TYPES)}


def generate_sft_jsonl_from_env(
    env_url: str,
    out_jsonl: Path,
    samples: int = 120,
    task_id: str = "phase2_core",
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for _ in range(samples):
        r = requests.post(f"{env_url.rstrip('/')}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        briefing = _extract_briefing(payload)
        if not briefing:
            continue
        action = _legal_action_heuristic(briefing)
        prompt = (
            "You are Ghostexec AI Chief-of-Staff.\n"
            "Output one valid GhostexecAction JSON only.\n\n"
            f"{briefing}"
        )
        rows.append({"prompt": prompt, "completion": json.dumps(action, ensure_ascii=True)})
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} SFT rows to {out_jsonl}")


def run_sft_then_grpo(
    model_name: str,
    env_url: str,
    sft_jsonl: Path,
    out_dir: Path,
    env_reward_scale: float,
    local_reward_scale: float,
    max_sft_steps: int,
    max_grpo_steps: int,
    complexity_curriculum: str,
    curriculum_ramp_ratio: float,
    *,
    sft_lr: float,
    sft_grad_accum: int,
    grpo_lr: float,
    grpo_grad_accum: int,
    grpo_beta: float,
    reward_ema_decay: float,
) -> None:
    try:
        from datasets import load_dataset
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing training deps. Install unsloth, trl, datasets, transformers before running."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    def _trainable_lora_sum_abs(model) -> float:
        total = 0.0
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora" not in n.lower():
                continue
            total += float(p.detach().abs().sum().item())
        return total

    policy, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    policy = FastLanguageModel.get_peft_model(
        policy,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    ds = load_dataset("json", data_files=str(sft_jsonl), split="train")
    sft_cfg = SFTConfig(
        output_dir=str(out_dir / "sft"),
        max_steps=max_sft_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=sft_grad_accum,
        learning_rate=sft_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        logging_steps=5,
        save_steps=max(10, max_sft_steps),
        report_to=[],
    )
    sft_trainer = SFTTrainer(
        model=policy,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=sft_cfg,
        dataset_text_field="prompt",
        formatting_func=lambda ex: [f"{p}\n\n{c}" for p, c in zip(ex["prompt"], ex["completion"])],
    )
    sft_before = _trainable_lora_sum_abs(policy)
    sft_trainer.train()
    sft_after = _trainable_lora_sum_abs(sft_trainer.model)
    sft_delta = abs(sft_after - sft_before)
    print(f"SFT LoRA delta(abs-sum): {sft_delta:.6f}")
    if sft_delta <= 1e-6:
        raise RuntimeError("SFT appears not to have updated LoRA weights (delta too small).")
    sft_dir = out_dir / "sft_adapter"
    sft_trainer.model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"SFT complete. Adapter saved: {sft_dir}")

    def _extract_json(text: str) -> dict[str, Any] | None:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _env_step_reward_from_completion(text: str) -> float:
        payload = _extract_json(text)
        if payload is None:
            return -0.25
        payload.setdefault("action_type", "do_nothing")
        try:
            r = requests.post(f"{env_url.rstrip('/')}/reset", json={"task_id": "phase2_core"}, timeout=30)
            r.raise_for_status()
            s = requests.post(
                f"{env_url.rstrip('/')}/step",
                json={"action": payload},
                timeout=30,
            )
            s.raise_for_status()
            raw = s.json()
        except Exception:
            return 0.0
        rew = raw.get("reward")
        if rew is None and isinstance(raw.get("observation"), dict):
            rew = raw["observation"].get("reward", 0.0)
        try:
            return float(rew)
        except Exception:
            return 0.0

    progress = {"step": 0, "total": max(1, max_grpo_steps)}
    reward_ema_state = {"env": None}

    class _ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
            progress["step"] = int(getattr(state, "global_step", progress["step"]))
            return control

    def _progress_frac() -> float:
        return min(1.0, progress["step"] / progress["total"])

    def _curriculum_phase_weight() -> float:
        frac = _progress_frac()
        ramp = max(0.05, min(1.0, curriculum_ramp_ratio))
        if complexity_curriculum == "off":
            return 1.0
        # easy_to_full: start with strong scaffold guidance, then smoothly
        # transition to full env-dominant optimization.
        if frac >= ramp:
            return 0.0
        return max(0.0, 1.0 - (frac / ramp))

    def _annealed_local_scale() -> float:
        frac = _progress_frac()
        base = local_reward_scale * (1.20 - 0.70 * frac)
        return base * (1.0 + 0.70 * _curriculum_phase_weight())

    def _annealed_env_scale() -> float:
        w = _curriculum_phase_weight()
        # Slightly downweight env reward in early easy phase to reduce variance,
        # then recover to full strength by the end of ramp.
        return env_reward_scale * (1.0 - 0.30 * w)

    def env_reward(completions, **_):
        scale = _annealed_env_scale()
        raw = [scale * _env_step_reward_from_completion(str(c)) for c in completions]
        if reward_ema_decay <= 0.0:
            return raw
        batch_mean = sum(raw) / max(len(raw), 1)
        prev = reward_ema_state["env"]
        d = max(0.0, min(1.0, reward_ema_decay))
        if prev is None:
            smoothed_mean = batch_mean
        else:
            smoothed_mean = (1.0 - d) * prev + d * batch_mean
        reward_ema_state["env"] = smoothed_mean
        delta = smoothed_mean - batch_mean
        return [r + delta for r in raw]

    def format_reward(completions, **_):
        scale = _annealed_local_scale()
        outs: list[float] = []
        for c in completions:
            txt = str(c).strip()
            obj = _extract_json(txt)
            if obj is None:
                outs.append(-0.20 * scale)
                continue
            if obj.get("action_type") not in LEGAL_ACTION_TYPES:
                outs.append(-0.20 * scale)
                continue
            # Encourage concise, parseable schema-correct JSON.
            length_pen = -0.04 * scale if len(txt) > 500 else 0.0
            outs.append(0.12 * scale + length_pen)
        return outs

    def semantic_action_reward(completions, prompts=None, **_):
        scale = _annealed_local_scale()
        outs: list[float] = []
        for i, c in enumerate(completions):
            obj = _extract_json(str(c))
            if obj is None:
                outs.append(-0.10 * scale)
                continue
            at = str(obj.get("action_type", ""))
            ptxt = str(prompts[i] if prompts and i < len(prompts) else "").lower()
            bonus = 0.0
            if "critical" in ptxt and at == "reply_email":
                bonus += 0.08
            if "clash" in ptxt and at in ("reschedule_meeting", "cancel_meeting"):
                bonus += 0.08
            if ("overdue" in ptxt or "due soon" in ptxt) and at in ("complete_task", "delegate_task"):
                bonus += 0.08
            outs.append(scale * bonus)
        return outs

    def anti_idle_reward(completions, **_):
        scale = _annealed_local_scale()
        outs = []
        for c in completions:
            txt = str(c).lower()
            outs.append((-0.20 if "do_nothing" in txt else 0.02) * scale)
        return outs

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir / "grpo"),
        learning_rate=grpo_lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grpo_grad_accum,
        max_steps=max_grpo_steps,
        logging_steps=5,
        num_generations=2,
        beta=grpo_beta,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        report_to=[],
    )
    grpo_trainer = GRPOTrainer(
        model=sft_trainer.model,
        processing_class=tokenizer,
        reward_funcs=[env_reward, format_reward, semantic_action_reward, anti_idle_reward],
        train_dataset=ds,
        args=grpo_cfg,
        callbacks=[_ProgressCallback()],
    )
    grpo_before = _trainable_lora_sum_abs(sft_trainer.model)
    grpo_trainer.train()
    progress["step"] = progress["total"]
    grpo_after = _trainable_lora_sum_abs(grpo_trainer.model)
    grpo_delta = abs(grpo_after - grpo_before)
    print(f"GRPO LoRA delta(abs-sum): {grpo_delta:.6f}")
    if grpo_delta <= 1e-6:
        raise RuntimeError("GRPO appears not to have updated LoRA weights (delta too small).")
    final_dir = out_dir / "grpo_adapter"
    grpo_trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"GRPO complete. Adapter saved: {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT warmup before GRPO.")
    parser.add_argument(
        "--model-name",
        default="",
        help="Optional explicit model id. If omitted, --model-preset is used.",
    )
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default="small_iter_fast",
        help="Recommended compute-aware preset. small_iter_fast is best for iteration speed.",
    )
    parser.add_argument(
        "--training-preset",
        choices=sorted(TRAINING_PRESETS.keys()),
        default="hackathon_turbo",
        help="Compute-aware run preset. hackathon_turbo is best default for iterative winning loops.",
    )
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sft-jsonl", type=Path, default=Path("outputs/sft_from_env.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/train_runs/sft_then_grpo"))
    parser.add_argument("--generate-sft-from-env", action="store_true")
    parser.add_argument("--sft-samples", type=int, default=120)
    parser.add_argument("--max-sft-steps", type=int, default=60)
    parser.add_argument("--max-grpo-steps", type=int, default=120)
    parser.add_argument("--env-reward-scale", type=float, default=1.0)
    parser.add_argument("--local-reward-scale", type=float, default=0.35)
    parser.add_argument(
        "--complexity-curriculum",
        choices=["off", "easy_to_full"],
        default="easy_to_full",
        help="Reward curriculum: easy_to_full starts with stronger local scaffold and anneals to env-dominant.",
    )
    parser.add_argument(
        "--curriculum-ramp-ratio",
        type=float,
        default=0.60,
        help="Fraction of GRPO steps used to ramp from easy scaffold to full env weighting.",
    )
    parser.add_argument(
        "--reward-ema-decay",
        type=float,
        default=-1.0,
        help="EMA decay in [0,1] for env reward smoothing; -1 uses training preset default.",
    )
    args = parser.parse_args()
    model_name = args.model_name.strip() or MODEL_PRESETS[args.model_preset]
    p = TRAINING_PRESETS[args.training_preset]
    max_sft_steps = int(p["max_sft_steps"])
    max_grpo_steps = int(p["max_grpo_steps"])
    env_reward_scale = float(p["env_reward_scale"])
    local_reward_scale = float(p["local_reward_scale"])
    complexity_curriculum = str(p["complexity_curriculum"])
    curriculum_ramp_ratio = float(p["curriculum_ramp_ratio"])
    sft_samples = int(p["sft_samples"])
    sft_lr = float(p["sft_lr"])
    sft_grad_accum = int(p["sft_grad_accum"])
    grpo_lr = float(p["grpo_lr"])
    grpo_grad_accum = int(p["grpo_grad_accum"])
    grpo_beta = float(p["grpo_beta"])
    reward_ema_decay = float(p["reward_ema_decay"])
    if args.max_sft_steps != 60:
        max_sft_steps = args.max_sft_steps
    if args.max_grpo_steps != 120:
        max_grpo_steps = args.max_grpo_steps
    if args.env_reward_scale != 1.0:
        env_reward_scale = args.env_reward_scale
    if args.local_reward_scale != 0.35:
        local_reward_scale = args.local_reward_scale
    if args.complexity_curriculum != "easy_to_full":
        complexity_curriculum = args.complexity_curriculum
    if args.curriculum_ramp_ratio != 0.60:
        curriculum_ramp_ratio = args.curriculum_ramp_ratio
    if args.sft_samples != 120:
        sft_samples = args.sft_samples
    if args.reward_ema_decay >= 0.0:
        reward_ema_decay = float(args.reward_ema_decay)
    print(f"Model preset: {args.model_preset} -> {model_name}")
    print(
        "Training preset:"
        f" {args.training_preset} -> sft={max_sft_steps}, grpo={max_grpo_steps},"
        f" env_scale={env_reward_scale}, local_scale={local_reward_scale},"
        f" curriculum={complexity_curriculum}, ramp={curriculum_ramp_ratio}"
    )

    if args.generate_sft_from_env or not args.sft_jsonl.exists():
        generate_sft_jsonl_from_env(
            env_url=args.env_url,
            out_jsonl=args.sft_jsonl,
            samples=sft_samples,
            task_id="phase2_core",
        )

    run_sft_then_grpo(
        model_name=model_name,
        env_url=args.env_url,
        sft_jsonl=args.sft_jsonl,
        out_dir=args.out_dir,
        env_reward_scale=env_reward_scale,
        local_reward_scale=local_reward_scale,
        max_sft_steps=max_sft_steps,
        max_grpo_steps=max_grpo_steps,
        complexity_curriculum=complexity_curriculum,
        curriculum_ramp_ratio=curriculum_ramp_ratio,
        sft_lr=sft_lr,
        sft_grad_accum=sft_grad_accum,
        grpo_lr=grpo_lr,
        grpo_grad_accum=grpo_grad_accum,
        grpo_beta=grpo_beta,
        reward_ema_decay=reward_ema_decay,
    )


if __name__ == "__main__":
    main()
