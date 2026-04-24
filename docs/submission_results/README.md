# Submission artifacts (hackathon)

Judges and the README expect **committed** evidence of training, not only files under gitignored `outputs/`.

After a successful `training/ghostexec_unsloth_grpo_colab.ipynb` run, run:

```bash
uv run python scripts/export_submission_plots.py
```

This exports committable gatekeeper plots into this folder:

| Suggested filename | Source |
|--------------------|--------|
| `reward_curve.png` | `outputs/logs/episode_rewards.jsonl` |
| `loss_curve.png` | `outputs/training/**/trainer_state.json` |

You can also copy these optional diagnostics from `outputs/plots/` and link them from the root `README.md`:

| Suggested filename | Source (from a full notebook run) |
|--------------------|-----------------------------------|
| `grpo_diagnostics.png` | `outputs/plots/grpo_diagnostics.png` |
| `grpo_reward_curve.png` | `outputs/plots/grpo_reward_curve.png` |
| `before_after_eval.png` | `outputs/plots/before_after_eval.png` |
| `before_after_channels.png` | `outputs/plots/before_after_channels.png` |

Optional: add `before_after.csv` or a one-line summary table in the main README.

Keep files **small** (PNG at ~140 DPI is enough). Do not commit large videos here — use YouTube or Hugging Face post URLs in the README instead.
