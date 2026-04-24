"""Random-policy baseline for held-out Ghostexec evaluation."""

from __future__ import annotations

import random
from typing import get_args
from typing import Any

from .eval_harness import EvalReport, evaluate

try:
    from ghostexec.models import GhostexecAction, GhostexecObservation
except ImportError:
    from models import GhostexecAction, GhostexecObservation


def _random_action(obs: GhostexecObservation, rng: random.Random) -> GhostexecAction:
    meta: dict[str, Any] = obs.metadata or {}
    state: dict[str, Any] = meta.get("world_state") or {}
    email_ids = [str(e.get("id", "")) for e in state.get("emails", []) if isinstance(e, dict)]
    task_ids = [str(t.get("id", "")) for t in state.get("tasks", []) if isinstance(t, dict)]
    meeting_ids = [str(m.get("id", "")) for m in state.get("meetings", []) if isinstance(m, dict)]
    contact_names = [str(c.get("name", "")) for c in state.get("contacts", []) if isinstance(c, dict)]

    action_type = rng.choice(list(get_args(GhostexecAction.model_fields["action_type"].annotation)))
    kwargs: dict[str, str] = {"action_type": action_type}

    if action_type in {"reply_email", "archive_email"} and email_ids:
        kwargs["email_id"] = rng.choice(email_ids)
    if action_type == "reply_email":
        kwargs["message_body"] = "Ack. I will follow up shortly."
    if action_type in {"reschedule_meeting", "cancel_meeting"} and meeting_ids:
        kwargs["meeting_id"] = rng.choice(meeting_ids)
        kwargs["reason"] = "Priority conflict."
    if action_type == "reschedule_meeting":
        kwargs["new_time"] = "2026-04-27T11:00:00Z"
    if action_type in {"complete_task", "delegate_task"} and task_ids:
        kwargs["task_id"] = rng.choice(task_ids)
    if action_type in {"delegate_task", "send_message"} and contact_names:
        kwargs["contact_name"] = rng.choice(contact_names)
    if action_type == "send_message":
        kwargs["message"] = "Quick status check."

    return GhostexecAction(**kwargs)


def evaluate_random_policy(
    *,
    episodes_per_scenario: int = 3,
    max_steps: int = 8,
    seed: int = 2026,
) -> EvalReport:
    """Evaluate a random legal-action baseline with deterministic sampling."""
    rng = random.Random(seed)

    def policy(obs: GhostexecObservation, _: random.Random) -> GhostexecAction:
        return _random_action(obs, rng)

    return evaluate(policy, episodes_per_scenario=episodes_per_scenario, max_steps=max_steps, base_seed=seed)


__all__ = ["evaluate_random_policy"]
