# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for GhostExec — all world and API types live here."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import Action as _OpenEnvAction
    from openenv.core.env_server.types import Observation as _OpenEnvObservation
except Exception:
    _OpenEnvAction = BaseModel  # type: ignore[assignment]
    _OpenEnvObservation = BaseModel  # type: ignore[assignment]


def _is_pydantic_model_class(cls: object) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False


# Some OpenEnv builds expose dataclass-style Action/Observation that do not accept
# additional keyword fields, which breaks GhostexecAction/GhostexecObservation
# construction in Colab. Fall back to BaseModel in that case.
ActionBase = _OpenEnvAction if _is_pydantic_model_class(_OpenEnvAction) else BaseModel
ObservationBase = (
    _OpenEnvObservation if _is_pydantic_model_class(_OpenEnvObservation) else BaseModel
)

# --- Aliases for scenario / world strings ---

EmailPriority = Literal["critical", "high", "normal", "low"]
SenderRelationship = Literal["VIP", "personal", "professional", "unknown"]
ContactRelationship = Literal[
    "board_member",
    "spouse",
    "investor",
    "direct_report",
    "client",
    "friend",
    "team_member",
]
CommPreference = Literal["email", "text", "call"]
Mood = Literal["happy", "neutral", "annoyed", "angry", "furious"]
TaskStatus = Literal["pending", "in-progress", "done", "overdue"]
Effort = Literal["low", "medium", "high"]
MeetingPriority = Literal["critical", "high", "normal", "low"]

GhostexecActionType = Literal[
    "reply_email",
    "archive_email",
    "reschedule_meeting",
    "cancel_meeting",
    "complete_task",
    "delegate_task",
    "send_message",
    "do_nothing",
]


class Email(BaseModel):
    """Single inbox message."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sender: str
    subject: str
    body: str
    read: bool = False
    replied: bool = False
    priority: EmailPriority
    sender_relationship: SenderRelationship


class Meeting(BaseModel):
    """Calendar block."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    start: str = Field(..., description="ISO 8601 start datetime")
    duration_minutes: int = Field(..., ge=1)
    attendees: list[str] = Field(default_factory=list)
    location: str = ""
    priority: MeetingPriority = "normal"
    cancelled: bool = False


class Contact(BaseModel):
    """Stakeholder in the exec's network."""

    model_config = ConfigDict(extra="forbid")

    name: str
    relationship_type: ContactRelationship
    communication_preference: CommPreference
    importance: int = Field(..., ge=1, le=5)
    mood: Mood = "neutral"


class Task(BaseModel):
    """To-do item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    deadline: str = Field(..., description="ISO 8601 deadline")
    owner: str
    status: TaskStatus = "pending"
    effort: Effort = "medium"
    delegated_to: str | None = None


class WorldState(BaseModel):
    """Full simulated world — JSON-serialisable."""

    model_config = ConfigDict(extra="forbid")

    simulation_time: str = Field(..., description="Current simulated instant, ISO 8601")
    stress: int = Field(default=0, ge=0, le=100)
    active_conflicts: list[str] = Field(default_factory=list)
    action_log: list[str] = Field(default_factory=list)
    episode_active: bool = True
    episode_end_reason: str | None = None
    max_episode_steps: int = Field(default=48, ge=1, le=10_000)
    emails: list[Email] = Field(default_factory=list)
    meetings: list[Meeting] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)


class GhostexecAction(ActionBase):
    """
    Legal agent actions (Phase 3). Unknown HTTP payloads default to do_nothing
    so older clients do not crash deserialization.
    """

    action_type: GhostexecActionType = Field(
        default="do_nothing",
        description="Which legal action to execute this step",
    )
    email_id: str = ""
    message_body: str = ""
    meeting_id: str = ""
    new_time: str = ""
    reason: str = ""
    task_id: str = ""
    contact_name: str = ""
    message: str = Field(default="", description="Optional note for action_log (legacy / debug)")

    @model_validator(mode="before")
    @classmethod
    def _default_action_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_type" not in data:
            data = {**data, "action_type": "do_nothing"}
        return data


class GhostexecObservation(ObservationBase):
    """
    Primary LLM-facing field is `echoed_message`: full plain-text briefing (Phase 3).
    """

    # Keep these fields explicit for compatibility with OpenEnv builds where
    # Observation is not a pydantic base carrying done/reward/metadata.
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    echoed_message: str = Field(
        default="",
        description="Human-readable briefing text for the LLM (not JSON)",
    )
    message_length: int = Field(default=0, description="Byte length of echoed_message for quick checks")


class RewardBreakdown(BaseModel):
    """Phase 4 reward components (logged and exposed in observation metadata)."""

    model_config = ConfigDict(extra="forbid")

    conflict_raw: float = 0.0
    critical_queue_bonus: float = 0.0
    conflict: float = 0.0
    relationship: float = 0.0
    task: float = 0.0
    shaping_synergy: float = 0.0
    shaping_tradeoff: float = 0.0
    shaping_potential: float = 0.0
    shaping_scaffold: float = 0.0
    shaping_quality: float = 0.0
    shaping_total: float = 0.0
    shaping_to_base_ratio: float = 0.0
    weighted_base: float = 0.0
    output_scale: float = 1.0
    invalid_step_adjustment: float = 0.0
    episode_completion_bonus: float = 0.0
    catastrophic_penalty: float = 0.0
    do_nothing_floor: float = 0.0
    final: float = 0.0
