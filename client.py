# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ghostexec Environment Client."""

from typing import Any, Dict

try:
    # OpenEnv newer layout.
    from openenv.client import EnvClient
except ImportError:
    # Backward compatibility with older OpenEnv versions.
    from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import GhostexecAction, GhostexecObservation


class GhostexecEnv(
    EnvClient[GhostexecAction, GhostexecObservation, State]
):
    """
    Client for the Ghostexec Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: GhostexecAction) -> Dict[str, Any]:
        payload = action.model_dump(mode="json")
        if not payload.get("metadata"):
            payload.pop("metadata", None)
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[GhostexecObservation]:
        obs_data = payload.get("observation", {})
        observation = GhostexecObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
