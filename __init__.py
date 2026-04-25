# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ghostexec Environment."""

from .models import GhostexecAction, GhostexecObservation

# Importing ghostexec.models in notebooks should not require websocket client deps.
# Keep client import optional so package imports survive OpenEnv layout differences.
try:
    from .client import GhostexecEnv
except Exception:  # pragma: no cover - import-compat shim
    GhostexecEnv = None  # type: ignore[assignment]

__all__ = ["GhostexecAction", "GhostexecObservation"]
if GhostexecEnv is not None:
    __all__.append("GhostexecEnv")
