# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Ghostexec Environment.

This module creates an HTTP server that exposes the GhostexecEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path

try:
    import openenv.core.env_server.http_server as _openenv_http
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# OpenEnv's serialize_observation drops `metadata` from the JSON body; Ghostexec
# trainers and live tests rely on step_ok / ids inside observation.metadata.
_orig_serialize_observation = _openenv_http.serialize_observation


def _ghostexec_serialize_observation(observation):  # type: ignore[no-untyped-def]
    payload = _orig_serialize_observation(observation)
    inner = payload.get("observation")
    if isinstance(inner, dict):
        meta = getattr(observation, "metadata", None) or {}
        inner["metadata"] = _openenv_http._make_json_serializable(meta)
    return payload


_openenv_http.serialize_observation = _ghostexec_serialize_observation

from openenv.core.env_server.http_server import create_app  # noqa: E402


def _configure_openenv_readme_path() -> None:
    """Ensure OpenEnv's web UI can resolve README (Gradio accordion).

    OpenEnv tries /app/README.md first, then ENV_README_PATH. The Dockerfile
    copies README to /app/README.md; this also sets ENV_README_PATH when a
    preset points at a missing file (common on some hosts) so the loader does
    not skip valid fallbacks.
    """
    cur = os.environ.get("ENV_README_PATH")
    if cur:
        try:
            p = Path(cur)
            if p.is_file() and p.stat().st_size > 0:
                return
        except OSError:
            pass
        os.environ.pop("ENV_README_PATH", None)

    _here = Path(__file__).resolve()
    for candidate in (
        Path("/app/env/README.md"),
        Path("/app/README.md"),
        _here.parent.parent / "README.md",
        Path.cwd() / "README.md",
    ):
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                os.environ["ENV_README_PATH"] = str(candidate)
                return
        except OSError:
            continue


_configure_openenv_readme_path()

try:
    # Editable / normal install (package name `ghostexec`).
    from ghostexec.models import GhostexecAction, GhostexecObservation
    from ghostexec.server.ghostexec_environment import GhostexecEnvironment
except ImportError:
    # Plain `uvicorn server.app:app` from repo root: top-level `models` + `server` package.
    from models import GhostexecAction, GhostexecObservation
    from server.ghostexec_environment import GhostexecEnvironment


# Create the app with web interface and README integration
app = create_app(
    GhostexecEnvironment,
    GhostexecAction,
    GhostexecObservation,
    env_name="ghostexec",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def _patch_openapi_ghostexec_examples(schema: dict) -> None:
    """Replace OpenEnv's generic observation examples with GhostExec's plain-text briefing shape."""
    briefing = (
        "=== GHOSTEXEC BRIEFING — Tue 21 Apr 2026 08:00 ===\n\n"
        "UNREAD EMAILS (…): …\n\n"
        "CALENDAR CONFLICTS IN NEXT 4 HOURS: …\n\n"
        "CONTACTS TO WATCH: …\n\n"
        "OVERDUE OR DUE-SOON TASKS: …\n\n"
        "EXEC STRESS LEVEL: 52/100\n"
        "STEPS REMAINING: 48"
    )
    obs = {"echoed_message": briefing, "message_length": len(briefing)}
    reset_ex = {"observation": obs, "reward": 0.0, "done": False}
    step_ex = {"observation": obs, "reward": -0.42, "done": False}
    for path, example in (("/reset", reset_ex), ("/step", step_ex)):
        try:
            cell = (
                schema["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]
            )
            if isinstance(cell, dict):
                cell["example"] = example
        except KeyError:
            continue


_OPENAPI_HTTP_EPISODE_SENTINEL = "Ghostexec / OpenEnv HTTP"

_OPENAPI_HTTP_EPISODE_NOTE = f"""
---
## {_OPENAPI_HTTP_EPISODE_SENTINEL}

Each `POST /reset` and `POST /step` may run on a **new** environment instance, so
separate HTTP requests do **not** share one in-memory episode across calls. A lone
`POST /step` still applies your action once (after internal scenario load). For
**many steps on the same episode**, use **WebSocket `/ws`**: open a connection,
reset once, then send many step messages on that same socket. See **ghostexec/README.md**
for details.
"""


def _patch_openapi_ghostexec_http_note(schema: dict) -> None:
    """Document HTTP statelessness vs /ws so Swagger and OpenAPI clients see it."""
    try:
        info = schema.get("info")
        if not isinstance(info, dict):
            return
        desc = info.get("description") or ""
        if _OPENAPI_HTTP_EPISODE_SENTINEL in desc:
            return
        info["description"] = desc + _OPENAPI_HTTP_EPISODE_NOTE
    except (TypeError, KeyError):
        return


_fastapi_openapi = type(app).openapi.__get__(app, type(app))


def _ghostexec_openapi() -> dict:
    if app.openapi_schema is None:
        _fastapi_openapi()
        _patch_openapi_ghostexec_examples(app.openapi_schema)
        _patch_openapi_ghostexec_http_note(app.openapi_schema)
    return app.openapi_schema  # type: ignore[return-value]


app.openapi = _ghostexec_openapi  # type: ignore[method-assign]


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m ghostexec.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn ghostexec.server.app:app --workers 4
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="GhostExec OpenEnv HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Listen port")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
