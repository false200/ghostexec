# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# End-to-end stack test: FastAPI/OpenEnv HTTP + WebSocket, GhostExec env,
# training smoke, and (optionally) GhostexecEnv client over ASGI TestClient.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from ghostexec.models import GhostexecAction
from ghostexec.server.app import app
from ghostexec.server.ghostexec_environment import GhostexecEnvironment

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "phase2_core.json"
MONDAY = ROOT / "scenarios" / "monday_morning.json"


def _train_subprocess(train_args: list[str]) -> tuple[list[str], dict[str, str] | None]:
    """Prefer `uv run python` so the child process sees the project venv; else set PYTHONPATH."""
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python", str(ROOT / "training" / "train.py"), *train_args], None
    prev = os.environ.get("PYTHONPATH", "")
    merged = os.pathsep.join([str(ROOT), prev]).strip(os.pathsep)
    return [sys.executable, str(ROOT / "training" / "train.py"), *train_args], {**os.environ, "PYTHONPATH": merged}


def _http_paths(client: TestClient) -> set[str]:
    paths: set[str] = set()
    for r in app.routes:
        p = getattr(r, "path", None)
        if isinstance(p, str) and p:
            paths.add(p)
    return paths


def test_server_app_import_matches_uvicorn_server_string() -> None:
    """`uvicorn server.app:app` loads `server.app` with cwd on path (no `ghostexec.` prefix)."""
    rc = subprocess.run(
        [sys.executable, "-c", "import server.app; assert server.app.app is not None"],
        cwd=str(ROOT),
        check=False,
    )
    assert rc.returncode == 0, "import server.app must work from ghostexec repo root"


def test_openapi_docs_and_schema_discovery() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec.get("openapi")
        assert "paths" in spec and spec["paths"]

        for path in ("/docs", "/redoc"):
            resp = client.get(path)
            assert resp.status_code == 200
            assert len(resp.text) > 100


def test_openapi_examples_match_ghostexec_observation_shape() -> None:
    spec = app.openapi()
    for path in ("/reset", "/step"):
        ex = spec["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["example"]
        obs = ex["observation"]
        assert "echoed_message" in obs and "message_length" in obs
        assert "status" not in obs and "data" not in obs
        assert "reward" in ex and "done" in ex


def test_openapi_info_documents_http_vs_websocket_episode() -> None:
    """Runtime-visible API docs: HTTP reset/step are not one persistent episode; /ws is."""
    spec = app.openapi()
    desc = spec.get("info", {}).get("description") or ""
    assert "Ghostexec / OpenEnv HTTP" in desc
    assert "/ws" in desc and "WebSocket" in desc


def test_all_registered_get_post_routes_smoke() -> None:
    """Smoke every stable OpenEnv HTTP route (simulation mode, no Gradio /web)."""
    with TestClient(app, raise_server_exceptions=True) as client:
        paths = _http_paths(client)
        assert "/health" in paths
        assert "/metadata" in paths
        assert "/schema" in paths
        assert "/state" in paths
        assert "/reset" in paths
        assert "/step" in paths
        assert "/ws" in paths
        assert "/mcp" in paths

        h = client.get("/health")
        assert h.status_code == 200
        assert h.json().get("status") == "healthy"

        meta = client.get("/metadata")
        assert meta.status_code == 200
        body = meta.json()
        assert body.get("name") in ("ghostexec", "GhostexecEnvironment")
        assert "description" in body

        st = client.get("/state")
        assert st.status_code == 200
        assert "step_count" in st.json()

        sch = client.get("/schema")
        assert sch.status_code == 200
        sj = sch.json()
        assert "action" in sj and "observation" in sj and "state" in sj
        assert sj["action"].get("title") or sj["action"].get("properties")


def test_http_reset_and_step_return_valid_payloads() -> None:
    """
    Stateless HTTP: each request builds a fresh env (OpenEnv design).
    POST /step on a new instance loads the scenario then applies the action (primed reset).
    """
    with TestClient(app, raise_server_exceptions=True) as client:
        reset = client.post("/reset", json={})
        assert reset.status_code == 200
        rj = reset.json()
        assert "observation" in rj
        obs = rj["observation"]
        assert "echoed_message" in obs
        assert "GHOSTEXEC BRIEFING" in (obs.get("echoed_message") or "")

        step = client.post(
            "/step",
            json={
                "action": {
                    "action_type": "reply_email",
                    "email_id": "e05",
                    "message_body": "On it.",
                }
            },
        )
        assert step.status_code == 200
        sj = step.json()
        assert "observation" in sj
        assert sj.get("reward") is not None or sj["observation"].get("reward") is not None


def test_http_step_invalid_action_422() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        bad = client.post("/step", json={"action": "not-an-object"})
        assert bad.status_code == 422


def test_mcp_jsonrpc_tools_list() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        r = client.post("/mcp", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "result" in data or "error" in data


def test_websocket_full_episode_reset_step_state_close() -> None:
    with TestClient(app, raise_server_exceptions=True) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            msg = ws.receive_json()
            assert msg.get("type") == "observation"
            data = msg.get("data") or {}
            assert "observation" in data
            inner = data["observation"]
            assert "echoed_message" in inner
            assert "GHOSTEXEC BRIEFING" in inner.get("echoed_message", "")

            ws.send_json(
                {
                    "type": "step",
                    "data": {
                        "action_type": "reschedule_meeting",
                        "meeting_id": "m02",
                        "new_time": "2026-04-21T18:00:00",
                    },
                }
            )
            msg2 = ws.receive_json()
            assert msg2.get("type") == "observation"
            d2 = msg2.get("data") or {}
            assert d2.get("reward") is not None

            ws.send_json({"type": "state"})
            msg3 = ws.receive_json()
            assert msg3.get("type") == "state", msg3
            st = msg3.get("data") or {}
            assert st.get("step_count", 0) >= 1

            ws.send_json({"type": "close", "data": {}})


def test_inprocess_env_matches_ws_briefing_shape() -> None:
    env = GhostexecEnvironment(SCENARIO)
    obs = env.reset()
    assert "BRIEFING" in obs.echoed_message
    o2 = env.step(
        GhostexecAction(
            action_type="reschedule_meeting",
            meeting_id="m02",
            new_time="2026-04-21T18:00:00",
        )
    )
    assert o2.reward is not None
    assert o2.metadata.get("step_ok") is True


def test_monday_morning_scenario_reward_signal() -> None:
    assert MONDAY.is_file()
    env = GhostexecEnvironment(MONDAY)
    env.reset()
    r = env.step(GhostexecAction(action_type="do_nothing")).reward
    assert isinstance(r, float)


def test_ghostexec_env_client_against_live_url_if_set() -> None:
    """
    GhostexecEnv opens a real TCP WebSocket; Starlette TestClient uses the
    non-resolvable host ``testserver`` on some platforms, so this only runs when
    ``GHOSTEXEC_WS_BASE_URL`` points at a live server (e.g. local uvicorn).
    """
    base = os.environ.get("GHOSTEXEC_WS_BASE_URL", "").strip().rstrip("/")
    if not base:
        pytest.skip("Set GHOSTEXEC_WS_BASE_URL (e.g. http://127.0.0.1:8000) to test GhostexecEnv client.")

    from ghostexec.client import GhostexecEnv

    sync_client = GhostexecEnv(base_url=base).sync()
    with sync_client:
        res = sync_client.reset()
        assert res.observation.echoed_message
        res2 = sync_client.step(GhostexecAction(action_type="do_nothing"))
        assert res2.observation.echoed_message


def test_training_pipeline_smoke() -> None:
    logf = ROOT / "outputs" / "training" / "_integration_train_smoke.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    if logf.exists():
        logf.unlink()
    cmd, env = _train_subprocess(
        [
            "--backend",
            "local",
            "--agent",
            "smart",
            "--episodes",
            "5",
            "--max-steps",
            "6",
            "--log-path",
            str(logf),
            "--checkpoint-dir",
            str(ROOT / "outputs" / "training" / "_integration_ckpt"),
        ]
    )
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)
    lines = [ln for ln in logf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 5


def test_full_pytest_suite_excluding_this_file() -> None:
    """Re-run the entire tests/ tree (avoids recursive self-invocation)."""
    env_pytest_addopts = os.environ.get("PYTEST_ADDOPTS", "")
    if "-x" in env_pytest_addopts or "--maxfail" in env_pytest_addopts:
        pytest.skip("Skipping nested full suite when fail-fast is enabled.")
    uv = shutil.which("uv")
    if uv:
        cmd = [
            uv,
            "run",
            "pytest",
            str(ROOT / "tests"),
            "-q",
            "--ignore=tests/test_complete_integration.py",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT))
    else:
        prev = os.environ.get("PYTHONPATH", "")
        nested_env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(ROOT), prev]).strip(os.pathsep),
        }
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests"),
            "-q",
            "--ignore=tests/test_complete_integration.py",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), env=nested_env)
    assert proc.returncode == 0, "Full test suite failed; see output above."

