# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Exhaustive / adversarial probes against a RUNNING GhostExec HTTP server.
# Default: http://127.0.0.1:8000  (override with GHOSTEXEC_LIVE_BASE_URL).
# Skips all tests if /health is unreachable.

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

import pytest

BASE = os.environ.get("GHOSTEXEC_LIVE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _req(
    method: str,
    path: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> tuple[int, bytes]:
    url = BASE + path
    h = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(h, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except (ConnectionResetError, OSError):
            body = b""
        return e.code, body


@pytest.fixture(scope="module")
def live() -> str:
    try:
        code, _ = _req("GET", "/health", timeout=3.0)
    except OSError as e:
        pytest.skip(f"Live server not reachable at {BASE!r}: {e}")
    if code != 200:
        pytest.skip(f"Live /health returned {code} at {BASE!r}")
    return BASE


def test_get_core_docs(live: str) -> None:
    for path, min_len in [
        ("/health", 10),
        ("/metadata", 20),
        ("/state", 10),
        ("/schema", 500),
        ("/openapi.json", 1000),
        ("/docs", 200),
        ("/redoc", 200),
    ]:
        code, body = _req("GET", path)
        assert code == 200, f"{path} -> {code}"
        assert len(body) >= min_len, f"{path} body tiny"


def test_wrong_http_methods_on_control_routes(live: str) -> None:
    assert _req("GET", "/reset")[0] == 405
    assert _req("GET", "/step")[0] == 405
    assert _req("PUT", "/reset", data=b"{}")[0] in (405, 422)
    code, _ = _req("DELETE", "/health")
    assert code in (405, 404)
    assert _req("GET", "/this-path-should-not-exist-ghostexec")[0] == 404


def test_reset_payload_variants(live: str) -> None:
    for label, payload in [
        ("empty", {}),
        ("seed", {"seed": 42}),
        ("episode_id", {"episode_id": "probe-episode-1"}),
        ("extra_ignored", {"seed": 1, "unknown_future_field_xyz": True}),
    ]:
        code, body = _req(
            "POST",
            "/reset",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert code == 200, f"reset {label}: {code}"
        j = json.loads(body.decode())
        assert "observation" in j and "done" in j
        obs = j["observation"]
        assert "echoed_message" in obs


def test_step_valid_action_types(live: str) -> None:
    cases: list[tuple[str, dict[str, Any]]] = [
        ("do_nothing", {"action_type": "do_nothing"}),
        (
            "reply_email",
            {"action_type": "reply_email", "email_id": "e14", "message_body": "Live exhaustive probe."},
        ),
        ("archive_email", {"action_type": "archive_email", "email_id": "e09"}),
        (
            "reschedule_meeting",
            {
                "action_type": "reschedule_meeting",
                "meeting_id": "m02",
                "new_time": "2026-04-21T18:00:00",
            },
        ),
        (
            "cancel_meeting",
            {"action_type": "cancel_meeting", "meeting_id": "m10", "reason": "probe cancel"},
        ),
        ("complete_task", {"action_type": "complete_task", "task_id": "t07"}),
        (
            "delegate_task",
            {
                "action_type": "delegate_task",
                "task_id": "t08",
                "contact_name": "Jordan Lee",
            },
        ),
        (
            "send_message",
            {
                "action_type": "send_message",
                "contact_name": "Jamie Liu",
                "message_body": "Exhaustive live test ping.",
            },
        ),
    ]
    for name, action in cases:
        code, body = _req(
            "POST",
            "/step",
            data=json.dumps({"action": action}).encode(),
            headers={"Content-Type": "application/json"},
        )
        assert code == 200, f"step {name}: HTTP {code} {body[:200]!r}"
        j = json.loads(body.decode())
        assert "observation" in j
        meta = (j.get("observation") or {}).get("metadata") or {}
        assert "step_ok" in meta, f"step {name}: missing step_ok"


def test_step_invalid_contracts(live: str) -> None:
    assert _req("POST", "/step", data=b"not-json", headers={"Content-Type": "application/json"})[0] in (
        400,
        422,
    )
    assert (
        _req(
            "POST",
            "/step",
            data=json.dumps({"action": "not-a-dict"}).encode(),
            headers={"Content-Type": "application/json"},
        )[0]
        == 422
    )
    assert (
        _req(
            "POST",
            "/step",
            data=json.dumps({"action": {"action_type": "reply_email", "email_id": "nope", "message_body": "x"}}).encode(),
            headers={"Content-Type": "application/json"},
        )[0]
        == 200
    )
    j = json.loads(
        _req(
            "POST",
            "/step",
            data=json.dumps(
                {"action": {"action_type": "reply_email", "email_id": "nope", "message_body": "x"}}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )[1].decode()
    )
    assert j["observation"]["metadata"].get("step_ok") is False

    assert (
        _req(
            "POST",
            "/step",
            data=json.dumps({"action": {"action_type": "complete_task", "task_id": "t09"}}).encode(),
            headers={"Content-Type": "application/json"},
        )[0]
        == 200
    )
    j2 = json.loads(
        _req(
            "POST",
            "/step",
            data=json.dumps({"action": {"action_type": "complete_task", "task_id": "t09"}}).encode(),
            headers={"Content-Type": "application/json"},
        )[1].decode()
    )
    assert j2["observation"]["metadata"].get("step_ok") is False


def test_step_unicode_and_long_message(live: str) -> None:
    long_body = ("Line note.\n" * 80) + " café naïve résumé 日本語"
    code, body = _req(
        "POST",
        "/step",
        data=json.dumps(
            {"action": {"action_type": "reply_email", "email_id": "e05", "message_body": long_body}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    assert code == 200


def test_step_wrong_content_type(live: str) -> None:
    code, _ = _req(
        "POST",
        "/step",
        data=b"action_type=do_nothing",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert code in (400, 415, 422)


def test_reset_invalid_json(live: str) -> None:
    code, _ = _req("POST", "/reset", data=b"{", headers={"Content-Type": "application/json"})
    assert code in (400, 422)


def test_mcp_variants(live: str) -> None:
    assert _req("POST", "/mcp", data=b"{", headers={"Content-Type": "application/json"})[0] == 200
    body = _req(
        "POST",
        "/mcp",
        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "bogus/thing", "params": {}}).encode(),
        headers={"Content-Type": "application/json"},
    )[1].decode()
    j = json.loads(body)
    assert "error" in j or "result" in j


def test_openapi_lists_expected_paths(live: str) -> None:
    _, raw = _req("GET", "/openapi.json")
    spec = json.loads(raw.decode())
    paths = spec.get("paths") or {}
    for p in ("/health", "/reset", "/step", "/schema", "/metadata", "/state", "/mcp"):
        assert p in paths, f"missing path {p} in OpenAPI"


def test_websocket_dead_ends(live: str) -> None:
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets not installed")

    async def _run() -> None:
        ws_url = live.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            await ws.send("{ not json")
            e1 = json.loads(await ws.recv())
            assert e1.get("type") == "error"

            await ws.send(json.dumps({"type": "nosuch", "data": {}}))
            e2 = json.loads(await ws.recv())
            assert e2.get("type") == "error"

            await ws.send(json.dumps({"type": "reset", "data": {}}))
            ok = json.loads(await ws.recv())
            assert ok.get("type") == "observation"

            await ws.send(
                json.dumps({"type": "step", "data": {"action_type": "reply_email", "email_id": "missing"}})
            )
            bad = json.loads(await ws.recv())
            assert bad.get("type") == "observation"
            meta = (bad.get("data") or {}).get("observation", {}).get("metadata") or {}
            assert meta.get("step_ok") is False

            await ws.send(json.dumps({"type": "state"}))
            st = json.loads(await ws.recv())
            assert st.get("type") == "state"

            await ws.send(json.dumps({"type": "close", "data": {}}))

    asyncio.run(_run())
