"""Hard API dead-test: 500+ calls with reward-consistency checks."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ghostexec.server.app import app

W_CONFLICT = 0.35
W_REL = 0.35
W_TASK = 0.30
OUTPUT_SCALE = 0.48


def _step_payload_for(i: int) -> dict[str, Any]:
    templates: list[dict[str, Any]] = [
        {"action": {"action_type": "do_nothing"}},
        {"action": {"action_type": "reply_email", "email_id": "e01", "message_body": "On it now."}},
        {"action": {"action_type": "reply_email", "email_id": "e14", "message_body": "Acknowledged."}},
        {"action": {"action_type": "reply_email", "email_id": "nope_999", "message_body": "x"}},
        {"action": {"action_type": "archive_email", "email_id": "e09"}},
        {"action": {"action_type": "archive_email", "email_id": "bad_id"}},
        {
            "action": {
                "action_type": "reschedule_meeting",
                "meeting_id": "m02",
                "new_time": "2026-04-21T18:00:00",
            }
        },
        {
            "action": {
                "action_type": "reschedule_meeting",
                "meeting_id": "m03",
                "new_time": "2026-04-21T09:30:00",  # overlap -> invalid semantic
            }
        },
        {"action": {"action_type": "cancel_meeting", "meeting_id": "m10", "reason": "dead test"}},
        {"action": {"action_type": "cancel_meeting", "meeting_id": "m99", "reason": "dead test"}},
        {"action": {"action_type": "complete_task", "task_id": "t07"}},
        {"action": {"action_type": "complete_task", "task_id": "t09"}},  # already done
        {"action": {"action_type": "delegate_task", "task_id": "t08", "contact_name": "Jordan Lee"}},
        {"action": {"action_type": "delegate_task", "task_id": "t08", "contact_name": "Nobody"}},
        {
            "action": {
                "action_type": "send_message",
                "contact_name": "Jamie Liu",
                "message_body": "Quick sync please.",
            }
        },
        {
            "action": {
                "action_type": "send_message",
                "contact_name": "Nobody",
                "message_body": "hello",
            }
        },
    ]
    return templates[i % len(templates)]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


def test_api_surface_all_endpoints(client: TestClient) -> None:
    # Core GET endpoints.
    for path in ("/health", "/metadata", "/state", "/schema", "/openapi.json", "/docs", "/redoc"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"

    # Control routes: method contracts.
    assert client.get("/reset").status_code == 405
    assert client.get("/step").status_code == 405
    assert client.put("/reset", json={}).status_code in (405, 422)
    assert client.get("/this-path-should-not-exist-ghostexec").status_code == 404

    # Reset variants.
    for body in ({}, {"seed": 42}, {"episode_id": "dead-api-001"}, {"seed": 1, "future_field": True}):
        rr = client.post("/reset", json=body)
        assert rr.status_code == 200
        j = rr.json()
        assert "observation" in j and "done" in j

    # MCP endpoint variants.
    mcp_ok = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert mcp_ok.status_code == 200
    mcp_bad_json = client.post("/mcp", content="{", headers={"Content-Type": "application/json"})
    assert mcp_bad_json.status_code == 200


@pytest.mark.parametrize("idx", range(520))
def test_api_reward_dead_520_cases(client: TestClient, idx: int) -> None:
    # Keep each case independent and deterministic.
    rr = client.post("/reset", json={"episode_id": f"dead-{idx:04d}", "seed": 42})
    assert rr.status_code == 200

    payload = _step_payload_for(idx)
    rs = client.post("/step", json=payload)
    assert rs.status_code == 200, f"idx={idx} payload={payload} status={rs.status_code}"

    body = rs.json()
    assert "observation" in body and "reward" in body and "done" in body
    obs = body["observation"]
    meta = obs.get("metadata") or {}
    bd = meta.get("reward_breakdown") or {}

    # Structural contracts.
    assert isinstance(obs.get("echoed_message", ""), str) and obs.get("echoed_message")
    assert "step_ok" in meta
    assert "step_detail" in meta
    assert "final" in bd
    assert "weighted_base" in bd

    # Reward identity: top-level reward must equal breakdown.final.
    reward = float(body["reward"])
    final = float(bd["final"])
    assert reward == pytest.approx(final, abs=1e-9)

    # Aggregation formula must hold exactly (within floating tolerance).
    conflict = float(bd.get("conflict", 0.0))
    relationship = float(bd.get("relationship", 0.0))
    task = float(bd.get("task", 0.0))
    weighted_inner = W_CONFLICT * conflict + W_REL * relationship + W_TASK * task
    expected_weighted = OUTPUT_SCALE * weighted_inner
    assert float(bd["weighted_base"]) == pytest.approx(expected_weighted, abs=1e-9)

    expected_final = (
        float(bd.get("weighted_base", 0.0))
        + float(bd.get("invalid_step_adjustment", 0.0))
        + float(bd.get("episode_completion_bonus", 0.0))
        + float(bd.get("catastrophic_penalty", 0.0))
        + float(bd.get("do_nothing_floor", 0.0))
    )
    assert final == pytest.approx(expected_final, abs=1e-9)

    action_type = payload["action"]["action_type"]
    if action_type == "do_nothing":
        assert float(bd.get("do_nothing_floor", 0.0)) == pytest.approx(-0.15, abs=1e-12)
        assert reward < 0

    if meta.get("step_ok") is False:
        assert float(bd.get("invalid_step_adjustment", 0.0)) == pytest.approx(-0.25, abs=1e-12)

