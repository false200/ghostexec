"""Run 500+ LIVE HTTP API reward dead-tests against a running GhostExec server.

Usage:
    uv run python scripts/run_live_api_dead_500.py --url http://127.0.0.1:8002 --cases 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
import urllib.error
import urllib.request

W_CONFLICT = 0.35
W_REL = 0.35
W_TASK = 0.30
OUTPUT_SCALE = 0.48


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 20.0,
) -> tuple[int, str]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


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


def _assert_api_surface(base_url: str) -> None:
    for path in ("/health", "/metadata", "/state", "/schema", "/openapi.json", "/docs", "/redoc"):
        code, _ = _request(base_url, "GET", path)
        assert code == 200, f"{path} -> {code}"
    assert _request(base_url, "GET", "/reset")[0] == 405
    assert _request(base_url, "GET", "/step")[0] == 405
    assert _request(base_url, "GET", "/this-path-should-not-exist-ghostexec")[0] == 404
    assert _request(base_url, "POST", "/mcp", body={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})[0] == 200


def main() -> int:
    p = argparse.ArgumentParser(description="Run live 500+ reward dead-tests.")
    p.add_argument("--url", default="http://127.0.0.1:8002", help="Base server URL")
    p.add_argument("--cases", type=int, default=500, help="Number of /reset+/step cases")
    args = p.parse_args()

    base_url = args.url.rstrip("/")
    cases = max(1, args.cases)

    _assert_api_surface(base_url)

    out_dir = Path("outputs") / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"api_dead_live_{cases}.jsonl"

    passed = 0
    failed = 0
    failures: list[str] = []

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(cases):
            rec: dict[str, Any] = {"idx": idx, "ok": False, "error": None}
            try:
                rc, rb = _request(
                    base_url,
                    "POST",
                    "/reset",
                    body={"episode_id": f"live-dead-{idx:04d}", "seed": 42},
                )
                assert rc == 200, f"reset status {rc}"

                payload = _step_payload_for(idx)
                rec["action"] = payload["action"]
                sc, sb = _request(base_url, "POST", "/step", body=payload)
                assert sc == 200, f"step status {sc}"
                body = json.loads(sb)

                obs = body["observation"]
                meta = obs.get("metadata") or {}
                bd = meta.get("reward_breakdown") or {}

                reward = float(body["reward"])
                final = float(bd["final"])
                assert reward == final, "reward != breakdown.final"

                c = float(bd.get("conflict", 0.0))
                r = float(bd.get("relationship", 0.0))
                t = float(bd.get("task", 0.0))
                expected_weighted = OUTPUT_SCALE * (W_CONFLICT * c + W_REL * r + W_TASK * t)
                assert float(bd["weighted_base"]) == expected_weighted, "weighted_base mismatch"

                expected_final = (
                    float(bd.get("weighted_base", 0.0))
                    + float(bd.get("invalid_step_adjustment", 0.0))
                    + float(bd.get("episode_completion_bonus", 0.0))
                    + float(bd.get("catastrophic_penalty", 0.0))
                    + float(bd.get("do_nothing_floor", 0.0))
                )
                assert final == expected_final, "final aggregation mismatch"

                if payload["action"]["action_type"] == "do_nothing":
                    assert float(bd.get("do_nothing_floor", 0.0)) == -0.15, "do_nothing floor mismatch"
                    assert reward < 0, "do_nothing should be negative"

                if meta.get("step_ok") is False:
                    assert float(bd.get("invalid_step_adjustment", 0.0)) == -0.25, "invalid penalty mismatch"

                rec["ok"] = True
                rec["reward"] = reward
                rec["step_ok"] = meta.get("step_ok")
                passed += 1
            except Exception as e:  # noqa: BLE001
                rec["ok"] = False
                rec["error"] = str(e)
                failed += 1
                if len(failures) < 10:
                    failures.append(f"idx={idx}: {e}")
            finally:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Live API dead-test complete: passed={passed} failed={failed} total={cases}")
    print(f"Report: {out_path}")
    if failures:
        print("First failures:")
        for row in failures:
            print(" -", row)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

