#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# CLI: hit GhostExec HTTP endpoints (live URL or --local in-process app).
#
#   uv run python scripts/http_endpoint_smoke.py --local
#   uv run python scripts/http_endpoint_smoke.py --url http://127.0.0.1:8000

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urljoin

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_curl(base: str) -> None:
    print("# --- copy/paste (bash) ---")
    for method, path in [
        ("GET", "/health"),
        ("GET", "/metadata"),
        ("GET", "/state"),
        ("GET", "/schema"),
        ("GET", "/openapi.json"),
    ]:
        print(f"curl -sS -X {method} '{base.rstrip('/')}{path}' | head -c 200 && echo")
    print(
        "curl -sS -X POST '{base}/reset' -H 'Content-Type: application/json' -d '{{}}' | head -c 300 && echo".format(
            base=base.rstrip("/")
        )
    )
    print(
        "curl -sS -X POST '{base}/step' -H 'Content-Type: application/json' "
        "-d '{{\"action\":{{\"action_type\":\"do_nothing\"}}}}' | head -c 300 && echo".format(base=base.rstrip("/"))
    )
    print(
        "# Note: HTTP uses a new env per request — not one multi-step episode; use WebSocket /ws for that."
    )


class LiveClient:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        url = urljoin(self.base + "/", path.lstrip("/"))
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.status, resp.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")


class LocalClient:
    def __init__(self) -> None:
        from fastapi.testclient import TestClient

        from ghostexec.server.app import app

        self._client = TestClient(app, raise_server_exceptions=True)

    def request(
        self,
        method: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        hdrs = headers or {}
        kwargs: dict[str, Any] = {}
        if data is not None:
            kwargs["content"] = data
            kwargs["headers"] = hdrs
        r = self._client.request(method, path, **kwargs)
        return r.status_code, r.text


def main() -> int:
    p = argparse.ArgumentParser(description="GhostExec HTTP endpoint smoke (CLI).")
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Live server base URL (ignored with --local).",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Use in-process FastAPI TestClient (no server required).",
    )
    p.add_argument(
        "--print-curl",
        action="store_true",
        help="Print example curl commands and exit 0.",
    )
    args = p.parse_args()

    if args.print_curl:
        _print_curl(args.url)
        return 0

    client: LiveClient | LocalClient
    label: str
    if args.local:
        client = LocalClient()
        label = "local TestClient"
    else:
        client = LiveClient(args.url)
        label = args.url

    def check_get(path: str) -> None:
        code, body = client.request("GET", path)
        ok = 200 <= code < 300
        status = "OK" if ok else "FAIL"
        print(f"[{status}] GET {path} -> HTTP {code} (body ~{len(body)} chars)")
        if not ok:
            raise SystemExit(1)

    print(f"GhostExec HTTP smoke ({label})\n")

    for path in (
        "/health",
        "/metadata",
        "/state",
        "/schema",
        "/openapi.json",
        "/docs",
        "/redoc",
    ):
        check_get(path)

    body = json.dumps({}).encode()
    hdrs = {"Content-Type": "application/json"}
    code, txt = client.request("POST", "/reset", data=body, headers=hdrs)
    print(f"[{'OK' if code == 200 else 'FAIL'}] POST /reset -> HTTP {code}")
    if code != 200:
        raise SystemExit(1)
    j = json.loads(txt)
    em = (j.get("observation") or {}).get("echoed_message", "")[:50]
    print(f"      briefing prefix: {em!r}")

    step_payload = json.dumps({"action": {"action_type": "do_nothing"}}).encode()
    code2, txt2 = client.request("POST", "/step", data=step_payload, headers=hdrs)
    print(f"[{'OK' if code2 == 200 else 'FAIL'}] POST /step do_nothing -> HTTP {code2}")
    if code2 != 200:
        raise SystemExit(1)

    print(
        "\nNote: OpenEnv HTTP may use a new env per request, so separate POSTs do not advance "
        "one long episode; each POST /step runs a single action on a fresh instance. "
        "Multi-step learning on one episode: WebSocket /ws (see ghostexec/README.md)."
    )

    code3, _ = client.request("POST", "/mcp", data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode(), headers=hdrs)
    print(f"[{'OK' if code3 == 200 else 'FAIL'}] POST /mcp tools/list -> HTTP {code3}")
    if code3 != 200:
        raise SystemExit(1)

    code4, _ = client.request("GET", "/reset")
    print(f"[{'OK' if code4 == 405 else 'FAIL'}] GET /reset (expect 405) -> HTTP {code4}")
    if code4 != 405:
        raise SystemExit(1)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
