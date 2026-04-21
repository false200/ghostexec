# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Parse model completions into GhostexecAction for Colab / GRPO reward loops.

from __future__ import annotations

import json
import re
from typing import Any

try:
    from ghostexec.models import GhostexecAction
except ImportError:
    from models import GhostexecAction


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"```\s*$")


def _strip_markdown_fences(text: str) -> str:
    s = text.strip()
    lines = s.splitlines()
    if not lines:
        return s
    if _FENCE_RE.match(lines[0]):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _FENCE_END_RE.search(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Take first top-level JSON object substring from text."""
    t = _strip_markdown_fences(text)
    if not t:
        return None
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start : i + 1]
                try:
                    obj = json.loads(chunk)
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def parse_completion_to_action(text: str) -> GhostexecAction | None:
    """
    Parse a model completion into GhostexecAction.

    Accepts raw JSON, ```json ... ``` fences, or text with an embedded object.
    Returns None if no valid action dict can be parsed.
    """
    data = _extract_json_object(text)
    if not data:
        return None
    try:
        return GhostexecAction.model_validate(data)
    except Exception:
        return None
