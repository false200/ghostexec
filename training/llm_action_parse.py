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


def completion_to_str(completion: Any) -> str:
    """
    Turn a TRL / chat completion into plain text for JSON extraction.

    GRPO may pass ``str`` or structured values (e.g. list of ``{"role","content"}`` dicts);
    lists used to reach ``_strip_markdown_fences`` caused ``AttributeError: 'list' object has no attribute 'strip'``.
    """
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, (list, tuple)):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                c = item.get("content")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict):
                            t = block.get("text")
                            if block.get("type") == "text" and isinstance(t, str):
                                parts.append(t)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if isinstance(completion, dict):
        c = completion.get("content")
        if isinstance(c, str):
            return c.strip()
        return json.dumps(completion, ensure_ascii=False)
    return str(completion).strip()


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


def parse_completion_to_action(completion: Any) -> GhostexecAction | None:
    """
    Parse a model completion into GhostexecAction.

    Accepts raw JSON, ```json ... ``` fences, or text with an embedded object.
    ``completion`` may be a string or a chat-style list of message dicts (TRL GRPO).
    Returns None if no valid action dict can be parsed.
    """
    text = completion_to_str(completion)
    data = _extract_json_object(text)
    if not data:
        return None
    try:
        return GhostexecAction.model_validate(data)
    except Exception:
        return None
