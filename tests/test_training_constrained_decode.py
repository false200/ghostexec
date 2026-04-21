# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/constrained_decode.py."""

from __future__ import annotations

import importlib.util

import pytest

from training.constrained_decode import (
    ghostexec_action_json_schema,
    outlines_json_generator,
)


def test_action_schema_exposes_action_type_enum() -> None:
    schema = ghostexec_action_json_schema()
    assert schema.get("type") == "object"
    props = schema.get("properties", {})
    assert "action_type" in props
    at = props["action_type"]
    # Either "enum" (inline) or "$ref" to a $defs enum — both are acceptable.
    has_enum = "enum" in at
    has_ref = "$ref" in at or "allOf" in at
    assert has_enum or has_ref, f"action_type schema has no enum / ref: {at}"


def test_outlines_json_generator_raises_when_outlines_missing() -> None:
    if importlib.util.find_spec("outlines") is not None:
        pytest.skip("outlines is installed; no-import path not exercised")
    with pytest.raises(ImportError):
        outlines_json_generator(object())
