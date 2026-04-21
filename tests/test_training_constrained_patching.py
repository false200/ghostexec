# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for training/constrained_decode.py patching helpers.

Uses a tiny fake model + tokenizer so we don't need HF / torch / outlines
installed to exercise the wrapping logic itself.
"""

from __future__ import annotations

import importlib.util

import pytest

from training.constrained_decode import (
    patch_model_for_json_generation,
    unpatch_model_generation,
)


class _FakeModel:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, object] = {}

    def generate(self, *args: object, **kwargs: object) -> str:
        self.call_kwargs = dict(kwargs)
        return "ok"


class _FakeTokenizer:
    vocab_size = 4

    def __call__(self, *a: object, **kw: object) -> dict[str, object]:  # pragma: no cover
        return {}


def _any_backend_installed() -> bool:
    return (
        importlib.util.find_spec("lmformatenforcer") is not None
        or importlib.util.find_spec("outlines") is not None
    )


def test_patch_requires_a_backend() -> None:
    if _any_backend_installed():
        pytest.skip("constrained-decode backend is installed; no-backend path not exercised")
    model = _FakeModel()
    with pytest.raises(ImportError):
        patch_model_for_json_generation(model, _FakeTokenizer())


def test_explicit_backend_lmfe_missing_raises() -> None:
    if importlib.util.find_spec("lmformatenforcer") is not None:
        pytest.skip("lm-format-enforcer installed")
    with pytest.raises(ImportError):
        patch_model_for_json_generation(_FakeModel(), _FakeTokenizer(), backend="lmfe")


def test_explicit_backend_outlines_missing_raises() -> None:
    if importlib.util.find_spec("outlines") is not None:
        pytest.skip("outlines installed")
    with pytest.raises(ImportError):
        patch_model_for_json_generation(_FakeModel(), _FakeTokenizer(), backend="outlines")


def test_unpatch_is_safe_when_never_patched() -> None:
    model = _FakeModel()
    # Nothing to unpatch; should just return False and not crash.
    assert unpatch_model_generation(model) is False


def test_patch_and_unpatch_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a backend IS available, we still exercise the wrap/unwrap bookkeeping."""
    if not _any_backend_installed():
        pytest.skip("no constrained-decode backend installed")
    # Stub out the actual backend builders so we don't need a real tokenizer.
    import training.constrained_decode as cd

    monkeypatch.setattr(
        cd,
        "build_json_prefix_allowed_tokens_fn",
        lambda tok: (lambda batch_id, input_ids: [0, 1, 2, 3]),
    )
    monkeypatch.setattr(
        cd,
        "build_json_logits_processor",
        lambda tok: None,
    )
    model = _FakeModel()
    original = model.generate
    unpatch = cd.patch_model_for_json_generation(model, _FakeTokenizer())
    assert model.generate is not original
    assert getattr(model, "_ghostexec_json_patched", False) is True
    model.generate()
    assert "prefix_allowed_tokens_fn" in model.call_kwargs
    unpatch()
    assert model.generate is original
    assert getattr(model, "_ghostexec_json_patched", False) is False
