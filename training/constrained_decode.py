# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Constrained / structured generation helpers for Ghostexec.
#
# Why: during GRPO, every unparsable completion is wasted compute. If every
# sample is guaranteed to match the ``GhostexecAction`` JSON schema, the reward
# only reflects *policy* quality, which massively speeds up useful-steps per
# minute in early training.
#
# What this module exposes:
#   - ``ghostexec_action_json_schema()``: schema dict from the Pydantic model.
#   - ``build_json_prefix_allowed_tokens_fn(tokenizer)``: HF-compatible
#     ``prefix_allowed_tokens_fn`` via ``lm-format-enforcer`` (preferred; works
#     with Unsloth's tokenizer out of the box).
#   - ``build_json_logits_processor(tokenizer)``: HF ``LogitsProcessorList``
#     via ``outlines`` as a secondary backend.
#   - ``patch_model_for_json_generation(model, tokenizer)``: monkey-patches
#     ``model.generate`` so every call (including the ones TRL's GRPOTrainer
#     makes) enforces the schema. Returns an ``unpatch`` callable.
#   - ``outlines_json_generator(model)``: standalone ``outlines.generate.json``
#     helper for non-TRL inference / eval.

from __future__ import annotations

import functools
from typing import Any, Callable

try:
    from ghostexec.models import GhostexecAction
except ImportError:
    from models import GhostexecAction


def ghostexec_action_json_schema() -> dict[str, Any]:
    """JSON schema for ``GhostexecAction`` (what the model must produce)."""
    return GhostexecAction.model_json_schema()


# --- Backend 1: lm-format-enforcer (preferred — prefix_allowed_tokens_fn path) ---


def build_json_prefix_allowed_tokens_fn(tokenizer: Any) -> Callable[..., list[int]]:
    """Return a HuggingFace-compatible ``prefix_allowed_tokens_fn`` enforcing the schema.

    Uses ``lm-format-enforcer`` because its transformers integration accepts a
    vanilla HF tokenizer (no custom wrapper needed, so it plays nicely with
    Unsloth's patched tokenizer).

    Raises ``ImportError`` if ``lm-format-enforcer`` is not installed.
    """
    try:
        from lmformatenforcer import JsonSchemaParser  # type: ignore
        from lmformatenforcer.integrations.transformers import (  # type: ignore
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "lm-format-enforcer is not installed. "
            "`pip install lm-format-enforcer` to use JSON-constrained decoding."
        ) from exc
    parser = JsonSchemaParser(ghostexec_action_json_schema())
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


# --- Backend 2: outlines (LogitsProcessorList path) ---


def build_json_logits_processor(tokenizer: Any) -> Any:
    """Return a HF ``LogitsProcessorList`` that enforces the schema (outlines backend).

    Raises ``ImportError`` if ``outlines`` is not installed.
    """
    try:
        from outlines.processors import JSONLogitsProcessor  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "outlines is not installed. "
            "`pip install outlines` to use JSON-constrained decoding."
        ) from exc
    # outlines >=0.1 accepts HF tokenizers directly via its adapter; fall back
    # to passing the raw tokenizer for older versions.
    otok: Any = tokenizer
    try:
        from outlines.models.transformers import TransformerTokenizer  # type: ignore

        otok = TransformerTokenizer(tokenizer)
    except Exception:  # pragma: no cover - depends on outlines version
        otok = tokenizer
    proc = JSONLogitsProcessor(ghostexec_action_json_schema(), otok)
    from transformers import LogitsProcessorList  # type: ignore

    return LogitsProcessorList([proc])


# --- model.generate patching (what you actually plug into the Colab notebook) ---


def patch_model_for_json_generation(
    model: Any,
    tokenizer: Any,
    *,
    backend: str = "auto",
) -> Callable[[], None]:
    """Wrap ``model.generate`` to enforce GhostexecAction JSON on every call.

    Parameters
    ----------
    model:
        Any HF-compatible model (Unsloth's ``FastLanguageModel`` works).
    tokenizer:
        Matching tokenizer.
    backend:
        ``"auto"`` (default, prefers lm-format-enforcer then outlines),
        ``"lmfe"``, or ``"outlines"``.

    Returns
    -------
    unpatch : callable
        Call it to restore the original ``model.generate``.

    Raises
    ------
    ImportError
        If no supported backend is installed.
    """
    if getattr(model, "_ghostexec_json_patched", False):
        # Already patched; return a no-op unpatch so callers can still chain.
        return lambda: None

    pre_fn: Callable[..., list[int]] | None = None
    logits_procs: Any = None

    def _try_lmfe() -> bool:
        nonlocal pre_fn
        try:
            pre_fn = build_json_prefix_allowed_tokens_fn(tokenizer)
            return True
        except ImportError:
            return False

    def _try_outlines() -> bool:
        nonlocal logits_procs
        try:
            logits_procs = build_json_logits_processor(tokenizer)
            return True
        except ImportError:
            return False

    if backend == "lmfe":
        if not _try_lmfe():
            raise ImportError("lm-format-enforcer not installed.")
    elif backend == "outlines":
        if not _try_outlines():
            raise ImportError("outlines not installed.")
    else:  # auto
        if not _try_lmfe() and not _try_outlines():
            raise ImportError(
                "No constrained-decoding backend installed. "
                "`pip install lm-format-enforcer` (preferred) or `pip install outlines`."
            )

    original_generate = model.generate

    @functools.wraps(original_generate)
    def constrained_generate(*args: Any, **kwargs: Any) -> Any:
        if pre_fn is not None and "prefix_allowed_tokens_fn" not in kwargs:
            kwargs["prefix_allowed_tokens_fn"] = pre_fn
        if logits_procs is not None:
            existing = kwargs.get("logits_processor")
            if existing is None:
                kwargs["logits_processor"] = logits_procs
            else:
                # Append without mutating caller's list in place.
                from transformers import LogitsProcessorList  # type: ignore

                merged = LogitsProcessorList(list(existing) + list(logits_procs))
                kwargs["logits_processor"] = merged
        return original_generate(*args, **kwargs)

    model.generate = constrained_generate  # type: ignore[method-assign]
    model._ghostexec_json_patched = True
    model._ghostexec_original_generate = original_generate

    def unpatch() -> None:
        if getattr(model, "_ghostexec_json_patched", False):
            model.generate = model._ghostexec_original_generate  # type: ignore[method-assign]
            model._ghostexec_json_patched = False
            try:
                del model._ghostexec_original_generate
            except AttributeError:
                pass

    return unpatch


def unpatch_model_generation(model: Any) -> bool:
    """Restore the original ``model.generate`` if previously patched. Returns True if it was patched."""
    if getattr(model, "_ghostexec_json_patched", False):
        model.generate = model._ghostexec_original_generate  # type: ignore[method-assign]
        model._ghostexec_json_patched = False
        try:
            del model._ghostexec_original_generate
        except AttributeError:
            pass
        return True
    return False


# --- outlines direct generator (for eval / inference outside TRL) ---


def outlines_json_generator(model: Any) -> Any:
    """Return an ``outlines`` JSON generator bound to the Ghostexec action schema.

    Parameters
    ----------
    model:
        An ``outlines.models.*`` instance (e.g. ``outlines.models.transformers(hf_model, tok)``).

    Raises ``ImportError`` if ``outlines`` is not installed.
    """
    try:
        import outlines  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "outlines is not installed. `pip install outlines` to use constrained decoding."
        ) from exc
    return outlines.generate.json(model, GhostexecAction)


__all__ = [
    "build_json_logits_processor",
    "build_json_prefix_allowed_tokens_fn",
    "ghostexec_action_json_schema",
    "outlines_json_generator",
    "patch_model_for_json_generation",
    "unpatch_model_generation",
]
