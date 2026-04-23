# Copyright (c) Meta Platforms, Inc. and affiliates. Training helper for notebooks.
"""Some conda / SageMaker PyTorch wheels ship ``torch._inductor`` without a
``config`` submodule. ``unsloth_zoo`` imports ``inspect.getsource(torch._inductor.config)``
at import time; this helper attaches a real submodule or a tiny file-backed stub
so ``from unsloth import ...`` does not crash.

Call :func:`apply_unsloth_torch_inductor_shim` once per process before importing
``unsloth`` / ``unsloth_zoo``.

Call :func:`require_cuda_for_unsloth` immediately after the shim and before
``from unsloth import ...``. ``unsloth_zoo`` hard-requires a torch accelerator
(``torch.cuda.is_available()`` on NVIDIA); CPU-only PyTorch raises a confusing
``NotImplementedError`` otherwise.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types


def apply_unsloth_torch_inductor_shim() -> None:
    import torch

    ind = getattr(torch, "_inductor", None)
    if ind is None or getattr(ind, "config", None) is not None:
        return

    try:
        cfg = importlib.import_module("torch._inductor.config")
    except Exception:
        cfg = None
    if cfg is not None:
        ind.config = cfg
        return

    # ``unsloth_zoo.temporary_patches.common`` filters ``torch.compile`` option keys by
    # substring checks against ``inspect.getsource(torch._inductor.config)``.
    stub_src = """# epilogue_fusion max_autotune shape_padding trace enabled triton cudagraphs debug dce
# memory_planning coordinate_descent_tuning graph_diagram compile_threads group_fusion
# disable_progress verbose_progress multi_kernel use_block_ptr enable_persistent_tma_matmul
# autotune_at_compile_time cooperative_reductions cuda compile_opt_level enable_cuda_lto
# combo_kernels benchmark_combo_kernel combo_kernel_foreach_dynamic_shapes
"""
    fd, path = tempfile.mkstemp(prefix="torch_inductor_config_shim_", suffix=".py", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(stub_src)
    mod = types.ModuleType("torch._inductor.config")
    mod.__file__ = path
    mod.__name__ = "torch._inductor.config"
    mod.__package__ = "torch._inductor"
    sys.modules.setdefault("torch._inductor.config", mod)
    ind.config = mod


def require_cuda_for_unsloth() -> None:
    """Raise with actionable text if PyTorch cannot see an NVIDIA-style GPU."""
    import torch

    if torch.cuda.is_available():
        return
    _cuda_built = getattr(torch.version, "cuda", None)
    _cvd_repr = repr(os.environ["CUDA_VISIBLE_DEVICES"]) if "CUDA_VISIBLE_DEVICES" in os.environ else "(not set)"
    _cpu_only = _cuda_built is None
    _hints = []
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        _hints.append(
            "CUDA_VISIBLE_DEVICES is set to an empty string (hides all GPUs). "
            "Run the first notebook cell that clears it, then **Kernel → Restart** "
            "so torch loads after the fix."
        )
    if _cpu_only:
        _hints.append(
            "torch.version.cuda is None → this PyTorch build is **CPU-only**. "
            "Install CUDA PyTorch in the same conda env, then restart the kernel, e.g.: "
            "`conda install -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia` "
            "(12.4 wheels work with newer drivers such as 12.8; see pytorch.org/get-started/locally)."
        )
    else:
        _hints.append(
            "Driver may see a GPU (`nvidia-smi`) while this torch build does not — "
            "reinstall matching CUDA torch from pytorch.org / conda nvidia channel."
        )
    raise RuntimeError(
        "Unsloth requires a CUDA GPU: torch.cuda.is_available() is False. "
        f"torch={torch.__version__!s} torch.version.cuda={_cuda_built!r} "
        f"CUDA_VISIBLE_DEVICES={_cvd_repr}. "
        + " ".join(_hints)
    )
