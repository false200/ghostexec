# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Put repo root on sys.path before test collection (supports `uv run pytest` without editable install).

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
