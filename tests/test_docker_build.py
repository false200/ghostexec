"""Opt-in Docker build smoke test for Phase 1 deployment readiness."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    shutil.which("docker") is None or os.environ.get("GHOSTEXEC_RUN_DOCKER_BUILD") != "1",
    reason="Set GHOSTEXEC_RUN_DOCKER_BUILD=1 and ensure docker is installed to run this test.",
)
def test_server_dockerfile_builds():
    image_tag = "ghostexec-env:ci"
    build_cmd = ["docker", "build", "-t", image_tag, "."]
    built = subprocess.run(
        build_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert built.returncode == 0, (
        "docker build failed\n"
        f"stdout:\n{built.stdout}\n"
        f"stderr:\n{built.stderr}\n"
    )

    inspect_cmd = ["docker", "image", "inspect", image_tag]
    inspected = subprocess.run(
        inspect_cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert inspected.returncode == 0, (
        f"image inspect failed for {image_tag}\n"
        f"stdout:\n{inspected.stdout}\n"
        f"stderr:\n{inspected.stderr}\n"
    )
