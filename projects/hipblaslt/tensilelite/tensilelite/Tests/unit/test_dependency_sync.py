# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Keep the source checkout's dependency declarations and exports synchronized."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_dependency_sync():
    project = Path(__file__).resolve().parents[3]
    # Installed test artifacts and wheels omit the top-level CMakeLists.txt.
    # Use a build-source marker so missing dependency files still fail in checkouts.
    if not (project / "CMakeLists.txt").is_file():
        pytest.skip("Dependency synchronization is checked only in source checkouts")

    # A deleted pyproject must fail here instead of letting uv discover a parent project.
    assert (project / "pyproject.toml").is_file(), "Missing source pyproject.toml"
    # Tox provisions uv on PATH; uv-based development already has it.
    assert shutil.which("uv"), (
        "Dependency synchronization needs uv on PATH. Run these tests through tox "
        "or install uv before running pytest directly."
    )
    result = subprocess.run(
        [sys.executable, str(project / "scripts/generate_requirements.py"), "--check"],
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, result.stdout
