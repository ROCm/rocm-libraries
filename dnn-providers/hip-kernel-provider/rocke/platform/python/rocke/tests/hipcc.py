# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: check that hipcc accepts lowered HIP source for one target.

Only the device front end runs (``-fsyntax-only``), so no GPU is needed. A host
without hipcc reports a pytest skip instead of passing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

__all__ = ["assert_hipcc_accepts"]


def assert_hipcc_accepts(source: str, arch: str, tmp_path: Path) -> None:
    """Compile ``source`` for ``arch`` with hipcc; fail with its diagnostics."""
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    path = tmp_path / "kernel.hip"
    path.write_text(source)
    result = subprocess.run(
        [
            "hipcc",
            f"--offload-arch={arch}",
            "--cuda-device-only",
            "-fsyntax-only",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
