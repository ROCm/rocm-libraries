################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Subprocess smoke tests for bin/geko (argparse and early validation only).

Skip with pytest --skip-geko-bin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_GEKO = REPO_ROOT / "bin" / "geko"


def _run_bin(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BIN_GEKO), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.geko_bin
def test_bin_geko_help_exits_zero() -> None:
    r = _run_bin("--help")
    assert r.returncode == 0
    assert "--hipblaslt" in r.stdout
    assert "--workload-log" in r.stdout or "--list" in r.stdout


@pytest.mark.geko_bin
def test_bin_geko_rejects_missing_workload_source(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    r = _run_bin("--bench", "--hipblaslt", str(hip))
    assert r.returncode != 0


@pytest.mark.geko_bin
def test_bin_geko_rejects_nonexistent_hipblaslt_dir(tmp_path: Path) -> None:
    missing = tmp_path / "not_a_hip_dir"
    r = _run_bin(
        "--bench",
        "--hipblaslt",
        str(missing),
        "--devices",
        "0",
        "--inline",
        "16",
        "16",
        "1",
        "16",
        "B",
        "B",
        "S",
        "N",
        "N",
    )
    assert r.returncode != 0


@pytest.mark.geko_bin
def test_bin_geko_rejects_missing_workload_file(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    ghost = tmp_path / "nope.yaml"
    r = _run_bin(
        "--bench",
        "--hipblaslt",
        str(hip),
        "--devices",
        "0",
        "--workload-log",
        str(ghost),
    )
    assert r.returncode != 0


@pytest.mark.geko_bin
def test_bin_geko_rejects_missing_devices(tmp_path: Path) -> None:
    hip = tmp_path / "hip"
    hip.mkdir()
    r = _run_bin(
        "--bench",
        "--hipblaslt",
        str(hip),
        "--inline",
        "16",
        "16",
        "1",
        "16",
        "B",
        "B",
        "S",
        "N",
        "N",
    )
    assert r.returncode != 0
    assert "--devices" in r.stderr or "-d" in r.stderr
