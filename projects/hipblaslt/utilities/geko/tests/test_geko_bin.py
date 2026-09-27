# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
