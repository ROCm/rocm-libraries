# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the dnn-benchmarking setup script."""

import subprocess
from pathlib import Path


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup.sh"


def test_setup_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SETUP_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_setup_script_usage_mentions_cuda_mode() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--torch-mode <rocm|cuda|cpu|existing|none>" in result.stdout


def test_setup_script_rejects_force_build_with_cuda_mode() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--torch-mode", "cuda", "--force-build"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--force-build is not supported with --torch-mode cuda" in result.stderr


def test_setup_script_rejects_unknown_torch_mode() -> None:
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--torch-mode", "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "rocm, cuda, cpu, existing, none" in result.stderr
