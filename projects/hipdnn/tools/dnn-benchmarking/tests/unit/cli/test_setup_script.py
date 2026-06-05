# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the dnn-benchmarking setup script's cheap UX paths."""

import os
import subprocess
from pathlib import Path

import pytest


SETUP_SCRIPT = Path(__file__).resolve().parents[3] / "setup.sh"


def run_setup(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DNN_BENCH_WORKSPACE"] = str(tmp_path / "workspace")
    return subprocess.run(
        ["bash", str(SETUP_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_setup_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SETUP_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_help_documents_rocm_cpu_and_existing_torch_modes(tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--help")

    assert result.returncode == 0
    assert "--torch-mode <rocm|cpu|existing|none>" in result.stdout
    assert "Does not require a system" in result.stdout
    assert "CPU/non-ROCm torch uses installed ROCm/hipDNN" in result.stdout
    assert "--skip-torch-install" not in result.stdout
    assert "--install-dir" not in result.stdout


@pytest.mark.parametrize(
    "option",
    [
        "--rocm-prefix",
        "--torch-mode",
        "--torch-index-url",
        "--gpu-arch",
    ],
)
def test_value_options_report_missing_argument(tmp_path: Path, option: str) -> None:
    result = run_setup(tmp_path, option)

    assert result.returncode == 1
    assert f"ERROR: {option} requires a value." in result.stderr
    assert "Usage:" in result.stdout


def test_value_options_reject_next_option_as_missing_argument(tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--rocm-prefix", "--torch-mode")

    assert result.returncode == 1
    assert "ERROR: --rocm-prefix requires a value." in result.stderr
    assert "Usage:" in result.stdout
    assert not (tmp_path / "workspace" / ".venv").exists()


def test_existing_torch_mode_requires_existing_venv(tmp_path: Path) -> None:
    result = run_setup(tmp_path, "--torch-mode", "existing")

    assert result.returncode == 1
    assert "requires an existing virtual environment" in result.stderr
    assert not (tmp_path / "workspace" / ".venv").exists()


@pytest.mark.parametrize(
    ("legacy_arg", "args"),
    [
        ("--skip-torch-install", ("--skip-torch-install",)),
        ("--install-dir", ("--install-dir", "/tmp/rocm")),
    ],
)
def test_removed_legacy_args_are_unknown(
    tmp_path: Path, legacy_arg: str, args: tuple[str, ...]
) -> None:
    result = run_setup(tmp_path, *args)

    assert result.returncode == 1
    assert f"Unknown argument: {legacy_arg}" in result.stdout
    assert not (tmp_path / "workspace" / ".venv").exists()
