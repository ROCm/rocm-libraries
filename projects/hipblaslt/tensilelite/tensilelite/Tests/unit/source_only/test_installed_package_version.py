# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check package initialization and CLI versions from a real installed wheel."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _run(*args, cwd, env):
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_installed_wheel_cli_reports_distribution_version(tmp_path):
    source = tmp_path / "source"
    runpy.run_path(str(_SOURCE_ROOT / "scripts/stage_release_source.py"))[
        "stage_source"
    ](_SOURCE_ROOT, source)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "TENSILELITE_ROCM_VERSION",
            "TENSILELITE_SOURCE_REVISION",
            "TENSILELITE_ROCM_ADAPTER",
            "THEROCK_PACKAGE_VERSION",
            "ROCM_PATH",
            "ROCM_VERSION",
            "HIP_PATH",
            "PYTHONPATH",
        )
    }
    build_environment = dict(
        environment,
        TENSILELITE_SOURCE_REVISION=_SHA,
        TENSILELITE_ROCM_VERSION="7.12.0rc1",
    )
    wheel_dir = tmp_path / "wheels"
    _run(
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        "--disable-pip-version-check",
        "--wheel-dir",
        str(wheel_dir),
        str(source),
        cwd=tmp_path,
        env=build_environment,
    )
    (wheel,) = wheel_dir.glob("*.whl")
    installed = tmp_path / "installed"
    _run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--target",
        str(installed),
        "--disable-pip-version-check",
        str(wheel),
        cwd=tmp_path,
        env=environment,
    )
    rocm = tmp_path / "rocm"
    (rocm / ".info").mkdir(parents=True)
    (rocm / ".info/version").write_text("7.12.0\n")
    runtime_environment = dict(
        environment, PYTHONPATH=str(installed), ROCM_PATH=str(rocm)
    )
    values = json.loads(
        _run(
            sys.executable,
            "-c",
            "import json, tensilelite; from importlib.metadata import version; "
            "print(json.dumps([version('tensilelite'), tensilelite.__version__, "
            "tensilelite.GENERATOR_VERSION, tensilelite.__file__]))",
            cwd=tmp_path,
            env=runtime_environment,
        )
    )
    assert values[:3] == ["5.0.0+rocm7.12.0rc1", "5.0.0+rocm7.12.0rc1", "5.0.0"]
    assert Path(values[3]).is_relative_to(installed)
    cli_version = _run(
        sys.executable,
        "-m",
        "tensilelite",
        "--version",
        cwd=tmp_path,
        env=runtime_environment,
    )
    assert cli_version == values[0] + "\n"
