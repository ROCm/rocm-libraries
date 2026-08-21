# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.unit

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_CMAKE = _TENSILELITE_ROOT.parent / "cmake" / "hipblaslt_python.cmake"


def _resolve_version(
    tmp_path: Path, *, package_version: str | None = None
) -> subprocess.CompletedProcess:
    script = tmp_path / "resolve.cmake"
    package_line = (
        f'set(THEROCK_PACKAGE_VERSION "{package_version}")\n'
        if package_version is not None
        else ""
    )
    script.write_text(
        "set(HIPBLASLT_ENABLE_THEROCK ON)\n"
        f"{package_line}"
        f'include("{_PYTHON_CMAKE.as_posix()}")\n'
        "hipblaslt_resolve_build_rocm_version(resolved)\n"
        'message(STATUS "resolved=${resolved}")\n',
        encoding="utf-8",
    )
    return subprocess.run(
        ["cmake", "-P", str(script)],
        capture_output=True,
        text=True,
    )


def test_therock_uses_warned_provisional_identity_before_validation(tmp_path):
    result = _resolve_version(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "resolved=0.0.0" in result.stdout
    assert "temporary 0.0.0 ROCm identity" in " ".join(result.stderr.split())


def test_therock_prefers_forwarded_package_identity(tmp_path):
    result = _resolve_version(tmp_path, package_version="10.1.0a20260813")

    assert result.returncode == 0, result.stderr
    assert "resolved=10.1.0a20260813" in result.stdout
