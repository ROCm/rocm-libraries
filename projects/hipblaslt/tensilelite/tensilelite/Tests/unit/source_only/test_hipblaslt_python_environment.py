# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]
_CMAKE_HELPER = _HIPBLASLT_ROOT / "cmake" / "hipblaslt_python.cmake"


def _configured_environment(tmp_path: Path, *, therock: bool) -> str:
    script = tmp_path / "environment.cmake"
    script.write_text(
        "\n".join(
            [
                f'set(HIPBLASLT_ENABLE_THEROCK {"ON" if therock else "OFF"})',
                'set(HIPBLASLT_BUILD_ROCM_ROOT "/graph/rocm")',
                'set(HIPBLASLT_BUILD_ROCM_VERSION "10.1.0.dev0+abcdef")',
                'set(ENV{PATH} "/graph/bin:/ambient/bin")',
                f'include("{_CMAKE_HELPER.as_posix()}")',
                "hipblaslt_tensilelite_python_environment(environment)",
                'message(STATUS "environment=${environment}")',
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def test_therock_environment_uses_package_identity_and_captured_path(tmp_path):
    environment = _configured_environment(tmp_path, therock=True)

    assert "THEROCK_PACKAGE_VERSION=10.1.0.dev0+abcdef" in environment
    assert "TENSILELITE_ROCM_VERSION=10.1.0.dev0+abcdef" in environment
    assert "PATH=/graph/bin:/ambient/bin" in environment
    assert "ROCM_PATH=" not in environment


def test_standalone_environment_retains_selected_rocm_path(tmp_path):
    environment = _configured_environment(tmp_path, therock=False)

    assert "ROCM_PATH=/graph/rocm" in environment
    assert "TENSILELITE_ROCM_VERSION=10.1.0.dev0+abcdef" in environment
    assert "THEROCK_PACKAGE_VERSION=" not in environment
