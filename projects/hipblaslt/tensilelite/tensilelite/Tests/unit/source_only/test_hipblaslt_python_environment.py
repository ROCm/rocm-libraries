# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]
_CMAKE_HELPER = _HIPBLASLT_ROOT / "cmake" / "hipblaslt_python.cmake"


def _configured_environment(
    tmp_path: Path, *, therock: bool, version: str = "10.1.0.dev0+abcdef"
) -> str:
    script = tmp_path / "environment.cmake"
    script.write_text(
        "\n".join(
            [
                f'set(HIPBLASLT_ENABLE_THEROCK {"ON" if therock else "OFF"})',
                'set(hip_VERSION "7.2.53211")',
                'set(HIPBLASLT_BUILD_ROCM_ROOT "/graph/rocm")',
                f'set(HIPBLASLT_BUILD_ROCM_VERSION "{version}")',
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


def test_therock_environment_marks_the_adapter_and_passes_a_real_version(tmp_path):
    environment = _configured_environment(tmp_path, therock=True)

    assert "TENSILELITE_ROCM_ADAPTER=therock-ci" in environment
    assert "THEROCK_PACKAGE_VERSION=10.1.0.dev0+abcdef" in environment
    assert "ROCM_PATH=" not in environment
    assert "TENSILELITE_ROCM_VERSION" not in environment


def test_therock_environment_without_a_version_still_marks_the_adapter(tmp_path):
    environment = _configured_environment(tmp_path, therock=True, version="")

    assert "TENSILELITE_ROCM_ADAPTER=therock-ci" in environment
    assert "THEROCK_PACKAGE_VERSION" not in environment


@pytest.mark.parametrize("therock", [True, False])
def test_generator_environment_carries_the_hip_version(tmp_path, therock):
    # A TheRock build has no /opt/rocm, ROCM_PATH or HIP_PATH for
    # tensilelite.Toolchain.Component to search when it is imported, so the
    # version must come from the configure, as in develop's command.
    environment = _configured_environment(tmp_path, therock=therock)

    assert "ROCM_VERSION=7.2.53211" in environment


def test_standalone_environment_retains_selected_rocm_path(tmp_path):
    environment = _configured_environment(tmp_path, therock=False)

    assert "ROCM_PATH=/graph/rocm" in environment
    assert "TENSILELITE_ROCM_ADAPTER" not in environment
    assert "THEROCK_PACKAGE_VERSION=" not in environment
    assert "TENSILELITE_ROCM_VERSION" not in environment
