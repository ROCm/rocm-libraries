# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CMAKE_MODULE = (
    Path(__file__).parents[4] / "cmake" / "tensilelite_supported_architectures.cmake"
)


def _run_cmake(tmp_path, body, *, asan=False):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is required for TensileLite target-normalization tests")
    script = tmp_path / "test.cmake"
    script.write_text(
        f'set(HIPBLASLT_ENABLE_ASAN {"ON" if asan else "OFF"})\n'
        f'include("{_CMAKE_MODULE}")\n'
        f"{body}\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [cmake, "-P", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("target", ["gfx90c", "gfx90c:xnack+", "gfx90c:xnack-"])
def test_cmake_accepts_all_gfx90c_target_forms(tmp_path, target):
    result = _run_cmake(
        tmp_path, f'tensilelite_validate_gpu_targets("{target}")'
    )
    assert result.returncode == 0, result.stderr


def test_sanitizer_normalizes_bare_gfx90c_to_xnack_plus(tmp_path):
    result = _run_cmake(
        tmp_path,
        'tensilelite_offload_target(actual "gfx90c")\n'
        'if(NOT actual STREQUAL "gfx90c:xnack+")\n'
        '  message(FATAL_ERROR "unexpected target: ${actual}")\n'
        "endif()",
        asan=True,
    )
    assert result.returncode == 0, result.stderr


def test_sanitizer_rejects_explicit_gfx90c_xnack_minus(tmp_path):
    result = _run_cmake(
        tmp_path,
        'tensilelite_offload_target(actual "gfx90c:xnack-")',
        asan=True,
    )
    assert result.returncode != 0
    assert "explicitly disables XNACK" in result.stderr


def test_sanitizer_leaves_unrelated_architecture_unchanged(tmp_path):
    result = _run_cmake(
        tmp_path,
        'tensilelite_offload_target(actual "gfx1100")\n'
        'if(NOT actual STREQUAL "gfx1100")\n'
        '  message(FATAL_ERROR "unexpected target: ${actual}")\n'
        "endif()",
        asan=True,
    )
    assert result.returncode == 0, result.stderr
