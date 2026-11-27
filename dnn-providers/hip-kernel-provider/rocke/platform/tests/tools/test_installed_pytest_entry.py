# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke_installed_pytest_entry writes the one shape of an installed pytest entry.

The function lives in tests/cmake/InstalledPytestEntry.cmake. These tests run it
in CMake's script mode with stand-in command pieces, so they need cmake but no
configure.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "cmake" / "InstalledPytestEntry.cmake"

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="needs cmake on PATH"
)

# The policies of the platform project, and the command pieces its
# CMakeLists.txt defines, as stand-ins.
_PIECES = """
cmake_minimum_required(VERSION 3.16)
set(_ROCKE_INSTALLED_PYTEST "python -m pytest -p shard")
set(_ROCKE_INSTALLED_PYTEST_WORKERS "-p workers")
set(_ROCKE_INSTALLED_PYTEST_GPU_TEST_LIMIT "--timeout=120 -p gpu_policy")
set(_ROCKE_INSTALLED_GPU_RESOURCE_LOCK "rocke_gpu")
"""


def _run(call: str, scratch: Path) -> subprocess.CompletedProcess:
    """Run ``call`` (a rocke_installed_pytest_entry call that sets ``entry``)
    in script mode, printing ``entry`` to stderr."""
    script = scratch / "entry.cmake"
    script.write_text(
        f'{_PIECES}include("{MODULE.as_posix()}")\n{call}\nmessage("${{entry}}")\n'
    )
    return subprocess.run(
        ["cmake", "-P", str(script)], capture_output=True, text=True, check=False
    )


def test_an_entry_has_the_shared_command_its_properties_and_quoted_arguments(
    tmp_path,
):
    result = _run(
        "rocke_installed_pytest_entry(entry rocke_x_pytest WORKERS GPU\n"
        "  TESTS a.py dir/b.py::test_c\n"
        '  ARGUMENTS -m "not exhaustive" -v\n'
        "  LABELS ckc python gpu\n"
        '  ENVIRONMENT "PYTHONPATH=lib"\n'
        "  TIMEOUT 1800)",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.strip("\n").splitlines() == [
        (
            'add_test(rocke_x_pytest python -m pytest -p shard -p workers "a.py" '
            '"dir/b.py::test_c" "-m" "not exhaustive" "-v" '
            "--timeout=120 -p gpu_policy)"
        ),
        "set_tests_properties(rocke_x_pytest PROPERTIES",
        '  LABELS "ckc;python;gpu"',
        '  ENVIRONMENT "PYTHONPATH=lib"',
        "  RESOURCE_LOCK rocke_gpu",
        "  TIMEOUT 1800)",
    ]


def test_the_compiler_options_extend_the_entrys_environment_list(tmp_path):
    result = _run(
        "rocke_installed_pytest_entry(entry rocke_x_pytest\n"
        "  COMPILER_TOOLS COMPILER_REQUIREMENT\n"
        "  TESTS a.py LABELS ckc ENVIRONMENT PYTHONPATH=lib TIMEOUT 900)",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stderr.splitlines()
    assert lines[:4] == [
        'set(_rocke_x_pytest_environment "PYTHONPATH=lib")',
        "list(APPEND _rocke_x_pytest_environment",
        "     ${_rocke_compiler_tool_environment}",
        "     ${_rocke_compiler_requirement_environment})",
    ]
    assert '  ENVIRONMENT "${_rocke_x_pytest_environment}"' in lines


@pytest.mark.parametrize("missing", ["LABELS", "ENVIRONMENT", "TIMEOUT"])
def test_an_entry_without_a_required_property_stops_the_configure(tmp_path, missing):
    given = {
        "LABELS": "LABELS ckc",
        "ENVIRONMENT": "ENVIRONMENT A=1",
        "TIMEOUT": "TIMEOUT 9",
    }
    del given[missing]
    result = _run(
        f"rocke_installed_pytest_entry(entry rocke_x_pytest TESTS a.py "
        f"{' '.join(given.values())})",
        tmp_path,
    )
    assert result.returncode != 0
    assert f"rocke_x_pytest): {missing} is required" in " ".join(result.stderr.split())


def test_an_unknown_argument_stops_the_configure(tmp_path):
    result = _run(
        "rocke_installed_pytest_entry(entry rocke_x_pytest TESTS a.py "
        "LABELS ckc ENVIRONMENT A=1 TIMEOUT 9 SKIP_RETURN_CODE 77)",
        tmp_path,
    )
    assert result.returncode != 0
    assert "rocke_x_pytest): unknown arguments SKIP_RETURN_CODE;77" in " ".join(
        result.stderr.split()
    )
