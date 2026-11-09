# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: find a C++ test program the build made, named by an environment variable.

A Python test that compares the C++ engine with the Python one through a small C++
program reads the program's path from an environment variable (for example
``ROCKE_INTEXPR_CLI``). ``tests/run_all.py`` sets it from a configured build and
the installed ``rocke_pytest_*`` ctest entries set it. With the variable unset the
test skips; a value that names no file is a broken setup, so it fails instead of
skipping.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

__all__ = ["assert_refuses_arguments", "built_program"]


def built_program(env_var: str, target: str) -> Path:
    """The program ``env_var`` names; skips when it is unset, fails when it names
    no file. ``target`` is the CMake target, named in the skip reason."""
    executable = os.environ.get(env_var)
    if not executable:
        pytest.skip(f"Set {env_var} to the built {target}")
    path = Path(executable).resolve()
    if not path.is_file():
        pytest.fail(f"{env_var} names {path}, not a file")
    return path


def assert_refuses_arguments(program: Path, *arguments: str) -> None:
    """Assert that ``program`` run with ``arguments`` exits 2 with a usage line on
    stderr: a test program must refuse a mistyped mode instead of falling through
    to its self-test and passing."""
    finished = subprocess.run(
        [str(program), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert finished.returncode == 2, (arguments, finished.returncode, finished.stderr)
    assert finished.stderr.startswith(f"usage: {program}"), finished.stderr
