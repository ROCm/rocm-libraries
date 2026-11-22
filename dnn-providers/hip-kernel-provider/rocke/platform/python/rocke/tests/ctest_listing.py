# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CTest entries registered in a directory, as ``ctest --show-only=json-v1`` lists them.

One home for reading a registration without running it: the runners and checks
that need entry names, commands, labels or properties ask CTest rather than
parsing ``CTestTestfile.cmake``. CTest omits an entry's ``command`` when it
cannot find the program, so list a built tree, with the test interpreter on
``PATH`` for script entries.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from ..assets import platform_root

__all__ = ["configure_installed_ctest", "list_ctest_entries"]


def list_ctest_entries(
    test_dir: Path,
    *,
    config: str | None = None,
    name_pattern: str | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict]:
    """The json-v1 ``tests`` of ``test_dir``, optionally for one build
    ``config`` and only names matching the ``name_pattern`` regex (``-R``).
    ``env`` replaces the environment CTest evaluates the registration in."""
    command = ["ctest", "--test-dir", str(test_dir)]
    if config is not None:
        command += ["-C", config]
    command.append("--show-only=json-v1")
    if name_pattern is not None:
        command += ["-R", name_pattern]
    listed = subprocess.run(
        command, check=True, capture_output=True, text=True, env=env
    )
    return json.loads(listed.stdout)["tests"]


def configure_installed_ctest(
    scratch: Path, *cmake_arguments: str
) -> tuple[Path, Path]:
    """Configure the platform source tree with BUILD_TESTING into
    ``scratch/build`` and return that build directory and ``scratch/installed-ctest``,
    a directory holding its installed CTestTestfile as an install lays it out.
    Nothing is built. A -D in ``cmake_arguments`` overrides the default
    ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF, because CMake keeps the last value."""
    build = scratch / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(platform_root()),
            "-B",
            str(build),
            "-DBUILD_TESTING=ON",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
            *cmake_arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ctest_dir = scratch / "installed-ctest"
    ctest_dir.mkdir()
    shutil.copy2(
        build / "CTestTestfile.cmake.install", ctest_dir / "CTestTestfile.cmake"
    )
    return build, ctest_dir
