# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure and build the platform's CMake project.

One home for the recipe the runners and tests share: which source tree, which
interpreter the build steps and CTest entries run, which generator a fresh tree
gets, and how many compile jobs a build may start.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ..assets import platform_root
from .host_cpus import available_cpus

__all__ = ["build_platform_targets", "configure_platform"]


def configure_platform(
    build_root: Path, *cmake_arguments: str, config: str = "Release"
) -> None:
    """Configure the platform source tree into ``build_root`` for ``config``,
    with this interpreter as the one its build steps and CTest entries run.

    A fresh tree uses Ninja when it is on PATH; an existing tree keeps its
    generator, which CMake cannot change. A -D in ``cmake_arguments`` overrides
    a default above, because CMake keeps the last value. CMake's progress goes
    nowhere; its errors reach stderr. Raises ``subprocess.CalledProcessError``
    when the configure fails."""
    command = [
        "cmake",
        "-S",
        str(platform_root()),
        "-B",
        str(build_root),
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if not (build_root / "CMakeCache.txt").exists() and shutil.which("ninja"):
        command.append("-GNinja")
    command += cmake_arguments
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def build_platform_targets(
    build_root: Path, targets: Sequence[str], *, config: str = "Release"
) -> None:
    """Build ``targets`` of the configured tree ``build_root`` with this
    process's share of the CPUs. The build's output is shown: the compiler
    reports errors there. Raises ``subprocess.CalledProcessError`` when the
    build fails."""
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_root),
            "--config",
            config,
            "--target",
            *targets,
            "--parallel",
            str(available_cpus()),
        ],
        check=True,
    )
