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

__all__ = [
    "build_platform_targets",
    "cmake_cache_entries",
    "configure_platform",
    "generator_arguments",
]


def generator_arguments(build_root: Path) -> list[str]:
    """The generator argument of a configure into ``build_root``: ``-GNinja``
    for a fresh tree when ``ninja`` is on PATH, else none. An existing tree
    keeps its generator, which CMake cannot change; a fresh tree without Ninja
    gets CMake's default."""
    if (build_root / "CMakeCache.txt").exists() or shutil.which("ninja") is None:
        return []
    return ["-GNinja"]


def cmake_cache_entries(build_root: Path) -> dict[str, str]:
    """The ``NAME:TYPE=value`` entries of the configured tree ``build_root``'s
    CMakeCache.txt, each value by its name: what the configure recorded, such
    as ``CMAKE_COMMAND``, the CMake program that configured the tree. Raises
    ``FileNotFoundError`` when ``build_root`` has no cache."""
    entries = {}
    cache = build_root / "CMakeCache.txt"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            entries[key.split(":", 1)[0]] = value
    return entries


def configure_platform(
    build_root: Path, *cmake_arguments: str, config: str = "Release"
) -> None:
    """Configure the platform source tree into ``build_root`` for ``config``,
    with this interpreter as the one its build steps and CTest entries run.

    :func:`generator_arguments` picks the generator. A -D in
    ``cmake_arguments`` overrides a -D this function passes, because CMake
    keeps the last value. CMake's progress goes nowhere; its errors reach
    stderr. Raises ``subprocess.CalledProcessError`` when the configure fails."""
    command = [
        "cmake",
        "-S",
        str(platform_root()),
        "-B",
        str(build_root),
        f"-DCMAKE_BUILD_TYPE={config}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        *generator_arguments(build_root),
        *cmake_arguments,
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def build_platform_targets(
    build_root: Path, targets: Sequence[str], *, config: str = "Release"
) -> None:
    """Build ``targets`` of the configured tree ``build_root`` with this
    process's share of the CPUs; with no ``targets``, the tree's default
    target, which builds every program the configure defines. The build's
    output is shown: the compiler reports errors there. Raises
    ``subprocess.CalledProcessError`` when the build fails."""
    target_arguments = ["--target", *targets] if targets else []
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_root),
            "--config",
            config,
            *target_arguments,
            "--parallel",
            str(available_cpus()),
        ],
        check=True,
    )
