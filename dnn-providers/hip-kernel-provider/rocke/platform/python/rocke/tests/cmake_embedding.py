# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure the rocKE CMake project embedded in a wrapper project, as the provider embeds it.

rocke/ declares no ``cmake_minimum_required`` of its own: it is meant to be added
with ``add_subdirectory`` by a parent project. CMake 4 refuses such a project as a
top level, so every test that configures rocke/ goes through this wrapper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["configure_embedded_rocke"]


def configure_embedded_rocke(
    rocke: Path, build: Path, *cmake_arguments: str
) -> subprocess.CompletedProcess[str]:
    """Write a wrapper project beside ``build`` that adds the rocKE tree at
    ``rocke`` as a subdirectory, configure it into ``build`` with
    ``cmake_arguments`` and return the finished process. Nothing is built; the
    caller checks the return code."""
    embedding = build.parent / "embedding source"
    embedding.mkdir(parents=True, exist_ok=True)
    (embedding / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(rocke_embedding_contract LANGUAGES C CXX)\n"
        f'add_subdirectory("{rocke.as_posix()}" rocke)\n'
    )
    return subprocess.run(
        ["cmake", "-S", str(embedding), "-B", str(build), *cmake_arguments],
        capture_output=True,
        check=False,
        text=True,
    )
