# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The CMake parity-emitter target compiles its .c source as C++20."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _run(command):
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("compiler", ["clang++", "g++"])
def test_cmake_parity_emitter_compiles_with_warnings_as_errors(tmp_path, compiler):
    compiler_path = shutil.which(compiler)
    if not compiler_path or not shutil.which("cmake") or not shutil.which("ninja"):
        pytest.skip(f"CMake, Ninja and {compiler} are required")

    platform = Path(__file__).resolve().parents[3]
    build = tmp_path / "build"
    _run(
        [
            "cmake",
            "-S",
            str(platform),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_CXX_COMPILER={compiler_path}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS=-Werror",
            "-DBUILD_TESTING=ON",
            "-DROCKE_BUILD_PYBIND=OFF",
            "-DROCKE_BUILD_SHARED_ENGINE=OFF",
            "-DROCKE_INSTALL_TESTS=OFF",
        ]
    )

    # Compile the actual generated object recipe, including the platform's
    # warning flags, without linking or rebuilding the full engine archive.
    suffix = ".obj" if os.name == "nt" else ".o"
    target = (
        "tests/CMakeFiles/rocke_optimization_barrier_emit.dir/"
        f"instances/parity/optimization_barrier_emit.c{suffix}"
    )
    _run(["cmake", "--build", str(build), "--target", target, "--verbose"])
    assert (build / target).is_file()
