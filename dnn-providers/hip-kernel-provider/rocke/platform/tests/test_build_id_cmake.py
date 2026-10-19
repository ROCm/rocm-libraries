# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The engine build id follows the inputs of the generated arch-domain table.

``rocke_compute_build_id`` runs while CMake configures, and the build rewrites
the C++ table whenever a column or the waivers change. A build directory that
already exists must therefore compute the id again, or the new table carries the
old id. Each test configures a small project that includes the real module,
edits one input, builds, and reads the id back. No compiler is needed.
"""

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

_PLATFORM = Path(__file__).resolve().parents[1]
_MODULE = _PLATFORM / "cmake/rocke_build_id.cmake"
_CMAKE = shutil.which("cmake")
# Ninja where there is one: it is the generator that CI uses. Any generator
# re-runs CMake for a configure dependency.
_GENERATOR = ["-G", "Ninja"] if shutil.which("ninja") else []
pytestmark = pytest.mark.skipif(
    _CMAKE is None or not _MODULE.exists(),
    reason="needs CMake and the platform CMake modules (absent when installed)",
)

# The files that the module hashes besides the engine sources: the columns, the
# waivers, and the reader and generator that turn them into the C++ table.
_TABLE_INPUTS = [
    "python/rocke/core/arch/data/intrinsic_arch_domain.llvm22.json",
    "python/rocke/core/arch/data/arch_domain_waivers.json",
    "python/rocke/core/arch/domain.py",
    "tools/gen_arch_domain_cpp.py",
]

_CMAKELISTS = """\
cmake_minimum_required(VERSION 3.16)
project(build_id_probe NONE)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/rocke_build_id.cmake")
rocke_compute_build_id(ROCKE_BUILD_ID "${CMAKE_CURRENT_SOURCE_DIR}/engine")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/id.txt" "${ROCKE_BUILD_ID}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/configures.txt" "x")
add_custom_target(probe ALL COMMAND "${CMAKE_COMMAND}" -E true)
"""


def _age(path, seconds):
    """Move the modification time of everything under ``path`` into the past.

    A build is stale when an input is newer than the build system, so the
    build system must be older than the edit by more than the file system's
    time resolution. Dating the edit ahead instead would leave Ninja looking at
    an input newer than the file that it has just written.
    """
    past = time.time() - seconds
    for item in [path, *path.rglob("*")]:
        os.utime(item, (past, past))


@pytest.fixture
def project(tmp_path):
    engine = tmp_path / "engine"
    for relative in [
        "cpp/core/engine.cpp",
        "cpp/include/rocke/engine.h",
        "python/rocke/core/arch/data/intrinsic_arch_domain.llvm20.json",
        *_TABLE_INPUTS,
    ]:
        path = engine / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n")
    # A copy, so that the age of the module in the checkout cannot make the
    # build system look stale.
    (tmp_path / "cmake").mkdir()
    shutil.copyfile(_MODULE, tmp_path / "cmake/rocke_build_id.cmake")
    (tmp_path / "CMakeLists.txt").write_text(_CMAKELISTS)
    _age(tmp_path, 1000)
    return tmp_path


def _configure(project, build):
    subprocess.run(
        [_CMAKE, *_GENERATOR, "-S", str(project), "-B", str(build)],
        check=True,
        capture_output=True,
        text=True,
    )
    _age(build, 1000)


def _build(build):
    subprocess.run(
        [_CMAKE, "--build", str(build)], check=True, capture_output=True, text=True
    )


def _build_id(build):
    return (build / "id.txt").read_text()


def _configure_runs(build):
    return len((build / "configures.txt").read_text())


@pytest.mark.parametrize("relative", _TABLE_INPUTS)
def test_editing_a_table_input_recomputes_the_id(project, relative):
    build = project / "build"
    _configure(project, build)
    before = _build_id(build)
    assert len(before) == 16, before

    _build(build)
    assert _configure_runs(build) == 1, "a build of an unchanged tree re-ran CMake"

    with (project / "engine" / relative).open("a") as handle:
        handle.write("edited\n")
    _build(build)
    assert _configure_runs(build) == 2, "the build did not re-run CMake after the edit"
    after = _build_id(build)
    assert after != before, "the id is the same after an edit to an input"

    # The id that the build computed must be the one that a new build directory
    # computes for the same tree.
    fresh = project / "fresh"
    _configure(project, fresh)
    assert _build_id(fresh) == after
