# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CMake's file-API replies give the files a configure read and each target's files.

``rocke.tests.cmake_file_api`` asks a configure for the replies and reads them.
The cases configure a small real project (no compiler: LANGUAGES NONE) that
includes a CMake file from inside its tree and one from beside it, with the
generator a fresh platform tree gets (``rocke.tests.platform_build``).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from rocke.tests.cmake_file_api import read_codemodel, request_codemodel
from rocke.tests.lanes import BUILD_VARIANT, skip_until
from rocke.tests.platform_build import generator_arguments

PROJECT = """\
cmake_minimum_required(VERSION 3.16)
project(file_api_demo LANGUAGES NONE)
include("${CMAKE_CURRENT_SOURCE_DIR}/inside.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/../beside.cmake")
add_custom_target(base SOURCES base.txt)
add_custom_target(top SOURCES top.txt)
add_dependencies(top base)
add_custom_target(other SOURCES other.txt)
"""


@pytest.fixture(scope="module")
def configured(tmp_path_factory):
    """The project's source and build trees, configured for Release."""
    scratch = tmp_path_factory.mktemp("file_api")
    source = scratch / "project"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(PROJECT)
    (source / "inside.cmake").write_text("set(INSIDE 1)\n")
    (scratch / "beside.cmake").write_text("set(BESIDE 1)\n")
    for name in ("base.txt", "top.txt", "other.txt"):
        (source / name).write_text("text\n")
    build = scratch / "build"
    generator = generator_arguments(build)
    # The tree is fresh, so no generator argument means ninja is not on PATH.
    if shutil.which("cmake") is None or not generator:
        skip_until(
            BUILD_VARIANT,
            reason="needs cmake and ninja on PATH: it configures with Ninja, the "
            "generator a fresh platform tree gets",
        )
    request_codemodel(build)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            *generator,
        ],
        check=True,
        capture_output=True,
    )
    return source.resolve(), build


def test_the_inputs_are_the_project_files_the_configure_read(configured):
    source, build = configured
    model = read_codemodel(build, "Release")
    # CMake's own modules are left out; a file outside the tree is kept.
    assert model.input_files == {
        source / "CMakeLists.txt",
        source / "inside.cmake",
        source.parent / "beside.cmake",
    }


def test_a_target_links_the_targets_it_depends_on(configured):
    source, build = configured
    model = read_codemodel(build, "Release")
    assert model.linked_targets(["top"]) == {"top", "base"}
    assert model.targets["top"].sources == {source / "top.txt"}
    assert model.targets["top"].dependencies == {"base"}


def test_an_unknown_target_is_refused(configured):
    _, build = configured
    model = read_codemodel(build, "Release")
    with pytest.raises(
        KeyError,
        match=r"reading CMake targets failed - target='absent', expected a target "
        r"the configure defined",
    ):
        model.linked_targets(["top", "absent"])


def test_a_configuration_the_replies_lack_is_refused(configured):
    _, build = configured
    with pytest.raises(
        KeyError,
        match=r"reading the CMake file API failed - config='Debug', expected one of "
        r"\['Release'\]",
    ):
        read_codemodel(build, "Debug")


def test_a_tree_configured_without_the_request_is_refused(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match=r"reading the CMake file API failed - no reply in .*, expected "
        r"request_codemodel before the configure",
    ):
        read_codemodel(tmp_path, "Release")
