# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.platform_build.generator_arguments gives a fresh build tree Ninja
when ``ninja`` is on PATH and CMake's default otherwise, and leaves an existing
tree its generator. Each case puts PATH on a directory that holds a ``ninja``
or holds nothing.

cmake_cache_entries reads what a real configure of a small project (no
compiler: LANGUAGES NONE) recorded in its CMakeCache.txt."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rocke.tests.lanes import BUILD_VARIANT, skip_until
from rocke.tests.platform_build import cmake_cache_entries, generator_arguments


def _tool_directory(directory: Path, *, with_ninja: bool) -> str:
    """``directory``, created, holding an executable ``ninja`` when ``with_ninja``."""
    directory.mkdir()
    if with_ninja:
        ninja = directory / ("ninja.exe" if os.name == "nt" else "ninja")
        ninja.write_text("")
        ninja.chmod(0o755)
    return str(directory)


def test_a_fresh_tree_gets_ninja_when_it_is_on_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _tool_directory(tmp_path / "bin", with_ninja=True))
    assert generator_arguments(tmp_path / "build") == ["-GNinja"]


def test_a_fresh_tree_without_ninja_gets_the_cmake_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _tool_directory(tmp_path / "bin", with_ninja=False))
    assert generator_arguments(tmp_path / "build") == []


@pytest.mark.parametrize("with_ninja", [True, False])
def test_an_existing_tree_keeps_its_generator(tmp_path, monkeypatch, with_ninja):
    monkeypatch.setenv("PATH", _tool_directory(tmp_path / "bin", with_ninja=with_ninja))
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
    assert generator_arguments(build) == []


def test_cache_entries_are_what_the_configure_recorded(tmp_path):
    if shutil.which("cmake") is None:
        skip_until(BUILD_VARIANT, reason="cmake is not on PATH")
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(cache_demo LANGUAGES NONE)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), "-DDEMO_VALUE:STRING=a=b"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    entries = cmake_cache_entries(build)
    # A value keeps every "=" after the first, which ends the name and type.
    assert entries["DEMO_VALUE"] == "a=b"
    assert Path(entries["CMAKE_HOME_DIRECTORY"]).resolve() == source.resolve()
    assert Path(entries["CMAKE_COMMAND"]).is_file()


def test_an_unconfigured_tree_has_no_cache_entries(tmp_path):
    with pytest.raises(FileNotFoundError):
        cmake_cache_entries(tmp_path)
