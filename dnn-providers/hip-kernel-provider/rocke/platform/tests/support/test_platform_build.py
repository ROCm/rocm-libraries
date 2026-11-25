# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.platform_build.generator_arguments gives a fresh build tree Ninja
when ``ninja`` is on PATH and CMake's default otherwise, and leaves an existing
tree its generator. Each case puts PATH on a directory that holds a ``ninja``
or holds nothing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from rocke.tests.platform_build import generator_arguments


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
