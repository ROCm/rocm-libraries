# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.cmake_generator.ninja_generator_or_skip gives a test's configure
Ninja where cmake and ninja are on PATH, and skips the test, waiting for the
build-variant lane, where either is missing (CI's test container has no build
program). Each case puts PATH on a directory that holds the named tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from rocke.tests.cmake_generator import ninja_generator_or_skip


def _tools_on_path(directory: Path, *names: str) -> str:
    """``directory``, created, holding an executable file for each of
    ``names``."""
    directory.mkdir()
    for name in names:
        tool = directory / (f"{name}.exe" if os.name == "nt" else name)
        tool.write_text("")
        tool.chmod(0o755)
    return str(directory)


def test_a_test_configure_with_cmake_and_ninja_gets_ninja(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", _tools_on_path(tmp_path / "bin", "cmake", "ninja"))
    assert ninja_generator_or_skip(tmp_path / "build") == ["-GNinja"]


@pytest.mark.parametrize(
    "tools", [(), ("cmake",), ("ninja",)], ids=["none", "cmake-only", "ninja-only"]
)
def test_a_test_configure_without_cmake_or_ninja_waits_for_the_build_variant_lane(
    tmp_path, monkeypatch, tools
):
    monkeypatch.setenv("PATH", _tools_on_path(tmp_path / "bin", *tools))
    with pytest.raises(pytest.skip.Exception, match="waits for build-variant"):
        ninja_generator_or_skip(tmp_path / "build")
