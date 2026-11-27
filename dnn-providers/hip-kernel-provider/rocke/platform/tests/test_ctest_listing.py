# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``list_ctest_entries`` reads a real CTest registration without running it,
and ``python_on_path_environment`` lets it list the commands of entries that
run a bare ``python``."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from rocke.tests.ctest_listing import (
    list_ctest_entries,
    list_installed_ctest_entries,
    python_on_path_environment,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ctest") is None, reason="needs ctest on PATH"
)
needs_posix_symlinks = pytest.mark.skipif(
    sys.platform == "win32", reason="the python links here are POSIX symlinks"
)


def _register(test_dir: Path) -> None:
    python = Path(sys.executable).as_posix()
    (test_dir / "CTestTestfile.cmake").write_text(
        f'add_test(rocke_first "{python}" -c "pass")\n'
        'set_tests_properties(rocke_first PROPERTIES LABELS "quick;standard")\n'
        f'add_test(rocke_second "{python}" -c "pass")\n',
        encoding="utf-8",
    )


def test_lists_every_entry_with_its_command_and_properties(tmp_path) -> None:
    _register(tmp_path)
    tests = list_ctest_entries(tmp_path)
    assert [test["name"] for test in tests] == ["rocke_first", "rocke_second"]
    assert tests[0]["command"][1:] == ["-c", "pass"]
    assert {"name": "LABELS", "value": ["quick", "standard"]} in tests[0]["properties"]


def test_name_pattern_selects_entries(tmp_path) -> None:
    _register(tmp_path)
    tests = list_ctest_entries(tmp_path, name_pattern="^rocke_second$")
    assert [test["name"] for test in tests] == ["rocke_second"]


def _path_with_ctest_only(scratch: Path) -> str:
    """A PATH on which ctest resolves and python does not."""
    tools = scratch / "ctest-only"
    tools.mkdir()
    (tools / "ctest").symlink_to(shutil.which("ctest"))
    return str(tools)


def test_a_label_pattern_lists_only_the_entries_with_a_matching_label(tmp_path):
    _register(tmp_path)
    tests = list_ctest_entries(tmp_path, label_pattern="^standard$")
    assert [test["name"] for test in tests] == ["rocke_first"]
    assert list_ctest_entries(tmp_path, label_pattern="^no-such-label$") == []


@needs_posix_symlinks
def test_a_bare_python_entry_lists_its_command_with_python_put_on_path(tmp_path):
    """Without python on PATH CTest drops the entry's command; listed as an
    installed tree, the entry has its command, run by this interpreter."""
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "CTestTestfile.cmake").write_text(
        "add_test(rocke_bare python -m pytest)\n", encoding="utf-8"
    )
    without_python = {"PATH": _path_with_ctest_only(tmp_path)}

    unlisted = list_ctest_entries(registration, env=without_python)
    assert "command" not in unlisted[0]

    listed = list_installed_ctest_entries(registration, tmp_path, env=without_python)
    assert listed[0]["command"][1:] == ["-m", "pytest"]
    assert Path(listed[0]["command"][0]).resolve() == Path(sys.executable).resolve()


@needs_posix_symlinks
def test_an_environment_with_python_on_path_is_returned_unchanged(tmp_path):
    """Where python resolves already, the helper changes nothing and makes no
    link."""
    interpreter_dir = tmp_path / "bin"
    interpreter_dir.mkdir()
    (interpreter_dir / "python").symlink_to(sys.executable)
    environment = {"PATH": str(interpreter_dir), "OTHER": "kept"}
    assert python_on_path_environment(tmp_path / "scratch", environment) == environment
    assert not (tmp_path / "scratch").exists()
