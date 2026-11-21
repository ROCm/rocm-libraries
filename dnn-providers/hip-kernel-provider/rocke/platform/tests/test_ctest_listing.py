# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``list_ctest_entries`` reads a real CTest registration without running it."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from rocke.tests.ctest_listing import list_ctest_entries

pytestmark = pytest.mark.skipif(
    shutil.which("ctest") is None, reason="needs ctest on PATH"
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
