# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which installed CTest entries carry the compiler-tool requirement.

``ROCKE_REQUIRE_COMPILER_TOOLS=1`` turns a missing-tool skip into a failure,
so it belongs only on entries whose compiler checks must run. The platform's
``test_engine_linkage.py`` pins the same contract from a fresh configure; this
check reads the CTest file CI actually runs, from inside the MLA entry, and
skips anywhere else (a source-tree run has no CTest file in its cwd).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ._compiler_tools import REQUIRE_TOOLS_ENV

# Entries that append _rocke_compiler_requirement_environment
# (platform/CMakeLists.txt). Add an entry here when it starts appending it.
_REQUIRING_ENTRIES = {"rocke_library_mla_pytest"}


def test_only_requiring_entries_carry_the_flag():
    ctest_file = Path.cwd() / "CTestTestfile.cmake"
    if not ctest_file.is_file() or not all(
        name in ctest_file.read_text() for name in _REQUIRING_ENTRIES
    ):
        pytest.skip("not running from the installed CTest tree")
    ctest = shutil.which("ctest")
    if ctest is None:
        pytest.skip("ctest not on PATH")
    listed = subprocess.run(
        [ctest, "--show-only=json-v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    flagged = set()
    for test in json.loads(listed.stdout)["tests"]:
        for prop in test.get("properties", []):
            if (
                prop["name"] == "ENVIRONMENT"
                and f"{REQUIRE_TOOLS_ENV}=1" in prop["value"]
            ):
                flagged.add(test["name"])
    # The CMake condition that fills the requirement list, evaluated in the
    # environment ctest just listed under. Not this process's own flag: that
    # comes from the entry under test.
    required = sys.platform == "linux" and bool(os.environ.get("ROCM_PATH"))
    expected = _REQUIRING_ENTRIES if required else set()
    assert flagged == expected
