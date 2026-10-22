# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static checks that the IR validity gate stays wired into CI.

TheRock selects CTest entries by name from the provider category YAML. An entry
that the CMake file registers but the YAML does not name never runs in CI, and
nothing else reports that. These checks read the source files, so they skip in
an installed test tree, which holds neither.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PLATFORM = Path(__file__).resolve().parents[1]
_PROVIDER = _PLATFORM.parent.parent
_CMAKE = _PLATFORM / "CMakeLists.txt"
_CATEGORIES = _PROVIDER / "ROCKE_ENGINE_test_categories_external.yaml"
_ENTRY = "rocke_ir_validity"
_GATE_ENTRY = "rocke_golden_static"


def _read(path: Path) -> str:
    if not path.is_file():
        pytest.skip(f"{path.name} is not part of this tree")
    return path.read_text()


def test_the_installed_ctest_file_registers_the_gate_in_strict_mode():
    """Without --strict a host with no LLVM tools would pass the entry."""
    text = _read(_CMAKE)
    match = re.search(r"add_test\(" + _ENTRY + r' python (.*?)\)\\n"', text)
    assert match, f"{_CMAKE.name} registers no {_ENTRY} entry"
    assert "check_ir_validity.py" in match.group(1)
    assert match.group(1).rstrip().endswith("--strict")


def test_the_entry_has_a_timeout_and_the_library_on_its_path():
    text = _read(_CMAKE)
    block = re.search(
        r"set_tests_properties\(" + _ENTRY + r" PROPERTIES(.*?)\)\\n\"", text, re.S
    )
    assert block, f"{_ENTRY} has no set_tests_properties block"
    assert "TIMEOUT" in block.group(1)
    assert "tests/library" in block.group(1)


def test_the_tool_and_every_helper_it_imports_install_together():
    """The tool imports its helper modules by name; a missing one is an ImportError.

    The helper list comes from the tool's own imports, so a new helper that the
    CMake file does not install fails here and not in CI.
    """
    tool = _PLATFORM / "tools" / "check_ir_validity.py"
    helpers = re.findall(r"^from (_(?!_)\w+) import", _read(tool), re.M)
    assert helpers, "the tool imports no helper module; update this check"
    text = _read(_CMAKE)
    for name in ["check_ir_validity", *helpers]:
        assert re.search(
            rf'"[^"]*/tools/{name}\.py"', text
        ), f"{_CMAKE.name} does not install tools/{name}.py"


def test_every_category_that_runs_the_golden_gate_also_runs_this_one():
    """TheRock runs only the names a category lists."""
    text = _read(_CATEGORIES)
    golden = len(re.findall(rf'"{_GATE_ENTRY}"', text))
    assert golden, f"no category names {_GATE_ENTRY}"
    assert len(re.findall(rf'"{_ENTRY}"', text)) == golden, (
        f"{_CATEGORIES.name} must name {_ENTRY} in each category that names "
        f"{_GATE_ENTRY}"
    )
