# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keep C++ test registration distributed across concern-owned CMake files."""

import re
from pathlib import Path

import pytest


_INCLUDE_RE = re.compile(r"(?im)^[ \t]*include\s*\(\s*([^\s)]+)\s*\)")
_REGISTRATION_RE = re.compile(r"\badd_(?:test|executable)\s*\(", re.IGNORECASE)
_FRAGMENTS = (
    "CoreTests.cmake",
    "PortableIrTests.cmake",
    "InstanceTests.cmake",
    "TestCategoryLabels.cmake",
)
_EXPECTED_INCLUDES = tuple(
    f'"${{CMAKE_CURRENT_LIST_DIR}}/cmake/{fragment}"' for fragment in _FRAGMENTS
)


def _assert_orchestrator_contract(text: str) -> None:
    active_includes = tuple(_INCLUDE_RE.findall(text))
    assert active_includes == _EXPECTED_INCLUDES
    assert _REGISTRATION_RE.search(text) is None


def test_cmake_test_registration_is_concern_owned():
    platform = Path(__file__).resolve().parents[2]
    orchestrator = platform / "tests" / "CMakeLists.txt"
    if not orchestrator.is_file():
        pytest.skip("source CMake project is absent from the installed test artifact")

    for fragment in _FRAGMENTS:
        assert (platform / "tests" / "cmake" / fragment).is_file(), fragment

    _assert_orchestrator_contract(orchestrator.read_text())


@pytest.mark.parametrize(
    "mutation",
    (
        lambda text: text.replace(
            'include("${CMAKE_CURRENT_LIST_DIR}/cmake/InstanceTests.cmake")',
            '# include("${CMAKE_CURRENT_LIST_DIR}/cmake/InstanceTests.cmake")',
        ),
        lambda text: text
        + '\ninclude("${CMAKE_CURRENT_LIST_DIR}/cmake/UnexpectedTests.cmake")\n',
        lambda text: text + "\nADD_TEST(unowned test_command)\n",
        lambda text: text + "\nadd_executable (unowned test.cpp)\n",
    ),
)
def test_orchestrator_contract_rejects_inactive_extra_or_inline_registration(mutation):
    orchestrator = Path(__file__).resolve().parents[1] / "CMakeLists.txt"
    with pytest.raises(AssertionError):
        _assert_orchestrator_contract(mutation(orchestrator.read_text()))
