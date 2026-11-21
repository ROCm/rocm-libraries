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


_ADD_TEST_NAME_RE = re.compile(r"\badd_test\s*\(\s*NAME\s+(\w+)", re.IGNORECASE)


def test_every_registered_cpp_test_is_in_a_provider_tier():
    """TheRock CI selects provider tests by tier label, so a C++ test that no
    tier of the provider's categories lists never runs there."""
    platform = Path(__file__).resolve().parents[2]
    categories = platform.parents[1] / "ROCKE_ENGINE_test_categories_external.yaml"
    fragments = platform / "tests" / "cmake"
    if not (categories.is_file() and fragments.is_dir()):
        pytest.skip("the provider tier lists or the CMake fragments are absent here")
    yaml = pytest.importorskip("yaml", reason="reading the tier lists needs PyYAML")
    patterns = [
        pattern
        for tier in yaml.safe_load(categories.read_text())["test_categories"].values()
        for pattern in tier["test_patterns"]
    ]
    registered = {
        name
        for fragment in _FRAGMENTS
        for name in _ADD_TEST_NAME_RE.findall((fragments / fragment).read_text())
    }
    tiered = {
        name
        for name in registered
        if any(re.fullmatch(pattern, name) for pattern in patterns)
    }
    untiered = sorted(registered - tiered)
    assert not untiered, f"{categories.name} lists no tier for {untiered}"
