# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for test discovery functions.

These tests verify that we can discover tests from executables
and parse their output correctly.
"""

from pathlib import Path

import pytest
from rrtest.core import (
    discover_catch2_tests,
    discover_ctest_tests,
    discover_gtest_tests,
)


@pytest.fixture
def build_dir():
    """Fixture to provide build directory."""
    import os

    if "ROCROLLER_BUILD_DIR" in os.environ:
        return Path(os.environ["ROCROLLER_BUILD_DIR"])

    return Path("build")


def test_discover_gtest_tests_no_filters(build_dir):
    """Test discovering GTest tests without any filters."""
    executable = "test/rocroller-tests"

    tests = discover_gtest_tests(executable, build_dir=build_dir)

    assert isinstance(tests, set), "discover_gtest_tests should return a set"
    assert len(tests) > 0, "Should discover GTest tests"

    for test in tests:
        assert test.framework == "gtest", f"Test {test.name} should be gtest framework"


def test_discover_gtest_tests_with_excludes(build_dir):
    """Test discovering GTest tests with exclude filters."""
    executable = "test/rocroller-tests"

    all_tests = discover_gtest_tests(executable, build_dir=build_dir)

    filtered_tests = discover_gtest_tests(
        executable, include=[], exclude=["*GPU*"], build_dir=build_dir
    )

    assert len(filtered_tests) < len(
        all_tests
    ), "Filtered tests should be fewer than all tests"

    for test in filtered_tests:
        assert "GPU" not in test.name, f"Test {test.name} should not contain 'GPU'"


def test_discover_catch2_tests_no_filters(build_dir):
    """Test discovering Catch2 tests without any filters."""
    executable = "test/rocroller-tests-catch"

    tests = discover_catch2_tests(executable, build_dir=build_dir)

    assert isinstance(tests, set), "discover_catch2_tests should return a set"
    assert len(tests) > 0, "Should discover Catch2 tests"

    for test in tests:
        assert (
            test.framework == "catch2"
        ), f"Test {test.name} should be catch2 framework"


def test_discover_catch2_tests_with_filters(build_dir):
    """Test discovering Catch2 tests with filters."""
    executable = "test/rocroller-tests-catch"

    all_tests = discover_catch2_tests(executable, build_dir=build_dir)

    filtered_tests = discover_catch2_tests(
        executable, include=["[kernel-graph]"], exclude=[], build_dir=build_dir
    )

    assert len(filtered_tests) <= len(
        all_tests
    ), "Filtered tests should be <= all tests"


def test_discover_ctest_tests_no_filters(build_dir):
    """Test discovering CTest tests without any filters."""
    executable = "ctest"

    tests = discover_ctest_tests(executable, build_dir=build_dir)

    assert isinstance(tests, set), "discover_ctest_tests should return a set"
    assert len(tests) > 0, "Should discover CTest tests"

    for test in tests:
        assert test.framework == "ctest", f"Test {test.name} should be ctest framework"
