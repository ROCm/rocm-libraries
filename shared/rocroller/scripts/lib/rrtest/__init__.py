# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
rrtest - rocRoller Test Utilities

A simple Python library for discovering and filtering rocRoller tests.
"""

from .core import (
    Test,
    discover_catch2_tests,
    discover_ctest_tests,
    discover_gtest_tests,
    get_test_commands,
    list_tests,
)

__version__ = "0.2.0"

__all__ = [
    "Test",
    "discover_catch2_tests",
    "discover_ctest_tests",
    "discover_gtest_tests",
    "get_test_commands",
    "list_tests",
]
