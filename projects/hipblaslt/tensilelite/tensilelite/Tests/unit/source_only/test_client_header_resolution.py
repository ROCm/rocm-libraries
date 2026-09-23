# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression checks for case-insensitive hipBLASLt client header lookup."""

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]


def test_tensilelite_utility_header_uses_a_distinct_include_path():
    """Windows must not select ``Utility.hpp`` for lowercase client includes."""
    clients_cmake = (_HIPBLASLT_ROOT / "clients/CMakeLists.txt").read_text(encoding="utf-8")
    utility_source = (_HIPBLASLT_ROOT / "clients/common/src/utility.cpp").read_text(
        encoding="utf-8"
    )

    assert (
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/client>" in clients_cmake
    )
    assert (
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../tensilelite/client/include>"
        not in clients_cmake
    )
    assert '#include "utility.hpp"' in utility_source
    assert '#include "include/Utility.hpp"' in utility_source
