# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Keep C++ test registration distributed across concern-owned CMake files."""

from pathlib import Path

import pytest


def test_cmake_test_registration_is_concern_owned():
    platform = Path(__file__).resolve().parents[2]
    orchestrator = platform / "tests" / "CMakeLists.txt"
    if not orchestrator.is_file():
        pytest.skip("source CMake project is absent from the installed test artifact")

    text = orchestrator.read_text()
    fragments = (
        "CoreTests.cmake",
        "PortableIrTests.cmake",
        "InstanceTests.cmake",
        "TestCategoryLabels.cmake",
    )

    positions = []
    for fragment in fragments:
        path = platform / "tests" / "cmake" / fragment
        assert path.is_file(), fragment
        positions.append(text.index(f"cmake/{fragment}"))

    assert positions == sorted(positions)
    assert "add_test(" not in text
    assert "add_executable(" not in text
