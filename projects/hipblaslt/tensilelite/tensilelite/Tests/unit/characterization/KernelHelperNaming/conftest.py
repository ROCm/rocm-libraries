################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Fixtures for the KernelHelperNaming suite: a real Solution from the vendored
logic fixture (its naming functions read solution state)."""

import copy
from pathlib import Path

import pytest

import tensilelite.LibraryIO as LibraryIO

_FIXTURE = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


@pytest.fixture(scope="session")
def _solution(assembler, isa_info_map):
    return LibraryIO.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    ).solutions[0]


@pytest.fixture
def solution(_solution):
    return copy.deepcopy(_solution)
