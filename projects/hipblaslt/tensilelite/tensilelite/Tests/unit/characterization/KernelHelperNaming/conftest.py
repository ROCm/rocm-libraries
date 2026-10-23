################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Fixtures for the KernelHelperNaming suite: a real Solution from the vendored
logic fixture (its naming functions read solution state)."""

import copy
from pathlib import Path

import pytest

from tensilelite.Common.Architectures import SUPPORTED_ISA
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults
import tensilelite.LibraryIO as LibraryIO

_FIXTURE = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


@pytest.fixture(scope="session")
def _solution():
    cxx = validateToolchain("amdclang++")
    iim = makeIsaInfoMap(SUPPORTED_ISA, cxx)
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    asm = makeAssemblyToolchain(cxx, bundler, "default").assembler
    return LibraryIO.parseLibraryLogicFile(str(_FIXTURE), asm, False, False, False, iim, False).solutions[0]


@pytest.fixture
def solution(_solution):
    return copy.deepcopy(_solution)
