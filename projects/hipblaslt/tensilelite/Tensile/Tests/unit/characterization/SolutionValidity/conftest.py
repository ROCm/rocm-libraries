################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Fixtures for the SolutionValidity characterization suite.

Provides fully-derived Solution states plus cap map / assembler so the
validity-reject arm tests can mutate individual parameters and call
Solution's static methods directly (CPU-only, no GPU, no compile).
"""

import copy
import os

import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
import Tensile.LibraryIO as LibraryIO

_DATA = os.path.join(os.path.dirname(__file__), "..", "_codegen", "data")

_LOGIC_FILES = {
    "gfx942_HSS": os.path.join(_DATA, "gfx942", "HSS_BH_Bias.yaml"),
    "gfx942_BBS": os.path.join(_DATA, "gfx942", "BBS_BH_Bias_Act.yaml"),
}


@pytest.fixture(scope="session")
def _cxx():
    return validateToolchain("amdclang++")


@pytest.fixture(scope="session")
def isa_info_map(_cxx):
    return makeIsaInfoMap(SUPPORTED_ISA, _cxx)


@pytest.fixture(scope="session")
def assembler(_cxx):
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(_cxx, bundler, "default").assembler


@pytest.fixture(scope="session")
def _base_solutions(assembler, isa_info_map):
    """Parse vendored logic files, return {label: solution._state} dict."""
    out = {}
    for label, path in _LOGIC_FILES.items():
        if not os.path.exists(path):
            continue
        logic = LibraryIO.parseLibraryLogicFile(
            path, assembler, False, False, False, isa_info_map, False
        )
        sols = logic.solutions
        sol0 = (list(sols.values()) if isinstance(sols, dict) else list(sols))[0]
        out[label] = sol0._state
    return out


@pytest.fixture
def hss_state(_base_solutions):
    """Deep copy of the gfx942 HSS solution state (safe to mutate)."""
    return copy.deepcopy(_base_solutions["gfx942_HSS"])
