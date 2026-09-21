################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Fixtures for the Solution-derivation parameter sweep (slice 3b).

Provides fully-derived base ``Solution`` states parsed from a few curated logic
files (different arch/dtype), plus the cap map and rocm version, so the sweep
can vary individual parameters and re-run ``Solution.assignDerivedParameters``
to exercise derivation/rejection branches the tuned configs never hit.
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

# (label, relative path under _codegen/data) base solutions to sweep.
_BASES = {
    "gfx942_HSS": "gfx942/HSS_BH_Bias.yaml",
    "gfx942_BBS": "gfx942/BBS_BH_Bias_Act.yaml",
    "gfx950_HHS": "gfx950/HHS.yaml",
    "gfx90a_HSS": "gfx90a/HSS.yaml",
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
def base_states(assembler, isa_info_map):
    """Map label -> a fully-derived solution _state (deep-copied per use)."""
    out = {}
    for label, rel in _BASES.items():
        path = os.path.join(_DATA, rel)
        if not os.path.exists(path):
            continue
        logic = LibraryIO.parseLibraryLogicFile(
            path, assembler, False, False, False, isa_info_map, False
        )
        sols = logic.solutions
        sol0 = (list(sols.values()) if isinstance(sols, dict) else list(sols))[0]
        out[label] = copy.deepcopy(sol0._state)
    return out
