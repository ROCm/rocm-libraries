################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Shared fixtures for the Solution-derivation slice-3 suite. Reuses the
LibraryIO cap fixtures and parses the vendored logic fixture into real Solution
objects (whose `_state` seeds the derivation statics)."""

from pathlib import Path

import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
import Tensile.LibraryIO as LibraryIO

_FIXTURE = Path(__file__).parent.parent / "LibraryIO" / "data" / "logic_gfx942_HSS_BH.yaml"


@pytest.fixture(scope="session")
def cxx_compiler():
    return validateToolchain("amdclang++")


@pytest.fixture(scope="session")
def isa_info_map(cxx_compiler):
    return makeIsaInfoMap(SUPPORTED_ISA, cxx_compiler)


@pytest.fixture(scope="session")
def assembler(cxx_compiler):
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx_compiler, bundler, "default").assembler


@pytest.fixture(scope="session")
def _base_solution(assembler, isa_info_map):
    logic = LibraryIO.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    return logic.solutions[0]


@pytest.fixture
def real_state(_base_solution):
    """A deep copy of a real, fully-derived solution state (safe to mutate)."""
    import copy
    return copy.deepcopy(_base_solution._state)
