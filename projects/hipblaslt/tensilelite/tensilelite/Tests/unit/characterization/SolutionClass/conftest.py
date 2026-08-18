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

"""Shared fixtures for the Solution-class slice-2 suite.

Building a real ``Solution`` requires the toolchain-derived capability map and a
real ``Assembler`` (``Solution`` reads ``assembler.rocm_version``). We reuse the
LibraryIO suite's session fixtures (``cxx_compiler`` / ``isa_info_map`` /
``assembler``) and parse the LibraryIO vendored logic fixture once to obtain
genuine ``Solution`` objects — avoiding hand-authoring a self-consistent kernel.
"""

from pathlib import Path

import pytest

from Tensile.Activation import ActivationType
from Tensile.Common.Architectures import SUPPORTED_ISA
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
import Tensile.LibraryIO as LibraryIO

# Reuse the LibraryIO suite's committed, vendored logic fixture (read-only).
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
def solutions(assembler, isa_info_map):
    """Real ``Solution`` objects parsed from the vendored logic fixture."""
    logic = LibraryIO.parseLibraryLogicFile(
        str(_FIXTURE), assembler, False, False, False, isa_info_map, False
    )
    return logic.solutions


@pytest.fixture
def solution(solutions):
    return solutions[0]


# Fields whose values are environment- or toolchain-derived (and would make a
# raw state snapshot non-reproducible). We pin the schema (key set) and a curated
# set of stable fields instead of dumping every value.
def _render(v):
    if isinstance(v, DataType):
        return f"<DataType {v.toName()}>"
    if isinstance(v, ActivationType):
        return f"<ActivationType {str(v)}>"
    if isinstance(v, ProblemType):
        return f"<ProblemType {str(v)}>"
    if isinstance(v, (list, tuple)):
        return [_render(x) for x in v]
    if isinstance(v, dict):
        return {k: _render(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    return v


# A curated set of stable, toolchain-independent scalar fields to snapshot by
# value (the full 300+-key state may carry env-coupled derived values).
_STABLE_FIELDS = (
    "KernelLanguage", "MacroTile0", "MacroTile1", "DepthU",
    "MatrixInstM", "MatrixInstN", "MatrixInstB",
    "GlobalSplitU", "WorkGroup", "ThreadTile", "UseCustomMainLoopSchedule",
)


@pytest.fixture
def solution_summary():
    def _summarize(sol):
        return {
            "name": str(sol),
            "num_keys": len(sol),
            "keys": sorted(sol.keys()),
            "problem_type": str(sol["ProblemType"]),
            "stable": {f: _render(sol[f]) for f in _STABLE_FIELDS if f in sol},
        }

    return _summarize
