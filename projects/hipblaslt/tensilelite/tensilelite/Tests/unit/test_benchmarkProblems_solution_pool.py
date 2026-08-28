################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
################################################################################
"""Unit tests for the solution-pool helpers in BenchmarkProblems.

The solution-pool path lets a benchmark load pre-tuned solutions from a library
logic file ("pool file") instead of generating them from ForkParameters. These
tests exercise the four helpers added for that feature against a small, real
library logic fixture (``test_data/solution_pool_gfx950.yaml``)
"""

import copy
import os
import types

import pytest

pytestmark = pytest.mark.unit

import Tensile.BenchmarkProblems as bp
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.SolutionStructs.Solution import Solution
from Tensile.Toolchain.Assembly import makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

POOL_FILE = os.path.join(os.path.dirname(__file__), "test_data", "solution_pool_gfx950.yaml")
_POOL_ISA = IsaVersion(9, 5, 0)  # the fixture is a gfx950 GEMM library


@pytest.fixture(scope="module")
def pool_env():
    """A GPU-free toolchain + ISA capability map matching the gfx950 fixture.

    Built once per module from the local amdclang++ install; nothing here needs
    a GPU. Also seeds the global parameters so Solution construction can run.
    """
    cxxCompiler, _cCompiler, offloadBundler = validateToolchain(
        ToolchainDefaults.CXX_COMPILER,
        ToolchainDefaults.C_COMPILER,
        ToolchainDefaults.OFFLOAD_BUNDLER,
    )
    isaInfoMap = makeIsaInfoMap([_POOL_ISA], cxxCompiler)
    assignGlobalParameters({}, isaInfoMap)
    assembler = makeAssemblyToolchain(cxxCompiler, offloadBundler, "default").assembler
    return types.SimpleNamespace(
        assembler=assembler,
        isaInfoMap=isaInfoMap,
        debugConfig=types.SimpleNamespace(
            splitGSU=False,
            printSolutionRejectionReason=False,
            printIndexAssignmentInfo=False,
        ),
    )


@pytest.fixture(autouse=True)
def _sequential_pool(monkeypatch):
    # Construct the pool in-process: fast, deterministic, and avoids
    # oversubscribing CI (the suite already runs under pytest-xdist).
    # monkeypatch restores the previous value after each test.
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def test_parsePoolFile_returns_problem_type_and_solutions(pool_env):
    ptStr, poolFile, data = bp._parsePoolFile(POOL_FILE)

    assert poolFile == POOL_FILE
    assert isinstance(ptStr, str) and ptStr  # non-empty ProblemType key
    assert "ProblemType" in data
    assert len(data["Solutions"]) == 1


def test_loadSolutionPool_groups_files_by_problem_type(pool_env):
    # The same file twice -> both entries indexed under one ProblemType key.
    index = bp._loadSolutionPool([POOL_FILE, POOL_FILE])

    assert len(index) == 1
    (ptStr, entries), = index.items()
    assert ptStr == bp._parsePoolFile(POOL_FILE)[0]
    assert len(entries) == 2
    for poolFile, data in entries:
        assert poolFile == POOL_FILE
        assert data["Solutions"]


def test_loadSolutionPool_empty_list_returns_empty_index(pool_env):
    assert bp._loadSolutionPool([]) == {}


def test_constructAllPoolSolutions_builds_solution_objects(pool_env):
    poolEntries = next(iter(bp._loadSolutionPool([POOL_FILE]).values()))

    solutions = bp._constructAllPoolSolutions(
        poolEntries, pool_env.assembler, pool_env.debugConfig, pool_env.isaInfoMap
    )

    assert len(solutions) == 1
    assert all(isinstance(s, Solution) for s in solutions)
    assert solutions[0]["ProblemType"] is not None


def test_constructAllPoolSolutions_merges_distinct_solutions_across_files(pool_env):
    # The fixture holds a single solution, so build two *distinct* pool entries
    # from it (differing WorkGroupMapping). This verifies the merge combines
    # different solutions across files rather than just doubling one.
    _, _, data = bp._parsePoolFile(POOL_FILE)
    problemType = data["ProblemType"]
    solA = copy.deepcopy(data["Solutions"][0]); solA["WorkGroupMapping"] = 1
    solB = copy.deepcopy(data["Solutions"][0]); solB["WorkGroupMapping"] = 4
    poolEntries = [
        ("poolA.yaml", {"ProblemType": problemType, "Solutions": [solA]}),
        ("poolB.yaml", {"ProblemType": problemType, "Solutions": [solB]}),
    ]

    solutions = bp._constructAllPoolSolutions(
        poolEntries, pool_env.assembler, pool_env.debugConfig, pool_env.isaInfoMap
    )

    assert len(solutions) == 2
    assert all(isinstance(s, Solution) for s in solutions)
    # Both distinct solutions survive the merge (AssignedDerivedParameters keeps
    # WorkGroupMapping as-is, so the differing values come straight through).
    assert sorted(s["WorkGroupMapping"] for s in solutions) == [1, 4]


def test_constructAllPoolSolutions_flattens_multiple_solutions_in_one_file(pool_env):
    # A single pool file containing multiple solutions -> all are constructed.
    _, _, data = bp._parsePoolFile(POOL_FILE)
    problemType = data["ProblemType"]
    sol0 = copy.deepcopy(data["Solutions"][0]); sol0["WorkGroupMapping"] = 1
    sol1 = copy.deepcopy(data["Solutions"][0]); sol1["WorkGroupMapping"] = 4
    poolEntries = [("pool.yaml", {"ProblemType": problemType, "Solutions": [sol0, sol1]})]

    solutions = bp._constructAllPoolSolutions(
        poolEntries, pool_env.assembler, pool_env.debugConfig, pool_env.isaInfoMap
    )

    assert len(solutions) == 2
    assert sorted(s["WorkGroupMapping"] for s in solutions) == [1, 4]


def test_constructAllPoolSolutions_keeps_duplicate_solutions(pool_env):
    # Identical solutions are intentionally NOT de-duplicated here: this helper
    # only gathers candidates, and duplicate kernels are collapsed later by the
    # kernel builder (writeSolutionsAndKernels dedups by BaseName). So the same
    # file listed twice yields two separate Solution objects at this stage.
    poolEntries = next(iter(bp._loadSolutionPool([POOL_FILE, POOL_FILE]).values()))

    solutions = bp._constructAllPoolSolutions(
        poolEntries, pool_env.assembler, pool_env.debugConfig, pool_env.isaInfoMap
    )

    assert len(solutions) == 2
    assert solutions[0] is not solutions[1]  # distinct objects, not aliased


def test_create_solution_from_state_builds_one_solution(pool_env):
    _, _, data = bp._parsePoolFile(POOL_FILE)
    state = copy.deepcopy(data["Solutions"][0])
    problemType = ProblemType(data["ProblemType"], False)

    solution = bp._create_solution_from_state(
        state,
        problemType,
        pool_env.debugConfig.splitGSU,
        pool_env.debugConfig.printSolutionRejectionReason,
        pool_env.debugConfig.printIndexAssignmentInfo,
        pool_env.assembler,
        pool_env.isaInfoMap,
        POOL_FILE,
    )

    assert isinstance(solution, Solution)
    assert solution["ProblemType"] is not None
    # The bookkeeping SolutionIndex is stripped before construction.
    assert "SolutionIndex" not in state


def test_create_solution_from_state_returns_none_on_bad_state(pool_env):
    _, _, data = bp._parsePoolFile(POOL_FILE)
    problemType = ProblemType(data["ProblemType"], False)

    # A state missing required solution parameters must be swallowed (logged and
    # returned as None) so one bad pool entry cannot abort the whole pool build.
    solution = bp._create_solution_from_state(
        {"this": "is not a valid solution state"},
        problemType,
        False,
        False,
        False,
        pool_env.assembler,
        pool_env.isaInfoMap,
        POOL_FILE,
    )

    assert solution is None
