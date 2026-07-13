################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Solution-derivation parameter sweep (slice 3b).

For each base solution and each (parameter -> value) override, reset the
derivation flags and re-run ``Solution.assignDerivedParameters``. Each override
drives a derivation branch that the narrow tuned configs never select — many
override values are rejected (Valid=False) or raise, which is itself the
covered behaviour. The golden pins, per (base, param, value), the derivation
outcome (Valid + a few key derived scalars, or the exception type), which is
deterministic for a fixed toolchain/ISA.

Runs under global-state isolation (derivation mutates globalParameters /
validParameters) so it does not leak into other suites.
"""

import copy

import pytest

from codegen_harness import _isolated_globals  # shared isolation context

from Tensile.SolutionStructs import Solution as _SolutionMod
from Tensile.SolutionStructs.Solution import Solution as _Solution

pytestmark = pytest.mark.unit

# Parameters whose values gate distinct derivation branches, with a few values
# each (including some that will be rejected on the base config).
_SWEEP = {
    "PrefetchGlobalRead": [0, 1, 2],
    "PrefetchLocalRead": [0, 1, 2],
    "ScheduleIterAlg": [1, 2, 3],
    "GlobalSplitU": [1, 2, 4],
    "GlobalSplitUAlgorithm": ["SingleBuffer", "MultipleBuffer"],
    "StaggerU": [0, 4, 32],
    "WorkGroupMapping": [1, 4, 8],
    "DepthU": [16, 32, 64],
    "ExpandPointerSwap": [False, True],
    "1LDSBuffer": [0, 1],
    "StorePriorityOpt": [False, True],
    "GroupLoadStore": [False, True],
    "NonTemporalA": [0, 1, 2, 3],
    "NonTemporalB": [0, 1, 2, 3],
    "VectorWidthA": [1, 2, 4],
    "VectorWidthB": [1, 2, 4],
    "AssertSummationElementMultiple": [1, 2, 8],
    "AssertFree0ElementMultiple": [1, 2, 8],
    "LdsBlockSizePerPadA": [-1, 0, 128],
    "LdsBlockSizePerPadB": [-1, 0, 128],
    "StaggerUMapping": [0, 1, 2],
    "StaggerUStride": [0, 256],
    "WorkGroupMappingType": ["B", "Z"],
    "DirectToVgprA": [False, True],
    "DirectToVgprB": [False, True],
}

_KEYS = [
    "Valid", "WavefrontSize", "NumThreads", "MacroTile0", "MacroTile1",
    "DepthU", "GlobalSplitU", "PrefetchGlobalRead", "LoopIters",
]


def _derive(state, isa_info_map, rocm_version):
    state = copy.deepcopy(state)
    state["AssignedDerivedParameters"] = False
    state["AssignedProblemIndependentDerivedParameters"] = False
    try:
        _Solution.assignDerivedParameters(
            state, False, False, False, isa_info_map, rocm_version
        )
        return {k: state.get(k) for k in _KEYS}
    except Exception as exc:  # rejection-by-exception is real covered behaviour
        return {"exception": type(exc).__name__}


@pytest.mark.parametrize("param", sorted(_SWEEP.keys()))
def test_derivation_sweep(param, base_states, isa_info_map, assembler, snapshot):
    rocm = assembler.rocm_version
    out = {}
    with _isolated_globals():
        for label, base in base_states.items():
            for value in _SWEEP[param]:
                s = copy.deepcopy(base)
                s[param] = value
                out[f"{label}:{value}"] = _derive(s, isa_info_map, rocm)
    assert out == snapshot
