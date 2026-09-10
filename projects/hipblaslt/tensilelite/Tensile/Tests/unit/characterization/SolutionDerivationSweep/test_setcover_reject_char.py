# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Set-cover reject harvest -- Stream-K derivation rejection branches.

Targeted trip-cases for the Stream-K rejection guards in
``Solution.assignDerivedParameters`` that the tuned/emit configs never select.
Each case takes a fully-derived base solution, applies parameter overrides that
violate one or more Stream-K constraints, resets the derivation flags, and
re-runs derivation with ``printRejectionReason=False`` so the guard sets
``Valid=False`` instead of raising. The Stream-K guards do not early-return, so
one maximal-violation config trips several independent rejects in a single pass.

Trip-cases and their target guards were verified line-by-line under
``sys.settrace`` before authoring (work/mutcov-evidence/sol_reject_probe.py):
every case flips its intended currently-missing Solution.py lines. The golden
pins the deterministic derivation outcome (Valid plus a few derived scalars, or
the exception type) per case; every case asserts ``Valid is False`` since these
configs are genuinely invalid.

Runs under global-state isolation (derivation mutates globalParameters /
validParameters) so it does not leak into other suites.
"""

import copy
import os

import pytest

from codegen_harness import _isolated_globals  # shared isolation context

import Tensile.LibraryIO as LibraryIO
from Tensile.SolutionStructs.Solution import Solution as _Solution

pytestmark = pytest.mark.unit

# Bases loaded locally (NOT via the shared base_states fixture) so this suite
# does not perturb the parameter-sweep goldens that iterate over base_states.
_DATA = os.path.join(os.path.dirname(__file__), "..", "_codegen", "data")
_BASES = {
    "gfx950_SK": "gfx950/StreamK_F8F8S.yaml",
    "gfx942_BBS": "gfx942/BBS_BH_Bias_Act.yaml",
}


@pytest.fixture(scope="module")
def reject_bases(assembler, isa_info_map):
    out = {}
    for label, rel in _BASES.items():
        path = os.path.join(_DATA, rel)
        logic = LibraryIO.parseLibraryLogicFile(
            path, assembler, False, False, False, isa_info_map, False
        )
        sols = logic.solutions
        sol0 = (list(sols.values()) if isinstance(sols, dict) else list(sols))[0]
        out[label] = copy.deepcopy(sol0._state)
    return out

# (case_id, base_label, {overrides}) -- overrides support dotted keys for
# nested ProblemType.* fields. Base labels resolve via the base_states fixture.
_TRIPS = {
    "sk_cluster_maxviol": ("gfx950_SK", {
        "StreamK": 4, "ClusterDim": [2, 2],
    }),
    "sk_schedule_maxviol": ("gfx950_SK", {
        "StreamK": 1, "EnableMatrixInstruction": False, "KernelLanguage": "Source",
        "ProblemType.StridedBatched": False, "ProblemType.GroupedGemm": True,
        "ScheduleGlobalRead": 0, "ScheduleLocalWrite": 0, "BufferStore": False,
    }),
    "sk_atomic_maxviol": ("gfx942_BBS", {
        "StreamK": 3, "StreamKAtomic": 1, "LocalSplitU": 2,
    }),
    "sk_pap_maxviol": ("gfx950_SK", {
        "StreamK": 3, "PrefetchAcrossPersistent": 1, "BufferLoad": False,
        "PrefetchGlobalRead": 0, "DirectToVgprA": True, "BufferStore": False,
        "StoreRemapVectorWidth": 4, "ProblemType.NumIndicesSummation": 2,
        "ProblemType.Sparse": 1,
    }),
    "sk_debugloop": ("gfx950_SK", {
        "StreamK": 4, "DebugPersistentKernelLoopForever": True,
    }),
    "sk_ws_maxviol": ("gfx950_SK", {
        "StreamK": 3, "StreamKWorkStealing": True, "StreamKAtomic": 1,
        "DebugStreamK": 1,
    }),
}

_KEYS = [
    "Valid", "StreamK", "StreamKAtomic", "StreamKWorkStealing",
    "GlobalSplitU", "BufferStore", "EnableMatrixInstruction",
]


def _apply(state, overrides):
    for k, v in overrides.items():
        if "." in k:
            top, sub = k.split(".", 1)
            state[top][sub] = v
        else:
            state[k] = v


def _derive(state, isa_info_map, rocm):
    state = copy.deepcopy(state)
    state["AssignedDerivedParameters"] = False
    state["AssignedProblemIndependentDerivedParameters"] = False
    try:
        _Solution.assignDerivedParameters(
            state, False, False, False, isa_info_map, rocm
        )
        return {k: state.get(k) for k in _KEYS}
    except Exception as exc:  # rejection-by-exception is real covered behaviour
        return {"exception": type(exc).__name__}


@pytest.mark.parametrize("case", sorted(_TRIPS.keys()))
def test_setcover_reject(case, reject_bases, isa_info_map, assembler, snapshot):
    label, overrides = _TRIPS[case]
    base = reject_bases[label]
    rocm = assembler.rocm_version
    with _isolated_globals():
        s = copy.deepcopy(base)
        _apply(s, overrides)
        out = _derive(s, isa_info_map, rocm)
    assert out.get("Valid") is False
    assert out == snapshot
