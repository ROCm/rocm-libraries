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
pins the ordered rejection reasons together with deterministic derived state;
every case asserts ``Valid is False`` since these configs are genuinely invalid.

Runs under global-state isolation (derivation mutates globalParameters /
validParameters) so it does not leak into other suites.
"""

import copy
import pytest

from codegen_harness import _isolated_globals  # shared isolation context
from reject_harness import apply_overrides, derive_with_rejections

pytestmark = pytest.mark.unit

# (case_id, base_label, {overrides}) -- overrides support dotted keys for
# nested ProblemType.* fields. Base labels resolve via the base_states fixture.
_TRIPS = {
    "sk_cluster_maxviol": ("gfx950_SK", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "DynamicWorkQueue", "ClusterDim": [2, 2],
    }),
    "sk_schedule_maxviol": ("gfx950_SK", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid", "EnableMatrixInstruction": False, "KernelLanguage": "Source",
        "ProblemType.StridedBatched": False, "ProblemType.GroupedGemm": True,
        "ScheduleGlobalRead": 0, "ScheduleLocalWrite": 0, "BufferStore": False,
    }),
    "sk_atomic_maxviol": ("gfx942_BBS", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid", "StreamKAtomic": 1, "LocalSplitU": 2,
    }),
    "sk_pap_maxviol": ("gfx950_SK", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid", "PrefetchAcrossPersistent": 1, "BufferLoad": False,
        "PrefetchGlobalRead": 0, "DirectToVgprA": True, "BufferStore": False,
        "StoreRemapVectorWidth": 4, "ProblemType.NumIndicesSummation": 2,
        "ProblemType.Sparse": 1,
    }),
    "sk_debugloop": ("gfx950_SK", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "DynamicWorkQueue", "DebugPersistentKernelLoopForever": True,
    }),
    "sk_ws_maxviol": ("gfx950_SK", {
        "TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid", "WorkQueueStealing": True, "StreamKAtomic": 1,
        "DebugStreamK": 1,
    }),
}

_KEYS = [
    "Valid", "TileProcessingStrategy", "WorkAssignment", "StreamKAtomic", "WorkQueueStealing",
    "GlobalSplitU", "BufferStore", "EnableMatrixInstruction",
]


@pytest.mark.parametrize("case", sorted(_TRIPS.keys()))
def test_setcover_reject(case, reject_bases, isa_info_map, assembler, monkeypatch, snapshot):
    label, overrides = _TRIPS[case]
    base = reject_bases[label]
    rocm = assembler.rocm_version
    with _isolated_globals():
        s = copy.deepcopy(base)
        apply_overrides(s, overrides)
        out = derive_with_rejections(s, isa_info_map, rocm, monkeypatch, _KEYS)
    assert out.get("Valid") is False
    assert out["rejections"]
    assert out == snapshot
