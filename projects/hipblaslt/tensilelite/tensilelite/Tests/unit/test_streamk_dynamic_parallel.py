# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the parallel reduction of the StreamK Hybrid dynamic sub-mode.

When every tile is split the host sets bit 29 of the SKTiles argument. The
dynamic preLoop moves it into WorkAssignmentMode (3) and clears it from SKTiles;
every work item then selects the workspace slot of its part (SkPartialIdx =
StreamKPartialIdx) and the store sites branch to the static parallel-reduction
code on WorkAssignmentMode == 3. These tests pin those snippets on a fake
writer, the capability gating shared with the host, and that kernels without
the capability emit nothing new.
"""

import functools
import itertools
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from tensilelite.KernelWriterAssembly import KernelWriterAssembly

import rocisa
from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister
from rocisa.instruction import SAndB32, SBitcmp1B32, SCBranchSCC1, SCmpEQU32, SCSelectB32

from tensilelite.Common import IsaVersion
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Components.StreamK import StreamKHybrid
from tensilelite.ExecutionPolicy import usesStreamKDynamicParallel
from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _gfx950():
    isa = IsaVersion(9, 5, 0)
    with preserve_rocisa_kernel_state():
        makeIsaInfoMap([isa], shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
        rocisa.rocIsa.getInstance().setKernel(isa, 64)
        yield


_KERNEL = {
    "TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid",
    "DebugStreamK": 0, "StreamKAtomic": 0, "WavefrontSize": 64, "ISA": (9, 5, 0),
    "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2},
}


def _writer(pap=False):
    counter = itertools.count()

    @contextmanager
    def alloc_tmp(size, alignment=1, tag=""):
        yield ContinuousRegister(100, size)

    return SimpleNamespace(
        labels=SimpleNamespace(getNameInc=lambda n: "%s_%d" % (n, next(counter)),
                               getUniqueNamePrefix=lambda n: n),
        allocTmpSgpr=alloc_tmp,
        isPrefetchAcrossPersistentEnabled=lambda k: pap,
        states=SimpleNamespace(kernel=_KERNEL, version=(9, 5, 0)),
    )


def _params(inst):
    return [str(p) for p in inst.getParams()]


@pytest.mark.parametrize("debug,atomic,pap,expected", [
    (0, 0, False, True),
    (1, 0, False, False),
    (0, 1, False, False),
    (0, 0, True, False),
])
def test_capability_matches_codegen(debug, atomic, pap, expected):
    # The host sets the bit only for SupportStreamKDynamicParallel kernels; the
    # derivation (Solution.py) and the emitted code share one predicate.
    assert usesStreamKDynamicParallel(debug, atomic, pap) is expected
    kernel = dict(_KERNEL, DebugStreamK=debug, StreamKAtomic=atomic,
                  InternalSupportParams={"SupportStreamKDynamicParallel": expected})
    assert StreamKHybrid().usesDynamicParallel(_writer(pap), kernel) is expected


def test_codegen_rejects_a_mismatched_capability():
    kernel = dict(_KERNEL, InternalSupportParams={"SupportStreamKDynamicParallel": True})
    with pytest.raises(AssertionError, match="SupportStreamKDynamicParallel"):
        StreamKHybrid().usesDynamicParallel(_writer(pap=True), kernel)


def test_mode_extraction_moves_and_clears_the_bit():
    items = list(StreamKHybrid().extractDynamicParallelMode(_writer(), _KERNEL).flatitems())
    assert [type(i) for i in items] == [SBitcmp1B32, SCSelectB32, SAndB32]
    test, select, clear = items
    assert _params(test) == ["s[sgprSKTiles]", "29"]
    assert _params(select)[0] == "s[sgprWorkAssignmentMode]" and _params(select)[1] == "3"
    assert _params(select)[2] == "s[sgprWorkAssignmentMode]", "bit clear: stay plain dynamic (1)"
    assert _params(clear)[0] == "s[sgprSKTiles]" and _params(clear)[2] == "0xdfffffff"


def test_slot_is_the_part_index():
    items = list(StreamKHybrid().selectDynamicParallelSlot(_writer(), _KERNEL).flatitems())
    assert [type(i) for i in items] == [SCmpEQU32, SCSelectB32]
    assert _params(items[0]) == ["s[sgprWorkAssignmentMode]", "3"]
    assert _params(items[1]) == ["s[sgprSkPartialIdx]", "s[sgprStreamKPartialIdx]",
                                 "s[sgprSkPartialIdx]"]


@pytest.mark.parametrize("long", [False, True])
def test_branch_tests_the_mode(long):
    # long: the GSU0 store returns to the queue (PersistentLoopClose) through
    # the writer's long branch.
    w = _writer()
    w.longBranchScc1 = functools.partial(KernelWriterAssembly.longBranchScc1, w)
    mod = Module()
    StreamKHybrid().emitDynamicParallelBranch(w, _KERNEL, mod, Label("Target", ""), long=long)
    test, branch = mod.items()
    assert type(test) is SCmpEQU32
    assert _params(test) == ["s[sgprWorkAssignmentMode]", "3"]
    assert type(branch) is (Module if long else SCBranchSCC1)
    assert "label_Target" in str(branch)


@pytest.mark.parametrize("kernel", [
    dict(_KERNEL, StreamKAtomic=1),
    dict(_KERNEL, DebugStreamK=2),
])
def test_without_the_capability_nothing_is_emitted(kernel):
    w = _writer()
    hybrid = StreamKHybrid()
    assert not list(hybrid.extractDynamicParallelMode(w, kernel).flatitems())
    assert not list(hybrid.selectDynamicParallelSlot(w, kernel).flatitems())
    mod = Module()
    hybrid.emitDynamicParallelBranch(w, kernel, mod, Label("Target", ""))
    hybrid.emitDynamicParallelBranch(w, kernel, mod, Label("Target", ""), long=True)
    assert not list(mod.flatitems())

