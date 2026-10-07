# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the dynamic StreamK parallel reduction (SK5 Hybrid).

When every tile is split the host sets bit 29 of the SKTiles argument on the
dynamic sub-path. The dynamic preLoop selects WorkAssignmentMode 3 (from 1) on
it and clears it from SKTiles; every work item then selects the workspace slot of
its part (SkPartialIdx = StreamKPartialIdx) and the store sites branch to the
static parallel-reduction code on WorkAssignmentMode == 3. These tests pin
those snippets on a fake writer, the capability gating shared with the host,
and that kernels without the capability emit nothing new.
"""

import itertools
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

import rocisa
from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister
from rocisa.instruction import SAndB32, SBitcmp1B32, SCBranchSCC1, SCmpEQU32, SCSelectB32

from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Components import StreamK as StreamKModule
from Tensile.Components.StreamK import StreamK, StreamKHybrid
from Tensile.ExecutionPolicy import usesStreamKDynamicParallel
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

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
        labels=SimpleNamespace(getNameInc=lambda n: "%s_%d" % (n, next(counter))),
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


def test_the_bit_is_the_static_uso_bit_on_the_other_sub_path():
    # Bit 29 of slot 2 means uniform summation order on the static sub-path and
    # parallel reduction on the dynamic one; bit 30 selects the sub-path.
    assert StreamKModule._SK5_DYNAMIC_PARALLEL_BIT == StreamKModule._SK_USO_BIT == 29
    assert StreamKModule._SK5_MODE_DYNAMIC_PARALLEL == 3, \
        "nonzero: every hybrid dispatch tests WorkAssignmentMode == 0"


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


def test_branch_tests_the_mode():
    mod = Module()
    StreamKHybrid().emitDynamicParallelBranch(_writer(), _KERNEL, mod, Label("Target", ""))
    items = list(mod.flatitems())
    assert [type(i) for i in items] == [SCmpEQU32, SCBranchSCC1]
    assert _params(items[0]) == ["s[sgprWorkAssignmentMode]", "3"]
    assert "Target" in str(items[1])


def test_long_branch_tests_the_mode():
    # The GSU0 store returns to the queue (PersistentLoopClose) through the
    # writer's long branch, on the same named mode test.
    w = _writer()
    w.longBranchScc1 = lambda label, posNeg=0, comment="": SCBranchSCC1(
        labelName=label.getLabelName(), comment=comment)
    mod = Module()
    StreamKHybrid().emitDynamicParallelLongBranch(w, _KERNEL, mod, Label("PersistentLoopClose", ""))
    items = list(mod.flatitems())
    assert [type(i) for i in items] == [SCmpEQU32, SCBranchSCC1]
    assert _params(items[0]) == ["s[sgprWorkAssignmentMode]", "3"]
    assert "PersistentLoopClose" in str(items[1])

    mod = Module()
    StreamKHybrid().emitDynamicParallelLongBranch(w, dict(_KERNEL, StreamKAtomic=1), mod,
                                                  Label("PersistentLoopClose", ""))
    assert not list(mod.flatitems())


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
    assert not list(mod.flatitems())


def test_static_strategies_never_take_the_dynamic_path():
    # Static-only StreamK kernels (SK3) keep their code unchanged.
    assert StreamK.usesDynamicParallel(StreamKHybrid(), _writer(), _KERNEL) is False
    mod = Module()
    StreamK.emitDynamicParallelBranch(_FakeStatic(), _writer(), _KERNEL, mod, Label("T", ""))
    assert not list(mod.flatitems())


class _FakeStatic:
    usesDynamicParallel = StreamK.usesDynamicParallel
