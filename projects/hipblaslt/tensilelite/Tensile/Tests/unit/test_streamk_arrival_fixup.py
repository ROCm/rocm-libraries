# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the dynamic StreamK fixup by last arrival (SK5 Hybrid).

Every part of a split tile writes its partial and counts itself in on a
per-tile counter with ``s_atomic_inc`` bounded by ``SKSplit-1``; the part that
reads ``SKSplit-1`` (and so wraps the counter back to 0) fixes the tile up.
These tests pin the shape of that protocol on a fake writer: one atomic, its
bound, the LDS ticket broadcast that restores what it borrowed, that no
per-part ready flag is stored or polled, and that the last part sums the
partials in part order.
"""

import itertools
import re
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

import rocisa
from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister
from rocisa.instruction import (
    BufferLoadB32,
    BufferStoreB32,
    DSLoadB32,
    DSStoreB32,
    SAtomicInc,
    SBarrier,
    SCBranchSCC1,
    SLoadB32,
    SStoreB32,
    SSubU32,
)

from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Components import StreamK as StreamKModule
from Tensile.Components.StreamK import StreamK, StreamKHybrid
from Tensile.ExecutionPolicy import usesStreamKArrivalFixup
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _gfx950():
    # Instruction spelling (e.g. SMEM "glc" vs "scope:") follows rocisa's
    # active kernel, which earlier tests may have pointed at another ISA.
    isa = IsaVersion(9, 5, 0)
    with preserve_rocisa_kernel_state():
        makeIsaInfoMap([isa], shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
        rocisa.rocIsa.getInstance().setKernel(isa, 64)
        yield


class _Pool:
    def __init__(self, start=100):
        self._next = start

    def checkOut(self, n, *args, **kwargs):
        reg = self._next
        self._next += n
        return reg

    def checkOutAligned(self, n, align, *args, **kwargs):
        self._next += (-self._next) % align
        return self.checkOut(n)

    def checkIn(self, *args, **kwargs):
        return None


_KERNEL = {
    "TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid",
    "DebugStreamK": 0, "WavefrontSize": 64, "ISA": (9, 5, 0),
    "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2},
}


def _writer(hasSAtomic=True, pap=False):
    sgprs = _Pool()

    @contextmanager
    def alloc_tmp(size, alignment=1, tag=""):
        yield ContinuousRegister(sgprs.checkOut(size), size)

    w = SimpleNamespace(
        sgprPool=sgprs,
        vgprPool=_Pool(),
        labels=SimpleNamespace(getNameInc=lambda n, c=itertools.count(): "%s_%d" % (n, next(c))),
        allocTmpSgpr=alloc_tmp,
        isPrefetchAcrossPersistentEnabled=lambda k: pap,
        longBranchScc0=lambda label, posNeg=0, comment="": Module("longBranchScc0 " + label.getLabelName()),
        states=SimpleNamespace(
            kernel=_KERNEL,
            asmCaps={"HasSAtomic": hasSAtomic},
            archCaps={"NumXCD": 8, "CacheLineBytes": 128, "HasXCDSplitL2": True,
                      "HasInvWbDevFences": False, "RequiresXCntForVolatileVMEM": False,
                      "EnableXnackReplay": False},
            version=(9, 5, 0),
        ),
    )
    w.states.skArrivalFixupLabel = Label("SK_ArrivalFixup_test", "")
    return w


def _arrival(tmpSgprBlock=None, w=None):
    w = w or _writer()
    module = StreamKHybrid().emitArrival(w, _KERNEL, Label("SK_ArrivalDone_test", ""), tmpSgprBlock)
    return list(module.flatitems())


def test_one_atomic_bounded_by_split_minus_one():
    items = _arrival()
    atomics = [i for i in items if isinstance(i, SAtomicInc)]
    assert len(atomics) == 1, "exactly one arrival atomic per part"
    pos = items.index(atomics[0])
    bound = [i for i in items[:pos] if isinstance(i, SSubU32)
             and "sgprSKSplit" in str(i.getParams()[1])]
    assert bound, "the atomic's wrap bound must be SKSplit-1 so the last arrival resets it"
    assert "glc" in str(atomics[0]), "the ticket must come back from the atomic"


def test_no_ready_flag_traffic():
    items = _arrival()
    assert not any(isinstance(i, (BufferStoreB32, BufferLoadB32)) for i in items), (
        "the arrival protocol stores and polls no per-part ready flag")


def test_lds_mailbox_is_saved_and_restored():
    items = _arrival()
    loads = [i for i in items if isinstance(i, DSLoadB32)]
    stores = [i for i in items if isinstance(i, DSStoreB32)]
    # save + read ticket; publish ticket + restore
    assert len(loads) == 2 and len(stores) == 2
    assert items.index(loads[0]) < items.index(stores[0]), "save LDS before publishing"
    assert items.index(stores[1]) > items.index(loads[1]), "restore after every wave read it"
    # LDS idle -> ticket published -> ticket read -> mailbox restored
    assert sum(isinstance(i, SBarrier) for i in items) == 4


@pytest.mark.parametrize("hasSAtomic,pap,debug,expected", [
    (True, False, 0, True),
    (False, False, 0, False),
    (True, True, 0, False),
    (True, False, 1, False),
])
def test_uses_arrival_fixup_gating(hasSAtomic, pap, debug, expected):
    kernel = dict(_KERNEL, DebugStreamK=debug)
    assert StreamKHybrid().usesArrivalFixup(_writer(hasSAtomic, pap), kernel) is expected


@pytest.mark.parametrize("hasSAtomic,pap,debug,expected", [
    (True, False, 0, True),
    (False, False, 0, False),
    (True, True, 0, False),
    (True, False, 1, False),
])
def test_host_capability_matches_codegen(hasSAtomic, pap, debug, expected):
    # The host splits dynamic tiles only for SupportStreamKArrivalFixup
    # kernels; the derivation and the emitted protocol share one predicate.
    assert usesStreamKArrivalFixup(debug, hasSAtomic, pap) is expected
    kernel = dict(_KERNEL, DebugStreamK=debug,
                  InternalSupportParams={"SupportStreamKArrivalFixup": expected})
    assert StreamKHybrid().usesArrivalFixup(_writer(hasSAtomic, pap), kernel) is expected


def test_codegen_rejects_a_mismatched_capability():
    kernel = dict(_KERNEL, InternalSupportParams={"SupportStreamKArrivalFixup": True})
    with pytest.raises(AssertionError, match="SupportStreamKArrivalFixup"):
        StreamKHybrid().usesArrivalFixup(_writer(hasSAtomic=False), kernel)


def test_base_strategy_keeps_flags():
    assert StreamK.usesArrivalFixup(StreamKHybrid(), _writer(), _KERNEL) is False


def test_arrival_reuses_the_enclosing_scratch_block():
    w = _writer()
    before = w.sgprPool._next
    items = _arrival((40, 6), w)
    assert w.sgprPool._next == before, "no sgpr checked out when a block is lent"
    atomic = [i for i in items if isinstance(i, SAtomicInc)][0]
    assert "s[40:41]" in str(atomic), "counter address in the lent block's aligned pair"
    text = "".join(str(i) for i in items)
    assert "s_getpc_b64 s[40:41]" in text, "the long branch reuses the block too"
    used = {int(n) for n in re.findall(r"\bs\[?(\d+)", text)}
    assert used <= {40, 41, 42, 43}, "only the first 4 sgprs of the block are used"


def test_arrival_checks_out_its_own_sgprs_without_a_block():
    w = _writer()
    before = w.sgprPool._next
    _arrival(None, w)
    assert w.sgprPool._next > before


def _recording_hybrid():
    """StreamKHybrid whose fixup step only records how it was asked to run.

    Patched on the instance: a subclass would register as a second
    TileProcessingStrategy implementation for later tests.
    """
    sk = StreamKHybrid()
    sk.steps = []

    def fixupStep(writer, kernel, vectorWidths, elements, edges, tmpVgpr,
                  cvtVgprStruct, sPartialIdx, overwrite=False):
        sk.steps.append(overwrite)
        return Module("fixupStep overwrite=%s" % overwrite)

    sk.fixupStep = fixupStep
    return sk


def _arrival_store(monkeypatch):
    class _WA:
        def dispatch(self, writer, module, name, dynamic, static):
            dynamic(module)

        def flagsBaseOffset(self, writer, kernel):
            return 1024

    realComponent = StreamKModule.Component

    class _Component:
        WorkAssignment = SimpleNamespace(find=lambda writer: _WA())

        def __getattr__(self, name):
            return getattr(realComponent, name)

    # Swap the module's Component reference, not the registry's classes.
    monkeypatch.setattr(StreamKModule, "Component", _Component())
    sk = _recording_hybrid()
    w = _writer()
    kernel = dict(_KERNEL, StreamKAtomic=0)
    module = sk.storeBranches(w, kernel, Label("SK_Partials_test", ""), {False: 1},
                              {False: []}, 0, None)
    return sk, w, list(module.flatitems())


def test_dynamic_store_has_no_flag_spin_when_arriving(monkeypatch):
    _, w, items = _arrival_store(monkeypatch)
    assert not any(isinstance(i, (BufferStoreB32, BufferLoadB32, SLoadB32, SStoreB32))
                   for i in items), "the arrival fixup polls and resets no ready flag"
    assert any(isinstance(i, Label) and i.getLabelName() == w.states.skArrivalFixupLabel.getLabelName()
               for i in items), "storeBranches emits the fixup entry emitArrival jumps to"


def test_fixup_sums_partials_in_part_order(monkeypatch):
    # The sum is ((p0 + p1) + p2) + ... whichever part arrived last. A last
    # part >= 2 replaces its accumulators with p0 (the overwrite step) and
    # adds p1.. in a loop that includes its own partial; a last part 0 or 1
    # starts from its accumulators (p0 + p1 == p1 + p0) and adds the other.
    sk, _, items = _arrival_store(monkeypatch)
    assert sk.steps == [True, False], "one overwrite step, one accumulating loop body"
    text = "".join(str(i) for i in items)
    assert "s_cmp_ge_u32 s[sgprStreamKPartialIdx], 2" in text
    assert "s_max_u32" in text, "after p0/p1 the loop continues at p2"
    assert "s_cmp_le_u32" in text, "the loop runs through the tile's last partial"
    loopBranches = [i for i in items if isinstance(i, SCBranchSCC1)]
    assert len(loopBranches) == 1, "one fixup loop, no skip-own-part branch"
