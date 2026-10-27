# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the StreamK fixup by last arrival on the dynamic path
(DynamicWorkQueue, and Hybrid's dynamic sub-mode; protocol: see
StreamK.emitArrival), on a fake writer.
"""

import itertools
import re
import shutil
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

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
    SCBranchSCC0,
    SCBranchSCC1,
    SCmpGeU32,
    SCmpLeU32,
    SLoadB32,
    SMaxU32,
    SStoreB32,
    SSubU32,
    VAddF32,
    VAddF64,
    VAddPKF16,
    VAddU32,
    VFmaMixF32,
    VMovB32,
)

from tensilelite.Common import DataDirection, IsaVersion
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Components import StreamK as StreamKModule
from tensilelite.Components.StreamK import StreamKDynamic, StreamKHybrid, StreamKTwoTileDPFirst
from tensilelite.ExecutionPolicy import usesStreamKArrivalFixup
from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

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
    "DebugStreamK": 0, "WavefrontSize": 64, "ISA": (9, 5, 0), "LdsOffsetBias": 0,
    "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2,
                    "UseScaleAlphaVec": False, "UseScaleAB": ""},
}


def _writer(hasSAtomic=True, pap=False, useBias=DataDirection.NONE):
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
        longBranchScc0=lambda label, posNeg=0, comment="": SCBranchSCC0(labelName=label.getLabelName()),
        states=SimpleNamespace(
            kernel=_KERNEL,
            asmCaps={"HasSAtomic": hasSAtomic, "HasScalarStore": True},
            archCaps={"NumXCD": 8, "CacheLineBytes": 128, "HasXCDSplitL2": True,
                      "HasInvWbDevFences": False, "RequiresXCntForVolatileVMEM": False,
                      "EnableXnackReplay": False},
            version=(9, 5, 0),
            useBias=useBias,
            memTokenLdsBuffer0=0,
        ),
    )
    w.states.skArrivalFixupLabel = Label("SK_ArrivalFixup_test", "")
    return w


# The strategies with a dynamic path; both fix split tiles up by last arrival.
_DYNAMIC = [StreamKHybrid, StreamKDynamic]


@pytest.fixture(params=_DYNAMIC, ids=lambda c: c.__name__)
def strategy(request):
    return request.param


def _arrival(strategy, tmpSgprBlock=None, w=None, kernel=_KERNEL):
    w = w or _writer()
    module = strategy().emitArrival(w, kernel, Label("SK_ArrivalDone_test", ""), tmpSgprBlock)
    return list(module.flatitems())


def test_one_atomic_bounded_by_split_minus_one(strategy):
    items = _arrival(strategy)
    atomics = [i for i in items if isinstance(i, SAtomicInc)]
    assert len(atomics) == 1, "exactly one arrival atomic per part"
    pos = items.index(atomics[0])
    bound = [i for i in items[:pos] if isinstance(i, SSubU32)
             and "sgprSKSplit" in str(i.getParams()[1])]
    assert bound, "the atomic's wrap bound must be SKSplit-1 so the last arrival resets it"
    assert "glc" in str(atomics[0]), "the ticket must come back from the atomic"


def test_counter_indexed_by_the_strategys_split_tiles(strategy):
    # The counter's partial tile comes from the strategy's own split-tile count.
    text = "".join(str(i) for i in _arrival(strategy))
    assert "sgpr%s" % strategy.splitTilesSgpr in text
    other = ({c.splitTilesSgpr for c in _DYNAMIC} - {strategy.splitTilesSgpr}).pop()
    assert "sgpr%s" % other not in text


def test_no_ready_flag_traffic(strategy):
    items = _arrival(strategy)
    assert not any(isinstance(i, (BufferStoreB32, BufferLoadB32)) for i in items), (
        "the arrival protocol stores and polls no per-part ready flag")


def test_ticket_goes_through_the_work_queue_mailbox(monkeypatch, strategy):
    calls = []
    for name in ("_emitMailboxAddressAndWave0Skip", "_emitWorkItemMailbox"):
        real = getattr(StreamKModule, name)
        monkeypatch.setattr(StreamKModule, name,
                            lambda *a, _n=name, _r=real, **k: (calls.append(_n), _r(*a, **k)))
    items = _arrival(strategy)
    assert calls == ["_emitMailboxAddressAndWave0Skip", "_emitWorkItemMailbox"]
    ds = [i for i in items if isinstance(i, (DSLoadB32, DSStoreB32))]
    # publish ticket, read ticket; nothing to save without epilogue vectors
    assert [type(i) for i in ds] == [DSStoreB32, DSLoadB32]
    assert all(i.getMemToken() is not None for i in ds), "LDS0 tokens like the queue pop"
    # ticket published -> every wave read it
    assert sum(isinstance(i, SBarrier) for i in items) == 2


def _vectors_kernel(**problem):
    return dict(_KERNEL, ProblemType=dict(_KERNEL["ProblemType"], **problem))


@pytest.mark.parametrize("useBias,problem", [
    (DataDirection.READ, {}),
    (DataDirection.NONE, {"UseScaleAlphaVec": True}),
    (DataDirection.NONE, {"UseScaleAB": "Vector"}),
])
def test_epilogue_vectors_under_the_mailbox_are_saved_and_restored(strategy, useBias, problem):
    items = _arrival(strategy, w=_writer(useBias=useBias), kernel=_vectors_kernel(**problem))
    ds = [i for i in items if isinstance(i, (DSLoadB32, DSStoreB32))]
    # save, publish ticket, read ticket, restore
    assert [type(i) for i in ds] == [DSLoadB32, DSStoreB32, DSLoadB32, DSStoreB32]
    notLast = [i for i in items if isinstance(i, SCBranchSCC0)
               and i.labelName.endswith("SK_ArrivalDone_test")]
    assert len(notLast) == 1
    assert items.index(notLast[0]) < items.index(ds[-1]), "only the last part restores"
    # ticket published -> every wave read it -> restored (last part only)
    assert sum(isinstance(i, SBarrier) for i in items) == 3


def test_epilogue_vectors_past_the_mailbox_are_not_saved(strategy):
    kernel = dict(_KERNEL, LdsOffsetBias=_KERNEL["WavefrontSize"] * 4)
    items = _arrival(strategy, w=_writer(useBias=DataDirection.READ), kernel=kernel)
    assert sum(isinstance(i, (DSLoadB32, DSStoreB32)) for i in items) == 2


@pytest.mark.parametrize("hasSAtomic,pap,debug,expected", [
    (True, False, 0, True),
    (False, False, 0, False),
    (True, True, 0, False),
    (True, False, 1, False),
])
def test_uses_arrival_fixup_gating(strategy, hasSAtomic, pap, debug, expected):
    kernel = dict(_KERNEL, DebugStreamK=debug)
    assert strategy().usesArrivalFixup(_writer(hasSAtomic, pap), kernel) is expected


@pytest.mark.parametrize("hasSAtomic,pap,debug,expected", [
    (True, False, 0, True),
    (False, False, 0, False),
    (True, True, 0, False),
    (True, False, 1, False),
])
def test_host_capability_matches_codegen(strategy, hasSAtomic, pap, debug, expected):
    # Codegen and the host capability share one predicate.
    assert usesStreamKArrivalFixup(debug, hasSAtomic, pap) is expected
    kernel = dict(_KERNEL, DebugStreamK=debug,
                  InternalSupportParams={"SupportStreamKArrivalFixup": expected})
    assert strategy().usesArrivalFixup(_writer(hasSAtomic, pap), kernel) is expected


def test_codegen_rejects_a_mismatched_capability(strategy):
    kernel = dict(_KERNEL, InternalSupportParams={"SupportStreamKArrivalFixup": True})
    with pytest.raises(AssertionError, match="SupportStreamKArrivalFixup"):
        strategy().usesArrivalFixup(_writer(hasSAtomic=False), kernel)


def test_other_strategies_keep_flags():
    # No dynamic path: StaticGrid strategies keep per-part ready flags.
    assert StreamKTwoTileDPFirst().usesArrivalFixup(_writer(), _KERNEL) is False


def test_arrival_reuses_the_enclosing_scratch_block(strategy):
    w = _writer()
    before = w.sgprPool._next
    items = _arrival(strategy, (40, 6), w)
    assert w.sgprPool._next == before, "no sgpr checked out when a block is lent"
    atomic = [i for i in items if isinstance(i, SAtomicInc)][0]
    assert "s[40:41]" in str(atomic), "counter address in the lent block's aligned pair"
    text = "".join(str(i) for i in items)
    assert "s_getpc_b64 s[40:41]" in text, "the long branch reuses the block too"
    used = {int(n) for n in re.findall(r"\bs\[?(\d+)", text)}
    assert used <= {40, 41, 42, 43}, "only the first 4 sgprs of the block are used"


def test_arrival_checks_out_its_own_sgprs_without_a_block(strategy):
    w = _writer()
    before = w.sgprPool._next
    _arrival(strategy, None, w)
    assert w.sgprPool._next > before


def _recording(strategy):
    """``strategy`` whose fixup step only records how it was asked to run.

    Patched on the instance: a subclass would register as a second
    TileProcessingStrategy implementation for later tests.
    """
    sk = strategy()
    sk.steps = []

    def fixupStep(writer, kernel, vectorWidths, elements, edges, tmpVgpr,
                  cvtVgprStruct, sPartialIdx, overwrite=False):
        sk.steps.append(overwrite)
        return Module("fixupStep overwrite=%s" % overwrite)

    sk.fixupStep = fixupStep
    return sk


def _arrival_store(monkeypatch, strategy, hasSAtomic=True):
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
    sk = _recording(strategy)
    w = _writer(hasSAtomic)
    kernel = dict(_KERNEL, StreamKAtomic=0)
    module = sk.storeBranches(w, kernel, Label("SK_Partials_test", ""), {False: 1},
                              {False: []}, 0, None)
    return sk, w, list(module.flatitems())


def test_dynamic_store_has_no_flag_spin_when_arriving(monkeypatch, strategy):
    _, w, items = _arrival_store(monkeypatch, strategy)
    assert not any(isinstance(i, (BufferStoreB32, BufferLoadB32, SLoadB32, SStoreB32))
                   for i in items), "the arrival fixup polls and resets no ready flag"
    assert any(isinstance(i, Label) and i.getLabelName() == w.states.skArrivalFixupLabel.getLabelName()
               for i in items), "storeBranches emits the fixup entry emitArrival jumps to"


def test_dynamic_store_keeps_ready_flags_without_arrival(monkeypatch, strategy):
    # Without scalar atomics the dynamic path keeps per-part ready flags.
    sk, w, items = _arrival_store(monkeypatch, strategy, hasSAtomic=False)
    assert sk.steps == [False], "one accumulating fixup step, no overwrite"
    assert any(isinstance(i, (BufferLoadB32, SLoadB32)) for i in items), "reads the ready flag"
    assert not any(isinstance(i, SAtomicInc) for i in items)
    assert not any(isinstance(i, Label) and i.getLabelName().startswith("SK_ArrivalFixup")
                   for i in items)


def test_fixup_sums_partials_in_part_order(monkeypatch, strategy):
    # Order: see StreamK.arrivalStoreBranches.
    sk, _, items = _arrival_store(monkeypatch, strategy)
    assert sk.steps == [True, False], "one overwrite step, one accumulating loop body"
    ownPart = [i for i in items if isinstance(i, SCmpGeU32)
               and "sgprStreamKPartialIdx" in str(i.getParams()[0])]
    assert len(ownPart) == 1 and str(ownPart[0].getParams()[1]) == "2", (
        "parts 0 and 1 start from their own accumulators")
    assert any(isinstance(i, SMaxU32) for i in items), "after p0/p1 the loop continues at p2"
    assert any(isinstance(i, SCmpLeU32) for i in items), "the loop runs through the tile's last partial"
    loopBranches = [i for i in items if isinstance(i, SCBranchSCC1)]
    assert len(loopBranches) == 1, "one fixup loop, no skip-own-part branch"


# ---------------------------------------------------------------------------
# fixupBatch(overwrite=True) prefills the additive identity before each add;
# without overwrite nothing is prefilled.
# ---------------------------------------------------------------------------
_TYPES = ("Half", "BFloat16", "Single", "Int8", "Int32", "Double",
          "SingleComplex", "DoubleComplex")


def _dtype(name):
    checks = {"is%s" % t: (lambda t=t: t == name) for t in _TYPES}
    checks.update(isFloat8=lambda: False, isFloat8_fnuz=lambda: False,
                  isAnyBFloat8=lambda: False)
    return SimpleNamespace(**checks)


def _fixup_batch(compute, hpa=False, data=None, overwrite=True, regsPerData=1):
    dest = "BFloat16" if compute == "BFloat16" else compute
    kernel = {
        "StoreSyncOpt": 0, "EnableMatrixInstruction": True, "MIWaveGroup": [2, 2],
        "WorkGroup": [256, 1, 1], "WavefrontSize": 64, "LocalSplitU": 1,
        "BufferStore": True, "MIArchVgpr": True, "EnableMatrixInstructionStore": True,
        "ProblemType": {"ComputeDataType": _dtype(compute), "DataType": _dtype(data or compute),
                        "DestDataType": _dtype(dest), "HighPrecisionAccumulate": hpa},
    }
    bpe = {"Double": 8, "SingleComplex": 8, "DoubleComplex": 16}.get(compute, 4)
    writer = SimpleNamespace(
        db={"ConservativeWaitCnt": 0, "ForceEdgeStores": 0, "AssertNoEdge": 0,
            "ForceExpectedValue": 0, "ForceVSerial": 0, "CheckValueC": 0},
        readInput=lambda *args, **kwargs: Module("readInput"),
        vgprPool=_Pool(),
        states=SimpleNamespace(bpeCinternal=bpe, bpr=4, c=SimpleNamespace(startVgprValu=0),
                               interleaveStoreVmcnt=False, asmCaps={"HasWMMA": False},
                               mixinst=VFmaMixF32, serializedStore=False),
    )
    ss = SimpleNamespace(
        optSingleColVgpr=0, optSharedColVgpr=0, optSGPRUsage="", optSrdIncForRow=0,
        setupStoreElementsForBatch=lambda *args, **kwargs: None,
        elementAddr=[SimpleNamespace(addrCVgpr=10, addrDVgpr=11)], elementData=[20],
        elementMask=[30], elementSumIdx=[0], sharedColDVgprs=False,
        checkInTempVgprC=lambda: None,
        cfg=SimpleNamespace(numVgprsPerDataPerVI=regsPerData, halfDataRegPerVI=False),
    )
    cvt = SimpleNamespace(vgprBf16Mask=40, vgprFp32Nan=41, vgprBf16Inc=42)
    module = StreamKHybrid().fixupBatch(writer, kernel, ss, 0, False, 1, [(0, 0, 0, 0)], None, None,
                                  50, cvt, None, 60, None, None, overwrite=overwrite)
    return list(module.flatitems())


def _valuc_index(operand):
    """First ValuC register index of ``v[vgprValuC+N]`` or ``v[vgprValuC+N:...]``."""
    match = re.match(r"v\[vgprValuC\+(\d+)[\]:]", str(operand))
    assert match, f"not a ValuC operand: {operand}"
    return int(match.group(1))


def _prefills(items):
    return [i for i in items if isinstance(i, VMovB32) and "ValuC" in str(i.getParams()[0])]


@pytest.mark.parametrize("compute,hpa,data,regsPerData,addClass,identity", [
    ("Single", False, None, 1, VAddF32, ["0x80000000"]),
    ("Int32", False, None, 1, VAddU32, ["0"]),
    ("Single", False, "Int8", 1, VAddU32, ["0"]),
    ("Half", False, None, 1, VAddPKF16, ["0x80008000"]),
    ("Half", True, None, 1, VFmaMixF32, ["0x80000000"]),
    ("BFloat16", True, None, 1, VAddF32, ["0x80000000"]),
    ("Double", False, None, 2, VAddF64, ["0", "0x80000000"]),
    ("SingleComplex", False, None, 2, VAddF32, ["0x80000000", "0x80000000"]),
    ("DoubleComplex", False, None, 4, VAddF64, ["0", "0x80000000", "0", "0x80000000"]),
])
def test_overwrite_prefills_identity_before_add(compute, hpa, data, regsPerData, addClass, identity):
    items = _fixup_batch(compute, hpa, data, regsPerData=regsPerData)
    movs = _prefills(items)
    assert [str(m.getParams()[1]) for m in movs] == identity
    # The prefill directly precedes the first add, which reads what it wrote.
    first = items.index(movs[0])
    assert isinstance(items[first + len(movs)], addClass)
    assert _valuc_index(items[first + len(movs)].getParams()[0]) == _valuc_index(movs[0].getParams()[0])


@pytest.mark.parametrize("compute,hpa", [("Single", False), ("Half", False), ("Double", False)])
def test_accumulating_fixup_has_no_prefill(compute, hpa):
    assert _prefills(_fixup_batch(compute, hpa, overwrite=False,
                                  regsPerData=2 if compute == "Double" else 1)) == []
