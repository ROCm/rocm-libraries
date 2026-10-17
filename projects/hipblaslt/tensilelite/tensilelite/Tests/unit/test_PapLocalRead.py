# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for persistent first-PLR payload ownership and consumption."""

from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import SAddU32
from rocisa.label import LabelManager
from rocisa.register import RegisterPool

from Tensile.Components.PersistentLoop import PersistentKernelState
from _pap_isa import Machine

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isa(_isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx1250", wavesize=32)


def _writer(*, limit=1024, scales=True):
    writer = PersistentKernelState()
    writer.tPA, writer.tPB = {}, {}
    writer.papPgr2PipelineEnabled = lambda *args: True
    writer.states = SimpleNamespace(
        totalVgprs=678, regCaps={"MaxVgpr": limit}, memTokenPapPlr=7,
        a=SimpleNamespace(startVgprValu=286, numVgprValuPerBlock=64),
        b=SimpleNamespace(startVgprValu=416, numVgprValuPerBlock=128),
        mxsa=SimpleNamespace(startVgprValu=0, numVgprValuPerBlock=4),
        mxsb=SimpleNamespace(startVgprValu=10, numVgprValuPerBlock=8),
    )
    writer.labels = LabelManager()
    writer.vgprPool = RegisterPool(678, RegisterType.Vgpr, defaultPreventOverflow=False)
    kernel = dict(PrefetchAcrossPersistent=1,
                  ProblemType={"MXBlockA": 32 if scales else 0},
                  UnrollMajorLDSA=True, UnrollMajorLDSB=True,
                  UnrollMajorLDSMXSA=True, UnrollMajorLDSMXSB=True)
    return writer, kernel


def test_store_and_setup_borrow_only_non_payload_registers():
    writer, kernel = _writer()
    retained = {r for start, size in writer.papLocalReadPayloadRanges(kernel)
                for r in range(start, start + size)}
    assert len(retained) == 204
    # Exercise both phases' real allocator transitions. Every lendable slot is
    # checked out; an accidentally exposed operand or MX scale must be observed.
    for _ in range(2):
        writer.papReclaimValuRegisters(kernel, 0, 26, "ValuMXSAB")
        writer.papReclaimValuRegisters(kernel, 286, 386, "ValuAB")
        scratch = [writer.vgprPool.checkOut(1, "scratch") for _ in range(412 - 204)]
        assert not retained.intersection(scratch)
        assert set(scratch) == (set(range(26)) | set(range(286, 672))) - retained
        for reg in scratch:
            writer.vgprPool.checkIn(reg)
        writer.papReclaimValuRegisters(kernel, 0, 26, "ValuMXSAB", add=False)
        writer.papReclaimValuRegisters(kernel, 286, 386, "ValuAB", add=False)


@pytest.mark.parametrize("scales, expected", [(False, 192), (True, 204)])
def test_only_existing_first_buffers_are_retained(scales, expected):
    writer, kernel = _writer(scales=scales)
    assert sum(size for _, size in writer.papLocalReadPayloadRanges(kernel)) == expected
    assert writer.vgprPool.size() == 678


@pytest.mark.parametrize("change", ["budget", "format", "pap_off"])
def test_unsupported_resource_or_format_keeps_ordinary_entry(change):
    writer, kernel = _writer(limit=896 if change == "budget" else 1024)
    if change == "format":
        kernel["UnrollMajorLDSA"] = False
    if change == "pap_off":
        kernel["PrefetchAcrossPersistent"] = 0
    reads = Module()
    reads.add(SAddU32(sgpr("Reads"), sgpr("Reads"), 1))
    assert writer.papLocalReadPayloadRanges(kernel) == ()
    assert writer.papLocalReadEntry(kernel, reads, [Module()]) is reads
    assert not hasattr(writer.states, "papFirstPlrReads")


@pytest.mark.parametrize("state", [0, 1, 8, 16, 24, 0x8000001f])
def test_entry_consumes_only_plr_flag_and_preserves_pointer_update(state):
    writer, kernel = _writer()
    reads = Module()
    reads.add(SAddU32(sgpr("Reads"), sgpr("Reads"), 1))
    entry = writer.papLocalReadEntry(kernel, reads, [Module()])
    entry.add(SAddU32(sgpr("Pointer"), sgpr("Pointer"), 128))
    machine = Machine(PersistentPrefetchState=state, Reads=0, Pointer=0)
    assert machine.run(entry) is None
    assert machine["Reads"] == (0 if state & 16 else 1)
    assert machine["Pointer"] == 128
    assert machine["PersistentPrefetchState"] == state & ~16


def test_real_pack_work_falls_back_without_claiming_carried_payload():
    writer, kernel = _writer()
    reads, pack = Module(), Module()
    pack.add(SAddU32(sgpr("Pack"), sgpr("Pack"), 1))
    assert writer.papLocalReadEntry(kernel, reads, [pack]) is reads
    assert not hasattr(writer.states, "papFirstPlrReads")


def test_carried_reads_complete_at_common_first_operand_consumer():
    from Tensile.KernelWriter import _needsPreLoopLocalReadDrain
    kernel = {"UseCustomMainLoopSchedule": False}
    assert not _needsPreLoopLocalReadDrain(kernel, 1, False)
    assert _needsPreLoopLocalReadDrain(kernel, 1, False, papLocalReadIssued=True)
    assert not _needsPreLoopLocalReadDrain(kernel, 1, True, papLocalReadIssued=True)



def test_carried_completion_only_names_the_operand_completion_token():
    from rocisa.instruction import SBarrier
    writer, _ = _writer()
    writer.states.memTokenLdsBuffer0 = 0
    completion = writer.papLocalReadCompletion()
    barriers = [item for item in completion.flatitems() if isinstance(item, SBarrier)]
    assert len(barriers) == 1
    assert list(barriers[0].getMemToken().tokens) == [7]
