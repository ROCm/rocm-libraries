# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute emitted assignment branches and tile arithmetic without a GPU."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister, sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import SAddU32, SCBranchSCC0, SMovB32
from rocisa.register import RegisterPool

from Tensile.Component import Component
from Tensile.Components.TileProcessingStrategy import DataParallel
from Tensile.Components.PersistentLoop import PersistentLoopOn
from Tensile.Components.PersistentLoop import PersistentKernelState
from Tensile.Components.WorkAssignment import StaticPartition
from Tensile.Components.StreamK import StreamKDynamic
from Tensile.Components.TileProcessingStrategy import TileWork
from Tensile.Components.WorkAssignment import DynamicWorkQueue, Hybrid, StaticGrid
from _persistent_isa import Machine

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _gfx942_instruction_set(_isolate_rocisa_state):
    # The interpreter models gfx942 opcodes and its single barrier instruction.
    # Select that ISA independently of the target inherited from other tests;
    # the existing isolation fixture restores the incoming state afterward.
    from gpu_test_helpers import init_rocisa

    init_rocisa(target="gfx942", wavesize=64)


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return name + str(self.count)


class _Writer(PersistentKernelState):
    def __init__(self, kernel, tile_work):
        self.labels = _Labels()
        self.states = SimpleNamespace(kernel=kernel, currentTileWork=tile_work,
                                      archCaps={"WorkGroupIdFromTTM": False},
                                      rapInPapNextTilePrefetch=False, unrollIdx=0)
        self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
        self.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self.sgprPool.checkOutAligned(size, alignment or 1, tag)
        try:
            yield ContinuousRegister(base, size)
        finally:
            self.sgprPool.checkIn(base)

    def isPersistentConstantsToVgprEnabled(self, kernel):
        return False

    def isPrefetchAcrossPersistentEnabled(self, kernel):
        return True

    def longBranchScc0(self, label, posNeg):
        return SCBranchSCC0(labelName=label.getLabelName())

    def loopCounterName(self, kernel, index):
        return "LoopCounter"

    def calculateLoopNumIter(self, *args):
        module = Module("observable loop-counter setup")
        module.add(SMovB32(sgpr("LoopCounter"), 999))
        module.add(SMovB32(sgpr("OrigLoopCounter"), 998))
        return module

    def setupPrefetchAcrossPersistentLoads(self, *args, **kwargs):
        module = Module("observable data issue")
        module.add(SAddU32(sgpr("Loads"), sgpr("Loads"), 1))
        module.add(SMovB32(sgpr("PersistentPrefetchState"), 1))
        return module


def _kernel(assignment="StaticGrid"):
    return {"TileProcessingStrategy": "DataParallel" if assignment == "StaticGrid" else "StreamK",
            "WorkAssignment": assignment, "ClusterDim": [1, 1], "SpaceFillingAlgo": [],
            "WavefrontSize": 64, "PrefetchGlobalRead": 0, "ReuseAcrossPersistent": 0,
            "enableTDMA": False, "enableTDMB": False, "HalfPLR": False,
            "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2}}


@pytest.mark.parametrize("tiles_m,tiles_n,batches,grid", [
    (1, 1, 1, 7), (1, 7, 1, 7), (1, 8, 1, 7), (3, 4, 3, 7), (9, 5, 2, 16),
])
def test_native_emitted_grid_stride_covers_every_batched_tile_once(monkeypatch, tiles_m, tiles_n, batches, grid):
    kernel = _kernel()
    processing, assignment = DataParallel(), StaticGrid()
    writer = _Writer(kernel, processing.tileWork(kernel))
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: processing)
    monkeypatch.setattr(Component.XCCMapping, "find", lambda writer: lambda writer, kernel: Module("no XCC remap"))
    initialize = assignment.initialize(writer, kernel, processing)
    activate = assignment.activateReservedOrAcquire(writer, kernel, processing, {}, {})
    close = assignment.closeLoop(writer, kernel)
    peek = assignment.peekTileBatch(writer, kernel, "LookaheadBatch")
    covered = []
    total = tiles_m * tiles_n * batches
    for rank in range(grid):
        machine = Machine(WorkGroup0=rank, NumWorkGroups0=tiles_m, NumWorkGroups1=tiles_n,
                          PersistentGrid=grid)
        machine["SizesFree+2"] = batches
        if machine.run(initialize) == "label_KernelEnd":
            assert rank >= total
            continue
        while True:
            cursor = machine["NextTile"]
            machine.run(peek)
            assert machine["NextTile"] == cursor
            assert machine["LookaheadBatch"] == cursor // (tiles_m * tiles_n)
            machine.run(activate)
            tile = machine["WorkGroup2"] * tiles_m * tiles_n + machine["WorkGroup1"] * tiles_m + machine["WorkGroup0"]
            covered.append(tile)
            assert tile == cursor
            if machine.run(close) is None:
                break
    assert sorted(covered) == list(range(total))


def _fake_fetch(self, writer, kernel, **kwargs):
    item = writer.sgprPool.checkOut(1, "observed queue item")
    module = Module("observable stateful queue acquisition")
    module.add(SMovB32(sgpr(item), sgpr("QueueCursor")))
    module.add(SAddU32(sgpr("QueueCursor"), sgpr("QueueCursor"), 1))
    module.add(SAddU32(sgpr("Pops"), sgpr("Pops"), 1))
    return module, item


class _Processing:
    # Delegate the real eligibility and partition contract without registering
    # this test adapter as a selectable TileProcessingStrategy component.
    prefetchEligibility = StreamKDynamic.prefetchEligibility
    queuePartition = StreamKDynamic.queuePartition

    def tileWork(self, kernel):
        return TileWork("PersistentTileID", "LocalStart", "LocalEnd")

    def staticPartition(self):
        return StaticPartition("PersistentIteration", "PersistentIterationEnd", "skGrid", "Rank")

    def prefetchAcrossPersistentSetupNextTile(self, writer, kernel, *args, **kwargs):
        module = Module("borrow tile identity")
        for name in writer.papTileIdentityNames(kernel):
            module.add(SMovB32(sgpr(name), 12345))
        return module


@pytest.mark.parametrize("assignment_type,mode", [(DynamicWorkQueue, 1), (Hybrid, 1), (Hybrid, 0)])
@pytest.mark.parametrize("carrier", [0, 0x80000000, 1, 3, 0x80000001])
@pytest.mark.parametrize("exhausted", [False, True])
def test_emitted_reservation_and_prefetch_are_consumed_once(monkeypatch, assignment_type, mode, carrier, exhausted):
    kernel = _kernel(assignment_type.__name__)
    assignment, processing = assignment_type(), _Processing()
    writer = _Writer(kernel, processing.tileWork(kernel))
    monkeypatch.setattr(assignment_type, "fetchAndBroadcast", _fake_fetch)
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: processing)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda writer: assignment)
    prefetch = PersistentLoopOn().prefetch(writer, kernel, {}, {})
    item = 5 if exhausted else 2
    machine = Machine(QueueCursor=item, NextWorkItem=item, TotalItems=5,
                      PersistentPrefetchState=carrier, WorkAssignmentMode=mode,
                      PersistentIteration=item, PersistentIterationEnd=5,
                      WorkGroup0=11, WorkGroup1=12, WorkGroup2=13,
                      LocalStart=14, LocalEnd=15, LoopCounter=16, OrigLoopCounter=17)
    machine.registers["s[sgprAddressFlags:sgprAddressFlags+1]"] = 1
    identity = {name: machine[name] for name in writer.papTileIdentityNames(kernel)}
    machine.run(prefetch)
    machine.run(prefetch)
    assert machine["Pops"] == int(mode != 0 and carrier == 0)
    assert machine["Loads"] == int(not exhausted and not (carrier & 1))
    assert {name: machine[name] for name in identity} == identity
    assert machine["LoopCounter"] == 16 and machine["OrigLoopCounter"] == 17
    assert machine.barriers == int(not exhausted and not (carrier & 1))
    assert writer.states.rapInPapNextTilePrefetch is False
    if mode:
        acquire, acquired = assignment.acquireQueueItem(writer, kernel)
        exit_label = machine.run(acquire)
        assert machine["Pops"] == int(carrier == 0)
        assert exit_label == ("label_KernelEnd" if exhausted else None)
