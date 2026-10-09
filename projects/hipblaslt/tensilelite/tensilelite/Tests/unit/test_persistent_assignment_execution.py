# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute persistent assignment and prefetch control flow in small GPU kernels."""

import struct
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
from gpu_test_helpers import (
    GPU_MARKS, assemble_kernel, generate_kernel_asm, generate_load_params,
    init_rocisa, run_on_gpu, run_scalar_kernel,
)

pytestmark = [pytest.mark.unit, *GPU_MARKS]


@pytest.fixture(autouse=True)
def _gpu_instruction_set(_isolate_rocisa_state):
    init_rocisa(wavesize=64)


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return name + str(self.count)


class _Writer(PersistentKernelState):
    def __init__(self, kernel, tile_work, registers):
        self.labels = _Labels()
        self.states = SimpleNamespace(kernel=kernel, currentTileWork=tile_work,
                                      archCaps={"WorkGroupIdFromTTM": False},
                                      rapInPapNextTilePrefetch=False, unrollIdx=0)
        self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
        self.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
        # Keep launch arguments and result-export registers out of emitter temporaries.
        self.sgprPool.checkOut(6, "kernel arguments")
        self.vgprPool.checkOut(3, "lane ID and result export")
        self.sgprs = {}
        for name, size in registers:
            self.sgprs[name] = self.sgprPool.checkOutAligned(size, 2 if size > 1 else 1, name)

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
def test_data_parallel_emitted_grid_stride_covers_every_batched_tile_once(
    monkeypatch, tmp_path, tiles_m, tiles_n, batches, grid,
):
    kernel = _kernel()
    processing, assignment = DataParallel(), StaticGrid()
    registers = [(name, 1) for name in (
        "WorkGroup0", "WorkGroup1", "WorkGroup2", "NumWorkGroups0", "NumWorkGroups1",
        "PersistentGrid", "NextTile", "TotalTiles", "LookaheadBatch", "CursorBefore",
        "CursorAfterPeek", "TraceOffset", "TraceStep",
    )] + [("SizesFree", 3)]
    writer = _Writer(kernel, processing.tileWork(kernel), registers)
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: processing)
    monkeypatch.setattr(Component.XCCMapping, "find", lambda writer: lambda writer, kernel: Module("no XCC remap"))
    initialize = assignment.initialize(writer, kernel, processing)
    activate = assignment.activateReservedOrAcquire(writer, kernel, processing, {}, {})
    close = assignment.closeLoop(writer, kernel)
    peek = assignment.peekTileBatch(writer, kernel, "LookaheadBatch")

    total = tiles_m * tiles_n * batches
    # One extra trace slot detects an unintended iteration. The guard bounds a
    # broken loop before it can overwrite the result buffer or hang the GPU.
    slots = (total + grid - 1) // grid + 1
    outputs = ("CursorBefore", "CursorAfterPeek", "LookaheadBatch",
               "WorkGroup0", "WorkGroup1", "WorkGroup2")
    record_bytes = len(outputs) * 64 * 4
    args = [("output", 8, "global_buffer", "u32")]
    inputs = {"NumWorkGroups0": tiles_m, "NumWorkGroups1": tiles_n,
              "SizesFree+2": batches, "PersistentGrid": grid}
    loads = [(4, 2, 0, "output pointer")]
    for i, name in enumerate(inputs):
        args.append((f"input_{i}", 4, "by_value", "u32"))
        loads.append((name, 1, 8 + 4 * i, name))
    body = [
        str(generate_load_params(loads)),
        "s_mov_b32 s[sgprWorkGroup0], s2",  # Actual hardware workgroup rank.
        f"s_mul_i32 s[sgprTraceOffset], s2, {slots * record_bytes}",
        "s_mov_b32 s[sgprTraceStep], 0",
        str(initialize),
        "label_PersistentLoopStart:",
        f"s_cmp_ge_u32 s[sgprTraceStep], {slots}",
        "s_cbranch_scc1 label_KernelEnd",
        "s_mov_b32 s[sgprCursorBefore], s[sgprNextTile]",
        str(peek),
        "s_mov_b32 s[sgprCursorAfterPeek], s[sgprNextTile]",
        str(activate),
    ]
    for i, name in enumerate(outputs):
        body.extend([
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32 v1, 2, v0",
            "v_add_u32 v1, s[sgprTraceOffset], v1",
            f"v_add_u32 v1, {i * 64 * 4}, v1",
            f"v_mov_b32 v2, s[sgpr{name}]",
            "global_store_dword v1, v2, s[4:5]",
        ])
    body.extend([
        f"s_add_u32 s[sgprTraceOffset], s[sgprTraceOffset], {record_bytes}",
        "s_add_u32 s[sgprTraceStep], s[sgprTraceStep], 1",
        str(close),
        "label_KernelEnd:",
    ])
    asm = generate_kernel_asm("\n".join(body), writer, args, num_threads=64)
    (tmp_path / "grid_stride.s").write_text(asm)
    co = str(tmp_path / "grid_stride.co")
    assemble_kernel(asm, co)
    raw = run_on_gpu(co, grid * slots * record_bytes, scalars=tuple(inputs.values()),
                     num_threads=64, grid=grid)
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    covered = []
    for rank in range(grid):
        expected_tiles = list(range(rank, total, grid))
        for step in range(slots):
            offset = (rank * slots + step) * len(outputs) * 64
            record = words[offset:offset + len(outputs) * 64]
            if step >= len(expected_tiles):
                assert record == (0xffffffff,) * len(record), "workgroup must have exited"
                continue
            tile = expected_tiles[step]
            batch, within_batch = divmod(tile, tiles_m * tiles_n)
            n, m = divmod(within_batch, tiles_m)
            expected = (tile, tile, batch, m, n, batch)
            for i, value in enumerate(expected):
                assert record[i * 64:(i + 1) * 64] == (value,) * 64, (rank, step, outputs[i])
            covered.append(record[5 * 64] * tiles_m * tiles_n + record[4 * 64] * tiles_m + record[3 * 64])
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
def test_emitted_reservation_and_prefetch_are_consumed_once(
    monkeypatch, tmp_path, assignment_type, mode, carrier, exhausted,
):
    kernel = _kernel(assignment_type.__name__)
    assignment, processing = assignment_type(), _Processing()
    registers = [(name, 1) for name in (
        "QueueCursor", "NextWorkItem", "TotalItems", "PersistentPrefetchState",
        "WorkAssignmentMode", "PersistentIteration", "PersistentIterationEnd",
        "WorkGroup0", "WorkGroup1", "WorkGroup2", "LocalStart", "LocalEnd",
        "LoopCounter", "OrigLoopCounter", "Pops", "Loads", "Terminated",
    )] + [("AddressFlags", 2)]
    writer = _Writer(kernel, processing.tileWork(kernel), registers)
    # Count queue acquisition and load issue on the device. The real reservation,
    # eligibility, branching, barriers, and register save/restore remain intact.
    monkeypatch.setattr(assignment_type, "fetchAndBroadcast", _fake_fetch)
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda writer: processing)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda writer: assignment)
    loop = PersistentLoopOn()
    body = str(loop.prefetch(writer, kernel, {}, {}))
    body += str(loop.prefetch(writer, kernel, {}, {}))
    item = 5 if exhausted else 2
    inputs = dict(QueueCursor=item, NextWorkItem=item, TotalItems=5,
                  PersistentPrefetchState=carrier, WorkAssignmentMode=mode,
                  PersistentIteration=item, PersistentIterationEnd=5,
                  WorkGroup0=11, WorkGroup1=12, WorkGroup2=13,
                  LocalStart=14, LocalEnd=15, LoopCounter=16, OrigLoopCounter=17,
                  Pops=0, Loads=0, Terminated=0)
    inputs.update({"AddressFlags+0": 1, "AddressFlags+1": 0})
    identity = {name: inputs[name] for name in writer.papTileIdentityNames(kernel)}
    outputs = ["Pops", "Loads", *identity, "LoopCounter", "OrigLoopCounter", "Terminated"]
    if mode:
        acquire, acquired = assignment.acquireQueueItem(writer, kernel)
        writer.sgprs["Acquired"] = acquired
        outputs.append("Acquired")
        body += "s_mov_b32 s[sgprTerminated], 1\n" + str(acquire)
        body += "s_mov_b32 s[sgprTerminated], 0\nlabel_KernelEnd:\n"
    observed = run_scalar_kernel(writer, body, inputs, outputs, tmp_path)
    assert observed["Pops"] == (int(mode != 0 and carrier == 0),) * 64
    assert observed["Loads"] == (int(not exhausted and not (carrier & 1)),) * 64
    for name, value in identity.items():
        assert observed[name] == (value,) * 64
    assert observed["LoopCounter"] == (16,) * 64
    assert observed["OrigLoopCounter"] == (17,) * 64
    assert observed["Terminated"] == (int(mode != 0 and exhausted),) * 64
    assert writer.states.rapInPapNextTilePrefetch is False
    if mode:
        assert observed["Acquired"] == (item,) * 64
        writer.sgprPool.checkIn(acquired)
