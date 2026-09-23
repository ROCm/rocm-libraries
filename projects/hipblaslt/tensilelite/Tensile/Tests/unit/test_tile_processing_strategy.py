# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent strategy selection and ordinary writer defaults."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister, sgpr, vgpr
from rocisa.instruction import SCmpEQU32, VMovB32

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Components.PersistentLoop import PersistentLoopOff
from Tensile.Components.StreamK import StreamKDynamic, StreamKHybrid, StreamKTwoTileDPFirst
from Tensile.Components.TileProcessingStrategy import DataParallel, TileProcessingStrategy
from _persistent_isa import Machine

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("strategy,assignment,expected", [
    ("None", "StaticGrid", None),
    ("DataParallel", "StaticGrid", DataParallel),
    ("StreamK", "StaticGrid", StreamKTwoTileDPFirst),
    ("StreamK", "DynamicWorkQueue", StreamKDynamic),
    ("StreamK", "Hybrid", StreamKHybrid),
])
def test_only_persistent_policies_select_a_strategy(strategy, assignment, expected):
    writer = SimpleNamespace(states=SimpleNamespace(kernel={
        "TileProcessingStrategy": strategy, "WorkAssignment": assignment,
    }))
    selected = TileProcessingStrategy.find(writer)
    assert selected is None if expected is None else isinstance(selected, expected)


def _reject_strategy_lookup(*args, **kwargs):
    pytest.fail("Ordinary writer emission must not request a persistent strategy")


def test_strategy_resolution_follows_temporary_ordinary_kernel_state():
    # The optimized no-load loop temporarily emits an ordinary epilogue, then
    # restores the persistent policy. Resolution must follow that live state.
    kernel = {"TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid"}
    writer = SimpleNamespace(states=SimpleNamespace(kernel=kernel))
    assert isinstance(TileProcessingStrategy.find(writer), StreamKHybrid)
    kernel["TileProcessingStrategy"] = "None"
    assert TileProcessingStrategy.find(writer) is None
    kernel["TileProcessingStrategy"] = "StreamK"
    assert isinstance(TileProcessingStrategy.find(writer), StreamKHybrid)


def test_ordinary_lifecycle_does_not_create_persistent_work(monkeypatch):
    monkeypatch.setattr(TileProcessingStrategy, "find", _reject_strategy_lookup)
    writer = SimpleNamespace(states=SimpleNamespace())
    kernel = {"TileProcessingStrategy": "None"}
    loop = PersistentLoopOff()
    assert not list(loop.initialize(writer, kernel).flatitems())
    assert not list(loop.activateReservedOrAcquire(writer, kernel, {}, {}).flatitems())
    assert not hasattr(writer.states, "currentTileWork")


def test_ordinary_flat_addresses_copy_the_input_pointer(monkeypatch):
    monkeypatch.setattr(TileProcessingStrategy, "find", _reject_strategy_lookup)
    writer = SimpleNamespace(
        states=SimpleNamespace(preventVgprOverflowDuringNewTile=False),
        vgprPool=SimpleNamespace(checkOut=lambda *a, **kw: 10, checkIn=lambda *a: None),
    )
    kernel = {"TileProcessingStrategy": "None", "BufferLoad": 0}
    tensor = {"tensorChar": "A", "isSwizzled": False, "nrp": 0}
    instructions = list(KernelWriterAssembly.graAddresses(writer, kernel, tensor).flatitems())
    assert len(instructions) == 2
    for offset, instruction in enumerate(instructions):
        assert isinstance(instruction, VMovB32)
        assert [str(p) for p in instruction.getParams()] == [
            str(vgpr(10 + offset)), str(sgpr("AddressA+%u" % offset)),
        ]


@pytest.mark.parametrize("k,tail,max_unit,gsu,runtime_gsu,expected", [
    (0, False, 1, 0, 1, 0),
    (16, False, 1, 0, 1, 1),
    (31, False, 1, 0, 1, 1),
    (0, True, 1, 0, 1, 0),
    (16, True, 1, 0, 1, 1),
    (17, True, 1, 0, 1, 2),
    (20, True, 4, 0, 1, 2),
    (18, True, 4, 0, 1, 1),
    (17, True, 1, 2, 2, 1),
    (17, True, 1, 2, 1, 2),
])
def test_ordinary_loop_count_respects_tail_and_gsu(k, tail, max_unit, gsu, runtime_gsu, expected):
    writer = SimpleNamespace(
        states=SimpleNamespace(tailloopInNll=tail, tailloopInNllmaxUnit=max_unit, unrollIdx=0),
        gsuMaskHex=lambda kernel: "0x3fff",
    )
    module = KernelWriterAssembly._calculateOrdinaryLoopNumIter(
        writer, {"DepthU": 16, "GlobalSplitU": gsu}, "LoopCounter", 0,
        ContinuousRegister(100, 3),
    )
    machine = Machine(**{"SizesSum+0": k, "GSU": runtime_gsu})
    assert machine.run(module) is None
    assert machine["LoopCounter"] == expected


@pytest.mark.parametrize("gsu,runtime_gsu,arg_type,expected", [
    (0, 1, 0, 0x12345678),
    (0, 1, 3, 0),
    (2, 1, 3, 0),
    (2, 2, 3, 0x12345678),
])
def test_ordinary_srd_routes_general_batch_and_gsu(monkeypatch, gsu, runtime_gsu, arg_type, expected):
    monkeypatch.setattr(TileProcessingStrategy, "find", _reject_strategy_lookup)
    kernel = {
        "TileProcessingStrategy": "None", "GlobalSplitU": gsu,
        "GlobalSplitUAlgorithm": "MultipleBuffer", "_GlobalAccumulation": "MultipleBuffer",
        "ProblemType": {"SupportUserArgs": True},
    }

    @contextmanager
    def alloc_tmp(*args, **kwargs):
        yield ContinuousRegister(100, 1)

    writer = SimpleNamespace(
        states=SimpleNamespace(kernel=kernel), allocTmpSgpr=alloc_tmp,
        shiftSrd=lambda ch: Module("gfx950 requires no SRD shift"),
        cmpNamedArgTypeEq=lambda module, value, comment: module.add(
            SCmpEQU32(src0=sgpr("ArgType"), src1=value, comment=comment)),
    )
    module = KernelWriterAssembly.allocPostLoopSrd(writer, "D", kernel)
    machine = Machine(ArgType=arg_type, GSU=runtime_gsu)
    machine.registers[str(sgpr("AddressD+0", 2))] = 0x12345678
    machine.registers.update(BufferOOB=0xffffffff, Srd127_96=0)
    assert machine.run(module) is None
    assert machine.registers[str(sgpr("SrdD+0", 2))] == expected
