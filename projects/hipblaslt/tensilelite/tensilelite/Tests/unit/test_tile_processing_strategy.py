# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent strategy selection and ordinary writer defaults."""

from types import SimpleNamespace

import pytest
from rocisa.container import sgpr, vgpr
from rocisa.instruction import VMovB32

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Components.PersistentLoop import PersistentLoopOff
from Tensile.Components.StreamK import StreamKDynamic, StreamKHybrid, StreamKTwoTileDPFirst
from Tensile.Components.TileProcessingStrategy import DataParallel, TileProcessingStrategy

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
