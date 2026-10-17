# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute classic multicast pending publication and its entry-only lifetime."""

import pytest
from rocisa.code import Module
from rocisa.container import sgpr, vgpr
from rocisa.instruction import SAddU32
from Tensile.Component import Component
from Tensile.Components.WorkAssignment import StaticGrid
from test_PapPgr2Pipeline import _PipelineMachine, _setup
from test_PapShadowAddress import _components  # noqa: F401 - pytest autouse fixture

pytestmark = pytest.mark.unit


def _entry(monkeypatch, cluster=(2, 4), payload=True):
    kernel, tensors, writer = _setup(cluster=cluster)
    writer.states.unrollIdx = 0
    writer.loopCounter = lambda *args: sgpr("LoopCounterL")
    writer.papLocalReadPayloadRanges = lambda kernel: ((286, 64),) if payload else ()
    writer._wait = lambda kernel, a, b, count, *args: SAddU32(
        sgpr("FullWaits" if count == 0 else "PartialWaits"),
        sgpr("FullWaits" if count == 0 else "PartialWaits"), 1)
    writer.getWaitcntCodeForPGR = lambda *args: SAddU32(sgpr("ColdWaits"), sgpr("ColdWaits"), 1)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: StaticGrid())
    return kernel, tensors, writer


@pytest.mark.parametrize("cluster", [(2, 1), (2, 4)])
@pytest.mark.parametrize("q", [0, 1, 2, 3, 4, 6])
@pytest.mark.parametrize("state", [0, 1, 3, 9, 11, 25, 27])
def test_ready1_is_published_once_before_following_phase(monkeypatch, cluster, q, state):
    kernel, tensors, writer = _entry(monkeypatch, cluster)
    a, b = tensors["A"], tensors["B"]
    entry = writer.papPgr2EntryWait(kernel, a, b)
    merge = Module()
    merge.add(writer.papPgr2EntryMerge(kernel, a, b))
    reads = Module()
    reads.add(SAddU32(sgpr("ColdReads"), sgpr("ColdReads"), 1))
    merge.add(writer.papLocalReadEntry(kernel, reads, [Module()]))
    merge.add(writer.papClearPreparedAddress("routing reset preserves pending publication"))
    late = writer.papCompleteDeferredClusterReady(kernel, a, b)
    following = Module()
    StaticGrid()._clusterElectArriveSignal(
        writer, following, labelBase="FollowingPhase", electTag="next phase", wait=True)
    active = bool(state & 8)
    deferred = active and bool(state & 16) and q > 2
    for wave in range(4):
        for phantom in (0, 1):
            machine = _PipelineMachine(PersistentPrefetchState=state, LoopCounterL=q,
                                       PersistentPhantomTile=phantom)
            machine.registers[str(vgpr("Serial"))] = wave * 32
            phase = ([("s_barrier_signal", "-3")] if wave == 0 else [])
            phase += [("s_barrier_wait", "-3")]
            phases = lambda: [op for op in machine.barrier_operations if op[1] == "-3"]
            assert machine.run(entry) is None
            assert bool(machine["PersistentPrefetchState"] & 128) == deferred
            assert machine["FullWaits"] == int(active and not deferred)
            assert machine["PartialWaits"] == int(active)
            assert machine["ColdWaits"] == int(not active)
            assert phases() == phase * int(active and not deferred)
            # The real merge and first-read/routing bit clears must keep bit7.
            assert machine.run(merge) is None
            assert machine["PersistentPrefetchState"] == (128 if deferred else 0)
            assert machine["ColdReads"] == int(not bool(state & 16))
            assert machine.run(late) is None
            assert machine["PersistentPrefetchState"] == 0
            assert machine["FullWaits"] == int(active)
            assert phases() == phase * int(active)
            # The completion helper is idempotent after consuming its flag.
            assert machine.run(late) is None
            assert phases() == phase * int(active)
            assert machine.run(following) is None
            assert phases() == phase * (int(active) + 1)
            assert machine["PersistentPhantomTile"] == phantom


@pytest.mark.parametrize("key,value", [
    ("HalfPLR", 1), ("TileProcessingStrategy", "StreamK"), ("ScheduleIterAlg", 0),
    ("PrefetchGlobalRead", 1), ("NoTailLoop", False), ("ExpandPointerSwap", True),
    ("UnrollLoopSwapGlobalReadOrder", 1), ("DirectToVgprA", True),
    ("AdaptiveGemmNTAB", 1),
])
def test_unsupported_entry_keeps_deferral_disabled(monkeypatch, key, value):
    kernel, tensors, writer = _entry(monkeypatch)
    assert writer.papPgr2DeferredClusterReadyEnabled(kernel, tensors["A"], tensors["B"])
    kernel[key] = value
    assert not writer.papPgr2DeferredClusterReadyEnabled(kernel, tensors["A"], tensors["B"])
    assert not writer.papCompleteDeferredClusterReady(kernel, tensors["A"], tensors["B"]).flatitems()


@pytest.mark.parametrize("cluster,payload", [((1, 1), True), ((2, 4), False)])
def test_no_multicast_or_no_payload_has_no_pending_lifetime(monkeypatch, cluster, payload):
    kernel, tensors, writer = _entry(monkeypatch, cluster, payload)
    a, b = tensors["A"], tensors["B"]
    assert not writer.papPgr2DeferredClusterReadyEnabled(kernel, a, b)
    entry = writer.papPgr2EntryWait(kernel, a, b)
    machine = _PipelineMachine(PersistentPrefetchState=27, LoopCounterL=6)
    machine.registers[str(vgpr("Serial"))] = 0
    machine.run(entry)
    assert not machine["PersistentPrefetchState"] & 128
    assert machine["FullWaits"] == int(cluster != (1, 1))
    assert not writer.papCompleteDeferredClusterReady(kernel, a, b).flatitems()
    merged = Module()
    merged.add(writer.papPgr2EntryMerge(kernel, a, b))
    machine.run(merged)
    assert machine["PersistentPrefetchState"] == 16
