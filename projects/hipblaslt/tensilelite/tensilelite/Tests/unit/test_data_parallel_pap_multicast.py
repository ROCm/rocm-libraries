# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute PAP's cluster handoff and fallback control flow on each wave."""

import pytest
from rocisa.code import Module
from rocisa.container import vgpr

from Tensile.Component import Component
from Tensile.Components.PersistentLoop import PersistentLoopOn
from Tensile.Components.TileProcessingStrategy import DataParallel
from Tensile.Components.WorkAssignment import StaticGrid, RESERVED_MASK
from _pap_isa import Machine
from test_persistent_assignment_execution import _Writer, _kernel

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isa(_isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx1250", wavesize=32)


def _setup(monkeypatch, shape=(2, 2)):
    kernel = _kernel()
    kernel.update(ClusterDim=list(shape), PrefetchAcrossPersistent=1, WavefrontSize=32)
    processing, assignment = DataParallel(), StaticGrid()
    writer = _Writer(kernel, processing.tileWork(kernel), ())
    writer.states.asmCaps = {"HasClusterBarrier": True}
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda _: processing)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: assignment)
    monkeypatch.setattr(processing, "prefetchAcrossPersistentSetupNextTile",
                        lambda *args, **kwargs: Module("identity setup irrelevant to barriers"))
    return kernel, assignment, writer


@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("shape", [(2, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("skip_barrier", [False, True])
def test_pap_handoff_completes_one_cluster_phase_before_issue(monkeypatch, wave, shape, skip_barrier):
    kernel, assignment, writer = _setup(monkeypatch, shape)
    prefetch = PersistentLoopOn().prefetch(writer, kernel, {}, {}, skipBarrier=skip_barrier)
    close = assignment.persistentClusterNextTileArrive(writer, kernel)
    machine = Machine(NextTile=8, TotalTiles=32)
    machine.registers[str(vgpr("Serial"))] = wave * 32
    machine.run(prefetch)
    assert machine["Loads"] == 1
    expected = [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")]
    if wave == 0:
        expected.append(("s_barrier_signal", "-3"))
    expected.append(("s_barrier_wait", "-3"))
    assert machine.barrier_operations == expected
    # Re-entering PAP or closing the loop must not signal another phase: the
    # next iteration takes the primed branch around its first-load wait.
    machine.run(prefetch)
    machine.run(close)
    assert machine["Loads"] == 1
    assert machine.barrier_operations == expected
    text = str(prefetch)
    assert text.index("s_wait_dscnt") < text.index("s_barrier_signal -1")
    assert text.index("s_barrier_wait -3") < text.index("s[sgprLoads]")


@pytest.mark.parametrize("state", [0, RESERVED_MASK, 1, 3])
@pytest.mark.parametrize("has_next", [False, True])
@pytest.mark.parametrize("wave", range(4))
def test_loop_close_arrives_only_for_an_unprimed_next_tile(monkeypatch, state, has_next, wave):
    kernel, assignment, writer = _setup(monkeypatch)
    close = assignment.persistentClusterNextTileArrive(writer, kernel)
    machine = Machine(NextTile=8, TotalTiles=32 if has_next else 8, PersistentPrefetchState=state)
    machine.registers[str(vgpr("Serial"))] = wave * 32
    machine.run(close)
    expected = []
    if has_next and not state & 1:
        expected = [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")]
        if wave == 0:
            expected.append(("s_barrier_signal", "-3"))
    assert machine.barrier_operations == expected


def test_last_tile_skips_pap_without_a_cluster_phase(monkeypatch):
    kernel, assignment, writer = _setup(monkeypatch)
    machine = Machine(NextTile=32, TotalTiles=32)
    machine.run(PersistentLoopOn().prefetch(writer, kernel, {}, {}))
    machine.run(assignment.persistentClusterNextTileArrive(writer, kernel))
    assert machine["Loads"] == 0
    assert machine.barrier_operations == []


def test_noncluster_pap_keeps_its_local_handoff(monkeypatch):
    kernel, assignment, writer = _setup(monkeypatch, (1, 1))
    machine = Machine(NextTile=1, TotalTiles=2)
    machine.run(PersistentLoopOn().prefetch(writer, kernel, {}, {}))
    machine.run(assignment.persistentClusterNextTileArrive(writer, kernel))
    assert machine["Loads"] == 1
    assert machine.barrier_operations == [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")]


@pytest.mark.parametrize("current_phantom,next_phantom", [(0, 1), (1, 0)])
def test_pap_restores_the_current_peers_store_eligibility(monkeypatch, current_phantom, next_phantom):
    from rocisa.container import sgpr
    from rocisa.instruction import SMovB32
    kernel, _, writer = _setup(monkeypatch)
    processing = Component.TileProcessingStrategy.find(writer)
    setup = Module("next block may change this peer's store eligibility")
    setup.add(SMovB32(sgpr("PersistentPhantomTile"), next_phantom))
    monkeypatch.setattr(processing, "prefetchAcrossPersistentSetupNextTile", lambda *args, **kwargs: setup)
    machine = Machine(NextTile=8, TotalTiles=32, PersistentPhantomTile=current_phantom)
    machine.run(PersistentLoopOn().prefetch(writer, kernel, {}, {}))
    assert machine["Loads"] == 1
    assert machine["PersistentPhantomTile"] == current_phantom
    result = machine.run(processing.skipPhantomTileStore(writer, kernel))
    assert (result == "label_PersistentLoopClose") == bool(current_phantom)


@pytest.mark.parametrize("state", [0, 1, 2, 4, 5, 7, 13, 29, 0x80000004, 0x80000005])
def test_coordinate_handoff_publishes_only_prepared_issued_tiles(monkeypatch, state):
    _, _, writer = _setup(monkeypatch)
    writer.states.papNextTileVgpr = 12
    machine = Machine(PersistentPrefetchState=state, WorkGroup0=17, WorkGroup1=23,
                      WorkGroup2=2, PersistentPhantomTile=1, NextTile=99)
    machine.run(writer.papPublishNextTileCoordinates())
    assert bool(machine["PersistentPrefetchState"] & 32) == (state & 5 == 5)
    assert machine["PersistentPrefetchState"] & ~32 == state
    assert [machine[name] for name in ("WorkGroup0", "WorkGroup1", "WorkGroup2",
                                     "PersistentPhantomTile", "NextTile")] == [17, 23, 2, 1, 99]
