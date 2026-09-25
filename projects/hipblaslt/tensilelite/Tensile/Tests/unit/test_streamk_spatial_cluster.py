# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute the clustered StreamK scheduler and its reduction-address boundaries."""

from collections import Counter, defaultdict

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SCBranchSCC1

from Tensile.Component import Component
from Tensile.Common.DataType import DataType
from Tensile.Components.ClusterTileMapping import ClusterTileMapping
from Tensile.Components.StreamK import StreamKTwoTileDPFirst
from Tensile.Components.WorkAssignment import StaticGrid
from Tensile.ExecutionPolicy import normalize_execution_policy
from Tensile.CustomKernels import validateCustomPersistentArgs
from _persistent_isa import Machine
from test_persistent_assignment_execution import _Writer, _kernel

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isa(_isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx942", wavesize=64)


class Writer(_Writer):
    def acquirePersistentConstSgpr(self, kernel, name):
        return name

    def releasePersistentConstSgpr(self, name):
        pass

    def localReadResetOffsets(self, *args):
        return Module("no pointer reset in scalar model")

    def longBranchScc1(self, label, posNeg):
        return SCBranchSCC1(labelName=label.getLabelName())

    def shiftSrd(self, tc):
        return Module("no target SRD encoding in scalar model")


def _setup(monkeypatch, shape):
    kernel = _kernel()
    kernel.update(TileProcessingStrategy="StreamK", StreamKClusterMulticast=True,
                  ClusterDim=list(shape), StreamKAtomic=0)
    kernel["ProblemType"].update(ComputeDataType=DataType("s"), MXBlockA=0, MXBlockB=0)
    processing, assignment = StreamKTwoTileDPFirst(), StaticGrid()
    writer = Writer(kernel, processing.tileWork(kernel))
    writer.states.asmCaps = {"HasClusterBarrier": True}
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda _: processing)
    monkeypatch.setattr(Component.XCCMapping, "find", lambda _: lambda *args: Module("identity"))
    return kernel, processing, assignment, writer


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("tiles_m,tiles_n,batches,grid,sk_blocks,iters", [
    (1, 1, 1, 3, 1, 5),          # several clusters split one ragged block
    (5, 3, 2, 7, None, 5),       # global first-E ranges cross block boundaries
    (9, 5, 2, 3, 2, 7),          # DP prefix followed by split blocks
    (3, 3, 2, 17, 0, 1),         # whole-block fallback, idle clusters, K=0 ABI
])
def test_emitted_cluster_ranges_cover_real_tile_k_once(monkeypatch, shape, tiles_m, tiles_n,
                                                       batches, grid, sk_blocks, iters):
    kernel, processing, assignment, writer = _setup(monkeypatch, shape)
    cs, cn = shape
    peers = cs * cn
    blocks_m, blocks_n = (tiles_m + cs - 1) // cs, (tiles_n + cn - 1) // cn
    blocks = blocks_m * blocks_n * batches
    sk_blocks = blocks if sk_blocks is None else min(sk_blocks, blocks)
    # A DP prefix is a whole number of logical-grid waves, as required by static split.
    if 0 < sk_blocks < blocks:
        sk_blocks = blocks - ((blocks - sk_blocks) // grid) * grid
    initialize = assignment.initialize(writer, kernel, processing)
    activate = assignment.activateReservedOrAcquire(writer, kernel, processing, {}, {})
    close = assignment.closeLoop(writer, kernel)
    slot = ClusterTileMapping.physicalSlot(kernel, "PhysicalSlot", sgpr("Producer"))
    skip = processing.skipPhantomTileStore(writer, kernel)
    covered = Counter()
    producers = defaultdict(list)
    sequences = {}
    for cluster in range(grid):
        for peer in range(peers):
            machine = Machine(WorkGroup0=cluster * cs + peer % cs, WorkGroup1=peer // cs,
                              NumWorkGroups0=tiles_m, NumWorkGroups1=tiles_n,
                              ItersPerTile=iters, SKItersPerWG=sk_blocks * iters // grid,
                              skGrid=grid, skTiles=sk_blocks, Alpha=1)
            machine["SizesFree+2"] = batches
            machine.registers[str(sgpr("AddressFlags", 2))] = 1
            shift = iters.bit_length() - 1
            machine["MagicNumberItersPerTile"] = ((1 << (32 + shift)) + iters - 1) // iters
            if iters & (iters - 1) == 0:
                machine["MagicNumberItersPerTile"] = 0x80000000
                shift -= 1
            if iters == 1:
                machine["MagicNumberItersPerTile"] = 0
                shift = 0x80000000
            machine["MagicShiftItersPerTile"] = shift
            status = machine.run(initialize)
            assert machine["StreamKClusterPeer"] == peer
            assert machine["PersistentWorkGroupIndex"] == cluster
            if status == "label_KernelEnd":
                assert machine.barriers == 0, "idle clusters must not issue an unmatched arrive"
                sequences[cluster, peer] = []
                continue
            seq = []
            for _ in range(blocks + 2):
                start_iter = machine["PersistentIteration"]
                assert machine.run(activate) is None
                start, end = machine["StreamKLocalStart"], machine["StreamKLocalEnd"]
                block = start_iter // iters
                seq.append((block, start, end))
                batch, block_in_batch = divmod(block, blocks_m * blocks_n)
                bn, bm = divmod(block_in_batch, blocks_m)
                m, n = bm * cs + peer % cs, bn * cn + peer // cs
                phantom = m >= tiles_m or n >= tiles_n
                assert (machine["WorkGroup0"], machine["WorkGroup1"], machine["WorkGroup2"]) == (
                    min(m, tiles_m - 1), min(n, tiles_n - 1), batch)
                assert bool(machine["PersistentPhantomTile"]) == phantom
                assert (machine.run(skip) == "label_PersistentLoopClose") == phantom
                if not phantom:
                    for k in range(start, end):
                        covered[batch, m, n, k] += 1
                    machine["Producer"] = cluster
                    machine.run(slot)
                    assert machine["PhysicalSlot"] == cluster * peers + peer
                    producers[batch, m, n].append((cluster, peer, start, end, machine["PhysicalSlot"]))
                if machine.run(close) is None:
                    break
            else:
                pytest.fail("emitted schedule did not terminate")
            sequences[cluster, peer] = seq
    assert covered == Counter({(b, m, n, k): 1 for b in range(batches) for m in range(tiles_m)
                               for n in range(tiles_n) for k in range(iters)})
    for cluster in range(grid):
        assert all(sequences[cluster, peer] == sequences[cluster, 0] for peer in range(peers))
    # Build an independent tree for every real output. Each edge resolves an
    # emitted physical slot to an actual producer of that same tile and peer;
    # no phantom slot can become a dependency or a publication.
    for contributions in producers.values():
        contributions.sort()
        slots = {entry[4]: entry for entry in contributions}
        for i, (cluster, peer, _, _, _) in enumerate(contributions):
            stride = 1
            while i % (2 * stride) == 0 and i + stride < len(contributions):
                machine["Producer"] = cluster + stride
                machine["StreamKClusterPeer"] = peer
                machine.run(slot)
                assert machine["PhysicalSlot"] in slots
                assert slots[machine["PhysicalSlot"]][1] == peer
                stride *= 2


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
def test_workspace_and_flags_share_the_emitted_physical_slot(monkeypatch, shape):
    kernel, processing, _, writer = _setup(monkeypatch, shape)
    kernel.update(MacroTile0=1024, MacroTile1=1024, BufferStore=True)
    writer.states.bpeCinternal = 4
    workspace = processing.computeWorkspaceSrd(writer, kernel, sgpr("Producer"))
    flags = processing.flagOffset(kernel, "FlagOffset", sgpr("Producer"))
    for producer in (0, 1, 31, 255):
        for peer in range(shape[0] * shape[1]):
            machine = Machine(Producer=producer, StreamKClusterPeer=peer)
            machine.registers["BufferOOB"] = 0
            machine.registers["Srd127_96"] = 0
            machine.run(workspace)
            machine.run(flags)
            slot = producer * shape[0] * shape[1] + peer
            offset = machine["SrdWS+0"] + (machine["SrdWS+1"] << 32)
            assert offset == slot * 1024 * 1024 * 4
            assert machine["FlagOffset"] == slot * 4
            assert machine["Producer"] == producer


def test_abi2_is_explicit_and_custom_descriptors_are_rejected():
    state = normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                        "StreamKClusterMulticast": True})
    assert state["InternalSupportParams"]["PersistentLoopArgsVersion"] == 2
    with pytest.raises(ValueError, match="requires StreamKClusterMulticast"):
        normalize_execution_policy({"TileProcessingStrategy": "StreamK",
                                    "InternalSupportParams": {"PersistentLoopArgsVersion": 2}}, regenerate=False)
    with pytest.raises(ValueError, match="Custom kernels"):
        validateCustomPersistentArgs(state)
    for strategy, version in (("DataParallel", 1), ("StreamK", 0)):
        assert normalize_execution_policy({"TileProcessingStrategy": strategy})["InternalSupportParams"]["PersistentLoopArgsVersion"] == version


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
@pytest.mark.parametrize("blocks,grid,iters", [(1, 7, 19), (3, 7, 5), (7, 11, 9), (4, 3, 7)])
def test_emitted_tree_reduction_resolves_only_matching_peer_producers(monkeypatch, shape, blocks, grid, iters):
    from types import SimpleNamespace
    from rocisa.code import Label
    from rocisa.instruction import SMovB32

    kernel, processing, _, writer = _setup(monkeypatch, shape)
    kernel.update(StreamKFixupTreeReduction=1, DebugStreamK=0)

    def read_flag(writer, dst, soffset):
        module = Module("observed flag read")
        module.add(SMovB32(sgpr("PolledFlagOffset"), soffset))
        module.add(SMovB32(sgpr(dst), 1))
        return module

    def reset_flag(writer, src, soffset, comment=""):
        module = Module("observed flag reset")
        module.add(SMovB32(sgpr("ResetFlagOffset"), soffset))
        return module

    ordering = SimpleNamespace(readFlag=read_flag, acquireFence=lambda _: Module("acquire fence"))
    monkeypatch.setattr(Component.StreamKMemoryOrdering, "find", lambda _: ordering)
    monkeypatch.setattr(processing, "emitFlagStore", reset_flag)
    def observe_fixup(writer, kernel, widths, elements, edges, tmp, cvt, producer):
        module = ClusterTileMapping.physicalSlot(kernel, "PhysicalFixupSlot", sgpr(producer))
        module.add(SMovB32(sgpr("FixupSlot"), sgpr("PhysicalFixupSlot")))
        return module

    monkeypatch.setattr(processing, "fixupStep", observe_fixup)
    fixup = processing.storeBranchesCommon(writer, kernel, Label("PublishPartial", ""), [], [], 0, None)
    peers = shape[0] * shape[1]
    per_worker, extra = divmod(blocks * iters, grid)
    publication_counts = Counter()
    for block in range(blocks):
        contributions = {}
        for cluster in range(grid):
            start = cluster * per_worker + min(cluster, extra)
            end = start + per_worker + (cluster < extra)
            first, last = max(start, block * iters), min(end, (block + 1) * iters)
            if first < last:
                contributions[cluster] = set(range(first - block * iters, last - block * iters))
        for peer in range(peers):
            dependencies = {}
            publishers = set()
            for cluster, intervals in contributions.items():
                machine = Machine(NumWorkGroups0=blocks * shape[0], NumWorkGroups1=shape[1],
                                  ItersPerTile=iters, skTiles=blocks, skGrid=grid,
                                  SKItersPerWG=per_worker, StreamKClusterPeer=peer,
                                  PersistentWorkGroupIndex=cluster,
                                  PersistentIteration=(block + 1) * iters,
                                  StreamKLocalStart=min(intervals), StreamKLocalEnd=max(intervals) + 1)
                machine["SizesFree+2"] = 1
                result = machine.run(fixup)
                polls = [value for name, value in machine.writes if name == str(sgpr("PolledFlagOffset"))]
                resets = [value for name, value in machine.writes if name == str(sgpr("ResetFlagOffset"))]
                slots = [value for name, value in machine.writes if name == str(sgpr("FixupSlot"))]
                assert polls == resets == [slot * 4 for slot in slots]
                dependencies[cluster] = [slot // peers for slot in slots]
                for slot in slots:
                    assert slot % peers == peer
                    assert slot // peers in contributions
                    assert slot // peers > cluster, "tree dependency cycle"
                if result == "label_PublishPartial":
                    publishers.add(cluster)
                    publication_counts[cluster, peer] += 1
                else:
                    assert result is None
                    assert cluster == min(contributions)
            # Execute publications and reads in reverse topological order.
            # A missing producer or incorrect peer slot fails before a root
            # can claim a complete output. Slots cannot publish twice in a launch.
            reduced = {}
            for cluster in sorted(contributions, reverse=True):
                owned = set(contributions[cluster])
                for partner in dependencies[cluster]:
                    assert partner in publishers
                    assert owned.isdisjoint(reduced[partner])
                    owned.update(reduced[partner])
                reduced[cluster] = owned
            assert reduced[min(contributions)] == set(range(iters))
    assert max(publication_counts.values(), default=0) <= 1


@pytest.mark.parametrize("shadow", [False, True])
@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_prefetch_skip_consumes_cluster_arrive_exactly_once(monkeypatch, shadow, iterations):
    from rocisa.instruction import SCmpEQU32
    from Tensile.KernelWriterAssembly import KernelWriterAssembly

    kernel, _, assignment, writer = _setup(monkeypatch, (2, 2))
    kernel.update(EnableMatrixInstruction=False, StorePriorityOpt=False, SuppressNoLoadLoop=False)
    writer.states.doShadowInit = int(shadow)
    writer.isPrefetchAcrossPersistentEnabled = lambda _: False
    writer.checkLastIter = lambda _: SCmpEQU32(sgpr("LoopCounter"), 0)
    writer.longBranchScc1 = lambda label, **kwargs: SCBranchSCC1(labelName=label.getLabelName())
    writer.rapLabel = lambda name: name
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: assignment)
    module = KernelWriterAssembly.openSumAtLeastUnroll(writer, kernel, prefetch=True, isOptNLL=False)
    machine = Machine(LoopCounter=iterations)
    result = machine.run(module)
    assert machine.scc == (iterations == 0), "cluster wait must preserve branch predicate"
    if iterations == 0:
        assert machine.barriers == 1
        assert result == ("label_ShadowInitStart" if shadow else "label_PrefetchGlobalLastIterEnd")
    else:
        assert machine.barriers == 0, "first TDM load owns the wait on the compute path"
        assert result is None


@pytest.mark.parametrize("alpha,local_start,expected_waits", [(0, 0, 0), (0, 2, 1), (1, 0, 0), (1, 2, 0)])
def test_alpha_zero_nonowner_consumes_arrive_before_skipping_compute(monkeypatch, alpha, local_start, expected_waits):
    kernel, processing, _, writer = _setup(monkeypatch, (2, 2))
    scratch = writer.sgprPool.checkOutAligned(4, 2, "mapping")
    module = processing.finishStaticTile(writer, kernel, {}, {}, scratch)
    machine = Machine(NumWorkGroups0=2, NumWorkGroups1=2, Alpha=alpha,
                      StreamKLocalStart=local_start, StreamKLocalEnd=3, ItersPerTile=5)
    result = machine.run(module)
    assert machine.barriers == expected_waits
    assert (result == "label_PersistentLoopClose") == bool(expected_waits)
    if alpha == 0 and local_start == 0:
        assert machine["StreamKLocalEnd"] == 5


@pytest.mark.parametrize("shape", [(2, 1), (4, 1), (2, 2), (2, 4)])
def test_supported_shapes_match_host_capability(shape):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK", StreamKClusterMulticast=True,
                      StreamKFixupTreeReduction=1, ClusterDim=list(shape))
    assert _validateStreamKMulticast(state, False, _isa_map())


@pytest.mark.parametrize("overrides", [
    {"ClusterDim": [8, 1]}, {"ClusterDim": [4, 2]}, {"ClusterDim": [4, 4]},
    {"ClusterDim": [1, 2]}, {"PrefetchAcrossPersistent": 1}, {"PrefetchGlobalRead": 0},
    {"StreamKFixupTreeReduction": 0}, {"StreamKAtomic": 1}, {"UseSubtileImpl": 1},
    {"StoreRemapVectorWidth": 4}, {"ReuseAcrossPersistent": 1}, {"SpaceFillingAlgo": [1]},
    {"TDMInst": 2}, {"ProblemType": {"Sparse": 1}}, {"DebugStreamK": 1},
    {"ProblemType": {"OutputAmaxD": True}},
    {"ProblemType": {"Gradient": True, "UseBias": True}},
])
def test_unsupported_cluster_capabilities_fail_explicitly(overrides):
    from Tensile.SolutionStructs.Solution import _validateStreamKMulticast
    from test_streamk_multicast import _mc_state, _isa_map
    state = _mc_state(TileProcessingStrategy="StreamK", StreamKClusterMulticast=True,
                      StreamKFixupTreeReduction=1)
    state.update(overrides)
    assert not _validateStreamKMulticast(state, False, _isa_map())
