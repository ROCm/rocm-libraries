# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute the real two-group guards and compiler-token joins on the CPU."""

import pytest
from rocisa.code import Module
from rocisa.container import MemTokenData, sgpr, vgpr
from rocisa.instruction import SAddU32, SBarrier, SMovB32, SOrB32
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from Tensile.Component import Component
from Tensile.Components.PersistentLoop import PersistentLoopOn
from Tensile.Components.TileProcessingStrategy import DataParallel
from Tensile.Components.WorkAssignment import StaticGrid
from test_PapAddressPipeline import _writer, _MapperMachine
from test_PapShadowAddress import _components

pytestmark = pytest.mark.unit


class _PipelineMachine(_MapperMachine):
    def run(self, module):
        # This is an emitter scheduling boundary with no GPU instruction.
        lines = [line for line in str(module).splitlines()
                 if line.split("//", 1)[0].strip() != "scheduling_fence"]
        return super().run("\n".join(lines))

def _setup(scales=True, cluster=(1, 1)):
    kernel, tensors, writer = _writer(scales=scales, cluster=cluster)
    kernel.update(NoTailLoop=True, PrefetchLocalRead=1, LoopIters=2,
                  InnerUnroll=1, ForceUnrollSubIter=False,
                  NoLdsWriteCode=True)
    kernel["1LDSBuffer"] = False
    writer.states.numItersPLR = 1
    writer.states.doFullPackCodePrefetch = False
    writer.states.memTokenLdsBuffer0 = 0
    writer.states.ldsTensorTokenIdx = 1
    writer._nextLdsToken = lambda token: token ^ 1
    writer.sgprPool = RegisterPool(128, RegisterType.Sgpr, defaultPreventOverflow=False)
    writer.sgprPool.add(120, 8, "cluster election scratch")
    return kernel, tensors, writer


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("key,value", [("NoTailLoop", False), ("PrefetchLocalRead", 0),
                                      ("ForceUnrollSubIter", True), ("1LDSBuffer", True),
                                      ("ScheduleIterAlg", 0), ("PrefetchGlobalRead", 1),
                                      ("HalfPLR", True), ("TDMSplit", True)])
def test_unsupported_geometry_keeps_old_pipeline(scales, key, value):
    kernel, tensors, writer = _setup(scales)
    assert writer.papPgr2PipelineEnabled(kernel, tensors["A"], tensors["B"])
    kernel[key] = value
    assert not writer.papPgr2PipelineEnabled(kernel, tensors["A"], tensors["B"])


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("previous", [0, 1])
def test_alternative_emit_sites_do_not_advance_surrounding_tensor_token(scales, group, previous):
    kernel, tensors, writer = _setup(scales)
    writer.states.ldsTensorTokenIdx = previous
    seen = []

    def emit(kernel, mode, tensor):
        seen.append((tensor["tensorChar"], writer.states.ldsTensorTokenIdx))
        return Module()

    writer.globalReadDo = emit
    writer.papPgr2IssueGroup(kernel, tensors["A"], tensors["B"], group)
    assert seen == [(tc, group) for tc in (["A", "MXSA", "MXSB", "B"] if scales else ["A", "B"])]
    assert writer.states.ldsTensorTokenIdx == previous

    def failed_emit(*args):
        raise RuntimeError("emission failure")

    writer.globalReadDo = failed_emit
    with pytest.raises(RuntimeError, match="emission failure"):
        writer.papPgr2IssueGroup(kernel, tensors["A"], tensors["B"], group)
    assert writer.states.ldsTensorTokenIdx == previous


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize("prepared,has_next", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("bank", [0, 2])
def test_two_group_guards_publish_once_and_preserve_identity(monkeypatch, q, prepared, has_next, bank):
    kernel, tensors, writer = _setup()
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda _: DataParallel())
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: StaticGrid())

    def observed(name):
        module = Module()
        module.add(SAddU32(sgpr(name), sgpr(name), 1))
        return module

    writer.papTdmUpdateDescriptor = lambda *args, **kwargs: observed("DescriptorUpdates")
    writer.papPgr2AdvanceDescriptor = lambda *args: observed("DepthUIncrements")
    writer.tdmSwapLdsOffset = lambda *args: observed("DescriptorBankSwaps")
    writer.papPgr2IssueGroup = lambda kernel, a, b, group: observed("Group%uIssues" % group)
    writer.papTdmSaveLdsBank = lambda *args: SOrB32(sgpr("PersistentPrefetchState"),
                                                 sgpr("PersistentPrefetchState"), sgpr("Bank"))
    pipeline = Module()
    for group in (0, 1, 0, 1):
        pipeline.add(PersistentLoopOn().prefetchPgr2Group(writer, kernel, tensors["A"], tensors["B"], group))
    machine = _PipelineMachine(NextTile=1, TotalTiles=2 if has_next else 1,
                             OrigLoopCounter=q, Bank=bank,
                             PersistentPrefetchState=0x80000004 if prepared else 0,
                             WorkGroup0=31, WorkGroup1=32, WorkGroup2=33)
    next_coords = (41, 42, 43)
    for i, value in enumerate(next_coords):
        machine.registers[str(sgpr(writer.states.papAddressCarrier["coordinates"][i]))] = value
    assert machine.run(pipeline) is None
    active = prepared and has_next and q >= 2
    assert machine["Group0Issues"] == machine["Group1Issues"] == int(active)
    assert machine["DepthUIncrements"] == int(active)
    assert machine["DescriptorBankSwaps"] == 2 * int(active)  # data and paired MX
    assert machine["DescriptorUpdates"] == 2 * int(active)
    assert tuple(machine["WorkGroup%u" % i] for i in range(3)) == (31, 32, 33)
    assert tuple(machine.registers[str(sgpr(writer.states.papAddressCarrier["coordinates"][i]))]
                 for i in range(3)) == next_coords
    if active:
        assert machine["PersistentPrefetchState"] == 0x8000000d | bank
        assert machine.barrier_operations == [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")] * 3
    else:
        assert not machine["PersistentPrefetchState"] & 9
        # The test emits the NGLL site twice. Its common current-bank role
        # transition runs even when next-tile eligibility fails; no issue or
        # second-bank handoff runs on that path.
        assert machine.barrier_operations == [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")] * 2


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("state", [0, 1, 3, 4, 5, 9, 11, 25, 27])
def test_entry_wait_leaves_only_actual_younger_group(scales, state):
    kernel, tensors, writer = _setup(scales)
    writer._wait = lambda kernel, a, b, count, *args: SMovB32(sgpr("TensorWait"), count)
    writer.getWaitcntCodeForPGR = lambda *args: SMovB32(sgpr("TensorWait"), 0)
    machine = _PipelineMachine(PersistentPrefetchState=state)
    machine.run(writer.papPgr2EntryWait(kernel, tensors["A"], tensors["B"]))
    assert machine["TensorWait"] == ((2 if scales else 1) if state & 8 else 0)


@pytest.mark.parametrize("q", [2, 3, 4, 6])
@pytest.mark.parametrize("low", [0x1000, 0xfffffff0])
def test_group1_is_positive_depthu_even_at_current_wrap_counter(q, low):
    kernel, tensors, writer = _setup()
    machine = _PipelineMachine(LoopCounterL=2, StaggerUIter=2)
    for descriptor, increment, step in (("tdmAGroup0", "tdmABIncs", 256),
                                        ("tdmMXSAGroup0", "tdmMXSAMXSBIncs", 8)):
        machine[descriptor + "+2"] = low
        machine[descriptor + "+3"] = 0x80000012
        machine[increment] = step
    # These are the current-tile values the old generic helper selected.
    machine["WrapUA+0"] = ((1 - q) * 256) & 0xffffffff
    machine["WrapUMXSA+0"] = ((1 - q) * 8) & 0xffffffff
    module = writer.papPgr2AdvanceDescriptor(kernel)
    machine.run(module)
    for descriptor, step in (("tdmAGroup0", 256), ("tdmMXSAGroup0", 8)):
        actual = machine[descriptor + "+2"] | machine[descriptor + "+3"] << 32
        assert actual == (0x80000012 << 32 | low) + step


@pytest.mark.parametrize("feature,value", [("UseBias", 1), ("UseScaleAlphaVec", 1),
                                           ("UseScaleAB", "Vector"), ("OutputAmaxD", True)])
def test_unreviewed_or_lds_epilogue_keeps_two_group_pipeline_off(feature, value):
    kernel, tensors, writer = _setup()
    kernel["ProblemType"][feature] = value
    assert not writer.papPgr2PipelineEnabled(kernel, tensors["A"], tensors["B"])


def test_split_u_epilogue_keeps_two_group_pipeline_off():
    kernel, tensors, writer = _setup()
    kernel["LocalSplitU"] = 2
    assert not writer.papPgr2PipelineEnabled(kernel, tensors["A"], tensors["B"])


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("shape", [(2, 1), (2, 4)])
@pytest.mark.parametrize("q,prepared,has_next", [
    (0, True, True), (1, True, True), (2, False, True),
    (2, True, False), (2, True, True), (3, True, True), (4, True, True), (6, True, True),
])
def test_multicast_groups_release_and_publish_four_balanced_phases(
        monkeypatch, scales, shape, q, prepared, has_next):
    kernel, tensors, writer = _setup(scales, shape)
    assignment = StaticGrid()
    monkeypatch.setattr(Component.TileProcessingStrategy, "find", lambda _: DataParallel())
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: assignment)
    assert writer.papPgr2PipelineEnabled(kernel, tensors["A"], tensors["B"])
    writer.states.asmCaps["HasClusterBarrier"] = True

    def observed(name):
        return SAddU32(sgpr(name), sgpr(name), 1)

    def ready_join(kernel, comment, memoryToken):
        barrier = SBarrier(comment=comment)
        barrier.setMemToken(MemTokenData(memoryToken))
        return barrier

    writer._wait = lambda kernel, a, b, count, *args: SMovB32(sgpr("TensorWait"), count)
    writer._syncThreads = ready_join
    writer.getWaitcntCodeForPGR = lambda *args: SMovB32(sgpr("TensorWait"), 0)
    writer.papTdmUpdateDescriptor = lambda *args, **kwargs: observed("DescriptorUpdates")
    writer.papPgr2AdvanceDescriptor = lambda *args: observed("DepthUIncrements")
    writer.tdmSwapLdsOffset = lambda *args: observed("DescriptorBankSwaps")
    writer.papPgr2IssueGroup = lambda kernel, a, b, group: observed("Group%uIssues" % group)
    writer.papTdmSaveLdsBank = lambda *args: SOrB32(
        sgpr("PersistentPrefetchState"), sgpr("PersistentPrefetchState"), sgpr("Bank"))
    issue = Module()
    for group in (0, 1):
        issue.add(PersistentLoopOn().prefetchPgr2Group(
            writer, kernel, tensors["A"], tensors["B"], group))
    # No retained payload simulates the VGPR fallback. Group0 still needs its
    # remote-producer phase, and group1 is published only after store/setup.
    ready0 = writer.papPrefetchLocalRead(kernel, tensors["A"], tensors["B"])
    ready1 = writer.papPgr2EntryWait(kernel, tensors["A"], tensors["B"])
    close = assignment.persistentClusterNextTileArrive(writer, kernel)
    active = prepared and has_next and q >= 2
    for wave in range(4):
        for phantom in (0, 1):
            for bank in (0, 2):
                machine = _PipelineMachine(
                    NextTile=8 + (shape[0] * shape[1] - 1 if phantom else 0),
                    TotalTiles=32 if has_next else 8,
                    OrigLoopCounter=q, Bank=bank, PersistentPhantomTile=phantom,
                    PersistentPrefetchState=0x80000004 if prepared else 0,
                    WorkGroup0=31, WorkGroup1=32, WorkGroup2=33)
                machine.registers[str(vgpr("Serial"))] = wave * 32
                for i, value in enumerate((41, 42, 43)):
                    machine.registers[str(sgpr(writer.states.papAddressCarrier["coordinates"][i]))] = value
                assert machine.run(issue) is None
                assert machine["Group0Issues"] == machine["Group1Issues"] == int(active)
                assert tuple(machine["WorkGroup%u" % i] for i in range(3)) == (31, 32, 33)
                assert machine["PersistentPhantomTile"] == phantom
                assert machine.run(ready0) is None
                assert machine["TensorWait"] == ((2 if scales else 1) if active else 0)
                assert machine.run(ready1) is None
                waits = [value for register, value in machine.writes
                         if register == str(sgpr("TensorWait"))]
                younger = 2 if scales else 1
                assert waits == ([younger, 0, younger] if active else [0])
                phases = [op for op in machine.barrier_operations if op[1] == "-3"]
                expected_phase = ([("s_barrier_signal", "-3")] if wave == 0 else [])
                expected_phase += [("s_barrier_wait", "-3")]
                assert phases == expected_phase * (4 if active else 0)
                if active:
                    assert machine["PersistentPrefetchState"] == 0x8000000d | bank
                    before = list(machine.barrier_operations)
                    machine.run(close)
                    assert machine.barrier_operations == before
                    machine.run(issue)
                    assert machine["Group0Issues"] == machine["Group1Issues"] == 1
                    assert [op for op in machine.barrier_operations if op[1] == "-3"] == phases


def test_multicast_ready_tokens_separate_local_completion_from_cluster_group(monkeypatch):
    kernel, tensors, writer = _setup(cluster=(2, 4))
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: StaticGrid())
    for group in (0, 1):
        phase = writer.papPgr2ClusterPhase(kernel, group, "ready")
        barriers = [item for item in phase.flatitems() if isinstance(item, SBarrier)]
        assert len(barriers) == 3
        assert [list(item.getMemToken().tokens) for item in barriers] == (
            [[0, 1], [1], [1]] if group else [[0], [0], [0]])


@pytest.mark.parametrize("group", [0, 1])
def test_multicast_publication_keeps_producer_join_through_python_barrier_pass(monkeypatch, group):
    from Tensile.KernelWriterModules import syncThreads
    from test_ClusterBarrierPairing import _runPass, _waveParityKernel
    kernel, tensors, writer = _setup(cluster=(2, 4))
    kernel.update(NumThreads=128, _ScheduleIterAlg=0)
    monkeypatch.setattr(Component.WorkAssignment, "find", lambda _: StaticGrid())
    writer._wait = lambda kernel, a, b, count, *args: SMovB32(sgpr("TensorWait"), count)
    writer._syncThreads = lambda kernel, comment, memoryToken: syncThreads(
        kernel, {}, {"SeparateVscnt": False}, comment, memoryToken=memoryToken)
    readiness = writer.papPgr2GroupReady(kernel, tensors["A"], tensors["B"], group)
    checked = _runPass(_waveParityKernel(), readiness)
    assert checked.states.overflowedResources == 0
    barriers = [item for item in readiness.flatitems() if isinstance(item, SBarrier)]
    assert len(barriers) == 3
    assert [list(item.getMemToken().tokens) for item in barriers] == (
        [[0, 1], [1], [1]] if group else [[0], [0], [0]])
    for wave in range(4):
        machine = _PipelineMachine()
        machine.registers[str(vgpr("Serial"))] = wave * 32
        assert machine.run(readiness) is None
        assert machine["TensorWait"] == (2 if group == 0 else 0)
        expected = [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")]
        if wave == 0:
            expected.append(("s_barrier_signal", "-3"))
        expected.append(("s_barrier_wait", "-3"))
        assert machine.barrier_operations == expected


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("token", [0, 1])
def test_rotating_main_loop_readiness_survives_barrier_reconstruction(scales, token):
    from test_ClusterBarrierPairing import _fill, _read, _runPass, _waveParityKernel
    kernel, tensors, writer = _setup(scales)
    writer.states.unrollLoopCopies = 1
    writer.states.ldsReadTokenIdx = token
    readiness = writer.papPgr2MainLoopReadiness(kernel, tensors["A"], tensors["B"])
    fill, read = _fill(), _read()
    for instruction in (fill, read):
        instruction.setMemToken(MemTokenData([token]))
    module = Module("kernelBody")
    module.add(fill)
    module.add(readiness)
    module.add(read)
    checked = _runPass(_waveParityKernel(), module)
    assert checked.states.overflowedResources == 0
    barriers = [item for item in module.flatitems() if isinstance(item, SBarrier)]
    assert len(barriers) == 1  # the contract owns this join; no duplicate rebuild
    assert barriers[0].tensorcnt == (2 if scales else 1)
    assert list(barriers[0].getMemToken().tokens) == [token]
    assert writer.states.ldsReadTokenIdx == token


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("copies,pap", [(2, 1), (3, 1), (1, 0)])
def test_rotating_main_loop_contract_leaves_other_pipelines_unchanged(scales, copies, pap):
    kernel, tensors, writer = _setup(scales)
    kernel["PrefetchAcrossPersistent"] = pap
    writer.states.unrollLoopCopies = copies
    writer.states.ldsReadTokenIdx = 1
    assert writer.papPgr2MainLoopReadiness(kernel, tensors["A"], tensors["B"]) is None


def test_rotating_readiness_does_not_release_another_pending_bank():
    from test_ClusterBarrierPairing import _fill, _read, _runPass, _waveParityKernel
    kernel, tensors, writer = _setup(False)
    writer.states.unrollLoopCopies = 1
    writer.states.ldsReadTokenIdx = 1
    module = Module("kernelBody")
    for token in (0, 1):
        fill = _fill()
        fill.setMemToken(MemTokenData([token]))
        module.add(fill)
    module.add(writer.papPgr2MainLoopReadiness(kernel, tensors["A"], tensors["B"]))
    for token in (1, 0):
        read = _read()
        read.setMemToken(MemTokenData([token]))
        module.add(read)
    _runPass(_waveParityKernel(), module)
    barriers = [item for item in module.flatitems() if isinstance(item, SBarrier)]
    assert [(item.tensorcnt, list(item.getMemToken().tokens)) for item in barriers] == [
        (1, [1]), (-1, [0])]


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("contracted", [False, True])
def test_main_loop_schedule_issues_youngest_group_before_partial_readiness(scales, contracted):
    from types import SimpleNamespace
    from Tensile.KernelWriter import KernelWriter
    from rocisa.instruction import DSLoadInstruction, TensorLoadToLds
    from test_ClusterBarrierPairing import _fill, _read
    kernel, tensors, writer = _setup(scales)
    writer.states.unrollLoopCopies = 1
    writer.states.ldsReadTokenIdx = 1
    writer.states.scheduleIterAlg = 0
    producer = Module()
    for _ in range(2 if scales else 1):
        producer.add(_fill())
    reads = Module()
    reads.add(_read())
    sync = writer.papPgr2MainLoopReadiness(kernel, tensors["A"], tensors["B"]) if contracted else Module()
    if not contracted:
        sync.add(SBarrier())
    writer.codes = SimpleNamespace(perIterGlobalRead={1: producer},
                                   perIterLocalWrite={1: (0, Module())})
    result = KernelWriter._makeSubIterSchedule(
        writer, kernel, tensors["A"], tensors["B"], reads, 1,
        Module(), Module(), Module(), Module(), syncCode=sync)
    instructions = list(result.flatitems())
    issues = [i for i, item in enumerate(instructions) if isinstance(item, TensorLoadToLds)]
    barrier = next(i for i, item in enumerate(instructions) if isinstance(item, SBarrier))
    read = next(i for i, item in enumerate(instructions) if isinstance(item, DSLoadInstruction))
    assert len(issues) == (2 if scales else 1)
    if contracted:
        assert max(issues) < barrier < read
    else:
        assert barrier < read < min(issues)
