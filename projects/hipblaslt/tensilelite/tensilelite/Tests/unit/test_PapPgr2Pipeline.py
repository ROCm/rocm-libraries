# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute the real two-group guards and compiler-token joins on the CPU."""

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SAddU32, SMovB32, SOrB32
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

def _setup(scales=True):
    kernel, tensors, writer = _writer(scales=scales)
    kernel.update(NoTailLoop=True, PrefetchLocalRead=1, LoopIters=2,
                  InnerUnroll=1, ForceUnrollSubIter=False,
                  NoLdsWriteCode=True)
    kernel["1LDSBuffer"] = False
    writer.states.numItersPLR = 1
    writer.states.doFullPackCodePrefetch = False
    writer.states.memTokenLdsBuffer0 = 0
    writer.states.ldsTensorTokenIdx = 1
    writer._nextLdsToken = lambda token: token ^ 1
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
