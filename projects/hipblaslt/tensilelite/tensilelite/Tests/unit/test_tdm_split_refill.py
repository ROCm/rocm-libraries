################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""Split TDM refill: each descriptor set reads, then refills, in its own half of the
sub-iteration that reads a buffer for the last time."""

import itertools
from types import SimpleNamespace

import pytest
from rocisa.code import Module, TextBlock
from rocisa.container import sgpr
from rocisa.instruction import TensorLoadToLds

from Tensile.Components import DecouplePGR as DP
from Tensile.Components import TDMFuse as TF
from Tensile.Components.SIA import noSchedGlobalRead

pytestmark = pytest.mark.unit

_MX_STATES = {"mx": (32, 32), "no_mx": (0, 0)}


def _ks(fuse=2, mx="no_mx", halfPLR=0, **over):
    """A TDMFuse 1-3 kernel the split refill applies to: PGR2, ScheduleIterAlg 4, four waves."""
    mxA, mxB = _MX_STATES[mx]
    state = dict(TDMFuse=fuse, TDMInst=3, NumWaves=4, TDMSplit=0, UseSubtileImpl=False,
                 enableTDMA=True, enableTDMB=True, PrefetchGlobalRead=2, _ScheduleIterAlg=0,
                 _StinkyTofuOptLevel=3, InnerUnroll=1, HalfPLR=halfPLR, MIWaveTile=[8, 8],
                 EnableMatrixInstruction=True,
                 ProblemType={"MXBlockA": mxA, "MXBlockB": mxB, "Sparse": 0})
    state.update(over)
    return state


@pytest.mark.parametrize("fuse, mx", itertools.product((1, 2, 3), sorted(_MX_STATES)))
@pytest.mark.parametrize("halfPLR, order", [(0, ("B", "A")), (1, ("B", "A")), (2, ("A", "B"))])
def test_b_leads_unless_halfplr_halves_b(fuse, mx, halfPLR, order):
    assert DP.tdmSplitRefillOrder(_ks(fuse, mx, halfPLR)) == order


@pytest.mark.parametrize("over, reason", [
    ({"TDMFuse": 0}, "own descriptor sets"),
    ({"NumWaves": 1}, "one wave"),
    ({"enableTDMB": False}, "own TDM descriptor sets"),
    ({"TDMInst": 1}, "own TDM descriptor sets"),
    ({"TDMSplit": 1}, "own TDM descriptor sets"),
    ({"UseSubtileImpl": True}, "own TDM descriptor sets"),
    ({"PrefetchGlobalRead": 1}, "PrefetchGlobalRead=1"),
    ({"PrefetchGlobalRead": 3}, "PrefetchGlobalRead=3"),
    ({"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 1}, "decoupled pair"),
    ({"_ScheduleIterAlg": 1}, "ScheduleIterAlg=4"),
    ({"_StinkyTofuOptLevel": 0}, "ScheduleIterAlg=4"),
    ({"InnerUnroll": 2}, "InnerUnroll=2"),
    ({"ProblemType": {"MXBlockA": 0, "MXBlockB": 0, "Sparse": 1}}, "sparse"),
    ({"EnableMatrixInstruction": False}, "matrix instructions"),
    ({"HalfPLR": 3}, "HalfPLR=3"),
    ({"MIWaveTile": [1, 3]}, "odd number of WMMAs"),
] + [({key: True}, "%s is not supported" % key) for key in DP._SPLIT_REFILL_UNSUPPORTED_KEYS
     if key not in ("TDMSplit", "UseSubtileImpl")])
def test_each_unsupported_axis_names_its_reason(over, reason):
    state = _ks(**over)
    assert reason in DP.tdmSplitRefillUnsupportedReason(state)
    assert DP.tdmSplitRefillOrder(state) is None


@pytest.mark.parametrize("numItersPLR, coalescedReads, reason", [
    (0, (1, 1), "prefetched 0 sub-iterations"),
    (2, (1, 1), "prefetched 2 sub-iterations"),
    (1, (2, 1), "covers several sub-iterations"),
    (1, (1, 2), "covers several sub-iterations"),
])
def test_the_writer_must_prefetch_one_sub_iteration_per_read(numItersPLR, coalescedReads, reason):
    assert reason in DP.tdmSplitRefillUnsupportedReason(_ks(), numItersPLR, coalescedReads)
    assert DP.tdmSplitRefillOrder(_ks(), numItersPLR, coalescedReads) is None


@pytest.mark.parametrize("fuse, sides", [
    (1, {"A": "A", "MXSA": "A", "MXSB": "B", "B": "B"}),
    (2, {"A": "A", "MXSA": "A", "MXSB": "A", "B": "B"}),
    (3, {"A": "A", "MXSA": "B", "MXSB": "B", "B": "B"}),
])
def test_a_tensor_takes_the_lds_side_of_the_set_it_rides(fuse, sides):
    assert DP.tdmSplitRefillTokenSides(_ks(fuse, "mx")) == sides


@pytest.mark.parametrize("fuse, mx", itertools.product((1, 2, 3), sorted(_MX_STATES)))
def test_the_members_of_a_set_share_its_data_tensor_side(fuse, mx):
    state = _ks(fuse, mx)
    sides = DP.tdmSplitRefillTokenSides(state)
    for group in TF.liveGroups(state):
        owner = TF.tdmSetOwner(state, group[0])
        assert owner in ("A", "B")
        assert {sides[tc] for tc in group} == {owner}


# ---------------------------------------------------------------------------
# SIA.noSchedGlobalRead: where each set's tensor_load_to_lds goes.
# ---------------------------------------------------------------------------
def _marker(name, text):
    module = Module(name)
    module.add(TextBlock("%s\n" % text))
    return module


def _globalRead(tc, withLoad=True):
    """A data tensor's in-loop global read: non-TDM code, then its tensor_load_to_lds."""
    module = Module("globalRead%s" % tc)
    module.add(TextBlock("header_%s\n" % tc.lower()))
    if withLoad:
        load = Module("tdm issue %s" % tc)
        load.add(TensorLoadToLds(group0=sgpr(0, 4), group1=sgpr(4, 8), group2=None, group3=None,
                                 comment="tdm_%s" % tc.lower()))
        module.add(load)
    return module


def _schedule(state, loopIters, split=True, loadB=True):
    codes = SimpleNamespace(
        perIterGlobalRead=[Module("perIterGlobalRead%u" % i) for i in range(loopIters)],
        tdmRefill={"A": Module("tdmRefillA"), "B": Module("tdmRefillB")},
        tdmRefillHead=Module("tdmRefillHead"),
        dtlsM0UpdateA=Module(), dtlsM0UpdateB=Module(),
        dtlsM0UpdateMXSA=Module(), dtlsM0UpdateMXSB=Module(),
        globalReadA=_globalRead("A"), globalReadB=_globalRead("B", loadB),
        globalReadMXSA=Module("globalReadMXSA"), globalReadMXSB=Module("globalReadMXSB"),
        globalReadMetadata=Module(), gl2PrefetchIncrement=Module(), gl2Prefetch=Module())
    order = DP.tdmSplitRefillOrder(state) if split else None
    writer = SimpleNamespace(states=SimpleNamespace(numItersPLR=1, tdmSplitRefill=order or ()),
                             codes=codes)
    kernel = dict(state, LoopIters=loopIters, NoLdsWriteCode=True, TDMPlusLdsBuf=False)
    noSchedGlobalRead(writer, kernel, _marker("incA", "inc_a"), _marker("incB", "inc_b"))
    return codes


@pytest.mark.parametrize("loopIters", [2, 4])
@pytest.mark.parametrize("halfPLR", [0, 2])
def test_each_set_refills_from_its_owner_module(loopIters, halfPLR):
    codes = _schedule(_ks(halfPLR=halfPLR), loopIters)
    for tc, other in (("A", "B"), ("B", "A")):
        refill = str(codes.tdmRefill[tc])
        assert "tdm_%s" % tc.lower() in refill
        assert "tdm_%s" % other.lower() not in refill
    assert not any("tensor_load_to_lds" in str(m) for m in codes.perIterGlobalRead)


def test_sub_iteration_zero_code_moves_ahead_of_the_lead_refill():
    """LoopIters 2: sub-iteration 0 reads the buffer last, and its increments and the
    last-iteration disable must reach the descriptors before the lead set refills."""
    codes = _schedule(_ks(), 2)
    head = str(codes.tdmRefillHead)
    for text in ("inc_a", "inc_b", "header_a", "header_b"):
        assert text in head
    assert all(m.itemsSize() == 0 for m in codes.perIterGlobalRead[:1])


def test_a_later_last_read_keeps_sub_iteration_zero_code_in_place():
    codes = _schedule(_ks(), 4)
    assert codes.tdmRefillHead.itemsSize() == 0
    scheduled = str(codes.perIterGlobalRead[0])
    for text in ("inc_a", "inc_b", "header_a", "header_b"):
        assert text in scheduled


@pytest.mark.parametrize("split, loadB", [(False, True), (True, False)])
def test_without_a_refill_for_each_set_every_load_stays_deferred(split, loadB):
    codes = _schedule(_ks(), 2, split=split, loadB=loadB)
    assert all(m.itemsSize() == 0 for m in codes.tdmRefill.values())
    assert codes.tdmRefillHead.itemsSize() == 0
    assert "inc_a" in str(codes.perIterGlobalRead[0])
    deferred = str(codes.perIterGlobalRead[1])
    assert "tdm_a" in deferred
    assert ("tdm_b" in deferred) == loadB
