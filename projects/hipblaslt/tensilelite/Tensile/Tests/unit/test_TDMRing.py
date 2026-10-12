################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the deep TDM LDS ring predicate and the per-wave TDM in-flight bound."""
import collections

import pytest

from Tensile.Components.TDMRing import (
    TDM_INFLIGHT_PER_WAVE_LIMIT,
    _TDM_RING_UNSUPPORTED_FLAGS,
    tdmDeepRing,
    tdmInflightPerWaveBound,
    tdmInflightRejectReason,
    tdmRingFenceShape,
    tdmRingIncrementMaskThreshold,
    tdmRingIssuesPerStage,
    tdmRingLoopEntryThreshold,
    tdmRingNoLoadWait,
    tdmRingNullThreshold,
    tdmRingPrologueWait,
    tdmRingPublishWait,
    tdmRingRejectReason,
    tdmRingSideStages,
    tdmRingTailLead,
    tdmWaveIssueMembers,
)

pytestmark = pytest.mark.unit

_MX = {"MXBlockA": 32, "MXBlockB": 32, "Sparse": 0}
_DENSE = {"MXBlockA": 0, "MXBlockB": 0, "Sparse": 0}


def _tdmState(**overrides):
    """Four-wave MX PGR2 TDM state with the keys the bound reads already settled."""
    state = {
        "TDMInst": 3,
        "enableTDMA": True,
        "enableTDMB": True,
        "enableTDMMetadata": False,
        "NumWaves": 4,
        "TDMFuse": 0,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "ProblemType": dict(_MX),
        "PrefetchGlobalRead": 2,
        "NumLdsBlk": 2,
        "HalfPLR": 0,
        "SuppressNoLoadLoop": False,
        "NoTailLoop": False,
        "PrefetchAcrossPersistent": 0,
        "ReuseAcrossPersistent": 0,
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    "overrides, issues",
    [
        pytest.param({}, (("A", "MXSA"), ("B", "MXSB")) * 2, id="fuse0"),
        pytest.param({"TDMFuse": 1}, (("A", "MXSB"), ("MXSA", "B")) * 2, id="fuse1"),
        pytest.param({"TDMFuse": 2},
                     (("A", "B"), ("A", "B"), ("MXSA", "B"), ("MXSB", "B")), id="fuse2"),
        pytest.param({"TDMFuse": 3},
                     (("B", "A"), ("B", "A"), ("MXSA", "A"), ("MXSB", "A")), id="fuse3"),
        pytest.param({"ProblemType": dict(_DENSE)}, (("A",), ("B",)) * 2, id="dense"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True},
                     (("A", "Metadata"), ("B",)) * 2, id="sparse"),
        pytest.param({"TDMSplit": True}, (("A", "A", "MXSA"), ("B", "B", "MXSB")) * 2,
                     id="split"),
        pytest.param({"NumWaves": 1}, (("A", "MXSA", "MXSB", "B"),), id="one-wave"),
        pytest.param({"NumWaves": 1, "TDMSplit": True},
                     (("A", "A", "MXSA", "MXSB", "B", "B"),), id="one-wave-split"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True,
                      "TDMSplit": True},
                     (("A", "A", "Metadata"), ("B", "B")) * 2, id="sparse-split"),
    ],
)
def test_wave_issue_members(overrides, issues):
    assert tdmWaveIssueMembers(_tdmState(**overrides)) == issues


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({}, 4, id="mx-fuse0"),
        pytest.param({"TDMFuse": 1}, 4, id="mx-fuse1"),
        pytest.param({"TDMFuse": 2}, 4, id="mx-fuse2"),
        pytest.param({"TDMFuse": 3}, 4, id="mx-fuse3"),
        pytest.param({"ProblemType": dict(_DENSE)}, 2, id="dense"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True}, 4,
                     id="sparse"),
        pytest.param({"HalfPLR": 1}, 5, id="mx-halfplr"),
        pytest.param({"PrefetchAcrossPersistent": 1}, 6, id="mx-pap"),
        pytest.param({"HalfPLR": 1, "PrefetchAcrossPersistent": 1}, 8, id="mx-halfplr-pap"),
        pytest.param({"PrefetchGlobalRead": 1, "NumLdsBlk": 1}, 2, id="one-lds-buffer"),
        pytest.param({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1,
                      "PrefetchGlobalReadB": 2}, 4, id="decoupled-1-2-fuse0"),
        pytest.param({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1,
                      "PrefetchGlobalReadB": 2, "TDMFuse": 1}, 3, id="decoupled-1-2-fuse1"),
        pytest.param({"NumWaves": 1, "ProblemType": dict(_DENSE)}, 4, id="one-wave-dense"),
        pytest.param({"NumWaves": 1}, 8, id="one-wave-mx"),
        pytest.param({"NumWaves": 1, "HalfPLR": 1}, 11, id="one-wave-mx-halfplr"),
    ],
)
def test_bound_for_todays_configs(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, 6, id="ring3-mx"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4}, 8, id="ring4-mx"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "TDMFuse": 1}, 8,
                     id="ring4-mx-fuse1"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "ProblemType": dict(_DENSE)}, 4,
                     id="ring4-dense"),
        pytest.param({"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 2,
                      "PrefetchGlobalReadB": 4}, 8, id="ring-2-4-mx-fuse0"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "PrefetchAcrossPersistent": 1},
                     10, id="ring4-mx-pap"),
    ],
)
def test_deep_ring_within_limit(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "HalfPLR": 1,
                      "PrefetchAcrossPersistent": 1}, 16, id="ring4-mx-halfplr-pap"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "NumWaves": 1}, 16,
                     id="ring4-one-wave-mx"),
        pytest.param({"TDMSplit": True, "NumWaves": 1}, 12, id="split-one-wave-mx"),
        pytest.param({"TDMSplit": True, "HalfPLR": 1, "PrefetchAcrossPersistent": 1}, 12,
                     id="split-mx-halfplr-pap"),
        pytest.param({"NumWaves": 1, "TDMSplit": True, "HalfPLR": 1, "enableTDMMetadata": True,
                      "ProblemType": {**_DENSE, "Sparse": 1}}, 14,
                     id="split-sparse-one-wave-halfplr"),
    ],
)
def test_bound_over_limit_is_rejected(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert bound > TDM_INFLIGHT_PER_WAVE_LIMIT
    reason = tdmInflightRejectReason(state)
    assert reason is not None
    assert "up to %u TDM operations" % bound in reason
    assert "limit of %u" % TDM_INFLIGHT_PER_WAVE_LIMIT in reason


def test_tdm_plus_lds_buf_keeps_two_stages_in_flight():
    """Three LDS blocks, but PGR2 prefetches two stages, so the third block adds nothing."""
    state = _tdmState(TDMPlusLdsBuf=1, NumLdsBlk=3, NumWaves=1)
    assert tdmInflightPerWaveBound(state) == 8
    assert tdmInflightRejectReason(state) is None


def test_two_block_unsplit_path_is_not_rejected():
    """Over the limit by the bound, but on the two-block unsplit path existing solutions use."""
    state = _tdmState(NumWaves=1, HalfPLR=1, enableTDMMetadata=True,
                      ProblemType={**_MX, "Sparse": 1})
    assert tdmInflightPerWaveBound(state) == 14
    assert tdmInflightRejectReason(state) is None


def test_no_tail_loop_drops_the_tail_term():
    state = _tdmState(NumWaves=1, HalfPLR=1, NoTailLoop=True)
    assert tdmInflightPerWaveBound(state) == 8


def test_bound_is_zero_without_tdm():
    state = _tdmState(TDMInst=0, enableTDMA=False, enableTDMB=False)
    assert tdmInflightPerWaveBound(state) == 0
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, deep",
    [
        pytest.param({}, False, id="pgr2"),
        pytest.param({"PrefetchGlobalRead": 1, "NumLdsBlk": 1}, False, id="pgr1"),
        pytest.param({"TDMPlusLdsBuf": 1, "NumLdsBlk": 3}, False, id="tdmpluslds"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, True, id="ring3"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4}, True, id="ring4"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "enableTDMB": False}, False,
                     id="b-not-tdm"),
    ],
)
def test_deep_ring_is_independent_of_tdm_plus_lds_buf(overrides, deep):
    assert tdmDeepRing(_tdmState(**overrides)) is deep


def _ringState(**overrides):
    """Supported four-wave MX TDMFuse 0 ring at PrefetchGlobalRead 4 under ScheduleIterAlg 0,
    as Solution checks it: the custom main-loop schedule is still at its auto -1 there."""
    state = _tdmState(PrefetchGlobalRead=4, NumLdsBlk=4, ScheduleIterAlg=0,
                      _ScheduleIterAlg=0, _StinkyTofuOptLevel=0, PrefetchLocalRead=1,
                      LoopIters=2, EnableMatrixInstruction=True, MIInputPerThread=16,
                      LocalReadVectorWidthA=16, ClusterDim=[1, 1], UseCustomMainLoopSchedule=-1,
                      DepthU=256, AssertSummationElementMultiple=1, **{"1LDSBuffer": 0})
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({}, 8, id="ring4-mx"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, 6, id="ring3-mx"),
        pytest.param({"PrefetchGlobalRead": 5, "NumLdsBlk": 5}, 10, id="ring5-mx"),
        pytest.param({"ProblemType": dict(_DENSE)}, 4, id="ring4-dense"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3, "ProblemType": dict(_DENSE)}, 3,
                     id="ring3-dense"),
        pytest.param({"PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 4}, 8,
                     id="ring4-equal-pair"),
        pytest.param({"PrefetchLocalRead": 3, "LoopIters": 4}, 8, id="ring4-plr3"),
        pytest.param({"HalfPLR": 1}, 9, id="ring4-mx-halfplr"),
        pytest.param({"HalfPLR": 3, "SuppressNoLoadLoop": True, "PrefetchGlobalRead": 5, "NumLdsBlk": 5}, 11,
                     id="ring5-mx-halfplr-snll"),
        pytest.param({"HalfPLR": 2, "PrefetchGlobalRead": 11, "NumLdsBlk": 11, "ProblemType": dict(_DENSE)}, 11,
                     id="ring11-dense-halfplr"),
        pytest.param({"HalfPLR": 3, "PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 2},
                     9, id="ring-4-2-mx-halfplr"),
        pytest.param({"HalfPLR": 1, "PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 4},
                     9, id="ring-1-4-mx-halfplr"),
    ],
)
def test_ring_supported_subset(overrides, bound):
    state = _ringState(**overrides)
    assert tdmRingRejectReason(state) is None
    assert tdmInflightPerWaveBound(state) == bound
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, what",
    [
        pytest.param({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 0},
                     "PrefetchGlobalReadA=4 with PrefetchGlobalReadB=0", id="side-at-level-0"),
        pytest.param({"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 2,
                      "StaggerU": 32}, "StaggerU=32", id="divergent-stagger"),
        pytest.param({"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 2,
                      "StaggerU": 0, "AssertSummationElementMultiple": 512, "DepthU": 256},
                     "AssertSummationElementMultiple=512", id="divergent-asem"),
        pytest.param({"DirectToLdsA": 1}, "DirectToLds or DirectToVgpr", id="dtl"),
        pytest.param({"ScheduleIterAlg": 4, "_StinkyTofuOptLevel": 3}, "ScheduleIterAlg=4",
                     id="sia4"),
        pytest.param({"ScheduleIterAlg": 3, "_ScheduleIterAlg": 3}, "ScheduleIterAlg=3",
                     id="sia3"),
        pytest.param({"PrefetchLocalRead": 0}, "PrefetchLocalRead=0", id="plr0"),
        pytest.param({"PrefetchLocalRead": 2}, "PrefetchLocalRead=2 with LoopIters=2",
                     id="plr-wraps"),
        pytest.param({"LoopIters": 1}, "LoopIters=1", id="loopiters1"),
        pytest.param({"LocalReadVectorWidthA": 32}, "LoopIters=2", id="wide-read"),
        pytest.param({"1LDSBuffer": 1}, "1LDSBuffer=1", id="one-lds-buffer"),
        pytest.param({"HalfPLR": 1, "StaggerU": 32}, "HalfPLR with StaggerU=32", id="halfplr-stagger"),
        pytest.param({"PrefetchAcrossPersistent": 1}, "PrefetchAcrossPersistent", id="pap"),
        pytest.param({"SuppressNoLoadLoop": True}, "SuppressNoLoadLoop", id="suppress-nll"),
        pytest.param({"TDMSplit": True}, "TDMSplit", id="split"),
        pytest.param({"UseSubtileImpl": True}, "UseSubtileImpl", id="subtile"),
        pytest.param({"PrefetchGL2": 1}, "PrefetchGL2", id="gl2"),
        pytest.param({"UnrollLoopSwapGlobalReadOrder": 1}, "UnrollLoopSwapGlobalReadOrder",
                     id="ulsgro"),
        pytest.param({"UseCustomMainLoopSchedule": 1}, "UseCustomMainLoopSchedule", id="cms"),
        pytest.param({"UsePLRPack": 1}, "UsePLRPack", id="plr-pack"),
        pytest.param({"ExpertSchedulingMode": 1}, "ExpertSchedulingMode", id="esm"),
        pytest.param({"ClusterDim": [2, 1]}, "ClusterDim", id="cluster"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True},
                     "Sparse", id="sparse"),
        pytest.param({"LDSSegmentInterleave": 1}, "LDSSegmentInterleave=1", id="lds-seg"),
        pytest.param({"TDMFuse": 1}, "TDMFuse=1", id="fuse1"),
        pytest.param({"ProblemType": {**_MX, "MXBlockB": 0}}, "MX on one side only",
                     id="mx-a-only"),
    ],
)
def test_ring_rejects_outside_the_subset(overrides, what):
    reason = tdmRingRejectReason(_ringState(**overrides))
    assert reason is not None
    assert what in reason
    assert reason.startswith("TDM LDS ring (PrefetchGlobalRead=")


# Each flag the ring rejects when on, with a value that turns it on.
_RING_FLAGS_ON = {
    "PrefetchAcrossPersistent": 1,
    "ReuseAcrossPersistent": 1,
    "SuppressNoLoadLoop": True,
    "UseSubtileImpl": True,
    "TDMSplit": True,
    "PrefetchGL2": 1,
    "UnrollLoopSwapGlobalReadOrder": 1,
    "UseCustomMainLoopSchedule": 1,
    "ForceUnrollSubIter": True,
    "UsePLRPack": 1,
    "ExpertSchedulingMode": 2,
    "TDMPlusLdsBuf": 1,
    "ExpandPointerSwap": True,
}
_RING_FLAGS = [pytest.param(flag, id=flag) for flag in sorted(_RING_FLAGS_ON)]


def _ringFlagReason(flag):
    return "TDM LDS ring (PrefetchGlobalRead=4): %s is not supported yet" % flag


def test_ring_flag_table_covers_the_gate():
    assert set(_RING_FLAGS_ON) == set(_TDM_RING_UNSUPPORTED_FLAGS)


@pytest.mark.parametrize("flag", _RING_FLAGS)
def test_ring_rejects_a_flag_set_on(flag):
    """The reason names that flag alone: the auto custom main-loop schedule is not on."""
    reason = tdmRingRejectReason(_ringState(**{flag: _RING_FLAGS_ON[flag]}))
    assert reason == _ringFlagReason(flag)


@pytest.mark.parametrize("flag", _RING_FLAGS)
def test_ring_auto_value_of_a_flag(flag):
    """Only the custom main-loop schedule is still auto (-1) when Solution checks the ring, and
    Solution resolves it off on a ring afterwards; -1 on any other flag reads as on."""
    reason = tdmRingRejectReason(_ringState(**{flag: -1}))
    if flag == "UseCustomMainLoopSchedule":
        assert reason is None
    else:
        assert reason == _ringFlagReason(flag)


@pytest.mark.parametrize("flag", _RING_FLAGS)
def test_ring_accepts_a_flag_set_off(flag):
    for off in (0, False):
        assert tdmRingRejectReason(_ringState(**{flag: off})) is None


@pytest.mark.parametrize(
    "value, reason",
    [
        pytest.param(-1, None, id="auto"),
        pytest.param(0, None, id="off"),
        pytest.param(1, "TDM LDS ring (PrefetchGlobalRead=4): LDSSegmentInterleave=1 is not "
                     "supported yet", id="on"),
    ],
)
def test_ring_lds_segment_interleave(value, reason):
    """Auto reads as off, like the custom main-loop schedule's: Solution resolves it off on a
    ring before the check. Only an explicit 1 is rejected."""
    assert tdmRingRejectReason(_ringState(LDSSegmentInterleave=value)) == reason


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"PrefetchGlobalRead": 2, "NumLdsBlk": 2}, id="pgr2"),
        pytest.param({"PrefetchGlobalRead": 2, "NumLdsBlk": 3, "TDMPlusLdsBuf": 1},
                     id="tdmpluslds"),
        pytest.param({"enableTDMB": False}, id="b-not-tdm"),
    ],
)
def test_ring_rules_skip_non_ring_kernels(overrides):
    assert tdmRingRejectReason(_ringState(**overrides)) is None


@pytest.mark.parametrize(
    "overrides, perStage",
    [
        pytest.param({}, 2, id="mx-fuse0"),
        pytest.param({"ProblemType": dict(_DENSE)}, 1, id="dense"),
        pytest.param({"NumWaves": 1}, 4, id="one-wave-mx"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True}, None,
                     id="sparse-roles-differ"),
    ],
)
def test_ring_issues_per_stage(overrides, perStage):
    assert tdmRingIssuesPerStage(_ringState(**overrides)) == perStage


@pytest.mark.parametrize(
    "overrides, fence, prologue, publish, noLoad",
    [
        pytest.param({}, "fused", 6, 4, (4, 2, 0), id="s4-mx-fused"),
        pytest.param({"ProblemType": dict(_DENSE)}, "fused", 3, 2, (2, 1, 0),
                     id="s4-dense-fused"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, "fused", 4, 2, (2, 0),
                     id="s3-mx-fused"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3, "ProblemType": dict(_DENSE)},
                     "fused", 2, 1, (1, 0), id="s3-dense-fused"),
        pytest.param({}, "twoBarrier", 6, 6, (4, 2, 0), id="s4-mx-two-barrier"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, "twoBarrier", 4, 4, (2, 0),
                     id="s3-mx-two-barrier"),
    ],
)
def test_ring_waits_retire_only_the_oldest_stage(overrides, fence, prologue, publish, noLoad):
    """Prologue (S-1)*T, from the S*T it has in flight once every stage is issued; main loop
    (S-2)*T merged, (S-1)*T two-barrier; NGLL remainPgr r waits (r-1)*T."""
    state = _ringState(**overrides)
    assert tdmRingPrologueWait(state) == prologue
    assert tdmRingPublishWait(state, fence) == publish
    stages = state["PrefetchGlobalRead"]
    assert tuple(tdmRingNoLoadWait(state, r) for r in range(stages - 1, 0, -1)) == noLoad
    perStage = tdmRingIssuesPerStage(state)
    assert tdmInflightPerWaveBound(state) - prologue == perStage
    assert tdmInflightPerWaveBound(state) - publish == (2 if fence == "fused" else 1) * perStage


@pytest.mark.parametrize(
    "override, shape",
    [
        pytest.param("auto", "fused", id="auto"),
        pytest.param("fused", "fused", id="fused"),
        pytest.param("twoBarrier", "twoBarrier", id="two-barrier"),
    ],
)
def test_ring_fence_shape(override, shape):
    assert tdmRingFenceShape(override) == shape


def test_ring_fence_shape_rejects_unknown_override():
    with pytest.raises(ValueError):
        tdmRingFenceShape("threeBarrier")


def _ringReplay(state, tiles, fence, earlyExitDrain=True, prologueWait=None):
    """Replay a K of `tiles` K-tiles through the ring's branches and waits.

    Returns the stages in flight at each fence, the fences whose tile is still in flight after
    their wait, and the most TDMs one wave has in flight before the tail loop. It follows the
    writer: setupNewTile issues tile 0 and the prologue issues tiles 1..S-1, neither waiting.
    The prologue leaves for skipPGR{S}_1 at the stage equal to the loop counter
    (openPrefetchGlobalRead2orMore), which drains when `earlyExitDrain`; skipPGR{S}_{S} then
    waits for `prologueWait`, tdmRingPrologueWait by default, and its barrier publishes tile 0
    (closePrefetchGlobalRead2orMore). openLoop sends 1 tile to toPGR1, 2..S-1 tiles to
    NoGlobalLoadLoop_{S-2} and S tiles to NoGlobalLoadLoop_{S-1}; the main loop runs until
    S tiles remain. NoGlobalLoadLoop r >= 2 decrements the counter and, below S-1, leaves
    for toPGR1 at 1 (closeSumAtLeastUnroll). A fence publishes the tile the local-read
    prefetch reads next; the NoLoadLoop drains before the tail loop.

    The waves issuing A and those issuing B each wait on their own side's S_A or S_B slots.
    S above is the skeleton, PrefetchGlobalRead: the shallower side. Unless the prologue left
    early, the deeper side then issues the rest of its stages, and each of its fills past K is
    a null descriptor that still counts on tensorcnt. Fence counts are those of A's waves.
    """
    stages = state["PrefetchGlobalRead"]
    perStage = tdmRingIssuesPerStage(state)
    slots = {side: tdmRingSideStages(state, side) for side in ("A", "B")}
    if prologueWait is None:
        prologueWaits = {side: tdmRingPrologueWait(state, side) for side in slots}
    else:
        prologueWaits = dict.fromkeys(slots, prologueWait)
    inFlight = {side: collections.deque() for side in slots}
    issued = dict.fromkeys(slots, 0)
    fences, unsafe = [], []
    peak = 0

    def issue(side, computing=None):
        nonlocal peak
        # A main-loop fill reuses the slot of the tile whose reads its fence protected.
        assert computing is None or issued[side] == computing + slots[side]
        inFlight[side].append(issued[side] if issued[side] < tiles else None)
        issued[side] += 1
        peak = max(peak, len(inFlight[side]) * perStage)

    def issueBoth(computing=None):
        for side in slots:
            issue(side, computing)

    def wait(counts):
        for side, count in counts.items():
            while len(inFlight[side]) * perStage > count:
                inFlight[side].popleft()

    def publish(site, tile, counts):
        fences.append((site, len(inFlight["A"])))
        wait(counts)
        if any(tile in inFlight[side] for side in slots):
            unsafe.append((site, tile))

    counter = tiles
    issueBoth()
    leftEarly = False
    for stage in range(1, stages):
        if counter == stage:
            leftEarly = True
            break
        issueBoth()
    if leftEarly:
        fences.append(("EarlyExit", len(inFlight["A"])))
        if earlyExitDrain:
            wait(dict.fromkeys(slots, 0))
    else:
        for side in slots:
            while issued[side] < slots[side]:
                issue(side)
    publish("Prologue", 0, prologueWaits)
    computing = 0
    if counter == 1:
        first = 0
    elif counter <= stages - 1:
        first = stages - 2
    elif counter <= stages:
        first = stages - 1
    else:
        publishWaits = {side: tdmRingPublishWait(state, fence, side) for side in slots}
        while counter > stages:
            if fence == "twoBarrier":
                issueBoth(computing)
            publish("MainLoop", computing + 1, publishWaits)
            if fence == "fused":
                issueBoth(computing)
            computing += 1
            counter -= 1
        first = stages - 1
    for remainPgr in range(first, 0, -1):
        publish("NoGlobalLoadLoop_%u" % remainPgr, computing + 1,
                {side: tdmRingNoLoadWait(state, remainPgr, side) for side in slots})
        computing += 1
        if remainPgr >= 2:
            counter -= 1
            if remainPgr < stages - 1 and counter == 1:
                break
    fences.append(("NoLoadLoop", len(inFlight["A"])))
    wait(dict.fromkeys(slots, 0))
    fences.append(("TailLoop", len(inFlight["A"])))
    assert computing == tiles - 1, "the NoLoadLoop computes the last tile"
    return fences, unsafe, peak


def _ringStageCases():
    """Every (S, name, problem type) of the supported ring the in-flight limit admits."""
    for name, problemType in (("mx", _MX), ("dense", _DENSE)):
        for stages in range(3, TDM_INFLIGHT_PER_WAVE_LIMIT + 1):
            state = _ringState(PrefetchGlobalRead=stages, NumLdsBlk=stages,
                               ProblemType=dict(problemType))
            if tdmInflightRejectReason(state) is None:
                yield stages, name, problemType


_RING_REPLAYS = [
    pytest.param(stages, fence, problemType, id="s%u-%s-%s" % (stages, fence, name))
    for stages, name, problemType in _ringStageCases()
    for fence in ("fused", "twoBarrier")
]
_RING_STAGES = sorted({stages for stages, _, _ in _ringStageCases()})


@pytest.mark.parametrize("stages, fence, problemType", _RING_REPLAYS)
def test_ring_control_flow_lands_every_tile_it_publishes(stages, fence, problemType):
    """K = 1..2S+1 tiles: each fence has exactly the stages its wait assumes in flight, or none
    after the early-exit drain, and its wait lands the tile read next. With S or more tiles the
    prologue waits with all S stages in flight, the per-wave peak the R7 bound covers; an early
    exit has its N tiles in flight."""
    state = _ringState(PrefetchGlobalRead=stages, NumLdsBlk=stages, ProblemType=dict(problemType))
    assert tdmRingRejectReason(state) is None
    perStage = tdmRingIssuesPerStage(state)
    assert tdmInflightPerWaveBound(state) == stages * perStage <= TDM_INFLIGHT_PER_WAVE_LIMIT
    for tiles in range(1, 2 * stages + 2):
        fences, unsafe, peak = _ringReplay(state, tiles, fence)
        assert unsafe == [], (tiles, unsafe)
        assert peak == min(tiles, stages) * perStage, (tiles, peak)
        for site, inFlight in fences:
            if site == "EarlyExit":
                expected = tiles
            elif site == "Prologue":
                expected = stages if tiles >= stages else 0
            elif site == "MainLoop":
                expected = stages - 1 if fence == "fused" else stages
            elif site.startswith("NoGlobalLoadLoop_"):
                expected = int(site.rsplit("_", 1)[1]) if tiles >= stages else 0
            else:
                expected = 0
            assert inFlight == expected, (tiles, site, inFlight)


def _ringRaces(**replay):
    """Every unsafe fence as (S, tiles, site, tile), over every supported S, both fences, MX
    and dense. A callable `prologueWait` is given the state."""
    races = set()
    for stages, _, problemType in _ringStageCases():
        for fence in ("fused", "twoBarrier"):
            state = _ringState(PrefetchGlobalRead=stages, NumLdsBlk=stages,
                               ProblemType=dict(problemType))
            kwargs = dict(replay)
            if callable(kwargs.get("prologueWait")):
                kwargs["prologueWait"] = kwargs["prologueWait"](state)
            for tiles in range(1, 2 * stages + 2):
                races.update((stages, tiles) + race
                             for race in _ringReplay(state, tiles, fence, **kwargs)[1])
    return races


def test_ring_control_flow_without_the_early_exit_drain_races():
    """Positive control: without the drain an early exit has fewer than S stages in flight, so
    the prologue's wait for the oldest of S stages returns before tile 0 lands. 2..S-2 tiles
    also count down from NoGlobalLoadLoop_{S-2} with every tile they publish in flight."""
    assert _ringRaces(earlyExitDrain=False) == \
        {(stages, tiles, "Prologue", 0) for stages in _RING_STAGES
         for tiles in range(1, stages)} | \
        {(stages, tiles, "NoGlobalLoadLoop_%u" % (stages - 1 - tile), tile)
         for stages in _RING_STAGES for tiles in range(2, stages - 1) for tile in range(1, tiles)}


def test_ring_control_flow_with_a_prologue_wait_that_retires_nothing_races():
    """Positive control: a prologue wait of S*T keeps all S stages in flight, so every K of S or
    more tiles publishes tile 0 before it lands."""
    def allStages(state):
        return state["PrefetchGlobalRead"] * tdmRingIssuesPerStage(state)
    assert _ringRaces(prologueWait=allStages) == \
        {(stages, tiles, "Prologue", 0) for stages in _RING_STAGES
         for tiles in range(stages, 2 * stages + 2)}


@pytest.mark.parametrize("pgrA, pgrB", [(3, 2), (2, 4), (5, 3), (4, 5)])
@pytest.mark.parametrize("fence", ["fused", "twoBarrier"])
def test_divergent_ring_control_flow_lands_every_tile_it_publishes(pgrA, pgrB, fence):
    """K = 1..2*S_hi+2 tiles with S_A != S_B: each side's waits land the tile its fence
    publishes, and the deeper side peaks at all of its stages, null descriptors included,
    once the prologue does not leave early. Positive control: without the early-exit drain
    one tile races."""
    shallow, deep = sorted((pgrA, pgrB))
    state = _ringState(PrefetchGlobalRead=shallow, NumLdsBlk=shallow,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert tdmRingRejectReason(state) is None
    perStage = tdmRingIssuesPerStage(state)
    assert tdmInflightPerWaveBound(state) == deep * perStage <= TDM_INFLIGHT_PER_WAVE_LIMIT
    for tiles in range(1, 2 * deep + 3):
        _, unsafe, peak = _ringReplay(state, tiles, fence)
        assert unsafe == [], (tiles, unsafe)
        assert peak == (tiles if tiles < shallow else deep) * perStage, (tiles, peak)
    assert _ringReplay(state, 1, fence, earlyExitDrain=False)[1]


@pytest.mark.parametrize("pgrA, pgrB", [(4, 1), (1, 3), (5, 1)])
def test_one_slot_side_ring_lands_every_tile_it_publishes(pgrA, pgrB):
    """A side with one slot runs a PrefetchGlobalRead=1 loop under two barriers whatever the
    override: it refills its slot after the protect and waits for that fill before the publish,
    while the deeper side keeps all of its stages in flight. Positive control: a prologue wait
    that retires nothing races."""
    deep = max(pgrA, pgrB)
    state = _ringState(PrefetchGlobalRead=1, NumLdsBlk=1,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert tdmRingRejectReason(state) is None
    fence = tdmRingFenceShape("fused", state)
    assert fence == "twoBarrier"
    perStage = tdmRingIssuesPerStage(state)
    assert tdmInflightPerWaveBound(state) == deep * perStage
    for tiles in range(1, 2 * deep + 3):
        _, unsafe, peak = _ringReplay(state, tiles, fence)
        assert unsafe == [], (tiles, unsafe)
        assert peak == deep * perStage, (tiles, peak)
    assert _ringReplay(state, 1, fence, prologueWait=deep * perStage)[1]


def _halfPlrRingState(pgrA, pgrB, problemType=_MX):
    shallow = min(pgrA, pgrB)
    return _ringState(HalfPLR=1, PrefetchGlobalRead=shallow, NumLdsBlk=shallow,
                      PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB,
                      ProblemType=dict(problemType))


@pytest.mark.parametrize(
    "pgrA, pgrB, armed, masks, leads",
    [
        pytest.param(4, 4, True, (3, 3), {3: (0, 0), 4: (0, 0)}, id="equal"),
        pytest.param(5, 3, True, (4, 2), {2: (0, 0), 3: (1, 0), 4: (0, 0)}, id="pgr3-deep-a"),
        pytest.param(2, 4, True, (1, 4), {1: (0, 0), 2: (0, 2), 4: (0, 0)}, id="pgr2-wrap"),
        pytest.param(2, 4, False, (1, 3), {2: (0, 1), 3: (0, 0)}, id="pgr2"),
        pytest.param(4, 1, True, (4, 0), {0: (0, 0), 1: (3, 0), 4: (0, 0)}, id="one-slot-wrap"),
        pytest.param(4, 1, False, (3, 0), {1: (2, 0), 3: (0, 0)}, id="one-slot"),
    ],
)
def test_halfplr_ring_thresholds(pgrA, pgrB, armed, masks, leads):
    """Mask S_side - 1, one more on a deeper side under PrefetchGlobalRead 3 with the StaggerU
    wrap armed; lead max(0, mask - N) once N reaches PrefetchGlobalRead; null S_side; entry 0."""
    state = _halfPlrRingState(pgrA, pgrB)
    assert tuple(tdmRingIncrementMaskThreshold(state, side, armed) for side in "AB") == masks
    for tiles, lead in leads.items():
        assert tuple(tdmRingTailLead(state, side, tiles, armed) for side in "AB") == lead
    assert tuple(tdmRingNullThreshold(state, side) for side in "AB") == (pgrA, pgrB)
    assert tdmRingLoopEntryThreshold(state) == 0


def test_halfplr_ring_rules_keep_todays_values():
    """HalfPLR off a ring keeps its mask at 1; a ring keeps entering at PrefetchGlobalRead behind
    a NoLoadLoop, and SuppressNoLoadLoop on its own drops it as HalfPLR does."""
    assert tdmRingIncrementMaskThreshold(_tdmState(HalfPLR=1), "A", True) == 1
    assert tdmRingLoopEntryThreshold(_ringState()) == 4
    assert tdmRingLoopEntryThreshold(_ringState(SuppressNoLoadLoop=True)) == 0


def _halfPlrRingReplay(state, tiles, fence, armed, tail, **control):
    """Replay N = `tiles` K-tiles of a HalfPLR ring, StaggerU 0, into its tail; return what fails.

    Per side it follows the descriptor's tile, the slot each fill writes, the local reads not yet
    retired and the TDMs in flight on the waves issuing that side. The prologue fills tile 0 and
    advances (prefetch index PGR - 1, or 1 under 3; not with one slot); stage k = 1..PGR-1 leaves
    early, draining, when LoopCounter == k, else fills, and below PGR - 1 advances and swaps.
    Unless it left early, a deeper side then advances (no wrap test), swaps from stage 2 and fills
    each extra stage, null past K. The main loop runs above tdmRingLoopEntryThreshold. Body b
    advances: HalfPLR's mask zeroes the increment for good at or below
    tdmRingIncrementMaskThreshold, and an `armed` StaggerU wrap adds WrapU = 1 - N instead. The
    merged fence then waits, retires the reads, publishes tile b + 1, swaps and refills tile
    b + S_side, null at or below tdmRingNullThreshold but counted; two barriers retire, refill,
    wait and publish. Tile b + 1's two read halves follow, a dummy after the last tile. The tail
    removes the stagger, re-enables the descriptors, retires the dummy read, rewinds
    tdmRingTailLead tiles at the side's own increment and resets the slots: its TDM must load
    tile N into slot 0.
    """
    stages = state["PrefetchGlobalRead"]
    perStage = tdmRingIssuesPerStage(state)
    slots = {x: tdmRingSideStages(state, x) for x in "AB"}
    formulaArmed = control.get("formulaArmed", armed)
    mask = control.get("mask") or (lambda x: tdmRingIncrementMaskThreshold(state, x, formulaArmed))
    null = control.get("null") or (lambda x: tdmRingNullThreshold(state, x))
    lead = control.get("lead") or (lambda x, n: tdmRingTailLead(state, x, n, formulaArmed))
    entry = control.get("entry", tdmRingLoopEntryThreshold(state))
    waits = {x: tdmRingPublishWait(state, fence, x) + control.get("slack", 0) for x in slots}
    prologueWaits = {x: tdmRingPrologueWait(state, x) + control.get("prologueSlack", 0)
                     for x in slots}
    wrapCounter = (1 if stages == 1 else 2) if armed and stages < 3 else None
    wrapU = 1 - tiles
    firstHalves = (1,) if control.get("halfAfterFence") else (1, 2)
    sides = {x: {"tile": 0, "live": True, "enabled": True, "swaps": 0, "slots": {},
                 "halves": collections.defaultdict(set), "reads": set(),
                 "inFlight": collections.deque()} for x in slots}
    errors = []

    def advance(x, counter, prefetch, deepPrologue=False):
        side = sides[x]
        if prefetch == 0 and (control.get("maskDeep") or not deepPrologue) and counter <= mask(x):
            side["live"] = False
        if not deepPrologue and counter + prefetch == wrapCounter:
            side["tile"] += wrapU
        else:
            side["tile"] += side["live"]

    def fill(x, tile):
        side = sides[x]
        slot = side["swaps"] % slots[x]
        if side["enabled"] != (tile < tiles):
            errors.append(("null" if tile < tiles else "past K", x, tile))
        if side["enabled"]:
            if side["tile"] != tile:
                errors.append(("address", x, tile, side["tile"]))
            old = side["slots"].get(slot)
            if old is not None and (side["halves"][old] != {1, 2}
                                    or any(t % slots[x] == slot for t, _ in side["reads"])):
                errors.append(("WAR", x, tile, old))
            side["slots"][slot] = tile
        if side["enabled"] or not control.get("nullSkip"):
            side["inFlight"].extend([tile if side["enabled"] else None] * perStage)
        if len(side["inFlight"]) > slots[x] * perStage:
            errors.append(("in flight", x, tile))

    def read(x, tile, halves):
        side = sides[x]
        if tile < tiles and side["slots"].get(tile % slots[x]) != tile:
            errors.append(("read", x, tile))
        side["halves"][tile].update(halves)
        side["reads"].update((tile, half) for half in halves)

    def wait(counts):
        for x, count in counts.items():
            while len(sides[x]["inFlight"]) > count:
                sides[x]["inFlight"].popleft()

    def retireReads():
        for side in sides.values():
            side["reads"].clear()

    def publish(tile):
        errors.extend(("RAW", x, tile) for x in slots if tile in sides[x]["inFlight"])

    def refill(computing):
        for x in slots:
            sides[x]["swaps"] += 1
            fill(x, computing + slots[x])

    if tiles:
        for x in slots:
            fill(x, 0)
            if stages > 1:
                advance(x, tiles, 1 if stages < 3 else stages - 1)
            sides[x]["swaps"] += 1
        leftEarly = tiles < stages
        for stage in range(1, min(tiles, stages)):
            for x in slots:
                fill(x, stage)
                if stage < stages - 1:
                    advance(x, tiles, stages - 1 - stage)
                    sides[x]["swaps"] += 1
        if leftEarly:
            wait(dict.fromkeys(slots, 0))
        else:
            for x in slots:
                for stage in range(stages, slots[x]):
                    advance(x, tiles, 0, deepPrologue=True)
                    sides[x]["swaps"] += stage > 1
                    sides[x]["enabled"] &= tiles > stage
                    fill(x, stage)
                if control.get("leadAtPrologue"):
                    sides[x]["tile"] -= lead(x, tiles)
        wait(prologueWaits)
        publish(0)
        for x in slots:
            read(x, 0, firstHalves)
    for computing in range(max(0, tiles - entry)):
        counter = tiles - computing
        for x in slots:
            advance(x, counter, 0)
            sides[x]["enabled"] &= counter > null(x)
        if fence == "fused":
            wait(waits)
            retireReads()
            publish(computing + 1)
            refill(computing)
        else:
            if control.get("refillEarly"):
                refill(computing)
            retireReads()
            if not control.get("refillEarly"):
                refill(computing)
            wait(waits)
            publish(computing + 1)
        for x in slots:
            if control.get("halfAfterFence"):
                read(x, computing, (2,))
            read(x, computing + 1, firstHalves)
    for x in slots:
        if any(sides[x]["halves"][t] != {1, 2} for t in range(tiles)):
            errors.append(("unread", x))
    if not tail:
        return errors
    reenable = control.get("reenable", True)
    for x in slots:
        side = sides[x]
        if wrapCounter and tiles:
            side["tile"] += 1 - wrapU    # removeStagger at StaggerUIter = its own pf
        if tiles:
            side["enabled"] |= reenable
            if control.get("tailDscnt", True):
                side["reads"].clear()
        if slots["A"] != slots["B"]:
            side["enabled"] |= reenable
        if not control.get("leadAtPrologue"):
            side["tile"] -= 0 if control.get("leadIncs") and not side["live"] else lead(x, tiles)
        if not side["enabled"] or side["tile"] != tiles:
            errors.append(("tail", x, side["enabled"], side["tile"]))
        if any(t % slots[x] == 0 for t, _ in side["reads"]) or \
           any(t is not None and t % slots[x] == 0 for t in side["inFlight"]):
            errors.append(("tail slot 0 busy", x))
        if len(side["inFlight"]) + perStage > tdmInflightPerWaveBound(state):
            errors.append(("tail in flight", x))
    return errors


_HALFPLR_RING_PAIRS = {
    "equal": [(stages, stages) for stages in range(3, 12)],
    "unequal": [pair for lo in range(2, 7) for hi in range(lo + 1, 9) for pair in ((hi, lo), (lo, hi))],
    "one-slot": [pair for hi in range(3, 9) for pair in ((hi, 1), (1, hi))],
}


def _halfPlrRingReplays(kinds, control=None):
    """(case, errors) of every replay: each pair of `kinds` the in-flight limit admits, MX and
    dense, both fences (two barriers with one slot), the wrap armed or not, N = 0..max(16, 2S+3),
    with and without the tail. `control(state)` gives the replay's keyword overrides."""
    for kind in kinds:
        for problemType in (_MX, _DENSE):
            for pgrA, pgrB in _HALFPLR_RING_PAIRS[kind]:
                state = _halfPlrRingState(pgrA, pgrB, problemType)
                if tdmInflightRejectReason(state) is not None:
                    continue
                overrides = control(state) if control else {}
                for fence in sorted({tdmRingFenceShape(f, state) for f in ("fused", "twoBarrier")}):
                    for armed in (False, True):
                        for tiles in range(max(17, 2 * max(pgrA, pgrB) + 4)):
                            for tail in (False, True):
                                yield ((pgrA, pgrB, problemType["MXBlockA"], fence, armed, tiles, tail),
                                       _halfPlrRingReplay(state, tiles, fence, armed, tail, **overrides))


@pytest.mark.parametrize("kind", sorted(_HALFPLR_RING_PAIRS))
@pytest.mark.parametrize("leadAtPrologue", [False, True], ids=["lead-at-tail", "lead-at-prologue"])
def test_halfplr_ring_replay(kind, leadAtPrologue):
    """HalfPLR on a ring: every fill loads its own tile into a slot whose reads have retired, every
    read finds its tile, every publish has landed, at most S_side stages are in flight, and the
    tail loads tile N into slot 0 within the per-wave bound, whether a deeper side's lead is taken
    back at the tail or, as the writer does, at the end of its prologue."""
    failing = [(case, errors[0]) for case, errors in
               _halfPlrRingReplays((kind,), lambda st: {"leadAtPrologue": leadAtPrologue}) if errors]
    assert not failing, failing[:3]


@pytest.mark.parametrize(
    "kinds, control",
    [
        pytest.param(("equal",), lambda st: {"mask": lambda side: 1}, id="mask-1"),
        pytest.param(("unequal", "one-slot"), lambda st: {"formulaArmed": False}, id="no-wrap-term"),
        pytest.param(("unequal",), lambda st: {"maskDeep": True}, id="mask-in-deep-prologue"),
        pytest.param(("unequal", "one-slot"), lambda st: {"lead": lambda side, n: 0}, id="no-lead"),
        pytest.param(("unequal", "one-slot"), lambda st: {"leadIncs": True}, id="lead-at-masked-inc"),
        pytest.param(("unequal",), lambda st: {
            "lead": lambda side, n: 0 if n < st["PrefetchGlobalRead"]
            else tdmRingSideStages(st, side) - st["PrefetchGlobalRead"]}, id="todays-rewind"),
        pytest.param(("unequal", "one-slot"),
                     lambda st: {"null": lambda side: st["PrefetchGlobalRead"]}, id="null-at-pgr"),
        pytest.param(("equal",), lambda st: {"entry": st["PrefetchGlobalRead"]}, id="entry-at-pgr"),
        pytest.param(("one-slot",), lambda st: {"refillEarly": True}, id="refill-before-protect"),
        pytest.param(("equal",), lambda st: {"halfAfterFence": True}, id="half-after-fence"),
        pytest.param(("equal",), lambda st: {"tailDscnt": False}, id="tail-without-dscnt"),
        pytest.param(("equal",), lambda st: {"reenable": False}, id="tail-without-reenable"),
        pytest.param(("equal",), lambda st: {"slack": tdmRingIssuesPerStage(st)}, id="loop-wait-loose"),
        pytest.param(("equal",), lambda st: {"prologueSlack": tdmRingIssuesPerStage(st)},
                     id="prologue-wait-loose"),
        pytest.param(("equal",), lambda st: {"nullSkip": True}, id="nulls-not-counted"),
    ],
)
def test_halfplr_ring_replay_controls(kinds, control):
    """Positive controls: each fault the formulas and the writer guard against fails a replay."""
    assert any(errors for _, errors in _halfPlrRingReplays(kinds, control))
