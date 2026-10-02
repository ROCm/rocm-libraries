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
    TDM_RING_STAGES,
    _TDM_RING_UNSUPPORTED_FLAGS,
    tdmDeepRing,
    tdmInflightPerWaveBound,
    tdmInflightRejectReason,
    tdmRingFenceShape,
    tdmRingIssuesPerStage,
    tdmRingNoLoadWait,
    tdmRingPrologueWait,
    tdmRingPublishWait,
    tdmRingRejectReason,
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
                      **{"1LDSBuffer": 0})
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({}, 8, id="ring4-mx"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, 6, id="ring3-mx"),
        pytest.param({"ProblemType": dict(_DENSE)}, 4, id="ring4-dense"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3, "ProblemType": dict(_DENSE)}, 3,
                     id="ring3-dense"),
        pytest.param({"PrefetchGlobalReadA": 4, "PrefetchGlobalReadB": 4}, 8,
                     id="ring4-equal-pair"),
        pytest.param({"PrefetchLocalRead": 3, "LoopIters": 4}, 8, id="ring4-plr3"),
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
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3, "PrefetchGlobalReadA": 3,
                      "PrefetchGlobalReadB": 4},
                     "PrefetchGlobalReadA=3 with PrefetchGlobalReadB=4", id="divergent"),
        pytest.param({"PrefetchGlobalRead": 5, "NumLdsBlk": 5}, "3 or 4 slots", id="pgr5"),
        pytest.param({"DirectToLdsA": 1}, "DirectToLds or DirectToVgpr", id="dtl"),
        pytest.param({"NumWaves": 1}, "NumWaves=1", id="one-wave"),
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
        pytest.param({"HalfPLR": 1}, "HalfPLR", id="halfplr"),
        pytest.param({"PrefetchAcrossPersistent": 1}, "PrefetchAcrossPersistent", id="pap"),
        pytest.param({"StreamK": 3}, "StreamK", id="streamk"),
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
    "StreamK": 3,
    "PrefetchAcrossPersistent": 1,
    "ReuseAcrossPersistent": 1,
    "HalfPLR": 1,
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
    """
    stages = state["PrefetchGlobalRead"]
    perStage = tdmRingIssuesPerStage(state)
    if prologueWait is None:
        prologueWait = tdmRingPrologueWait(state)
    inFlight = collections.deque()
    fences, unsafe = [], []
    issued = peak = 0

    def issue(computing=None):
        nonlocal issued, peak
        # A main-loop fill reuses the slot of the tile whose reads its fence protected.
        assert computing is None or issued == computing + stages
        inFlight.append(issued)
        issued += 1
        peak = max(peak, len(inFlight) * perStage)

    def wait(count):
        while len(inFlight) * perStage > count:
            inFlight.popleft()

    def publish(site, tile, count):
        fences.append((site, len(inFlight)))
        wait(count)
        if tile in inFlight:
            unsafe.append((site, tile))

    counter = tiles
    issue()
    leftEarly = False
    for stage in range(1, stages):
        if counter == stage:
            leftEarly = True
            break
        issue()
    if leftEarly:
        fences.append(("EarlyExit", len(inFlight)))
        if earlyExitDrain:
            wait(0)
    publish("Prologue", 0, prologueWait)
    computing = 0
    if counter == 1:
        first = 0
    elif counter <= stages - 1:
        first = stages - 2
    elif counter <= stages:
        first = stages - 1
    else:
        publishWait = tdmRingPublishWait(state, fence)
        while counter > stages:
            if fence == "twoBarrier":
                issue(computing)
            publish("MainLoop", computing + 1, publishWait)
            if fence == "fused":
                issue(computing)
            computing += 1
            counter -= 1
        first = stages - 1
    for remainPgr in range(first, 0, -1):
        publish("NoGlobalLoadLoop_%u" % remainPgr, computing + 1,
                tdmRingNoLoadWait(state, remainPgr))
        computing += 1
        if remainPgr >= 2:
            counter -= 1
            if remainPgr < stages - 1 and counter == 1:
                break
    fences.append(("NoLoadLoop", len(inFlight)))
    wait(0)
    fences.append(("TailLoop", len(inFlight)))
    assert computing == tiles - 1, "the NoLoadLoop computes the last tile"
    return fences, unsafe, peak


_RING_REPLAYS = [
    pytest.param(stages, fence, problemType, id="s%u-%s-%s" % (stages, fence, name))
    for stages in TDM_RING_STAGES
    for fence in ("fused", "twoBarrier")
    for name, problemType in (("mx", _MX), ("dense", _DENSE))
]


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
    """Every unsafe fence as (S, tiles, site, tile), over S = 3, 4, both fences, MX and dense.
    A callable `prologueWait` is given the state."""
    races = set()
    for stages in TDM_RING_STAGES:
        for fence in ("fused", "twoBarrier"):
            for problemType in (_MX, _DENSE):
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
    the prologue's wait for the oldest of S stages returns before tile 0 lands. S=4 with 2 tiles
    also enters NoGlobalLoadLoop_2 with tile 1 in flight."""
    assert _ringRaces(earlyExitDrain=False) == \
        {(stages, tiles, "Prologue", 0) for stages in TDM_RING_STAGES
         for tiles in range(1, stages)} | {(4, 2, "NoGlobalLoadLoop_2", 1)}


def test_ring_control_flow_with_a_prologue_wait_that_retires_nothing_races():
    """Positive control: a prologue wait of S*T keeps all S stages in flight, so every K of S or
    more tiles publishes tile 0 before it lands."""
    def allStages(state):
        return state["PrefetchGlobalRead"] * tdmRingIssuesPerStage(state)
    assert _ringRaces(prologueWait=allStages) == \
        {(stages, tiles, "Prologue", 0) for stages in TDM_RING_STAGES
         for tiles in range(stages, 2 * stages + 2)}
