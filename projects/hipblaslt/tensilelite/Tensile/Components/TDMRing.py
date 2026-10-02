# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Deep TDM LDS rings (Decouple 4LDSBuffer): PrefetchGlobalRead[A/B] 3-4 on the TDM path.

A side prefetching S >= 3 stages rotates its TDM writes through S LDS slots. This is
independent of TDMPlusLdsBuf (3LDSB), which adds a third LDS buffer at PGR2 and still
keeps two stages in flight; that switch and its code paths are not shared with the
ring.

The per-wave in-flight bound covers every TDM configuration: gfx1250 B0 and gfx1251
TDM can deadlock with more than TDM_INFLIGHT_PER_WAVE_LIMIT operations outstanding
from one wave.
"""

from .DecouplePGR import dcpLdsSide, decouplePGRBlocks, pgrLevelsForTensors
from .TDMFuse import tdmGrouping, tdmMemberIsLive, tdmWavePartition, tdmWaveSeparated

TDM_INFLIGHT_PER_WAVE_LIMIT = 11
TDM_RING_STAGES = (3, 4)
TDM_RING_FENCE_SHAPES = ("auto", "fused", "twoBarrier")


def tdmDeepRing(ks):
    """True when TDM moves both A and B and a side prefetches three or more stages."""
    if not (ks.get("enableTDMA") and ks.get("enableTDMB")):
        return False
    _, pgrA, pgrB = pgrLevelsForTensors(ks)
    return max(pgrA, pgrB) >= 3


def _tdmIssueCopies(ks, tc):
    """One TDM per tensor; TDMSplit halves a data tensor, sparse or not, into two."""
    if ks.get("TDMSplit") and "MXS" not in tc and tc != "Metadata":
        return (tc, tc)
    return (tc,)


def tdmWaveIssueMembers(ks):
    """Per wave, the tensor behind each TDM it issues for one prefetch stage.

    Wave-separated, every wave issues each live set once, for the member its share
    names (counted even when that member is absent); sparse metadata rides the even
    waves. A single wave issues every live tensor itself.
    """
    metadata = ("Metadata",) if ks.get("enableTDMMetadata") else ()
    if not tdmWaveSeparated(ks):
        perWave = [[tc for tc in ("A", "MXSA", "MXSB", "B") if tdmMemberIsLive(ks, tc)]
                   + list(metadata)]
    else:
        perWave = []
        for wave in range(ks.get("NumWaves", 1)):
            members = []
            for group in tdmGrouping(ks).groups:
                if any(tdmMemberIsLive(ks, tc) for tc in group):
                    members += [tc for tc in group if wave in tdmWavePartition(ks, tc)[1]]
            if metadata and wave in tdmWavePartition(ks, "Metadata")[1]:
                members += metadata
            perWave.append(members)
    return tuple(tuple(issue for tc in members for issue in _tdmIssueCopies(ks, tc))
                 for members in perWave)


def tdmLdsBlocksBySide(ks):
    """LDS blocks each side's TDM writes rotate through: [A|MXSA] and [MXSB|B]."""
    decoupled, blkA, blkB = decouplePGRBlocks(ks)
    if decoupled and blkA != blkB:
        return {"A": blkA, "B": blkB}
    return {"A": ks["NumLdsBlk"], "B": ks["NumLdsBlk"]}


def tdmStagesBySide(ks):
    """TDM stages each side can have in flight: its LDS blocks, at most its prefetch level.

    TDMPlusLdsBuf rotates three blocks but still prefetches only two stages.
    """
    blocks = tdmLdsBlocksBySide(ks)
    _, pgrA, pgrB = pgrLevelsForTensors(ks)
    return {"A": max(1, min(blocks["A"], pgrA)), "B": max(1, min(blocks["B"], pgrB))}


def tdmInflightPerWaveBound(ks):
    """Most TDM operations one wave can have in flight, over every wave role.

    A role issuing T TDMs per stage, into sets with S stages in flight prefetched P
    deep, has at most max(sum S, L_exit + T, sum P + T with PAP, sum S + sum P with
    HalfPLR and PAP) in flight. L_exit is 0 behind a NoLoadLoop and sum S - 1
    without one; NoTailLoop drops that term. 0 unless TDM moves both A and B.
    """
    if not (ks.get("enableTDMA") and ks.get("enableTDMB")):
        return 0
    stages = tdmStagesBySide(ks)
    _, pgrA, pgrB = pgrLevelsForTensors(ks)
    levels = {"A": pgrA, "B": pgrB}
    noLoadLoop = not (ks.get("SuppressNoLoadLoop") or ks.get("HalfPLR")
                      or ks.get("ReuseAcrossPersistent"))
    bound = 0
    for issues in tdmWaveIssueMembers(ks):
        # Sparse rules out the divergent layout, so metadata sees equal sides.
        sides = ["A" if tc == "Metadata" else dcpLdsSide(tc) for tc in issues]
        perStage = len(sides)
        inRing = sum(stages[side] for side in sides)
        prefetched = sum(levels[side] for side in sides)
        terms = [inRing]
        if not ks.get("NoTailLoop"):
            terms.append((0 if noLoadLoop else inRing - 1) + perStage)
        if ks.get("PrefetchAcrossPersistent"):
            terms.append(prefetched + perStage)
            if ks.get("HalfPLR"):
                terms.append(inRing + prefetched)
        bound = max([bound] + terms)
    return bound


def tdmInflightRejectReason(ks):
    """Why the per-wave TDM in-flight limit rejects `ks`, or None.

    Existing solutions keep at most two unsplit TDM stages in flight, so the limit
    binds only on deep rings and TDMSplit.
    """
    bound = tdmInflightPerWaveBound(ks)
    if bound <= TDM_INFLIGHT_PER_WAVE_LIMIT:
        return None
    if not (tdmDeepRing(ks) or ks.get("TDMSplit")):
        return None
    blocks = tdmLdsBlocksBySide(ks)
    return ("TDM in-flight: a wave can have up to %u TDM operations outstanding, over the "
            "per-wave limit of %u (LDS blocks A=%u, B=%u; PrefetchGlobalRead=%u; NumWaves=%u)"
            % (bound, TDM_INFLIGHT_PER_WAVE_LIMIT, blocks["A"], blocks["B"],
               ks["PrefetchGlobalRead"], ks["NumWaves"]))


def tdmRingIssuesPerStage(ks):
    """TDMs every wave issues per ring stage, or None when wave roles issue different counts.

    The ring's tensorcnt immediates are wave-uniform and count exactly this many per stage:
    a count above what a wave issues would leave that wave's oldest stage unretired. The
    ring takes MX on both sides or neither, where the per-wave count is exact.
    """
    counts = {len(issues) for issues in tdmWaveIssueMembers(ks)}
    return counts.pop() if len(counts) == 1 else None


def tdmRingFenceShape(override):
    """Main-loop fence of a deep ring for the TDMRingFenceOverride global parameter.

    "auto" and "fused" give the merged fence: one barrier publishes the next tile and
    protects the slot refilled right after it. "twoBarrier" protects, fills, then publishes.
    """
    if override not in TDM_RING_FENCE_SHAPES:
        raise ValueError("TDMRingFenceOverride must be one of %s, got %r"
                         % (", ".join(TDM_RING_FENCE_SHAPES), override))
    return "twoBarrier" if override == "twoBarrier" else "fused"


def tdmRingPrologueWait(ks):
    """tensorcnt each wave waits for before the prologue barrier publishes tile 0.

    The prologue issues all S stages first, so only tile 0 must land. An early exit, with fewer
    than S tiles, drains at skipPGR{S}_1 before it reaches this wait.
    """
    return (ks["PrefetchGlobalRead"] - 1) * tdmRingIssuesPerStage(ks)


def tdmRingPublishWait(ks, fenceShape):
    """tensorcnt each wave waits for before the main-loop publish barrier of body i.

    Tiles i+1 .. i+S-1 are in flight at the merged fence, which fills tile i+S after its
    barrier; the two-barrier variant has issued tile i+S as well. Only tile i+1 must land.
    """
    stages = ks["PrefetchGlobalRead"]
    inFlight = stages - 1 if fenceShape == "fused" else stages
    return (inFlight - 1) * tdmRingIssuesPerStage(ks)


def tdmRingNoLoadWait(ks, remainPgr):
    """tensorcnt before the publish barrier of the NGLL that has remainPgr stages in flight."""
    return (remainPgr - 1) * tdmRingIssuesPerStage(ks)


def _tdmRingItersPLR(ks):
    """Sub-iterations the local-read prefetch runs ahead, as KernelWriter derives numItersPLR."""
    loopIters = ks.get("LoopIters", 0)
    inputPerThread = ks.get("MIInputPerThread", 0)
    readWidth = ks.get("LocalReadVectorWidthA", 0)
    if ks.get("EnableMatrixInstruction") and inputPerThread and readWidth >= inputPerThread:
        loopIters //= max(readWidth // inputPerThread, 1)
    return ks.get("PrefetchLocalRead", 0) % loopIters if loopIters else 0


_TDM_RING_UNSUPPORTED_FLAGS = (
    "StreamK", "PrefetchAcrossPersistent", "ReuseAcrossPersistent", "HalfPLR",
    "SuppressNoLoadLoop", "UseSubtileImpl", "TDMSplit", "PrefetchGL2",
    "UnrollLoopSwapGlobalReadOrder", "UseCustomMainLoopSchedule", "ForceUnrollSubIter",
    "UsePLRPack", "ExpertSchedulingMode", "TDMPlusLdsBuf",
)
# Auto values Solution has not resolved yet when it checks the ring (in depthUIteration);
# it resolves each of them off on a deep ring afterwards.
_TDM_RING_AUTO_OFF = {"UseCustomMainLoopSchedule": -1}
_TDM_RING_DIRECT_KEYS = (
    "DirectToLdsA", "DirectToLdsB", "DirectToVgprA", "DirectToVgprB",
    "DirectToLdsMXSA", "DirectToLdsMXSB", "DirectToVgprMXSA", "DirectToVgprMXSB",
)


def _tdmRingFlagOn(ks, key):
    """True when `key` is on in `ks`; the auto value of a flag the ring resolves off is off."""
    value = ks.get(key, 0)
    if key in _TDM_RING_AUTO_OFF and value == _TDM_RING_AUTO_OFF[key]:
        return False
    return value not in (0, False)


def tdmRingRejectReason(ks):
    """Why the deep TDM ring does not support `ks` yet, or None.

    Supported: the equal-depth ring at PrefetchGlobalRead 3-4 under ScheduleIterAlg 0, whose
    fences and waits the writer emits itself, with TDM A and B on two or more waves,
    TDMFuse 0 and MX on both sides or neither. None when `ks` is not a deep ring.
    """
    if not tdmDeepRing(ks):
        return None
    _, pgrA, pgrB = pgrLevelsForTensors(ks)
    pt = ks.get("ProblemType") or {}
    stages = ks["PrefetchGlobalRead"]

    def unsupported():
        if pgrA != pgrB:
            yield "PrefetchGlobalReadA=%u with PrefetchGlobalReadB=%u" % (pgrA, pgrB)
        if stages not in TDM_RING_STAGES:
            yield "PrefetchGlobalRead=%u (the ring has 3 or 4 slots)" % stages
        if any(ks.get(key) for key in _TDM_RING_DIRECT_KEYS):
            yield "DirectToLds or DirectToVgpr"
        if not tdmWaveSeparated(ks):
            yield "NumWaves=%u (the ring needs wave-separated TDM)" % ks.get("NumWaves", 1)
        if ks.get("_ScheduleIterAlg") != 0 or ks.get("_StinkyTofuOptLevel", 0) != 0:
            yield ("ScheduleIterAlg=%s (only 0, where the writer emits the ring waits)"
                   % ks.get("ScheduleIterAlg"))
        if ks.get("PrefetchLocalRead", 0) < 1 or _tdmRingItersPLR(ks) == 0:
            yield ("PrefetchLocalRead=%u with LoopIters=%u (the next tile is read after the "
                   "fence, so the local-read prefetch must run ahead within the tile)"
                   % (ks.get("PrefetchLocalRead", 0), ks.get("LoopIters", 0)))
        if ks.get("1LDSBuffer") == 1:
            yield "1LDSBuffer=1"
        flags = [key for key in _TDM_RING_UNSUPPORTED_FLAGS if _tdmRingFlagOn(ks, key)]
        if flags:
            yield ", ".join(flags)
        if any(dim != 1 for dim in (ks.get("ClusterDim") or [1, 1])):
            yield "ClusterDim != [1, 1]"
        if pt.get("Sparse"):
            yield "Sparse"
        # Solution resolves the auto -1 off on a ring before this check; only 1 is on.
        if ks.get("LDSSegmentInterleave") == 1:
            yield "LDSSegmentInterleave=1"
        if ks.get("TDMFuse", 0) != 0:
            yield "TDMFuse=%u (only 0)" % ks.get("TDMFuse", 0)
        if bool(pt.get("MXBlockA")) != bool(pt.get("MXBlockB")):
            yield "MX on one side only"
        if tdmRingIssuesPerStage(ks) is None:
            yield "wave roles issuing different TDM counts per stage"

    what = next(unsupported(), None)
    if what is None:
        return None
    return "TDM LDS ring (PrefetchGlobalRead=%u): %s is not supported yet" % (stages, what)
