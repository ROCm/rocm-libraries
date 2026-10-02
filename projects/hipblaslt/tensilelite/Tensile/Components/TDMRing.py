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


def tdmDeepRing(ks):
    """True when TDM moves both A and B and a side prefetches three or more stages."""
    if not (ks.get("enableTDMA") and ks.get("enableTDMB")):
        return False
    _, pgrA, pgrB = pgrLevelsForTensors(ks)
    return max(pgrA, pgrB) >= 3


def _tdmIssueCopies(ks, tc):
    """One TDM per tensor; TDMSplit halves a data tensor into two."""
    pt = ks.get("ProblemType") or {}
    if ks.get("TDMSplit") and "MXS" not in tc and not pt.get("Sparse"):
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
