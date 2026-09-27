# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-tensor PrefetchGlobalReadA/B (DecouplePGR).

Both keys must be set or both omitted. Omitting both is auto.

  omitted, omitted              auto: the legacy scalar PrefetchGlobalRead kernel first;
                                a divergent pair only if that kernel exceeds LDS and
                                pgrAutoIneligibleReason is None (gfx1250, TDMInst=3,
                                ScheduleIterAlg 0 or 4, PGR >= 2, concrete DepthU)
  (k, k) for k >= 0             PrefetchGlobalRead=k  (includes (0,0) and (1,1))
  (0, k) / (k, 0) for k >= 1    reject (a divergent pair at level 0 has no cadence)
  (0, 1) / (1, 0)               reject (both single-buffered)
  one key only                  reject
"""


import re
from typing import NamedTuple

from ..Common.DataType import DataType
from ..Common.Utilities import effectiveMatrixInstMN
from .TDMFuse import liveGroups, tdmBothTensors, tdmGroupingSeparatesAB, \
                     tdmSeparateABDescriptors


def pgrAutoPairCandidates(pgr):
    """Pairs from (`pgr`, `pgr`) down to (2,2), then (2,1)/(1,2)."""
    if pgr < 2:
        return []
    candidates = []
    for level in range(pgr, 1, -1):
        candidates.append((level, level))
    # Divergent pairs support at most two LDS blocks.
    candidates.append((2, 1))
    candidates.append((1, 2))
    return candidates


DCP_MAX_LDS_BLOCKS_DIVERGENT = 2


# gfx1250 DeviceLDS fallback when ISA data is unavailable here.
DCP_DEFAULT_MAX_LDS = 327680


def _macroTileFromMIGeometry(instM, instN, instBM, instBN, miWaveTile, miWaveGroup,
                             wavefrontSize):
    """Return MacroTile from MI geometry; MIBlock[0] == 4 needs unavailable ISA data."""
    for value in (instM, instN, instBM, instBN, wavefrontSize):
        if not isinstance(value, int) or value <= 0:
            return None
    if wavefrontSize % instN or (instM * instN) % wavefrontSize:
        return None
    threadTile0 = instBM * miWaveTile[0] * (instM * instN // wavefrontSize)
    threadTile1 = instBN * miWaveTile[1]
    subGroup0 = miWaveGroup[0] * (wavefrontSize // instN)
    subGroup1 = miWaveGroup[1] * instN
    return subGroup0 * threadTile0, subGroup1 * threadTile1


def macroTileFromMatrixInstruction(mi, wavefrontSize):
    """Return MacroTile after distributing MatrixInstB into MIBlock and MIWaveGroup."""
    if not isinstance(mi, (list, tuple)) or len(mi) < 9:
        return None
    if not isinstance(wavefrontSize, int) or wavefrontSize <= 0:
        return None
    if mi[0] == 4 or mi[0] <= 0 or mi[3] <= 0:
        return None
    waves = mi[7] * mi[8]
    wg0 = mi[4] * mi[0] * mi[7]
    if waves <= 0 or wg0 <= 0 or wg0 // mi[0] <= 0:
        return None
    instBM = min(wg0 // mi[0], mi[3])
    if instBM <= 0:
        return None
    instBN = mi[3] // instBM
    miwg0 = min((wg0 // mi[0]) // instBM, waves)
    if miwg0 <= 0:
        return None
    return _macroTileFromMIGeometry(mi[0], mi[1], instBM, instBN,
                                    (mi[5], mi[6]), (miwg0, waves // miwg0),
                                    wavefrontSize)


def autoPairCandidateIsLegal(pgrA, pgrB):
    """Return whether a pair is legal; filter before LDS ranking."""
    if pgrA == pgrB:
        return True
    if min(pgrA, pgrB) == 0:
        return False
    return max(ldsBlocksForPgrLevel(pgrA),
               ldsBlocksForPgrLevel(pgrB)) <= DCP_MAX_LDS_BLOCKS_DIVERGENT


def _asDataType(value):
    """DataType, or None if missing / not a name the DataType constructor accepts."""
    if value is None:
        return None
    if isinstance(value, DataType):
        return value
    try:
        return DataType(value)
    except Exception:
        return None


def _nonNegOrZero(value):
    """Treat unresolved padding as 0; auto LDS sizes are candidate-dependent lower bounds."""
    if value is None or value < 0:
        return 0
    return int(value)


def _ldsBytesAligned(depthU, macroTile, bpe, ldsPad=0, padInterval=0, align=64,
                     unrollMajor=False):
    """Same size math as calcLdsNumBytesAB."""
    if padInterval:
        raw = int(depthU * macroTile * bpe / padInterval * (padInterval + ldsPad * bpe))
    elif unrollMajor:
        raw = int((depthU + ldsPad) * macroTile * bpe)
    else:
        raw = int(depthU * (macroTile + ldsPad) * bpe)
    return (raw + align - 1) // align * align if align > 0 else raw


def _roundUpPow2(value):
    """Smallest power of two at or above `value`; 0 stays 0."""
    if value <= 0:
        return 0
    return 1 << (value - 1).bit_length()


def _maxLdsOrDefault(ks):
    """MaxLDS, or the default cap when it is missing or still -1."""
    maxLds = ks.get("MaxLDS", DCP_DEFAULT_MAX_LDS)
    if maxLds is None or maxLds < 0:
        return DCP_DEFAULT_MAX_LDS
    return maxLds


def _ldsAlignedBytes(ks, pt, mxTc, depthU, macroTile):
    """Aligned LDS bytes for A/B/MXSA/MXSB, matching calcLdsNumBytesAB."""
    if ks.get("DirectToVgpr%s" % mxTc):
        return 0
    tc = mxTc.replace("MXS", "")
    mxBlock = pt.get("MXBlock%s" % tc, 0) or 0
    if "MXS" in mxTc:
        if not mxBlock:
            return 0
        depthU = depthU // mxBlock
    mac = (_asDataType(pt.get("MacDataType%s" % tc))
           or _asDataType(pt.get("DataType%s" % tc))
           or _asDataType(pt.get("DataType")))
    if mac is None:
        return None
    if "MXS" in mxTc:
        bpe = 1
    elif ks.get("ConvertAfterDS"):
        bpe = (_asDataType(pt.get("DataType%s" % tc)) or mac).numBytes()
    else:
        bpe = mac.numBytes()
    align = 64 if mac.is6bitFloat() else int(64 / mac.numRegisters())
    return _ldsBytesAligned(
        depthU, macroTile, bpe,
        ldsPad=_nonNegOrZero(ks.get("LdsPad%s" % mxTc)),
        padInterval=_nonNegOrZero(ks.get("LdsBlockSizePerPad%s" % mxTc)),
        align=align,
        unrollMajor=ks.get("UnrollMajorLDS%s" % mxTc) in (1, True),
    )


def decouplePGRLdsBytesEstimate(ks, problemType=None):
    """Estimate candidate LDS bytes using the derived layout.

    Padding is unresolved and tail B is aligned, so the result is not exact.
    """
    depthU = ks["DepthU"]
    mt0 = ks["MacroTile0"]
    mt1 = ks["MacroTile1"]
    pt = problemType if problemType is not None else (ks.get("ProblemType") or {})

    ldsA = _ldsAlignedBytes(ks, pt, "A", depthU, mt0)
    ldsB = _ldsAlignedBytes(ks, pt, "B", depthU, mt1)
    if ldsA is None or ldsB is None:
        return None
    ldsMXSA = _ldsAlignedBytes(ks, pt, "MXSA", depthU, mt0)
    ldsMXSB = _ldsAlignedBytes(ks, pt, "MXSB", depthU, mt1)
    if ldsMXSA is None or ldsMXSB is None:
        return None

    _, nBlkA, nBlkB = decouplePGRBlocks(ks)
    if nBlkA != nBlkB:
        # setLdsOffsetsDecoupled: nBlkA x [A|MXSA] then nBlkB x [MXSB|B], packed.
        return nBlkA * (ldsA + ldsMXSA) + nBlkB * (ldsMXSB + ldsB)

    # setLdsOffsets: no metadata segment (Sparse is rejected), and one block is
    # sized as two.
    offsetB = ldsA + ldsMXSA + ldsMXSB
    offsetBlk = offsetB + ldsB
    if nBlkA == 2 and offsetBlk + _roundUpPow2(offsetBlk) <= _maxLdsOrDefault(ks):
        # The two-block xor swap rounds the block up, but only while the
        # rounded-up pair still fits -- the StoreSwapAddr test.
        offsetBlk = _roundUpPow2(offsetBlk)
    return (max(nBlkA, 2) - 1) * offsetBlk + offsetB + ldsB


def _macroTileFromState(state):
    """Resolve MacroTile from derived fields, MI fields, or MatrixInstruction."""
    mt0 = state.get("MacroTile0")
    mt1 = state.get("MacroTile1")
    if mt0 is not None and mt1 is not None:
        return mt0, mt1
    wavefrontSize = state.get("WavefrontSize")
    miBlock = state.get("MIBlock")
    miWaveTile = state.get("MIWaveTile")
    miWaveGroup = state.get("MIWaveGroup")
    if (isinstance(miBlock, (list, tuple)) and len(miBlock) == 6
            and isinstance(miWaveTile, (list, tuple)) and len(miWaveTile) == 2
            and isinstance(miWaveGroup, (list, tuple)) and len(miWaveGroup) == 2):
        if miBlock[0] == 4:
            return None
        instM, instN = effectiveMatrixInstMN(miBlock[0], miBlock[1],
                                             state.get("SourceSwap", False))
        return _macroTileFromMIGeometry(instM, instN, miBlock[4], miBlock[5],
                                        miWaveTile, miWaveGroup, wavefrontSize)
    mi = state.get("MatrixInstruction")
    if mi is not None:
        return macroTileFromMatrixInstruction(mi, wavefrontSize)
    return None


def _localReadWork(state):
    """Return per-thread A/B local-read elements for tie-breaking."""
    miWaveTile = state.get("MIWaveTile")
    inputA = state.get("MIInputPerThreadA")
    inputB = state.get("MIInputPerThreadB")
    if not (isinstance(miWaveTile, (list, tuple)) and len(miWaveTile) == 2):
        return None
    if not isinstance(inputA, int) or not isinstance(inputB, int):
        return None
    return miWaveTile[0] * inputA, miWaveTile[1] * inputB


def _thickSideRank(pair, state):
    """Prefer more local-read work on the thick side, then A-thick on exact ties."""
    blocksA = ldsBlocksForPgrLevel(pair[0])
    blocksB = ldsBlocksForPgrLevel(pair[1])
    if blocksA == blocksB:
        # Nothing is relocated. Top rank, so a tie never displaces an equal pair.
        return True, True
    thickIsA = blocksA > blocksB
    work = _localReadWork(state)
    placed = work is None or work[0] == work[1] or (work[0] > work[1]) == thickIsA
    return placed, thickIsA


def pgrAutoPairRanking(pgr, state, problemType=None):
    """Rank legal LDS-feasible pairs; retain successors for post-padding retries."""
    candidates = [pair for pair in pgrAutoPairCandidates(pgr)
                  if autoPairCandidateIsLegal(*pair)]
    if not candidates:
        return []
    macroTile = _macroTileFromState(state)
    if macroTile is None:
        return []
    depthU = state.get("DepthU")
    if not isinstance(depthU, int) or depthU <= 0:
        return []
    pt = problemType if problemType is not None else state.get("ProblemType")
    maxLds = _maxLdsOrDefault(state)
    probe = dict(state)
    probe["MacroTile0"], probe["MacroTile1"] = macroTile
    probe["DepthU"] = depthU
    ranked = []
    for pair in candidates:
        probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = pair
        lds = decouplePGRLdsBytesEstimate(probe, pt)
        if lds is None or lds > maxLds:
            continue
        ranked.append(((lds, _thickSideRank(pair, state)), pair))
    # Stable sort: a residual tie keeps pgrAutoPairCandidates' order.
    ranked.sort(key=lambda scored: scored[0], reverse=True)
    return [pair for _, pair in ranked]


def pgrPairKeysRejectReason(pgrA, pgrB):
    """Reject one-sided keys. Every both-set combination passes here."""
    if pgrA is None and pgrB is None:
        return None
    if (pgrA is None) != (pgrB is None):
        return ("PrefetchGlobalReadA/B: PrefetchGlobalReadA and PrefetchGlobalReadB must "
                "both be set or both omitted")
    return None


def pgrAutoIneligibleReason(state):
    """Why auto has no divergent candidate for this input state, or None.

    Only states develop can emit a divergent pair for qualify; anything else
    keeps the legacy scalar kernel and never enters the retry.
    """
    if state.get("PrefetchGlobalReadA") is not None \
            or state.get("PrefetchGlobalReadB") is not None:
        return "PrefetchGlobalReadA/B are set, so auto does not apply"
    if tuple(state.get("ISA", ()))[:2] != (12, 5):
        return "divergent pairs are gfx1250 only"
    if not tdmBothTensors(state):
        return "divergent pairs need TDMInst=3"
    if state.get("ScheduleIterAlg") not in (0, 4):
        return "divergent pairs need ScheduleIterAlg 0 or 4"
    pgr = state.get("PrefetchGlobalRead", 0)
    if not isinstance(pgr, int) or pgr < 2:
        return "PrefetchGlobalRead below 2 has no divergent pair"
    depthU = state.get("DepthU")
    if not isinstance(depthU, int) or depthU <= 0:
        return "auto needs a concrete DepthU"
    return None


def pgrAutoPairRequested(state):
    """Return whether auto may retry a divergent pair after an LDS refusal."""
    return pgrAutoIneligibleReason(state) is None


def pgrAutoDivergentRanking(state, problemType=None):
    """Divergent auto candidates, best first; the equal pair is the legacy kernel."""
    pt = problemType if problemType is not None else state.get("ProblemType")
    return [pair for pair in pgrAutoPairRanking(state.get("PrefetchGlobalRead", 0), state, pt)
            if pair[0] != pair[1]]


def resolvePrefetchGlobalReadAuto(state, attempt):
    """Pin the `attempt`-th divergent pair; False once the ranking is exhausted."""
    ranking = pgrAutoDivergentRanking(state)
    if attempt >= len(ranking):
        return False
    state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"] = ranking[attempt]
    return True


def pgrLevelsForTensors(ks):
    """Return (decoupled, pgrA, pgrB), using scalar PGR when keys are absent."""
    pgr = ks.get("PrefetchGlobalRead", 0)
    pgrA = ks.get("PrefetchGlobalReadA")
    pgrB = ks.get("PrefetchGlobalReadB")
    if pgrA is None and pgrB is None:
        return False, pgr, pgr
    return True, pgrA, pgrB


def ldsBlocksForPgrLevel(pgr):
    """LDS blocks for one per-tensor level; levels 0 and 1 both use one."""
    if pgr <= 1:
        return 1
    return pgr


DCP_LDS_SIDE = {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}


def dcpLdsSide(tc):
    """Return the fixed LDS side: [A|MXSA] or [MXSB|B].

    This is layout stride, not descriptor ownership.
    """
    return DCP_LDS_SIDE[tc]


def decouplePGRBlocks(ks):
    """Return (decoupled, numLdsBlkA, numLdsBlkB) without serializing derived state."""
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    return decoupled, ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB)


def equalPairDegeneratesToScalar(ks):
    """True when both per-tensor levels are the same real depth (including 0 and 1)."""
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    return bool(decoupled and pgrA == pgrB)


def divergentPairUnsupportedReason(ks):
    """Why a divergent pair cannot relocate its single-buffered fill, or None."""
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    if decoupled and min(pgrA, pgrB) == 0:
        # 0 and 1 both map to one block, so (0, N) would ship (1, N)'s
        # instructions under a second kernel name.
        return "level 0 has no distinct per-tensor cadence; use level 1 or higher"
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if max(numLdsBlkA, numLdsBlkB) > DCP_MAX_LDS_BLOCKS_DIVERGENT:
        return ("divergent pairs support at most two LDS blocks per tensor (A=%u, B=%u)"
                % (numLdsBlkA, numLdsBlkB))
    if ks["_ScheduleIterAlg"] != 0:
        # Only noSchedGlobalRead leaves a whole globalRead module to re-slot.
        return ("_ScheduleIterAlg=%u leaves no complete fill group to re-slot; use "
                "ScheduleIterAlg=0 or 4" % ks["_ScheduleIterAlg"])
    if ks["PrefetchLocalRead"] < 1:
        return "PrefetchLocalRead=0 leaves no late-fill slot; use 1 or higher"
    loopIters = ks["DepthU"] // ks["LocalSplitU"] // ks["InnerUnroll"]
    if ks.get("EnableMatrixInstruction", True):
        loopIters //= ks["MatrixInstK"]
    if ks["PrefetchLocalRead"] % loopIters == 0:
        return ("PrefetchLocalRead=%u is a multiple of LoopIters=%u; no late-fill slot "
                "remains" % (ks["PrefetchLocalRead"], loopIters))
    if ks["NumWaves"] <= 1:
        return "wave-separated TDM requires NumWaves > 1; got %u" % ks["NumWaves"]
    return None


def dcpThickThinIssueOrder(items):
    """Order items thickest-first; stable ties preserve tensor-count age order."""
    return tuple(item for _, item in sorted(items, key=lambda pair: -pair[0]))


def decoupledSingleBuffered(ks):
    """True when exactly one tensor has one LDS block and needs a refill barrier."""
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return decoupled and min(numLdsBlkA, numLdsBlkB) == 1 and max(numLdsBlkA, numLdsBlkB) > 1


def decoupledOneBlockBoth(ks):
    """True when both tensors occupy one LDS block in a per-tensor prefetch loop."""
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return decoupled and max(numLdsBlkA, numLdsBlkB) == 1 and bool(ks["PrefetchGlobalRead"])


def tdmWaveIssueOrder(ks, itemA, itemB):
    """Return A/B items in descending per-tensor LDS-block order."""
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return dcpThickThinIssueOrder(((numLdsBlkA, itemA), (numLdsBlkB, itemB)))


DCP_THICK_GATE_TOKENS = "tokens"


DCP_THICK_GATE_TEXT = "text"


def _dcpTokensGateSupported(_ks):
    """Separate descriptors leave one thick-tensor fill outstanding."""
    return 1


def _dcpTextGateSupported(ks):
    """Count live descriptor sets written by one fill; over-counting under-waits."""
    return len(liveGroups(ks))


# Tensor ops one fill leaves outstanding under each mechanism. Not a tunable: a
# count above that retires the gate with the previous fill still outstanding.
DCP_THICK_GATE_SUPPORTED = {DCP_THICK_GATE_TEXT: _dcpTextGateSupported,
                            DCP_THICK_GATE_TOKENS: _dcpTokensGateSupported}


class DcpThickGate(NamedTuple):
    """How a divergent decoupled pair's thick-tensor gate gets relaxed."""
    mechanism: str
    tensorcnt: int


def decoupledThickGateRelaxation(ks):
    """Return the resolved grouping's thick-tensor gate relaxation.

    N leaves N ops outstanding (larger is weaker); TOKENS is inserted and TEXT rewritten.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if not (decoupled and numLdsBlkA != numLdsBlkB):
        return None
    # The count is earned by skipping the thin side's one refill; a thin side
    # holding two blocks leaves nothing to skip.
    if min(numLdsBlkA, numLdsBlkB) != 1:
        return None
    if tdmGroupingSeparatesAB(ks):
        if not tdmSeparateABDescriptors(ks):
            return None
        mechanism = DCP_THICK_GATE_TOKENS
    else:
        mechanism = DCP_THICK_GATE_TEXT
    return DcpThickGate(mechanism, DCP_THICK_GATE_SUPPORTED[mechanism](ks))


def dcpThickGateFromTokenPasses(ks):
    """True when barrier rebuilding and wait-count insertion emit the relaxation."""
    gate = decoupledThickGateRelaxation(ks)
    return gate is not None and gate.mechanism == DCP_THICK_GATE_TOKENS


# Public: KernelWriter._dcpRelaxThickTextGate matches on it too.
DCP_TENSORCNT_RE = re.compile(r"^s_wait_tensorcnt\s+(\d+)(?:\s|$)")


_DCP_LDS_READ_RE = re.compile(r"^ds_(?:load|read)\w*\s")


def dcpIsFillLabel(line):
    """True for a DcpEarlyFill/DcpLateFill scan boundary."""
    return (("DcpEarlyFill" in line or "DcpLateFill" in line)
            and line.rstrip().endswith(":"))


def dcpThickGateUncoveredSites(lines, marker, relaxed, accepted):
    """Return thick-fill sites missing a sufficient gate before the next LDS read."""
    sites = [i for i, line in enumerate(lines)
             if marker in line and line.rstrip().endswith(":")]
    if not sites:
        return [(-1, "no %s label was emitted at all" % marker)]
    uncovered = []
    for i in sites:
        why = None
        for j in range(i + 1, len(lines)):
            candidate = lines[j]
            if dcpIsFillLabel(candidate):
                break
            if _DCP_LDS_READ_RE.match(candidate):
                why = ("reaches %s at line %d before any s_wait_tensorcnt"
                       % (candidate.split()[0], j))
                break
            gate = DCP_TENSORCNT_RE.match(candidate)
            if gate:
                if j not in accepted:
                    why = ("first gate is s_wait_tensorcnt %s at line %d, weaker "
                           "than the %d this pass relaxes to"
                           % (gate.group(1), j, relaxed))
                break
        if why:
            uncovered.append((i, why))
    return uncovered
