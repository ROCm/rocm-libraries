# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-tensor PrefetchGlobalReadA/B (DecouplePGR).

Both keys must be set or both omitted.
Auto (-1) is resolved before the DepthU candidates, so it needs a concrete DepthU.

  omitted, omitted              legacy scalar PrefetchGlobalRead
  (-1, -1) and PGR >= 2         auto: max-LDS pair, start at PrefetchGlobalRead
  (-1, -1) and PGR is 0 or 1    drop A/B, keep that scalar (no auto pair)
  (k, k) for k >= 0             PrefetchGlobalRead=k  (includes (0,0) and (1,1))
  (-1, k) / (k, -1) for k >= 1  auto over the -1 tensor with the other held at k
  (-1, 0) / (0, -1)             reject (a divergent pair at level 0 has no cadence)
  (0, 1) / (1, 0)               reject (both single-buffered)
  one key only                  reject
"""


import re
from typing import NamedTuple

from ..Common.DataType import DataType
from ..Common.Utilities import clusterEnabled, effectiveMatrixInstMN
from .TDMFuse import liveGroups, tdmGroupingSeparatesAB, tdmSeparateABDescriptors

PGR_SPECIAL_AUTO = -1
PGR_AUTO_DEFAULT_LEVEL = 2


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


def pgrAutoPairRanking(pgr, state, problemType=None, fixedA=None, fixedB=None):
    """Rank legal LDS-feasible pairs; retain successors for post-padding retries."""
    candidates = [pair for pair in pgrAutoPairCandidates(pgr)
                  if autoPairCandidateIsLegal(*pair)]
    if clusterEnabled(state.get("ClusterDim", [1, 1])):
        candidates = [pair for pair in candidates if pair[0] == pair[1]]
    if fixedA is not None:
        candidates = [pair for pair in candidates if pair[0] == fixedA]
    if fixedB is not None:
        candidates = [pair for pair in candidates if pair[1] == fixedB]
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


def pgrSpecialValueRejectReason(pgrA, pgrB):
    """Reject one-sided keys. Every both-set combination passes here."""
    if pgrA is None and pgrB is None:
        return None
    if (pgrA is None) != (pgrB is None):
        return ("PrefetchGlobalReadA/B: PrefetchGlobalReadA and PrefetchGlobalReadB must "
                "both be set or both omitted")
    return None


def pgrAutoPairRequested(state):
    """Return whether the per-tensor keys request an auto ranking."""
    pgrA = state.get("PrefetchGlobalReadA")
    pgrB = state.get("PrefetchGlobalReadB")
    pgr = state.get("PrefetchGlobalRead", 0)
    if pgrSpecialValueRejectReason(pgrA, pgrB):
        return False
    autoA = pgrA == PGR_SPECIAL_AUTO
    autoB = pgrB == PGR_SPECIAL_AUTO
    if autoA and autoB:
        return pgr not in (0, 1)
    return autoA != autoB


def resolvePrefetchGlobalReadSpecialValues(state, skip=0):
    """Resolve auto (-1); `skip` advances the ranking after an LDS refusal."""
    pgrA = state.get("PrefetchGlobalReadA")
    pgrB = state.get("PrefetchGlobalReadB")
    pgr = state.get("PrefetchGlobalRead", 0)
    reason = pgrSpecialValueRejectReason(pgrA, pgrB)
    if reason:
        return reason
    autoA = pgrA == PGR_SPECIAL_AUTO
    autoB = pgrB == PGR_SPECIAL_AUTO
    pairAuto = autoA and autoB
    oneSided = autoA != autoB
    if not (pairAuto or oneSided):
        return None
    if pairAuto and pgr in (0, 1):
        state.pop("PrefetchGlobalReadA", None)
        state.pop("PrefetchGlobalReadB", None)
        return None
    fixedA = fixedB = held = heldTc = None
    start = pgr
    if oneSided:
        if autoA:
            fixedB = held = pgrB
            heldTc = "B"
        else:
            fixedA = held = pgrA
            heldTc = "A"
        if held == 0:
            return ("PrefetchGlobalReadA/B: PrefetchGlobalRead%s=0 cannot be held while "
                    "the other tensor is auto; a divergent pair has no level 0. Use 1 "
                    "or more." % heldTc)
        # A divergent pair needs a side at 2, whatever the scalar says.
        start = max(start, held, PGR_AUTO_DEFAULT_LEVEL)
    depthU = state.get("DepthU")
    if not isinstance(depthU, int) or depthU <= 0:
        return ("PrefetchGlobalReadA/B: auto needs a concrete DepthU; DepthU=-1 picks its "
                "own candidates later. Name a DepthU, or drop the per-tensor keys.")
    ranking = pgrAutoPairRanking(start, state, state.get("ProblemType"),
                                 fixedA=fixedA, fixedB=fixedB)
    if skip >= len(ranking):
        clusterNote = ("; ClusterDim %s limits auto to equal pairs" % state["ClusterDim"]
                       if clusterEnabled(state.get("ClusterDim", [1, 1])) else "")
        if oneSided:
            return ("PrefetchGlobalReadA/B: auto found no LDS-feasible pair with "
                    "PrefetchGlobalRead%s held at %d, starting from %d; lower DepthU "
                    "or the macro tile%s" % (heldTc, held, start, clusterNote))
        return ("PrefetchGlobalReadA/B: auto found no LDS-feasible pair starting from "
                "PrefetchGlobalRead=%s; lower DepthU or the macro tile%s" % (pgr, clusterNote))
    state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"] = ranking[skip]
    return None


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
    return bool(decoupled and pgrA == pgrB and pgrA != PGR_SPECIAL_AUTO)


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


def dcpClusterRejectReason(ks):
    """Why a divergent pair cannot run at its ClusterDim, or None.

    The single-buffered tensor refills the block its own reads still use, so it
    stays self-only; the double-buffered tensor may multicast along its axis.
    """
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if numLdsBlkA == numLdsBlkB or not clusterEnabled(ks["ClusterDim"]):
        return None
    # ClusterDim[1] peers share A, ClusterDim[0] peers share B.
    thinTc, thinAxis = ("A", 1) if numLdsBlkA < numLdsBlkB else ("B", 0)
    if ks["ClusterDim"][thinAxis] != 1:
        return ("the single-buffered %s cannot have cluster peers; set ClusterDim[%u]=1"
                % (thinTc, thinAxis))
    if ks["Multicast"] and not ks["ClusterBarrier"]:
        return "multicast into the double-buffered tensor needs ClusterBarrier"
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


_DCP_CLUSTER_SIGNAL = "s_barrier_signal -3"
_DCP_CLUSTER_WAIT = "s_barrier_wait -3"
_DCP_ASM_LABEL_RE = re.compile(r"^([A-Za-z_.$][\w.$]*):")
_DCP_LC_CMP_RE = re.compile(r"^s_cmp_(eq|lg|le)_(u32|i32) s\[sgprLoopCounterL\], (?:0x0|0)$")
_DCP_LC_WRITE_RE = re.compile(r"^(\S+) s\[sgprLoopCounterL\],")
_DCP_WAVE_CMP_RE = re.compile(r"^s_(cmp_eq_u32|cmp_lg_u32|bitcmp1_b32|bitcmp0_b32) s\[sgprWaveIdx\], (\w+)$")
_DCP_LONG_JUMP_RE = re.compile(r"^s_add_i32 s\d+, (label_\w+), 4$")
_DCP_SCC_KEEPER_RE = re.compile(r"^(s_barrier|s_wait|s_nop|s_cbranch|s_branch|s_mov_b|s_cmov|s_set|"
                                r"s_sleep|s_delay|s_cselect|s_getpc|s_setpc|s_endpgm|s_load|"
                                r"s_buffer_load|s_prefetch|s_clause|s_dcache|s_icache|s_sendmsg)")


def _dcpAsmInstructions(asm):
    """(instructions, label -> instruction index) of kernel text; comments and directives dropped."""
    insts, labels, block = [], {}, None
    for raw in re.sub(r"/\*.*?\*/", " ", asm, flags=re.S).splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        if block:
            if line.startswith(block):
                block = None
            continue
        if line.startswith(".amdgpu_metadata"):
            block = ".end_amdgpu_metadata"
            continue
        if line.startswith(".amdhsa_kernel"):
            block = ".end_amdhsa_kernel"
            continue
        label = _DCP_ASM_LABEL_RE.match(line)
        if label:
            labels[label.group(1)] = len(insts)
            line = line[label.end():].strip()
            if not line:
                continue
        if not line.startswith("."):
            insts.append(" ".join(line.replace(",", ", ").split()).replace(" ,", ","))
    return insts, labels


def _dcpWaveZeroSccAfter(inst):
    """SCC wave 0 leaves after `inst` when it compares WaveIdx against a constant, else None."""
    cmp = _DCP_WAVE_CMP_RE.match(inst)
    if not cmp:
        return None
    op, value = cmp.group(1), int(cmp.group(2), 0)
    return {"cmp_eq_u32": value == 0, "cmp_lg_u32": value != 0,
            "bitcmp1_b32": False, "bitcmp0_b32": True}[op]


def _dcpWaveZeroPathProblems(insts, labels):
    """Wave 0 must alternate cluster signal and wait on every path through the kernel."""
    problems, seen, work = {}, set(), [(0, 0, "U", None)]
    while work:
        state = work.pop()
        if state in seen:
            continue
        seen.add(state)
        idx, posted, lc, scc = state
        if idx >= len(insts):
            if posted:
                problems.setdefault(("end", idx), "a cluster signal is still posted at the end")
            continue
        inst = insts[idx]
        mnemonic = inst.split(None, 1)[0]
        if inst == _DCP_CLUSTER_SIGNAL:
            if posted:
                problems.setdefault(("sig", idx), "instruction %d signals twice" % idx)
            posted = 1
        elif inst == _DCP_CLUSTER_WAIT:
            if not posted:
                problems.setdefault(("wait", idx), "instruction %d waits with no signal posted" % idx)
            posted = 0
        if mnemonic == "s_endpgm":
            if posted:
                problems.setdefault(("end", idx), "instruction %d ends with a signal posted" % idx)
            continue
        wave = _dcpWaveZeroSccAfter(inst)
        lcCmp = _DCP_LC_CMP_RE.match(inst)
        if wave is not None:
            scc = ("wave", wave)
        elif lcCmp and (lcCmp.group(1), lcCmp.group(2)) != ("le", "i32"):
            scc = ("lc", lcCmp.group(1) != "lg")
        elif mnemonic.startswith("s_") and not _DCP_SCC_KEEPER_RE.match(mnemonic):
            scc = None
        lcWrite = _DCP_LC_WRITE_RE.match(inst)
        if lcWrite and not lcWrite.group(1).startswith(("s_cmp", "s_bitcmp")):
            lc = "U"
        parts = inst.split()
        target = labels.get(parts[1]) if len(parts) > 1 else None
        if mnemonic == "s_branch":
            edges = [(target, lc)]
        elif mnemonic == "s_setpc_b64":
            jump = next((_DCP_LONG_JUMP_RE.match(insts[j]) for j in range(idx - 1, max(idx - 17, -1), -1)
                         if _DCP_LONG_JUMP_RE.match(insts[j])), None)
            edges = [(labels.get(jump.group(1)) if jump else None, lc)]
        elif mnemonic in ("s_cbranch_scc0", "s_cbranch_scc1"):
            takenIf = mnemonic == "s_cbranch_scc1"
            if scc is not None and scc[0] == "wave":
                edges = [(target, lc)] if scc[1] == takenIf else [(idx + 1, lc)]
            elif scc is not None:
                zeroWhenTaken = scc[1] == takenIf
                taken, fall = ("Z", "NZ") if zeroWhenTaken else ("NZ", "Z")
                edges = [(target, taken) if lc in ("U", taken) else (None, None),
                         (idx + 1, fall) if lc in ("U", fall) else (None, None)]
            else:
                edges = [(target, lc), (idx + 1, lc)]
        elif mnemonic.startswith("s_cbranch_"):
            edges = [(target, lc), (idx + 1, lc)]
        else:
            edges = [(idx + 1, lc)]
        for nxt, nextLc in edges:
            if nxt is None and nextLc is not None:
                problems.setdefault(("target", idx), "instruction %d jumps to an unknown label" % idx)
            elif nxt is not None:
                work.append((nxt, posted, nextLc, scc))
    return list(problems.values())


def dcpClusterHandshakeViolations(asm):
    """Why emitted kernel text breaks the decoupled-PGR cluster handshake; empty when it holds."""
    insts, labels = _dcpAsmInstructions(asm)
    joins = set(labels.values())
    problems = _dcpWaveZeroPathProblems(insts, labels)
    for i, inst in enumerate(insts):
        if not inst.startswith(("s_cbranch_scc0 ", "s_cbranch_scc1 ")) or i == 0:
            continue
        wave = _dcpWaveZeroSccAfter(insts[i - 1])
        end = labels.get(inst.split()[1], -1)
        if wave is None or end <= i:
            continue
        election = (inst.startswith("s_cbranch_scc0 ") and insts[i - 1] == "s_cmp_eq_u32 s[sgprWaveIdx], 0"
                    and end == i + 2 and insts[i + 1] == _DCP_CLUSTER_SIGNAL)
        if not election and any(t in (_DCP_CLUSTER_SIGNAL, _DCP_CLUSTER_WAIT) for t in insts[i + 1:end]):
            problems.append("instruction %d: a cluster barrier depends on the wave index" % i)

    signals = [i for i, inst in enumerate(insts) if inst == _DCP_CLUSTER_SIGNAL and i >= 2
               and insts[i - 1].startswith("s_cbranch_scc0 ") and "DcpCSigSkip" in insts[i - 1]]
    if not signals:
        problems.append("no decoupled-PGR cluster signal")
    for s in signals:
        k, stage = s - 3, "barrier"
        while k >= 0:
            inst = insts[k]
            if k + 1 in joins or inst.startswith(("s_cbranch", "s_branch", "ds_load", "ds_read")):
                problems.append("cluster signal %d: %s before its read barrier" % (s, inst.split()[0]))
                break
            if stage == "barrier" and inst == "s_barrier_wait -1":
                stage = "signal"
            elif stage == "signal" and inst == "s_barrier_signal -1":
                stage = "drain"
            elif stage == "drain" and re.match(r"^s_wait_(?:\w+_)?dscnt (?:0x0|0)$", inst):
                break
            k -= 1
        else:
            problems.append("cluster signal %d: no drained workgroup barrier ahead of it" % s)

    for b, inst in enumerate(insts):
        parts = inst.split()
        if len(parts) < 2 or not parts[0].startswith(("s_cbranch", "s_branch")):
            continue
        begin = labels.get(parts[1])
        if "LoopBegin" not in parts[1] or "TailLoop" in parts[1] or begin is None or begin >= b:
            continue
        loads = [i for i in range(begin, b) if insts[i].startswith("tensor_load_to_lds")]
        if not loads:
            continue
        mine = [s for s in signals if begin <= s < b]
        if len(mine) != 1:
            problems.append("loop %s holds %d cluster handshakes" % (parts[1], len(mine)))
            continue
        wait = next((i for i in range(mine[0], b) if insts[i] == _DCP_CLUSTER_WAIT), None)
        if wait is None or any(i > wait for i in loads):
            problems.append("loop %s issues a tensor load after its cluster wait" % parts[1])
    return problems


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
