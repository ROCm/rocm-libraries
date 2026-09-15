# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Per-tensor PrefetchGlobalReadA/B (DecouplePGR).

Both keys must be set or both omitted.

  omitted, omitted              legacy scalar PrefetchGlobalRead
  (-1, -1) and PGR >= 2         auto: max-LDS pair, start at PrefetchGlobalRead
  (-1, -1) and PGR == -1        auto: same, start at 2
  scalar PGR == -1 (no keys)    auto: start at 2
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
from ..Common.Utilities import effectiveMatrixInstMN
from .TDMFuse import tdmGroupingSeparatesAB, tdmSeparateABDescriptors

PGR_SPECIAL_AUTO = -1
PGR_AUTO_DEFAULT_LEVEL = 2


def pgrAutoStartLevel(pgr):
    """Candidate ceiling for auto. Missing / -1 means PGR_AUTO_DEFAULT_LEVEL."""
    if pgr is None or pgr == PGR_SPECIAL_AUTO:
        return PGR_AUTO_DEFAULT_LEVEL
    return pgr


def pgrAutoPairCandidates(pgr):
    """(pgrA, pgrB) from `pgr` down through (2,2), then (2,1)/(1,2). Empty if pgr < 2.

    (0,0)/(1,1) are scalar PGR, not auto. (0,1)/(1,0) are unsupported.
    """
    if pgr < 2:
        return []
    candidates = []
    for level in range(pgr, 1, -1):
        candidates.append((level, level))
    # Only the bottom pair can diverge: DCP_MAX_LDS_BLOCKS_DIVERGENT caps a
    # divergent pair at two blocks, so (N, N-1) above it is always refused.
    candidates.append((2, 1))
    candidates.append((1, 2))
    return candidates


DCP_MAX_LDS_BLOCKS_DIVERGENT = 2


# gfx1250's DeviceLDS. Stands in for an unset MaxLDS, which assignDerivedParameters
# resolves from the ISA info map this module cannot reach.
DCP_DEFAULT_MAX_LDS = 327680


def _macroTileFromMIGeometry(instM, instN, instBM, instBN, miWaveTile, miWaveGroup,
                             wavefrontSize):
    """MacroTile (0, 1) from derived MI geometry, or None if it cannot be derived.

    This repeats assignProblemIndependentDerivedParameters' own arithmetic for
    the MIBlock[0] != 4 branch; it is not a second formula for the same
    quantity.

    MIBlock[0] == 4 needs MIOutputVectorWidth, which comes from the ISA info map
    this module has no access to, so callers screen that shape out.
    """
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
    """MacroTile (0, 1) from a nine-item MatrixInstruction, or None.

    Derives MIBlockBM/BN and MIWaveGroup the way matrixInstructionToMIParameters
    does, then applies the shared tile arithmetic. MIWaveGroup cannot be read
    straight off mi[7]/mi[8]: MatrixInstB is distributed into MIBlockBM first
    and MIWaveGroup follows that distribution, so any MatrixInstB > 1 would come
    out too small and under-report LDS.
    """
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
    """Whether an auto candidate can survive the divergent-pair rules.

    Applied before ranking: a pair the later rules reject must not win on LDS
    and strand a legal pair that fits, because nothing retries.
    """
    if pgrA == pgrB:
        return True
    if min(pgrA, pgrB) == 0:
        return False
    return max(ldsBlocksForPgrLevel(pgrA),
               ldsBlocksForPgrLevel(pgrB)) <= DCP_MAX_LDS_BLOCKS_DIVERGENT


def _asDataType(value):
    """DataType, or None if missing / not a type Problem.py can name."""
    if value is None:
        return None
    if isinstance(value, DataType):
        return value
    try:
        return DataType(value)
    except Exception:
        return None


def _nonNegOrZero(value):
    """Unresolved (-1) or missing padding counts as 0.

    Not a simplification but the absence of an alternative: calcLdsPad derives
    LdsPad* from VectorWidth*, GlobalReadVectorWidth*, LocalReadVectorWidth*,
    UnrollMajorLDS* and the ISA info map, every one of which is still -1 or
    absent when auto runs, and the only sound upper bound -- LDS_MAX_PAD_BYTES
    over the smallest legal pad block -- is a multiple of the figure rather
    than a correction to it.

    So every size here is a lower bound on the derived one, short by exactly
    the padding, which runs around 1.5% of the packed block. It is short by
    that on every candidate alike, so the ranking does not move; what it can do
    is admit a pair whose derived size clears MaxLDS by less than twice the
    padding.
    """
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
    """Smallest power of two at or above `value`; 0 stays 0.

    The exact spelling of assignDerivedParameters' 2 ** ceil(log2(x)).
    """
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
    """One tensor's aligned LDS bytes. Same rules as calcLdsNumBytesAB.

    mxTc is A, B, MXSA, or MXSB. MX scale tiles are 1 byte. Align comes from
    MacDataType of A/B (6-bit stays 64, else 64/numRegisters).
    """
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
    """LDS bytes a candidate pair costs, which is what auto (-1,-1) ranks on.

    Mirrors whichever layout assignDerivedParameters will lay down for the
    pair: owner-grouped setLdsOffsetsDecoupled when the block counts diverge,
    legacy setLdsOffsets when they match. Solution.calcLdsNumBytesAB cannot be
    called from here -- it is nested inside assignDerivedParameters and needs
    LdsPad / _DepthU* / MacroTileA -- so the size math is repeated, with the
    padding unresolved (see _nonNegOrZero).

    Both real layouts close on the unaligned B size where this uses the aligned
    one, so the result is up to one LDS alignment (512B) high.
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
        # That layout has no power-of-two swap stride to round up to; it sets
        # StoreSwapAddr instead.
        return nBlkA * (ldsA + ldsMXSA) + nBlkB * (ldsMXSB + ldsB)

    # setLdsOffsets. A decoupled pair rejects Sparse, so the block carries no
    # metadata segment, and a single block is sized as two exactly as it is there.
    offsetB = ldsA + ldsMXSA + ldsMXSB
    offsetBlk = offsetB + ldsB
    if nBlkA == 2 and offsetBlk + _roundUpPow2(offsetBlk) <= _maxLdsOrDefault(ks):
        # Swapping between two blocks is an inlined xor, so the block is rounded
        # up to a power of two -- but only while the rounded-up pair still fits,
        # which is the StoreSwapAddr test. Three or more blocks never round up.
        offsetBlk = _roundUpPow2(offsetBlk)
    return (max(nBlkA, 2) - 1) * offsetBlk + offsetB + ldsB


def _macroTileFromState(state):
    """MacroTile (0, 1) for the LDS estimate, or None when it cannot be derived.

    Three sources in descending authority: MacroTile0/1 once derived; the MI
    parameters (MIBlock, MIWaveTile, MIWaveGroup), which is the normal tuning
    path because auto-selection runs before MacroTile exists; and a raw
    nine-item MatrixInstruction for callers that skipped that conversion.

    None means the solution cannot be sized, and the caller must reject rather
    than accept a candidate whose LDS it never measured.
    """
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
    """(A, B) local-read elements one thread issues per unrolled iteration.

    KernelWriter sizes a tensor's local reads as MIWaveTile[side] *
    numReadsPerUnroll, and numReadsPerUnroll is that tensor's
    MIInputPerThread, divided by the read instruction's width where the
    unroll-major path uses one. Those widths are not derived yet, so this
    counts elements and not instructions -- enough to say which side reads
    more, not how much more.

    None when the MI geometry is absent, which is the only state with nothing
    to compare.
    """
    miWaveTile = state.get("MIWaveTile")
    inputA = state.get("MIInputPerThreadA")
    inputB = state.get("MIInputPerThreadB")
    if not (isinstance(miWaveTile, (list, tuple)) and len(miWaveTile) == 2):
        return None
    if not isinstance(inputA, int) or not isinstance(inputB, int):
        return None
    return miWaveTile[0] * inputA, miWaveTile[1] * inputB


def _thickSideRank(pair, state):
    """Rank for separating candidates the LDS estimate ties; highest wins.

    (2, 1) and (1, 2) cost byte-identical LDS whenever A and B are the same
    size, and they are not the same kernel.
    _dcpScheduleSingleBufferedFillLate relocates the THIN tensor's refill
    behind the whole-workgroup write-after-read fence, where it cannot start
    until every ds_read of the iteration has retired, while the thick tensor
    keeps its fill at the loop head. So the thick side should be the one with
    more local-read work for the thin side's fill to hide behind.

    MIWaveTile decides that: it is in state at this point, and it does not tie
    merely because MacroTile does, since _macroTileFromMIGeometry folds
    MIWaveGroup and MIBlockBM/BN in beside it.

    The second element settles the residual exact tie on (2, 1). That is the
    arm leaving A thick, so tdmWaveIssueOrder keeps localReadSideOrder at
    A-then-B, so KernelWriter neither reverses local-read emission nor clears
    StinkyTofu's DsReadOrder, both of which (1, 2) causes.
    """
    blocksA = ldsBlocksForPgrLevel(pair[0])
    blocksB = ldsBlocksForPgrLevel(pair[1])
    if blocksA == blocksB:
        # Nothing is relocated, so there is no side to place. Top rank, so an
        # equal pair is never displaced by a divergent one that ties it.
        return True, True
    thickIsA = blocksA > blocksB
    work = _localReadWork(state)
    placed = work is None or work[0] == work[1] or (work[0] > work[1]) == thickIsA
    return placed, thickIsA


def pgrAutoPairRanking(pgr, state, problemType=None, fixedA=None, fixedB=None):
    """LDS-feasible pairs at or below `pgr`, best first. Empty if none fits.

    fixedA / fixedB pin one tensor by filtering the candidate list, so the
    ranking still maximises LDS over what remains. Legality is filtered BEFORE
    ranking: ranking first and rejecting afterwards loses the solution, since
    nothing steps down to the legal pair that also fits.

    Deepest-that-fits stays the objective. _thickSideRank only separates
    candidates that objective leaves tied, so no winner is decided by the order
    pgrAutoPairCandidates happens to append in.

    The whole ranking rather than just its head because these sizes are lower
    bounds -- they cannot see LDS padding (see _nonNegOrZero) -- so the head can
    still be over MaxLDS once the real pads are derived, and the caller has to
    be able to step to the next candidate in this order. A candidate already
    over MaxLDS on the lower bound is over it in the derivation too, so leaving
    those out costs nothing and makes an exhausted ranking a genuine no-fit.
    """
    candidates = [pair for pair in pgrAutoPairCandidates(pgr)
                  if autoPairCandidateIsLegal(*pair)]
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
    # Stable sort, so any residual tie keeps the order pgrAutoPairCandidates
    # appended in rather than one the sort invented.
    ranked.sort(key=lambda scored: scored[0], reverse=True)
    return [pair for _, pair in ranked]


def pgrAutoPairSelectMaxLds(pgr, state, problemType=None, fixedA=None, fixedB=None):
    """Best-ranked feasible pair at or below `pgr`, or None if none fits."""
    ranking = pgrAutoPairRanking(pgr, state, problemType,
                                 fixedA=fixedA, fixedB=fixedB)
    return ranking[0] if ranking else None


def pgrSpecialValueRejectReason(pgrA, pgrB):
    """Reject one-sided keys.

    Every both-set combination is left for resolvePrefetchGlobalReadSpecialValues:
    (-1, -1) searches both tensors, -1 against a real depth searches that one
    tensor with the other held fixed, and equal (k, k) for k >= 0 degenerates to
    scalar. A search that finds nothing rejects there, with the LDS reason.
    """
    if pgrA is None and pgrB is None:
        return None
    if (pgrA is None) != (pgrB is None):
        return ("PrefetchGlobalReadA/B: PrefetchGlobalReadA and PrefetchGlobalReadB must "
                "both be set or both omitted")
    return None


def pgrAutoPairRequested(state):
    """Whether auto has a ranking to search, so a refused pair has a successor.

    The same gate resolvePrefetchGlobalReadSpecialValues applies: both keys on
    -1, one key on -1 against a real depth, or the scalar on -1 with no keys.
    A pair-auto form with PrefetchGlobalRead 0 or 1 drops the keys instead of
    searching, so it is not auto here and has nothing to step to.
    """
    pgrA = state.get("PrefetchGlobalReadA")
    pgrB = state.get("PrefetchGlobalReadB")
    pgr = state.get("PrefetchGlobalRead", 0)
    if pgrSpecialValueRejectReason(pgrA, pgrB):
        return False
    autoA = pgrA == PGR_SPECIAL_AUTO
    autoB = pgrB == PGR_SPECIAL_AUTO
    if autoA and autoB:
        return pgr not in (0, 1)
    if autoA != autoB:
        return True
    return pgrA is None and pgrB is None and pgr == PGR_SPECIAL_AUTO


def resolvePrefetchGlobalReadSpecialValues(state, skip=0):
    """Resolve auto (-1). Returns reject reason or None.

    (-1, -1), or PrefetchGlobalRead=-1 with both keys omitted: pick the
    max-LDS pair starting at PrefetchGlobalRead if it is >= 2, else 2.
    (-1, -1) with PrefetchGlobalRead 0 or 1: drop the keys (no auto pair).
    -1 on one tensor against a real depth on the other: the same search,
    narrowed to the combinations that keep the fixed tensor at that depth. The
    ceiling rises to that depth too, so pinning a level cannot put it out of
    reach of its own search.

    `skip` drops that many leading candidates from the ranking, which is how a
    caller steps past a pair the real LDS check refused. Running past the end
    of the ranking reports the same no-fit as an empty one, so a shape where
    nothing fits cannot come back as a quietly different pair.
    """
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
    scalarAuto = pgrA is None and pgrB is None and pgr == PGR_SPECIAL_AUTO
    if not (pairAuto or scalarAuto or oneSided):
        return None
    if pairAuto and pgr in (0, 1):
        state.pop("PrefetchGlobalReadA", None)
        state.pop("PrefetchGlobalReadB", None)
        return None
    fixedA = fixedB = held = heldTc = None
    start = pgrAutoStartLevel(pgr)
    if oneSided:
        if autoA:
            fixedB = held = pgrB
            heldTc = "B"
        else:
            fixedA = held = pgrA
            heldTc = "A"
        if held == 0:
            # Not an LDS shortfall: no pair can keep a tensor at 0, because a
            # divergent pair at level 0 is unsupported. Say that, rather than
            # letting an empty candidate list be reported as "nothing fits" --
            # and reject for the same reason an explicit (0, N) is rejected.
            return ("PrefetchGlobalReadA/B: PrefetchGlobalRead%s=0 cannot be held while "
                    "the other tensor is auto: level 0 allocates the same single LDS "
                    "block as level 1 and pins the same scalar PrefetchGlobalRead, so a "
                    "divergent pair at level 0 is not supported" % heldTc)
        # The held level raises the ceiling, but a divergent pair needs a side at
        # 2, so the search must be allowed to reach it whatever the scalar says.
        start = max(start, held, PGR_AUTO_DEFAULT_LEVEL)
    ranking = pgrAutoPairRanking(start, state, state.get("ProblemType"),
                                 fixedA=fixedA, fixedB=fixedB)
    if skip >= len(ranking):
        if oneSided:
            return ("PrefetchGlobalReadA/B: auto found no LDS-feasible pair with "
                    "PrefetchGlobalRead%s held at %d, starting from %d"
                    % (heldTc, held, start))
        return ("PrefetchGlobalReadA/B: auto found no LDS-feasible pair starting from "
                "PrefetchGlobalRead=%s" % (pgr if pgr != PGR_SPECIAL_AUTO else start))
    state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"] = ranking[skip]
    return None


def pgrLevelsForTensors(ks):
    """(decoupled, pgrA, pgrB).

    Both keys omitted: legacy scalar PrefetchGlobalRead.
    Both keys present: the per-tensor pair. One-sided is rejected earlier.
    """
    pgr = ks.get("PrefetchGlobalRead", 0)
    pgrA = ks.get("PrefetchGlobalReadA")
    pgrB = ks.get("PrefetchGlobalReadB")
    if pgrA is None and pgrB is None:
        return False, pgr, pgr
    return True, pgrA, pgrB


def ldsBlocksForPgrLevel(pgr):
    """LDS blocks one per-tensor level allocates.

    Under TDM there is no VGPR staging buffer, so depth N is N blocks.
    Depth 0 and 1 both use one block.
    """
    if pgr <= 1:
        return 1
    return pgr


DCP_LDS_SIDE = {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}


def dcpLdsSide(tc):
    """Which of the two LDS block groups `tc` is laid into.

    setLdsOffsetsDecoupled lays out nBlkA x [A|MXSA] at strideA then
    nBlkB x [MXSB|B] at strideB whatever the descriptor grouping says, so
    this is the LDS block stride pairing and never descriptor ownership:
    routing it through tdmSetOwner would swap a scale by the other side's
    stride and gate its reads on the other side's token.
    """
    return DCP_LDS_SIDE[tc]


def decouplePGRBlocks(ks):
    """(decoupled, numLdsBlkA, numLdsBlkB).

    Derived on demand rather than stored in solution state, so nothing derived
    reaches the serialized library.
    """
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
        # ldsBlocksForPgrLevel maps 0 and 1 to the same single block and every
        # consumer reads block counts rather than levels, so (0, N) emits the
        # instructions of (1, N) under a different kernel name. Reject rather
        # than ship two names for one kernel.
        return ("level 0 has no per-tensor prefetch cadence: it allocates the same "
                "single LDS block as level 1 and pins the same scalar "
                "PrefetchGlobalRead, so (%u, %u) would emit the instructions of "
                "(%u, %u) under a different kernel name"
                % (pgrA, pgrB, max(pgrA, 1), max(pgrB, 1)))
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if max(numLdsBlkA, numLdsBlkB) > DCP_MAX_LDS_BLOCKS_DIVERGENT:
        return "more than two LDS blocks for a tensor is not supported"
    if ks["_ScheduleIterAlg"] != 0:
        return ("only ScheduleIterAlg=0 places the fill where it can be moved, "
                "and ScheduleIterAlg=4 derives to it")
    if ks["PrefetchLocalRead"] < 1:
        return ("PrefetchLocalRead must be at least 1 so a sub-iteration exists "
                "between the last local read and the pre-read sync")
    loopIters = ks["DepthU"] // ks["LocalSplitU"] // ks["InnerUnroll"]
    if ks.get("EnableMatrixInstruction", True):
        loopIters //= ks["MatrixInstK"]
    if (ks["PrefetchLocalRead"] >= loopIters
            and ks.get("ClusterLocalRead", 1)
            and not ks.get("ForceUnrollSubIter", False)):
        return ("PrefetchLocalRead=%u is not below LoopIters=%u, and is rewritten to 0 "
                "after this check, leaving no sub-iteration between the last local read "
                "and the pre-read sync" % (ks["PrefetchLocalRead"], loopIters))
    if ks["NumWaves"] <= 1:
        return ("the fill is re-slotted under a wave-parity guard, which needs the "
                "wave-separated TDM descriptor (NumWaves > 1); this solution has "
                "NumWaves=%u" % ks["NumWaves"])
    return None


def dcpThickThinIssueOrder(items):
    """Order the items one issue slot was handed, thickest first.

    `items` is a sequence of (ldsBlocks, item); more blocks means issue earlier.
    Thick-first is what makes the relaxed gate mean anything: s_wait_tensorcnt N
    is an age-ordered drain on one counter, so which tensor a count bypasses
    follows from issue order alone.

    The sort is stable, so an equal pair comes back in the order it was handed.
    """
    return tuple(item for _, item in sorted(items, key=lambda pair: -pair[0]))


def decoupledSingleBuffered(ks):
    """True when exactly one tensor is left on a single LDS block.

    That tensor has nowhere to put its next tile except on top of the copy the
    current iteration is still reading, so the write-after-read barriers have to
    fire even though the scalar PrefetchGlobalRead is nonzero.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return decoupled and min(numLdsBlkA, numLdsBlkB) == 1 and max(numLdsBlkA, numLdsBlkB) > 1


def decoupledOneBlockBoth(ks):
    """True when both tensors sit on one LDS block inside a prefetch loop.

    (0, 1) and (1, 0) hit this and are rejected: each fill overwrites the
    block the previous trip is still reading. Equal (1, 1) is the same LDS
    shape, but derivation drops those keys to scalar PrefetchGlobalRead=1
    before this runs on a new solution. PrefetchGlobalRead=0 is not this
    path (the no-prefetch branch already uses one block).
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return decoupled and max(numLdsBlkA, numLdsBlkB) == 1 and bool(ks["PrefetchGlobalRead"])


def tdmWaveIssueOrder(ks, itemA, itemB):
    """(first, second) of the two data tensors' items, thick tensor first.

    Each tensor is weighed by its own block count rather than by its wave's
    total. The two agree wherever A and B sit on separate waves, which is every
    arrangement except the shared-scale shape -- and that one is equal-pair
    only, so its weights tie and the stable sort keeps A first.
    """
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return dcpThickThinIssueOrder(((numLdsBlkA, itemA), (numLdsBlkB, itemB)))


DCP_THICK_GATE_TOKENS = "tokens"


DCP_THICK_GATE_TEXT = "text"


# The count each mechanism's grouping actually supports. Not a tunable: it is
# how many independent tensor ops the grouping leaves outstanding.
DCP_THICK_GATE_SUPPORTED = {DCP_THICK_GATE_TEXT: 2, DCP_THICK_GATE_TOKENS: 1}


class DcpThickGate(NamedTuple):
    """How a divergent decoupled pair's thick-tensor gate gets relaxed."""
    mechanism: str
    tensorcnt: int


def decoupledThickGateRelaxation(ks):
    """The relaxed thick-tensor gate a decoupled pair earns, or None.

    Sole owner of that decision; both emission paths read it.

    `s_wait_tensorcnt N` retires with N tensor ops still outstanding, so 0 is a
    full drain and a larger N is the weaker gate. A divergent pair lets the thin
    tensor's reads start while the thick tensor's last block is still landing.

      TOKENS, 1  separate descriptor sets take disjoint tensor tokens, so the
                 wait-count insertion pass emits the gate itself
      TEXT, 2    one shared set means one token and only a full drain, which
                 KernelWriter._dcpRelaxThickTextGate relaxes afterwards
      None       no thin side to start early, or one too deep to skip past

    Asked of the resolved grouping, not of TDMFuse, which can be declined.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if not (decoupled and numLdsBlkA != numLdsBlkB):
        return None
    # Both mechanisms earn their count by skipping past the thin side's one
    # refill, so a thin side holding two blocks leaves nothing to skip.
    if min(numLdsBlkA, numLdsBlkB) != 1:
        return None
    if tdmGroupingSeparatesAB(ks):
        if not tdmSeparateABDescriptors(ks):
            return None
        mechanism = DCP_THICK_GATE_TOKENS
    else:
        mechanism = DCP_THICK_GATE_TEXT
    return DcpThickGate(mechanism, DCP_THICK_GATE_SUPPORTED[mechanism])


_DCP_TENSORCNT_RE = re.compile(r"^s_wait_tensorcnt\s+(\d+)(?:\s|$)")


_DCP_LDS_READ_RE = re.compile(r"^ds_(?:load|read)\w*\s")


def dcpIsFillLabel(line):
    """True for a `DcpEarlyFill`/`DcpLateFill` label line.

    Ends one site's scan window. KernelWriter._dcpRelaxThickTextGate and the
    audit below have to stop at the same line or the audit judges a site by a
    gate the rewriter was never allowed to touch.
    """
    return (("DcpEarlyFill" in line or "DcpLateFill" in line)
            and line.rstrip().endswith(":"))


def dcpThickGateUncoveredSites(lines, marker, relaxed, accepted):
    """Thick-fill sites the text pass left un-relaxed, as [(lineIndex, why)].

    A coverage question rather than a retag count, because the waits come from
    the scheduler. `accepted` holds the line indices the pass rewrote or judged already
    strong enough; an empty return means no site reaches an LDS read through a
    gate weaker than the relaxation allows.

    Covered: a first gate this pass relaxed, one already at or below the relaxed
    count, or no gate and no read before the next fill label.
    Not covered: a `ds_load` first; a first gate weaker than the relaxed count,
    which would let the next gate down drain the thin tensor into the block its
    reads are about to touch; or no sites at all.
    """
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
            gate = _DCP_TENSORCNT_RE.match(candidate)
            if gate:
                if j not in accepted:
                    why = ("first gate is s_wait_tensorcnt %s at line %d, weaker "
                           "than the %d this pass relaxes to"
                           % (gate.group(1), j, relaxed))
                break
        if why:
            uncovered.append((i, why))
    return uncovered
