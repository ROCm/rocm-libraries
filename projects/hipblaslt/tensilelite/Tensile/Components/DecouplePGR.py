################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Per-tensor PrefetchGlobalReadA/B (DecouplePGR).

Both keys must be set or both omitted.

  omitted, omitted              legacy scalar PrefetchGlobalRead
  (-1, -1) and PGR >= 2         auto: max-LDS pair, start at PrefetchGlobalRead
  (-1, -1) and PGR == -1        auto: same, start at 2
  scalar PGR == -1 (no keys)    auto: start at 2
  (-1, -1) and PGR is 0 or 1    drop A/B, keep that scalar (no auto pair)
  (k, k) for k >= 0             PrefetchGlobalRead=k  (includes (0,0) and (1,1))
  (0, 1) / (1, 0)               reject (both single-buffered)
  one key only                  reject
  -1 mixed with a real depth    reject
"""


from ..Common.DataType import DataType

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
        candidates.append((level, level - 1))
        candidates.append((level - 1, level))
    return candidates


def macroTileFromMatrixInstruction(mi):
    """MacroTile (M, N) from a MatrixInstruction tuple, or None if unavailable.
    MacroTile0 = MT0 * MIWT0 * WG0, MacroTile1 = MT1 * MIWT1 * WG1
    """
    if not isinstance(mi, (list, tuple)) or len(mi) < 9:
        return None
    return mi[0] * mi[5] * mi[7], mi[1] * mi[6] * mi[8]


def _roundUpToMultiple(n, m):
    if m <= 0:
        return n
    return (n + m - 1) // m * m


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


def _macDataType(pt, tc):
    return (_asDataType(pt.get("MacDataType%s" % tc))
            or _asDataType(pt.get("DataType%s" % tc))
            or _asDataType(pt.get("DataType")))


def _nonNegOrZero(value):
    """Treat missing / auto (-1) padding as 0; auto-select runs before pad derivation."""
    if value is None or value < 0:
        return 0
    return int(value)


def _unrollMajor(ks, tc):
    return ks.get("UnrollMajorLDS%s" % tc) in (1, True)


def _ldsBytesAligned(depthU, macroTile, bpe, ldsPad=0, padInterval=0, align=64,
                     unrollMajor=False):
    """Same size math as calcLdsNumBytesAB."""
    if padInterval:
        raw = int(depthU * macroTile * bpe) // padInterval * (padInterval + ldsPad * bpe)
    elif unrollMajor:
        raw = int((depthU + ldsPad) * macroTile * bpe)
    else:
        raw = int(depthU * (macroTile + ldsPad) * bpe)
    return _roundUpToMultiple(raw, align)


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
    mac = _macDataType(pt, tc)
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
        unrollMajor=_unrollMajor(ks, mxTc),
    )


def decouplePGRLdsBytesEstimate(ks, problemType=None):
    """Owner-grouped LDS estimate for auto (-1,-1).

    Cannot call Solution.calcLdsNumBytesAB: that is a nested function inside
    assignDerivedParameters, and it needs LdsPad / _DepthU* / MacroTileA that
    do not exist yet when auto runs. Same size math, with pad treated as 0.
    Returns None when A/B element size cannot be resolved.
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
    spanA = ldsA + ldsMXSA
    blkA = spanA
    if blkA % 8:
        blkA += 8 - (blkA % 8)

    offBinB = ldsMXSB
    blkB = offBinB + ldsB
    if blkB % 8:
        blkB += 8 - (blkB % 8)

    if nBlkA != nBlkB:
        return max(nBlkA * blkA + nBlkB * blkB, nBlkA * blkA)

    interleaved = spanA + ldsMXSB + ldsB
    if interleaved % 8:
        interleaved += 8 - (interleaved % 8)
    return nBlkA * interleaved


def _macroTileFromState(state):
    mt0 = state.get("MacroTile0")
    mt1 = state.get("MacroTile1")
    if mt0 is not None and mt1 is not None:
        return mt0, mt1
    mi = state.get("MatrixInstruction")
    if mi is not None:
        return macroTileFromMatrixInstruction(mi)
    return None


def pgrAutoPairSelectMaxLds(pgr, state, problemType=None):
    candidates = pgrAutoPairCandidates(pgr)
    if not candidates:
        return None
    macroTile = _macroTileFromState(state)
    if macroTile is None:
        return candidates[0]
    pt = problemType if problemType is not None else state.get("ProblemType")
    maxLds = state.get("MaxLDS", 327680)
    if maxLds is None or maxLds < 0:
        maxLds = 327680
    probe = dict(state)
    probe["MacroTile0"], probe["MacroTile1"] = macroTile
    if "DepthU" not in probe:
        probe["DepthU"] = 32
    # Rank needs A/B element size; without it do not pretend this is F8F4.
    probe["PrefetchGlobalRead"] = pgr
    probe["PrefetchGlobalReadA"], probe["PrefetchGlobalReadB"] = candidates[0]
    if decouplePGRLdsBytesEstimate(probe, pt) is None:
        return candidates[0]
    bestPair = None
    bestLds = -1
    for pair in candidates:
        pgrA, pgrB = pair
        probe["PrefetchGlobalReadA"] = pgrA
        probe["PrefetchGlobalReadB"] = pgrB
        lds = decouplePGRLdsBytesEstimate(probe, pt)
        if lds is None or lds > maxLds:
            continue
        if lds > bestLds:
            bestLds = lds
            bestPair = pair
    return bestPair


def pgrSpecialValueRejectReason(pgrA, pgrB):
    """Reject one-sided keys, or -1 mixed with a real depth.

    (-1, -1) is auto. Equal (k, k) for k >= 0 is left for scalar degeneration.
    """
    if pgrA is None and pgrB is None:
        return None
    if (pgrA is None) != (pgrB is None):
        return ("PrefetchGlobalReadA/B: PrefetchGlobalReadA and PrefetchGlobalReadB must "
                "both be set or both omitted")
    if pgrA == pgrB:
        return None
    if pgrA == PGR_SPECIAL_AUTO or pgrB == PGR_SPECIAL_AUTO:
        return ("PrefetchGlobalReadA/B: special value %d on one tensor with %d on the "
                "other is not supported" % (pgrA, pgrB))
    return None


def resolvePrefetchGlobalReadSpecialValues(state):
    """Resolve auto (-1). Returns reject reason or None.

    (-1, -1), or PrefetchGlobalRead=-1 with both keys omitted: pick the
    max-LDS pair starting at PrefetchGlobalRead if it is >= 2, else 2.
    (-1, -1) with PrefetchGlobalRead 0 or 1: drop the keys (no auto pair).
    """
    pgrA = state.get("PrefetchGlobalReadA")
    pgrB = state.get("PrefetchGlobalReadB")
    pgr = state.get("PrefetchGlobalRead", 0)
    reason = pgrSpecialValueRejectReason(pgrA, pgrB)
    if reason:
        return reason
    pairAuto = pgrA == PGR_SPECIAL_AUTO and pgrB == PGR_SPECIAL_AUTO
    scalarAuto = pgrA is None and pgrB is None and pgr == PGR_SPECIAL_AUTO
    if not (pairAuto or scalarAuto):
        return None
    if pairAuto and pgr in (0, 1):
        state.pop("PrefetchGlobalReadA", None)
        state.pop("PrefetchGlobalReadB", None)
        return None
    start = pgrAutoStartLevel(pgr)
    selected = pgrAutoPairSelectMaxLds(start, state, state.get("ProblemType"))
    if selected is None:
        return ("PrefetchGlobalReadA/B: auto found no LDS-feasible pair starting from "
                "PrefetchGlobalRead=%s" % (pgr if pgr != PGR_SPECIAL_AUTO else start))
    state["PrefetchGlobalReadA"], state["PrefetchGlobalReadB"] = selected
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
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if max(numLdsBlkA, numLdsBlkB) > 2:
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
