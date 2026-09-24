################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""MX scale free-dim share (MXBlockFree) vs K-axis share (MXBlock).

MXBlock compresses K. MXBlockFreeA/B compresses M (A) or N (B): 1 keeps one
scale per row (the 1xK layout); 128 means 128 consecutive rows share one e8.
A yaml list ``MXBlockA: [free, k]`` is split into this pair by ProblemType.
"""

from math import log2
from typing import Mapping

# 2D MXS buffers are a few e8s. MacDataType F8's ldsAlign is 256 B
# (64 / 0.25 regs), which would round a 2-byte buffer back to the 1D size.
MXS_LDS_ALIGN_2D = 64


def mxFreeTile(kernel: Mapping, tc: str) -> int:
    """Free-dim MX share in scale rows. Non-MXS tensors return 1.

    Reads ``ProblemType.MXBlockFree{A,B}``. Missing or 0 is the 1xK layout.
    """
    if "MXS" not in tc:
        return 1
    pt = kernel.get("ProblemType")
    if not isinstance(pt, Mapping):
        return 1
    return max(1, int(pt.get(f"MXBlockFree{tc[-1]}", 1) or 1))


def mxTdmTileM(mt: int, mxTile: int) -> int:
    """How many scale-rows one MacroTile intersects.

    Ceil so an unaligned tile (e.g. MT=224, MXBlockFree=128) still covers
    the second group. Aligned MT is unchanged (256/128 == 2).
    """
    mxTile = max(1, int(mxTile))
    return (int(mt) + mxTile - 1) // mxTile


def mxTdmMSplitStride(mt: int, mxTile: int, numComp: int) -> int:
    """Scale-rows one TDM component advances when M-splitting.

    0 means the tile has fewer scale-rows than components: do not M-split
    (extra comps idle / share the same address). Never used for K-split.
    """
    return mxTdmTileM(mt, mxTile) // max(1, int(numComp))


def mxTdmTile0(sizeTile1: int, mxUnit: int, mxTile: int, numComp: int, kSplit: bool) -> int:
    """MXS TDM tile0: M/N extent of one issue, in scale e8s.

    1D (mxTile==1): MT * mxUnit, or MT * mxUnit / numComp when M-splitting.
    2D: those quantities divided by MXBlockFree. If that divide is 0
    (fewer scale-rows than TDM components), keep the full small tile so
    the issue is not 0-wide. Extra comps share the same address / LDS dest.
    """
    tileM = mxTdmTileM(sizeTile1, mxTile)
    if kSplit:
        return tileM * mxUnit
    per = tileM * mxUnit // max(1, int(numComp))
    return per if per > 0 else tileM * mxUnit


def mxTdmKRowPitch(size: int, mxTile: int) -> int:
    """2D TDM stride0: K-row pitch in scale e8s.

    1D sets stride0 = Size (elements along M or N in one K-block). 2D
    compresses that axis by MXBlockFree, so the same field is
    Size/MXBlockFree. Tile0 e8s in a K-block are packed with implicit
    stride 1; stride0 is the dim1 pitch, not the in-tile M step.
    A constant-1 stride0 makes dim1*stride0 a 12-byte window and the
    K-split MXSB pointer (Size/MXBlockFree) lands outside it.
    """
    mxTile = max(1, int(mxTile))
    if mxTile <= 1:
        raise ValueError("mxTdmKRowPitch is the 2D K-row pitch; 1D uses Size")
    return max(1, int(size) // mxTile)


def mxUnitsAre1(kernel: Mapping) -> bool:
    """True when every present MX side has mxUnit == 1 (MX128).

    mxUnit = MatrixInstK / MXBlockK. MX16, MX32, and non-MX stay false so
    TDM tensor_load and GL2 prefetch keep the pre-MXS-first issue order.
    """
    pt = kernel.get("ProblemType") or {}
    try:
        mi_k = int(kernel.get("MatrixInstK") or 0)
    except (TypeError, ValueError):
        return False
    units = []
    for side in ("A", "B"):
        block = pt.get("MXBlock%s" % side) or 0
        if isinstance(block, (list, tuple)):
            block = block[-1] if block else 0
        try:
            block = int(block)
        except (TypeError, ValueError):
            return False
        if block <= 0:
            continue
        if mi_k <= 0:
            return False
        units.append(mi_k // block)
    return bool(units) and all(u == 1 for u in units)


def mxIssueTpList(kernel: Mapping, tPA: Mapping, tPB: Mapping, *, includeMetadata: bool = False):
    """Tensor-parameter list for GL2 prefetch issue.

    mxUnit==1 issues MXSA, MXSB, then A, B. Otherwise A, B, then MXSA, MXSB.
    Metadata stays last when requested.
    """
    pt = kernel.get("ProblemType") or {}
    ab = [tPA, tPB]
    mx = []
    if pt.get("MXBlockA") and "MX" in tPA:
        mx.append(tPA["MX"])
    if pt.get("MXBlockB") and "MX" in tPB:
        mx.append(tPB["MX"])
    tps = (mx + ab) if mxUnitsAre1(kernel) else (ab + mx)
    if includeMetadata and kernel.get("enableTDMMetadata"):
        tps.append(tPA["tpsMetadata"] if tPA.get("is_sparse") else tPB["tpsMetadata"])
    return tps


def mxGl2CoalescedDim(mt: int, numTileWGs: int, mxUnit: int, mxTile: int = 1) -> int:
    """Return the MX GL2 coalesced size in e8s for all tile workgroups.

    The cluster free-dimension extent is converted to scale-row units.
    """
    clusterM = int(mt) * max(1, int(numTileWGs))
    return mxTdmTileM(clusterM, mxTile) * max(1, int(mxUnit))


def mxGl2TileOffset(tileIdx: int, mt: int, mxUnit: int, mxTile: int = 1) -> int:
    """Return the MXS GL2 free-dimension offset in e8s.

    The macro-tile index is converted to the M/N coordinate and then to the
    scale-row coordinate.
    """
    mxTile = max(1, int(mxTile))
    return ((int(tileIdx) * int(mt)) // mxTile) * max(1, int(mxUnit))


def mxLdsAlign(mxTile: int, macLdsAlign: int) -> int:
    """LDS alignment for an MXS buffer.

    1D keeps MacDataType's align (F8 is 256 B). 2D is a few e8s; that 256 B
    round-up would restore the 1D size, so use ``MXS_LDS_ALIGN_2D``.
    """
    return MXS_LDS_ALIGN_2D if max(1, int(mxTile)) > 1 else int(macLdsAlign)


def mxLdsNumBytes(mt: int, depthU: int, mxBlock: int, mxTile: int = 1,
                  ldsPad: int = 0, unrollMajor: bool = True,
                  padInterval: int = 0) -> int:
    """MXS LDS bytes for one WG buffer (before alignment).

    1D: MT × (DepthU/MXBlock). 2D: ceil(MT/MXBlockFree) × (DepthU/MXBlock).
    ``padInterval`` matches ``calcLdsNumBytesAB``: every block of that many
    bytes grows by ``ldsPad``. A 2D buffer smaller than one interval cannot
    use that formula (integer divide would be 0) and stays compact.
    """
    scaleRows = mxTdmTileM(mt, mxTile)
    mxDU = depthU // mxBlock
    if padInterval:
        raw = mxDU * scaleRows
        if raw >= padInterval:
            return int(raw / padInterval * (padInterval + ldsPad))
        ldsPad = 0
    if unrollMajor:
        return (mxDU + ldsPad) * scaleRows
    return mxDU * (scaleRows + ldsPad)


def mxLdsLsuStride(mt: int, mxTile: int, mxDU: int, lsu: int) -> int:
    """Bytes one LSU wave steps in MXS LDS.

    1D: MT × (mxDU/LSU). 2D: scale-rows × (mxDU/LSU), matching TDM K-split
    packing so LSU 1 lands on the second k-group, not MT bytes past the buffer.
    """
    lsu = max(1, int(lsu))
    return mxTdmTileM(mt, mxTile) * (int(mxDU) // lsu)


def mxLdsKStride(mt: int, mxTile: int, mxUnit: int, *, swizzled: bool,
                 unrollMajor: bool = True, ldsPad: int = 0) -> int:
    """LDS step between consecutive MX K-groups of the same scale-row.

    1D (mxTile==1) matches today's localReadInc: swizzled ``MT*mxUnit``,
    K-major ``mxUnit``, M-major ``(MT+pad)*mxUnit``. 2D replaces MT with
    scale-rows (``MT/MXBlockFree``), matching TDM k-split packing.
    """
    scaleRows = mxTdmTileM(mt, mxTile)
    if swizzled:
        return int(scaleRows * mxUnit)
    if unrollMajor:
        return int(mxUnit)
    return int((scaleRows + ldsPad) * mxUnit)


def mxLraFreeShift(mxTile: int) -> int:
    """log2(MXBlockFree) to turn a wave M-offset into a 2D scale-row index."""
    mxTile = max(1, int(mxTile))
    if mxTile <= 1:
        return 0
    if mxTile & (mxTile - 1):
        raise ValueError(f"MXBlockFree {mxTile} must be a power of 2")
    return int(log2(mxTile))


def mxLraScaleRow(waveId0: int, strideWave: int, mxTile: int) -> int:
    """LRA scale-row for a wave along M/N. Lanes of that wave share this index."""
    mxTile = max(1, int(mxTile))
    return (int(waveId0) * int(strideWave)) // mxTile


def mxTileSpanPartnerDelta(kernel: Mapping, tc: str, tile01: int) -> int:
    """Free-dim distance in M/N elements between TileSpan half-waves.

    Non-split (MIWaveGroup==1): MatrixInstT * VW (nIdx = wtid).
    Wave-split (MIWaveGroup>1): MatrixInstT * BM/BN * numWaves * VW
    (same product as LRA hiOffset without strideTile). After / MXBlockFree a
    non-multiple partnerΔ maps both half-waves onto the same scale row.
    """
    matrixInstT = kernel["MatrixInstM"] if tile01 == 0 else kernel["MatrixInstN"]
    vw = int(kernel.get(f"VectorWidth{tc}", 1) or 1)
    numWaves = int(kernel["MIWaveGroup"][tile01])
    if numWaves > 1:
        blockKey = "MatrixInstBM" if tile01 == 0 else "MatrixInstBN"
        blocks = int(kernel.get(blockKey, 1) or 1)
        return int(matrixInstT) * blocks * numWaves * vw
    return int(matrixInstT) * vw
