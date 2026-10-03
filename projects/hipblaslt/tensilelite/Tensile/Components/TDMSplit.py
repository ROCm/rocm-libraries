# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared arithmetic for TDMSplit descriptor, LDS, and region geometry.

The per-operand ``TDMSplitA`` and ``TDMSplitB`` parameters identify an axis
and factor. This module derives the remaining geometry without importing
rocisa or Solution state.
"""

from __future__ import annotations

from dataclasses import dataclass

# The two axes along which a tile can be split.
# MT is the operand's free axis; DU is the shared reduction axis.
AXIS_MT = "MT"
AXIS_DU = "DU"

# TDMSplit parameter value -> (MT factor, DU factor). The public parameter
# selects one axis with a fixed factor of two.
TDM_SPLIT_AXIS = {0: (1, 1), 1: (2, 1), 2: (1, 2)}


def split_of(kernel, tc) -> tuple:
    """`(factor, axis)` for operand `tc` — the ONE reading of `TDMSplitA`/`TDMSplitB`.

    Every gate, every descriptor field and every read-side term comes through here, because two
    readings of one parameter is how a gate and the thing it gates drift apart.  It lives in this
    module rather than in `Solution.py` so `Components/`, `Lowering/` and `SolutionStructs/` can
    all reach it without importing each other.

    MX-scale, sparse and metadata tensors are never split, and answering that here keeps the
    `not MXS and not Sparse` guard from being re-spelled at each site — it was, at nine, and they
    had already drifted (some also excluded metadata, some did not).  The back-compat spelling
    `TDMSplit` never reaches here: `assignProblemIndependentDerivedParameters` expands it to the
    per-operand pair and pops it."""
    if "MXS" in tc or tc == "Metadata" or kernel["ProblemType"]["Sparse"]:
        return (1, None)
    v = int(kernel.get("TDMSplit%s" % ("A" if tc.endswith("A") else "B"), 0) or 0)
    mt, du = TDM_SPLIT_AXIS.get(v, (1, 1))
    if mt > 1:
        return (mt, AXIS_MT)
    if du > 1:
        return (du, AXIS_DU)
    return (1, None)


def split_factors(kernel):
    """`(A_MT, B_MT, A_DU, B_DU)` for a split kernel, or None when neither operand is split.

    theta's positional layout, so `LoopModel/translate.py` and `LoopModel/bridge.py` need no
    separate spelling — `TDMSplitA/B` are a restricted way of writing the same tuple."""
    a = TDM_SPLIT_AXIS.get(int(kernel.get("TDMSplitA", 0) or 0), (1, 1))
    b = TDM_SPLIT_AXIS.get(int(kernel.get("TDMSplitB", 0) or 0), (1, 1))
    if a == (1, 1) and b == (1, 1):
        return None
    return (a[0], b[0], a[1], b[1])


def any_split(kernel) -> bool:
    """Return whether either operand is split."""
    return split_factors(kernel) is not None


def set_regions(kernel, tc) -> int:
    """How many storage regions the DESCRIPTOR SET carrying `tc` walks.

    One physical descriptor, one walk length: the longest any member wants.  A shorter member
    rides region 0 and is NULLed for the rest (`tdmMaskShortMembers`).  Ownership is a separate
    question, and asking `split_of` of the owner is wrong when the owner is the short member."""
    from .TDMFuse import tdmFusedGroups
    members = next((g for g in tdmFusedGroups(kernel) if tc in g), (tc,))
    return max(split_of(kernel, m)[0] for m in members)


def factor_axis(extent: int, factor: int, name: str) -> int:
    """Return the inner extent for ``extent = factor * inner`` or reject the split."""
    extent, factor = int(extent), max(1, int(factor))
    if extent < 1 or extent % factor:
        raise ValueError(
            f"TDMSplit must factorize {name} exactly: extent {extent} is not divisible "
            f"by split {factor}")
    return extent // factor


def split_packs_lds(axis, unrollMajor: bool) -> bool:
    """Does a split along `axis` make each region a SEPARATELY PACKED LDS block?

    A partition packs blocks exactly when it cuts the LDS image's INNER axis:

        unroll-major : LDS = [free][unroll]   -> inner is the reduction axis
        tlu          : LDS = [unroll][free]   -> inner is the free axis

    Cutting the OUTER axis leaves the parts a contiguous continuation of one array, so the two
    `tensor_load`s rebuild exactly the image one unsplit load builds and the reader needs to know
    nothing.  Cutting the INNER axis makes each part a stripe of every row, which two dense
    half-tile writes cannot reproduce — the hardware packs `[region][outer][inner/n]` instead, and
    the reader then owes a block base, a shortened row, and an in-region index.

    So the answer is a diagonal, not a property of either the axis or the layout alone:

        | axis | unroll-major | tlu    |
        | MT   | contiguous   | PACKED |
        | DU   | PACKED       | contiguous |

    On the MT/tlu cell, without the packed treatment every NT split kernel reads region 0's
    bytes for half its tiles; on MT/unroll-major, adding a region term breaks kernels that
    otherwise pass.  The diagonal is the rule, not either axis alone."""
    if axis is None:
        return False
    return (axis == AXIS_MT) == (not unrollMajor)


def _pad(off: int, blockSize: int, padBytes: int) -> int:
    """LDS byte offset with block padding applied, the same expansion the read leaf uses."""
    if blockSize and padBytes:
        off += (off // blockSize) * padBytes
    return off


@dataclass(frozen=True)
class TdmSplitGeometry:
    """Everything a split implies for ONE operand.  Build with `derive`."""

    #: How many regions this operand's tile is cut into (1 = unsplit).
    factor: int = 1
    #: `AXIS_MT`, `AXIS_DU`, or None when unsplit.
    axis: object = None
    #: Which descriptor dimension the split divisor goes on, 0 or 1.  dim0 is ALWAYS the
    #: contiguous dimension (`setTensorStride0` advances dim1), so which one carries the
    #: MacroTile is decided by the layout, never chosen.
    splitDim: int = 1
    #: Which stride the global region step uses: `'free'`, `'reduction'`, or None.
    strideAxis: object = None
    #: Bytes to advance the GLOBAL pointer per region, as the constant multiplying `strideAxis`'s
    #: stride.  bpe is folded in, matching the emitted `s_mul_i32 dst, stride, const`.
    globalConstBytes: int = 0
    #: Bytes to advance the LDS pointer per region, PADDED — the destination register holds
    #: post-pad addresses.
    ldsStepBytes: int = 0
    #: Whether the regions are separately packed in LDS (see `split_packs_lds`).
    packed: bool = False
    #: Per-region extent along **dim1**, in elements, or 0 when the split does not cut dim1.
    dim1SpanPerRegion: int = 0

    @property
    def isSplit(self) -> bool:
        return self.factor > 1


def derive(factor: int, axis, *, tlu: bool, mt: int, du: int, bpe: float,
           ldsBlockSizePerPad: int = 0, ldsPadBytes: int = 0) -> TdmSplitGeometry:
    """The one derivation.  `mt`/`du` are the operand's free and reduction tile extents in
    elements; `bpe` its bytes per element.

    `factor == 1` returns the inert geometry, and every consumer's `n == 1` path must then be
    arithmetically identical to the unsplit expression — that is what keeps unsplit kernels
    byte-for-byte unchanged by this module."""
    if factor <= 1 or axis is None:
        return TdmSplitGeometry()

    unrollMajor = not tlu
    # dim0 is the contiguous dimension; the free axis is contiguous exactly when `tlu`.
    freeDim = 0 if tlu else 1
    splitDim = freeDim if axis == AXIS_MT else (1 - freeDim)

    extent = mt if axis == AXIS_MT else du
    # Both layouts place region r at r * tileBytes / factor. Only the
    # reader's required displacement depends on whether the layout is packed.
    tileBytes = round(mt * du * bpe)

    return TdmSplitGeometry(
        factor=factor,
        axis=axis,
        splitDim=splitDim,
        strideAxis=("free" if axis == AXIS_MT else "reduction"),
        globalConstBytes=round(extent * bpe) // factor,
        ldsStepBytes=_pad(tileBytes // factor, ldsBlockSizePerPad, ldsPadBytes),
        packed=split_packs_lds(axis, unrollMajor),
        dim1SpanPerRegion=(extent // factor) if splitDim == 1 else 0,
    )


def seg_component_fold(waveCount: int, numComp: int, perWaveBytes: int, strideBytes: int):
    """`(wavesPerComp, withinBytes)` -- the write-side twin of `lds_geometry.component_fold`.

    `writeStrideBytes` is the jump between LDS COMPONENTS, so it may only be multiplied by a
    component index.  An operand carried by more waves than there are components has several waves
    inside one component; those step by `withinBytes`.  `(1, 0)` means one wave per component, where
    the flat `wId * strideBytes` is already correct.
    """
    wavesPerComp = max(1, int(waveCount) // max(1, int(numComp)))
    if wavesPerComp <= 1:
        return (1, 0)
    return (wavesPerComp, int(perWaveBytes))


def seg_interleave_row_shape(kernel, tc: str, tile01: int, groupShape: int) -> int:
    """The LDS row stride under `LDSSegmentInterleave`, or 0 for the baseline layout.

    The interleave rides the segment jump on the wave stride, so a wave's tiles land on different
    rows than `groupShape` -- which is what decides whether its share crosses a TDMSplit region.
    """
    if kernel.get("LDSSegmentInterleave") != 1:
        return 0
    off = kernel.get("LDSSegInterleaveOffsets") or {}
    vw = int(kernel.get("VectorWidth%s" % tc) or 0)
    if vw <= 0 or int(kernel["MIWaveTile"][tile01]) // vw <= 1:
        return 0
    wg = max(1, int(kernel["MIWaveGroup"][tile01]))
    if off.get("portSplit%s" % tc, False):
        return groupShape // wg
    if off.get("componentSplit", False) and off.get("activeTC") == tc:
        return groupShape // max(1, wg // max(1, int(kernel["NumWaves"]) // 2))
    return 0


def wave_region_span(kernel, tc: str) -> int:
    """How many TDMSplit regions ONE WAVE's tiles of `tc` actually occupy.

    NOT the split factor.  `nsplit` says how many regions the TILE is cut into -- how many
    `tensor_load`s the copy issues -- and that is a property of the whole macro-tile.  What the
    READ side needs is different: how many of those regions THIS WAVE reaches, because the wave's
    tiles are distributed by `VectorWidth` and `MIWaveGroup`, not laid out contiguously.

        row(t)    = (t // VW) * (MI * MIBlock * MIWaveGroup * VW) + (t % VW)
        region(t) = row(t) // (MacroTile // nsplit)

    The two coincide only when the wave's distribution lines up with the region cut, which is why
    this went unnoticed: at `MIWaveGroup[1,1]` the groups land exactly on the boundaries, so
    `t // tiles_per_region` is right.  At `MIWaveGroup[2,2]` the group jump is a WHOLE region, so
    a wave's tiles sit adjacent (rows 0 and 1) and are BOTH in region 0 -- while the index form
    splits them across regions and gives the second one the wrong region's memory token.
    `tool/region_derivation_check.check_tiles` is the independent derivation: it disagrees only
    at `MIWaveGroup[2,2] VW=2`, which is the cell this rule exists for.

    The COPY keeps `nsplit` -- it really does write every region.  Whether the LDS address needs
    a region term on M and N is decided by whether the waves' address distribution crosses the
    split region.
    """
    # READ THE FACTOR DIRECTLY, not through `split_of`: this is only ever asked about A and B,
    # where that guard is vacuous, and going through it would need a `ProblemType["Sparse"]` key
    # a hand-built theta dict has no reason to carry.
    mt, _du = TDM_SPLIT_AXIS.get(int(kernel.get("TDMSplit%s" % tc, 0) or 0), (1, 1))
    if mt <= 1:
        return 1                                # unsplit, or a DU split (every region present)
    t01 = 0 if tc == "A" else 1
    mi = list(kernel["MatrixInstruction"])
    miDim = int(mi[t01])
    miBlk = int(mi[3 + t01]) if len(mi) > 4 else 1
    wg = int((kernel.get("MIWaveGroup") or mi[7:9])[t01])
    wt = int((kernel.get("MIWaveTile") or mi[5:7])[t01])
    vw = int(kernel.get("VectorWidth%s" % tc) or 0)
    if vw <= 0:
        # AUTO, and not derived yet at theta-build time -- replay the rule that decides it: the
        # largest power of two that divides `MIWaveTile` and fits one register of elements.
        dt = ((kernel.get("ProblemType") or {}).get("MacDataType%s" % tc))
        regPer = dt.numRegisters() if hasattr(dt, "numRegisters") else None
        if regPer is None:
            return mt
        vw = max(1, int(4 // regPer) if regPer else 1)
        while vw > 1 and wt % vw:
            vw //= 2
    macroTile = wt * wg * miDim * max(1, miBlk)
    groupShape = miDim * max(1, miBlk) * wg * vw
    segShape = seg_interleave_row_shape(kernel, tc, t01, groupShape)
    if segShape:
        # portSplit/componentSplit move the component jump onto the wave base and stack the
        # TDMSplit regions inside each segment.  `tdmSplitLdsBoundary` divides the baseline
        # per-region footprint by numVectorsPerTile in exactly this arm, so the row classifier must
        # shorten BOTH its stride and its region extent.  Shortening only `groupShape` made
        # MIWT[8,8], VW[4,4], WG[2,2], TDMSplitA/B=1, LDSSI=1 look wave-relative for A (which carries
        # portSplitA) and coordinate-relative for B, even though both emitted read layouts are
        # coordinate-relative.
        numVectorsPerTile = max(1, wt // vw)
        groupShape = segShape
        rowsPerRegion = max(1, macroTile // mt // numVectorsPerTile)
    else:
        rowsPerRegion = max(1, macroTile // mt)
    rows = {((t // vw) * groupShape + (t % vw)) // rowsPerRegion for t in range(max(1, wt))}
    return max(1, min(mt, len(rows)))
