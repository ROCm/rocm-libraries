# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure arithmetic for per-tile LDS addresses."""

from __future__ import annotations

from ..Components.TDMSplit import AXIS_MT, split_packs_lds

# Bytes per register.
BPR = 4

# Lanes sharing one range of K positions in the unroll axis.
LANE_INTERLEAVE = 2


def read_fragments(blockWidth, bpeDS, lrvw: int, inputPerThUnroll: int, matrixInstK: int) -> tuple:
    """Return ``(unroll_elements, register_offset)`` pairs for one local-read act."""
    vwTrLoad = int(blockWidth * BPR / bpeDS)
    nInner = lrvw // vwTrLoad if vwTrLoad else 0
    nOuter = inputPerThUnroll // lrvw if lrvw else 0
    regSpan = int(inputPerThUnroll * bpeDS // BPR)
    if not (nInner and nOuter) or nOuter * nInner * int(blockWidth) != regSpan:
        raise NotImplementedError(
            "local-read fragments do not tile the tile's registers (%s x %s x blockWidth %s != "
            "%s; inputPerThUnroll=%s, lrvw=%s, vwTrLoad=%s, bpeDS=%s)"
            % (nOuter, nInner, blockWidth, regSpan, inputPerThUnroll, lrvw, vwTrLoad, bpeDS))
    if nOuter * nInner * vwTrLoad * LANE_INTERLEAVE != matrixInstK:
        raise NotImplementedError(
            "local-read fragments do not cover MatrixInstK (%s x %s x %s elems x %s lanes != %s; "
            "lrvw=%s, bpeDS=%s)"
            % (nOuter, nInner, vwTrLoad, LANE_INTERLEAVE, matrixInstK, lrvw, bpeDS))
    return tuple(((v + i * nInner * LANE_INTERLEAVE) * vwTrLoad,
                  int(blockWidth) * (v + nInner * i))
                 for i in range(nOuter) for v in range(nInner))


def region_split_is_packed(unrollMajor: bool, nsplit: int, axis=AXIS_MT) -> bool:
    """Return whether TDMSplit creates separately packed LDS regions."""
    return nsplit > 1 and split_packs_lds(axis, unrollMajor)


def region_row_elems(extent: int, ldsPad: int, unrollMajor: bool, nsplit: int, axis=AXIS_MT) -> int:
    """Row length, in elements, along the LDS image's INNER axis -- of ONE REGION when the split
    packs regions separately, of the whole tile otherwise.
    """
    packed = region_split_is_packed(unrollMajor, nsplit, axis)
    return (extent // max(1, nsplit) if packed else extent) + ldsPad


def fold_inner_offset(innerElems: int, extent: int, unrollMajor: bool, nsplit: int,
                      axis=AXIS_MT) -> tuple:
    """`(region, innerInRow)` -- split an INNER-axis element offset into the region it lands in and
    its position inside that region.  The counterpart of `region_row_elems`: that shortens the row,
    this says which row-block a coordinate past the end belongs to.
    """
    if not region_split_is_packed(unrollMajor, nsplit, axis):
        return (0, innerElems)
    span = max(1, extent // max(1, nsplit))
    return (int(innerElems) // span, int(innerElems) % span)


def addr_coord_on_split_axis(within: int, flat: int, ctx) -> int:
    """Select the within-region or flat coordinate for a split axis."""
    return within if region_split_is_packed(
        ctx.unrollMajor, ctx.nsplit, getattr(ctx, "splitAxis", AXIS_MT)) else flat


def free_position(ctx, t: int) -> int:
    """This wave-tile's ELEMENT POSITION on the free axis -- the same in either layout.

    The distribution is BY VECTOR GROUP: `vw` adjacent tiles, then the next group starts one
    wave-group further along.  Where that position lands in the LDS image is the layout's
    business (`tile_row`); the position itself is not, which is why `wave_region_span`,
    `localReadDo` and this module can all ask for it and get one answer.
    """
    vw = max(1, ctx.vectorWidth)
    group = ctx.MIWaveGroupShape[ctx.tile01] if ctx.MIWaveGroupShape else vw
    return (t // vw) * (getattr(ctx, "segRowShape", 0) or group) + (t % vw)


def macro_tile_extent(ctx) -> int:
    """The operand's free-axis extent, derived so a context need not carry it twice."""
    mt = getattr(ctx, "macroTile", 0)
    if mt:
        return int(mt)
    vw = max(1, ctx.vectorWidth)
    group = ctx.MIWaveGroupShape[ctx.tile01] if ctx.MIWaveGroupShape else vw
    return max(1, int(ctx.miWaveTileAxis) // vw) * int(group)


def tile_row(ctx, t: int) -> int:
    """Where wave-tile `t` sits along the LDS image's OUTER-indexed coordinate.

    unroll-major `[free][unroll]` : the free axis IS the outer one, so the row is the free
        position and the caller multiplies it by the row length.
    tile-major   `[unroll][free]` : the free axis is the INNER one and has stride 1, so the
        position folds into its region and the caller multiplies by 1.  The region term is
        `region_bytes`, exactly as `localReadDo`'s `splitRegionOfTile` computes it.

    Both arms return the free position; only the split fold differs, because only a packed
    split shortens the row this coordinate runs along.
    """
    pos = free_position(ctx, t)
    if ctx.unrollMajor:
        return pos
    _region, inRow = fold_inner_offset(pos, macro_tile_extent(ctx), False, ctx.nsplit,
                                       getattr(ctx, "splitAxis", AXIS_MT))
    return inRow


def component_fold(ctx, row: int) -> tuple:
    """`(rowInComponent, componentBytes)` -- LDSSegmentInterleave stores an operand's two
    components `segWriteStrideBytes` apart, so a row past the end of component 0 belongs to
    component 1 and owes that jump.  Post-pad, exactly like `region_bytes`.
    """
    cols = getattr(ctx, "segCompCols", 0)
    if cols <= 0:
        return (row, 0)
    return (row % cols, (row // cols) * int(ctx.segWriteStrideBytes))


def tiles_per_region(ctx) -> int:
    """How many of this operand's wave tiles live in ONE TDMSplit region."""
    if getattr(ctx, "splitAxis", AXIS_MT) != AXIS_MT:
        return max(1, ctx.miWaveTileAxis)
    return max(1, ctx.miWaveTileAxis // max(1, ctx.nsplit))


def region_bytes(ctx, region: int) -> int:
    """Return the byte displacement from the LDS base to a split region."""
    if (not region_split_is_packed(ctx.unrollMajor, ctx.nsplit, getattr(ctx, "splitAxis", AXIS_MT))
            or not ctx.splitBoundaryBytes):
        return 0
    return int(region) * int(ctx.splitBoundaryBytes)
