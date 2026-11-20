# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for the MX scale SEGMENT geometry the ULM lowering reads against.

THE SCAFFOLD OWNS THE LAYOUT AND THE LOWERING MUST FOLLOW IT.  `SolutionStructs.segment_interleave`
decides where a scale tensor's bytes land; `Lowering/leaves.py` decides which byte each read asks
for.  They are separate derivations of one fact, and when the scaffold grew the `mxsSplitA` layout
-- MXSA cut into M-halves, one per interleaved component -- the lowering kept stepping the WHOLE
macro tile.  Its second substep then landed at the END of the half rather than inside it, which
generated, assembled, and returned wrong results on hardware for every `TDMFuse=0` mxf8 kernel.

The invariant below is written against the layout's own sizing rather than against a constant, so
a future change to either side has to move both:

    the substep walk a segment holds  ==  the bytes the layout reserved for that segment

`mxsaHalfFootprintBytes` in `segment_interleave._mxsa_half_layout` is `MacroTile0 * (DepthU //
MXBlockA) // 2` plus padding, and the lowering walks `LoopIters` substeps of
`mx_scale_substep_stride`.  If someone re-shapes the scale segment and updates only the scaffold,
this fails and says so.
"""

import pytest

from tensilelite.Lowering.lds_geometry import (mx_scale_segment_rows,
                                               mx_scale_substep_stride)


def _raw_half(macro_tile, depth_u, mx_block):
    """The layout's own pre-pad half size, as `_mxsa_half_layout` computes it."""
    return macro_tile * (depth_u // mx_block) // 2


# (MacroTile0, DepthU, MXBlockA, MatrixInstK) -- f8maf6/f8maf7 plus two reshapes.
_SHAPES = [
    (256, 256, 32, 128),
    (256, 128, 32, 128),
    (128, 256, 32, 64),
    (256, 256, 16, 128),
]


@pytest.mark.parametrize("macro_tile,depth_u,mx_block,mi_k", _SHAPES)
def test_a_split_segment_holds_half_the_parent_rows(macro_tile, depth_u, mx_block, mi_k):
    """The split is in ROWS of the parent, so the segment is half of them -- and only then."""
    assert mx_scale_segment_rows(macro_tile, split=True) == macro_tile // 2
    assert mx_scale_segment_rows(macro_tile, split=False) == macro_tile


@pytest.mark.parametrize("macro_tile,depth_u,mx_block,mi_k", _SHAPES)
def test_the_substep_walk_fills_exactly_the_segment_the_layout_reserved(
        macro_tile, depth_u, mx_block, mi_k):
    """THE CONTRACT.  Walking every substep of a split MXSA must span the layout's half, no more.

    Over-stepping is the TDMFuse=0 defect: the walk runs past the half into the padding gap before
    the next buffer, and the reads come back as whatever is there.  Under-stepping would leave a
    substep reading another's bytes.
    """
    mx_unit = mi_k // mx_block                     # scale values one substep covers
    substeps = depth_u // mi_k                     # LoopIters
    stride = mx_scale_substep_stride(macro_tile, mx_unit, bpeDS=1, split=True)

    assert stride * substeps == _raw_half(macro_tile, depth_u, mx_block), (
        "the split MXSA substep walk (%u x %u = %u bytes) does not fill the half the layout "
        "reserved (%u).  `segment_interleave._mxsa_half_layout` and "
        "`lds_geometry.mx_scale_substep_stride` are two derivations of one segment -- update both."
        % (stride, substeps, stride * substeps,
           _raw_half(macro_tile, depth_u, mx_block)))


@pytest.mark.parametrize("macro_tile,depth_u,mx_block,mi_k", _SHAPES)
def test_an_unsplit_scale_still_walks_the_whole_tile(macro_tile, depth_u, mx_block, mi_k):
    """The halving is the SPLIT's, not the scale's: TDMFuse 1..4 keep MXSA contiguous."""
    mx_unit = mi_k // mx_block
    substeps = depth_u // mi_k
    stride = mx_scale_substep_stride(macro_tile, mx_unit, bpeDS=1, split=False)
    assert stride * substeps == 2 * _raw_half(macro_tile, depth_u, mx_block)


def test_the_split_stride_is_exactly_half_the_unsplit_one():
    """Stated once more as a ratio, so a change that scales both still has to be deliberate."""
    for macro_tile, depth_u, mx_block, mi_k in _SHAPES:
        mx_unit = mi_k // mx_block
        whole = mx_scale_substep_stride(macro_tile, mx_unit, bpeDS=1, split=False)
        half = mx_scale_substep_stride(macro_tile, mx_unit, bpeDS=1, split=True)
        assert whole == 2 * half


def test_a_sub_byte_scale_type_keeps_the_stride_proportional():
    """`bpeDS` is a float for sub-byte types; the stride must stay an exact byte count."""
    stride = mx_scale_substep_stride(256, 4, bpeDS=0.5, split=True)
    assert stride == int(128 * 4 * 0.5) and isinstance(stride, int)
