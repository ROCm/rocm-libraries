# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for SPARSE under UseLoopModel.

THE METADATA IS A COMPANION OPERAND.  A 2:4 problem's sparsity index rides the compressed
tensor's walk exactly as an MX scale rides its parent's: one fragment per parent tile, never
region-split, its own LDS ring and its own register ring, consumed by the same instruction as a
third source.  The only thing that makes it metadata rather than a scale is which instruction the
leaf emits -- `v_swmmac` versus `v_wmma ... matrix_*_scale` -- and no instruction carries both.

Two facts are pinned here because a change to either side alone is silently wrong:

1.  **Which tensor the metadata belongs to.**  `Sparse==2` compresses B, everything else A.  The
    letter of the operand name does NOT say it (unlike `MXSA`), so every site that needs the
    parent must ask, and all of them must agree.

2.  **K positions one stored entry covers.**  A dense tensor stores every position, a compressed
    tensor half of them, a metadata byte indexes eight.  `MIInputPerThread` is the DENSE count the
    instruction consumes and `MIInputPerThread{tc}` what the operand holds, so the ratio states
    all three -- and `read_fragments` must be given `MatrixInstK` divided by it, or a compressed
    tensor's fragments "do not cover MatrixInstK" and the kernel fails to generate.

If someone reshapes the metadata tile or changes how the solution derives
`MIInputPerThreadMetadata`, these fail and name what else has to move.
"""

import pytest

from tensilelite.Components.LoopModel.Registers import _parentSide
from tensilelite.Lowering.lds_geometry import read_fragments
from tensilelite.Lowering.leaves import LeafEmitters
from tensilelite.LoopModel.adapter import (COMPANIONS, METADATA, _is_companion,
                                           _metadata_frag_elems, _metadata_parent)


def _k_per_entry(mi_input_per_thread, mi_input_per_thread_tc):
    """The leaf's own rule, restated: the dense count over this operand's count."""
    return max(1, int(mi_input_per_thread) // max(1, mi_input_per_thread_tc))


# (MatrixInstK, MIInputPerThread, MIInputPerThreadA_sparse, MIInputPerThreadMetadata)
# The first row is the f8 16x16x128 wave32 shape the sparse gate generates.
_SHAPES = [
    (128, 64, 32, 8),
    (64, 32, 16, 4),
    (128, 32, 16, 4),
]


def test_metadata_is_a_companion_but_not_an_mx_scale():
    """It takes the companion branch everywhere, and its name does not end in a parent letter."""
    assert _is_companion(METADATA)
    assert METADATA in COMPANIONS
    assert all(_is_companion(name) for name in COMPANIONS)
    assert not _is_companion("A") and not _is_companion("B")
    # The MXS branch used to be spelled `"MXS" in name`; metadata must NOT match that spelling,
    # which is why the predicate is a named set rather than a substring test.
    assert "MXS" not in METADATA


@pytest.mark.parametrize("sparse,parent", [(1, "A"), (2, "B"), (0, "A")])
def test_the_parent_comes_from_the_problem_not_the_name(sparse, parent):
    """`Sparse==2` compresses B.  Nothing in the operand's name says so."""
    assert _metadata_parent({"Sparse": sparse}) == parent
    # ... and the register-side lookup used by the Valu allocation must give the same answer.
    assert _parentSide({"ProblemType": {"Sparse": sparse}}, METADATA) == parent
    # An MX scale's letter still decides its own, unchanged.
    assert _parentSide({"ProblemType": {"Sparse": sparse}}, "MXSA") == "A"
    assert _parentSide({"ProblemType": {"Sparse": sparse}}, "MXSB") == "B"


@pytest.mark.parametrize("mi_k,ipt,ipt_a,ipt_m", _SHAPES)
def test_k_positions_per_entry_is_the_dense_count_over_the_operands(mi_k, ipt, ipt_a, ipt_m):
    """One rule covers the dense tensor, the compressed one and the index."""
    assert _k_per_entry(ipt, ipt) == 1          # B, uncompressed
    assert _k_per_entry(ipt, ipt_a) == 2        # A, 2:4 compressed
    assert _k_per_entry(ipt, ipt_m) == 8        # the index: one byte per 8 K positions
    # What the read leaf hands `read_fragments`: the K span the operand's own entries cover.
    assert mi_k // _k_per_entry(ipt, ipt_a) == mi_k // 2
    assert mi_k // _k_per_entry(ipt, ipt_m) == mi_k // 8


@pytest.mark.parametrize("mi_k,ipt,ipt_a,ipt_m", _SHAPES)
def test_the_metadata_fragment_is_its_per_thread_byte_count(mi_k, ipt, ipt_a, ipt_m):
    """A metadata LDS entry IS a byte, so the tile's element count is the byte count.

    theta sizes the register ring off this; if the solution ever stores the index at another
    width, `elem_bytes=1` in `_build_operands` has to move with it.
    """
    assert _metadata_frag_elems({"MIInputPerThreadMetadata": ipt_m}) == ipt_m
    # Never zero: a companion with no fragment would get no ring at all.
    assert _metadata_frag_elems({}) == 1


# --- 3.  How a SWMMAC operand's K span is laid out in LDS -------------------------------------
#
# A dense WMMA lane holds its K span as single-read runs that alternate with the partner thread
# group's.  A SWMMAC lane holds TWO longer runs instead -- `blocksPerTGroupSMFMA` in
# `Components/LocalRead.py`.  Reading a SWMMAC operand at the dense granule therefore picks up
# the partner group's K positions, which is silently wrong for every output element rather than
# a generation failure, so it is pinned here.

# One `ds_load_b128` of bytes: the local read every f8 row below emits.
_B128, _BYTE, _MI_TO, _WAVE32 = 4, 1, 16, 32

# (MIInputPerThread{tc}, the K span this operand's own entries cover, fragments)
_SWMMAC_READS = [
    (64, 128, ((0, 0), (16, 4), (64, 8), (80, 12))),   # the dense operand: two runs of 32
    (32, 64, ((0, 0), (32, 4))),                       # the 2:4 compressed one: two runs of 16
]


def _lane_block(sparse, ipt, lrvw=16):
    return LeafEmitters._laneBlock({"WavefrontSize": _WAVE32, "ProblemType": {"Sparse": sparse}},
                                   _MI_TO, ipt, lrvw)


@pytest.mark.parametrize("ipt,k_span,expected", _SWMMAC_READS)
def test_a_swmmac_operand_is_read_as_two_runs(ipt, k_span, expected):
    """Both data operands -- the compressed one and its dense partner -- get two runs."""
    blockElems, laneInterleave = _lane_block(1, ipt)
    assert (blockElems, laneInterleave) == (ipt // 2, 2)
    assert read_fragments(_B128, _BYTE, blockElems, ipt, k_span, laneInterleave) == expected


def test_the_dense_granule_is_wrong_for_the_uncompressed_operand_only():
    """Why the defect survived review: on the COMPRESSED operand the two rules coincide.

    At MI 16x16x128 a compressed lane's run is 16 elements, which is exactly one `ds_load_b128`,
    so reading it at the dense granule lands on the same addresses.  Its dense partner's run is
    32, and there the two disagree -- so a fix checked only against A reads as correct.
    """
    dense_granule, lanes = _lane_block(0, 64)
    assert dense_granule == 16
    assert read_fragments(_B128, _BYTE, dense_granule, 64, 128, lanes) \
        == ((0, 0), (32, 4), (64, 8), (96, 12))        # NOT the SWMMAC layout
    assert read_fragments(_B128, _BYTE, dense_granule, 32, 64, lanes) \
        == read_fragments(_B128, _BYTE, *_lane_block(1, 32)[:1], 32, 64, lanes)


def test_a_dense_problem_keeps_the_single_read_run():
    """The granule a non-sparse operand gets is its LocalReadVectorWidth, unchanged."""
    for lrvw in (8, 16):
        assert _lane_block(0, 64, lrvw) == (lrvw, 2)
    # The interleave is derived, not assumed: it is the number of thread groups the wave splits
    # into over the instruction's tile axis.
    assert LeafEmitters._laneBlock({"WavefrontSize": 64, "ProblemType": {"Sparse": 0}},
                                   _MI_TO, 64, 16) == (16, 4)
