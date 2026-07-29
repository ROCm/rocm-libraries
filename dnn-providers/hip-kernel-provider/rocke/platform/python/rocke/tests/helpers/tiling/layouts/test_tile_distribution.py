# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the quantity-major (struct-of-arrays) make_tile_desc surface (offline, no GPU).

`make_tile_desc` returns a `TileDesc` (shape + the derived layout) in one call; the load-bearing
assertion is reproduction -- authoring a distribution as one axes-ordered list per geometric
quantity (columns = axes) must derive the SAME raw WarpDistributionEncoding a human would
hand-write (`td.layout`), proven against ck_tile's MakeADramTileDistribution figure.
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling.layouts import make_tile_desc


def test_reproduces_ck_tile_a_dram_distribution() -> None:
    """ck_tile MakeADramTileDistribution (M=256, K=32): the geometric table IS the input."""
    td = make_tile_desc(
        shape=[256, 32],
        thread_dist=[16, 4],
        wave_dist=[4, 1],
        thread_tile=[1, 8],
        block_repeat=[4, 1],
        wave_size=64,
    )
    assert td.shape == (256, 32)   # the TileDesc carries the shape (derivable from Hs products)
    enc = td.layout
    assert enc.replication_lengths == ()
    assert enc.hierarchical_lengths == ((4, 4, 16), (4, 8))
    assert enc.lane_to_rh_major == ((1,), (1, 2))   # P0 wave (M), P1 lane (M, K)
    assert enc.lane_to_rh_minor == ((1,), (2, 0))
    assert enc.register_to_rh_major == (1, 2)        # Y0 M-block_repeat, Y1 K-capture
    assert enc.register_to_rh_minor == (0, 1)


def test_c_accumulator_capture_on_first_axis() -> None:
    """A dense C accumulator (M capture, N lanes): capture rides the first column, still valid."""
    enc = make_tile_desc(
        shape=[16, 16],
        thread_dist=[4, 16],
        thread_tile=[4, 1],
        wave_size=64,
    ).layout
    assert enc.hierarchical_lengths == ((4, 4), (16,))  # M: lane4 then capture4 ; N: lane16
    assert enc.lane_to_rh_major == ((1, 2),)            # M lane then N lane (M-major)
    assert enc.lane_to_rh_minor == ((0, 0),)
    assert enc.register_to_rh_major == (1,)             # capture on M
    assert enc.register_to_rh_minor == (1,)             # M slot 1 (after its lane)


def test_defaults_fill_ones() -> None:
    td = make_tile_desc(shape=[64], thread_dist=[64], wave_size=64)
    assert td.shape == (64,)
    assert td.register_count == 1                        # no captures/block_repeats -> 1 reg per lane
    assert td.layout.hierarchical_lengths == ((64,),)
    assert td.layout.lane_to_rh_major == ((1,),)
    assert td.layout.register_to_rh_major == ()


def test_thread_broadcast_adds_replication_referenced_by_lane() -> None:
    """A full-duplicate half-wave: 16 data lanes x2 thread_broadcast = 32-lane wave, R referenced."""
    enc = make_tile_desc(shape=[16], thread_dist=[16], wave_size=32, thread_broadcast=2).layout
    assert enc.replication_lengths == (2,)
    assert enc.lane_to_rh_major == ((0, 1),)   # R (the duplicated half) is the high-order lane bit
    assert enc.lane_to_rh_minor == ((0, 0),)


def test_rdna3_wmma16_a_duplicated_inputs() -> None:
    """RDNA3 WMMA16x16x16 A on wave32: each lane holds a full 16-K row; the 16 M-rows fill lanes
    0-15 and are DUPLICATED onto lanes 16-31 -> thread_broadcast=2 as the high-order lane bit."""
    enc = make_tile_desc(
        shape=[16, 16],        # M, K
        thread_dist=[16, 1],    # M -> 16 lanes
        thread_tile=[1, 16],   # each lane grabs all 16 K contiguous
        wave_size=32,
        thread_broadcast=2,           # rows 0-15 duplicated onto lanes 16-31
    ).layout
    assert enc.replication_lengths == (2,)
    assert enc.hierarchical_lengths == ((16,), (16,))
    assert enc.lane_to_rh_major == ((0, 1),)   # R (half-wave) major, then the M lane
    assert enc.lane_to_rh_minor == ((0, 0),)
    assert enc.register_to_rh_major == (2,)    # thread_tile on K
    assert enc.register_to_rh_minor == (0,)


def test_mxfp6_scale_thread_broadcast_across_k_blocks() -> None:
    """gfx950 mxfp6 A-scale (per K-block, wave64): one E8M0 scale per M-row, duplicated across the
    4 k_blk lane-groups -- lanes m, m+16, m+32, m+48 share it -> thread_broadcast=4 (a separate operand
    tile from the fp6 mantissa MFMA)."""
    enc = make_tile_desc(shape=[16], thread_dist=[16], thread_broadcast=4, wave_size=64).layout
    assert enc.replication_lengths == (4,)
    assert enc.hierarchical_lengths == ((16,),)
    assert enc.lane_to_rh_major == ((0, 1),)   # R (the 4x duplicate) leads, then the M lane
    assert enc.lane_to_rh_minor == ((0, 0),)
    assert enc.register_to_rh_major == ()      # one scale value per lane, no registers


def test_thread_order_reorders_lane_significance() -> None:
    """Row-major A as an MMA operand: shape=(M,K) with K contiguous (column 1), but the atom wants
    K as the major lane -> thread_order=[1,0] flips the merge without touching the axis identities."""
    default = make_tile_desc(
        shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4], wave_size=64
    ).layout
    flipped = make_tile_desc(
        shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4], thread_order=[1, 0], wave_size=64
    ).layout
    assert default.lane_to_rh_major == ((1, 2),)   # column order: M (axis 0) is the major lane
    assert flipped.lane_to_rh_major == ((2, 1),)   # thread_order: K (axis 1) is the major lane
    # only the lane significance changed -- Hs and registers are identical
    assert flipped.hierarchical_lengths == ((16,), (4, 4))
    assert flipped.lane_to_rh_minor == ((0, 0),)
    assert flipped.register_to_rh_major == (2,)
    assert flipped.register_to_rh_minor == (1,)


def test_thread_order_default_is_column_order() -> None:
    a = make_tile_desc(shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4], wave_size=64)
    b = make_tile_desc(
        shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4], thread_order=[0, 1], wave_size=64
    )
    assert a == b   # [0,1] IS the column-order default (TileDesc equality: shape + layout)


def test_thread_order_must_be_permutation_of_lane_axes() -> None:
    with pytest.raises(ValueError, match="permutation of the axes that carry threads"):
        make_tile_desc(
            shape=[16, 16], thread_dist=[16, 4], thread_tile=[1, 4], thread_order=[0, 0], wave_size=64
        )


def test_wave_order_permutes_wave_layout() -> None:
    """wave_order reorders which wave gets which slice -- same convention (fastest right-most)."""
    default = make_tile_desc(shape=[4, 4], thread_dist=[2, 2], wave_dist=[2, 2], wave_size=4)
    flipped = make_tile_desc(
        shape=[4, 4], thread_dist=[2, 2], wave_dist=[2, 2], wave_order=[1, 0], wave_size=4
    )
    # P0 is the WAVE entry; its order is the wave merge across axes
    assert default.layout.lane_to_rh_major[0] == (1, 2)   # column order: axis 0 major wave
    assert flipped.layout.lane_to_rh_major[0] == (2, 1)   # wave_order flips it


def test_wave_broadcast_adjacent_pairs() -> None:
    """wave_broadcast=[1,2] duplicates across ADJACENT waves (W0==W1, W2==W3): R is the low wave
    bit. 2 distinct M-waves x 2 = 4 physical waves; A block-tile 32x16."""
    enc = make_tile_desc(
        shape=[32, 16], thread_tile=[1, 4], thread_dist=[16, 4], thread_order=[1, 0],
        wave_dist=[2, 1], wave_broadcast=[1, 2], wave_size=64,
    ).layout
    assert enc.replication_lengths == (2,)
    assert enc.lane_to_rh_major[0] == (1, 0)   # wave entry = [M-wave, R] -> wave = m_wave*2 + copy
    assert enc.lane_to_rh_minor[0] == (0, 0)


def test_wave_broadcast_whole_is_half_split() -> None:
    """wave_broadcast=2 (int) duplicates the whole wave-tile: R at the MAJOR wave bit -> W0==W2."""
    enc = make_tile_desc(
        shape=[32, 16], thread_tile=[1, 4], thread_dist=[16, 4], thread_order=[1, 0],
        wave_dist=[2, 1], wave_broadcast=2, wave_size=64,
    ).layout
    assert enc.replication_lengths == (2,)
    assert enc.lane_to_rh_major[0] == (0, 1)   # wave entry = [R, M-wave]
    assert enc.lane_to_rh_minor[0] == (0, 0)


def test_length_factor_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="!= shape"):
        make_tile_desc(shape=[16], thread_dist=[4], thread_tile=[2], wave_size=4)  # 4*2=8 != 16


def test_lanes_must_cover_wave_size() -> None:
    with pytest.raises(ValueError, match="wave_size"):
        make_tile_desc(shape=[16], thread_dist=[16], wave_size=64)  # 16 != 64


def test_rank_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="one entry per axis"):
        make_tile_desc(shape=[16, 16], thread_dist=[16], wave_size=64)


def test_thread_broadcast_size_count_sandwich() -> None:
    """thread_broadcast=[size, count] wedges the duplicate BETWEEN lane levels (the 'sandwich'):
    lanes 0-7 -> 8-15, lanes 16-23 -> 24-31, over thread_dist=[2, 8]."""
    enc = make_tile_desc(
        shape=[2, 8],
        thread_dist=[2, 8],
        wave_size=32,
        thread_broadcast=[8, 2],   # duplicate each 8-lane block, twice
    ).layout
    assert enc.replication_lengths == (2,)
    assert enc.hierarchical_lengths == ((2,), (8,))
    assert enc.lane_to_rh_major == ((1, 0, 2),)   # outer axis, R, inner axis
    assert enc.lane_to_rh_minor == ((0, 0, 0),)


def test_thread_broadcast_size_must_align_to_lane_boundary() -> None:
    with pytest.raises(ValueError, match="does not align to a lane boundary"):
        make_tile_desc(shape=[16], thread_dist=[16], wave_size=32, thread_broadcast=[8, 2])


def test_non_int_entry_rejected() -> None:
    with pytest.raises(TypeError, match="must be ints"):
        make_tile_desc(shape=[16], thread_dist=[True], wave_size=1)
