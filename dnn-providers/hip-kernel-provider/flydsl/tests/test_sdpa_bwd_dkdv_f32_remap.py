# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The gfx11 fp32 dK/dV kernel's rank-to-tile remap visits exactly the live Q tiles, once each.

`fmha_bwd_dkdv_f32_gfx11.py` runs one flat loop over `it` in `[0, group * n_blocks)` and, for a causal / windowed KV tile, remaps the per-head rank to a
Q tile: full tiles first, then the leading masked tiles, then the trailing masked tiles, each ascending. A wrong remap silently drops or doubles a tile,
so the check is exhaustive over small shapes. Pure Python: the mirror below follows the kernel's `step` prologue line by line (Int32 `//` lowers to a
floor division, as Python's does). The oracle is brute force over (q, kv) pairs and does not use the mirror.
"""

from __future__ import annotations

import itertools

import pytest

WINDOW_TOPLEFT = -2147483647  # 0x80000001
WINDOW_BOTRIGHT = -2147483646  # 0x80000002


def smin(a, b):
    return min(a, b)


def smax(a, b):
    return max(a, b)


def resolve_window(window_left, window_right, seqlen_q, seqlen_k):
    left = seqlen_q if window_left in (WINDOW_TOPLEFT, WINDOW_BOTRIGHT) else window_left
    right = 0 if window_right == WINDOW_TOPLEFT else window_right
    if window_right == WINDOW_BOTRIGHT:
        right = seqlen_k - seqlen_q
    return left, right


def kernel_tiles(sq, sk, window_left, window_right, BM, BN, kv_tile):
    """Mirror of the kernel's causal prologue: `(first_blk, n_blocks, pre, nfull)` of the workgroup owning KV tile `kv_tile`."""
    start_k = kv_tile * BN
    _alive = start_k < sk
    _wl, _wr = resolve_window(window_left, window_right, sq, sk)
    _wl = smin(smax(_wl, 0 - sk), sq)
    _wr = smin(smax(_wr, 0 - sq), sk)
    _span_w = _wl + _wr + 1
    _kv_end = smin(start_k + BN, sk)
    _q_lo = smax(start_k - _wr, 0)
    _q_hi = smin(_kv_end + _wl, sq)
    _first_blk = _q_lo // BM
    _any_live = _alive and (_span_w > 0) and (_q_hi > _q_lo)
    _n_blocks_i32 = ((_q_hi + (BM - 1)) // BM - _first_blk) if _any_live else 0
    _ff = 0 - ((0 - (_kv_end - 1 - _wr)) // BM)
    _lf = (start_k - (BM - 1) + _wl) // BM
    _pre = smin(smax(_ff - _first_blk, 0), _n_blocks_i32)
    _hi = smin(smax(_lf + 1 - _first_blk, _pre), _n_blocks_i32)
    _nfull = _hi - _pre
    return _first_blk, _n_blocks_i32, _pre, _nfull


def visited(sq, sk, window_left, window_right, BM, BN, kv_tile, group):
    """Mirror of the loop and of `step`'s head / tile decode: the `(head, tile)` of every iteration, in order."""
    _first_blk, _n_blocks_i32, _pre, _nfull = kernel_tiles(
        sq, sk, window_left, window_right, BM, BN, kv_tile
    )
    _nfull_pre = _nfull + _pre
    _neg_nfull = 0 - _nfull
    _nb_safe_i32 = smax(_n_blocks_i32, 1)
    _total_i32 = group * _n_blocks_i32
    out = []
    for it in range(_total_i32):
        _g = it // _nb_safe_i32
        _bi = it - _g * _nb_safe_i32
        _bi = (
            _first_blk
            + _bi
            + (_pre if _bi < _nfull else (_neg_nfull if _bi < _nfull_pre else 0))
        )
        out.append((_g, _bi))
    return out


def visible(i, c, wl, wr):
    return i - wl <= c <= i + wr


def live_tiles(sq, sk, window_left, window_right, BM, BN, kv_tile):
    """Q tiles holding at least one visible, in-bounds (q, kv) pair for the KV tile."""
    wl, wr = resolve_window(window_left, window_right, sq, sk)
    k0, k1 = kv_tile * BN, min(kv_tile * BN + BN, sk)
    return {i // BM for i in range(sq) for c in range(k0, k1) if visible(i, c, wl, wr)}


def truly_full(sq, sk, window_left, window_right, BM, BN, kv_tile, t):
    wl, wr = resolve_window(window_left, window_right, sq, sk)
    k0 = kv_tile * BN
    return all(
        i < sq and c < sk and visible(i, c, wl, wr)
        for i in range(t * BM, t * BM + BM)
        for c in range(k0, k0 + BN)
    )


def windows(sq, sk, BM, BN):
    wls = {
        WINDOW_TOPLEFT,
        -sk - 3,
        -sk,
        -sk + 1,
        -BN,
        -1,
        0,
        1,
        BN - 1,
        BN,
        sq - 1,
        sq,
        sq + 5,
        *range(-3, 4),
    }
    wrs = {
        WINDOW_TOPLEFT,
        WINDOW_BOTRIGHT,
        -sq - 3,
        -sq,
        -sq + 1,
        -BM,
        -1,
        0,
        1,
        BM,
        sk - sq,
        sk - 1,
        sk,
        sk + 5,
        *range(-3, 4),
    }
    return itertools.product(sorted(wls), sorted(wrs))


def cases(geoms, max_len, stride):
    for BM, BN in geoms:
        lens = range(1, max_len + 1, stride)
        for sq, sk in itertools.product(lens, lens):
            for wl, wr in windows(sq, sk, BM, BN):
                for kv_tile in range(
                    (sk + BN - 1) // BN + 1
                ):  # +1: a dead workgroup past the sequence (varlen grid)
                    yield BM, BN, sq, sk, wl, wr, kv_tile


# (BM, BN): the fp32 ladder's loop-tile shapes, and small ones that make every region boundary (ragged, one-tile, empty) common.
SHAPES = [
    ((4, 4), 20, 1),
    ((8, 4), 20, 1),
    ((8, 8), 24, 1),
    ((16, 16), 40, 3),
    ((32, 16), 40, 3),
    ((32, 32), 40, 3),
]


@pytest.mark.parametrize(("geom", "max_len", "stride"), SHAPES, ids=lambda v: str(v))
def test_remap_is_a_bijection_onto_the_live_tiles(geom, max_len, stride):
    """Every iteration maps to a distinct (head, tile); together they are exactly the heads times the tiles [first, first + n_blocks), heads in order.

    The covered windows are clamped (wider than the sequence), negative (a band right of the diagonal), empty (n_blocks == 0, dead workgroups
    included) and ragged (Sq, Sk not multiples of the tile); every GQA group of 1, 2 and 3 is checked.
    """
    count = 0
    for BM, BN, sq, sk, wl, wr, kv_tile in cases([geom], max_len, stride):
        first, n, pre, nfull = kernel_tiles(sq, sk, wl, wr, BM, BN, kv_tile)
        assert 0 <= pre <= pre + nfull <= n, (BM, BN, sq, sk, wl, wr, kv_tile)
        live = live_tiles(sq, sk, wl, wr, BM, BN, kv_tile)
        assert live <= set(range(first, first + n)), (
            "the kernel's range misses a live tile",
            BM,
            BN,
            sq,
            sk,
            wl,
            wr,
            kv_tile,
        )
        for group in (1, 2, 3):
            got = visited(sq, sk, wl, wr, BM, BN, kv_tile, group)
            want = [(g, first + r) for g in range(group) for r in range(n)]
            assert len(set(got)) == len(got), (
                "a tile is visited twice",
                BM,
                BN,
                sq,
                sk,
                wl,
                wr,
                kv_tile,
                group,
            )
            assert sorted(got) == want, (
                "not a bijection",
                BM,
                BN,
                sq,
                sk,
                wl,
                wr,
                kv_tile,
                group,
            )
            heads = [g for g, _ in got]
            assert heads == sorted(heads), (
                "heads interleave",
                BM,
                BN,
                sq,
                sk,
                wl,
                wr,
                kv_tile,
                group,
            )
        count += 1
    assert count > 0


@pytest.mark.parametrize(("geom", "max_len", "stride"), SHAPES, ids=lambda v: str(v))
def test_full_tiles_are_unmasked_and_come_first(geom, max_len, stride):
    """Each head visits the `nfull` tiles before any other, and an in-bounds tile counted full is really unmasked.

    A ragged tile (rows past Sq or columns past Sk) may count as full: masking is by absolute position inside the step, so only the order is affected.
    """
    for BM, BN, sq, sk, wl, wr, kv_tile in cases([geom], max_len, stride):
        first, n, pre, nfull = kernel_tiles(sq, sk, wl, wr, BM, BN, kv_tile)
        full = set(range(first + pre, first + pre + nfull))
        for t in full:
            if (t + 1) * BM > sq or kv_tile * BN + BN > sk:
                continue
            assert truly_full(sq, sk, wl, wr, BM, BN, kv_tile, t), (
                BM,
                BN,
                sq,
                sk,
                wl,
                wr,
                kv_tile,
                t,
            )
        got = visited(sq, sk, wl, wr, BM, BN, kv_tile, 2)
        for g in range(2):
            flags = [t in full for h, t in got if h == g]
            assert flags == sorted(flags, reverse=True), (
                BM,
                BN,
                sq,
                sk,
                wl,
                wr,
                kv_tile,
                g,
            )


def test_a_workgroup_past_the_sequence_visits_nothing():
    assert kernel_tiles(8, 8, WINDOW_TOPLEFT, 0, 4, 4, 2)[1] == 0
    assert visited(8, 8, WINDOW_TOPLEFT, 0, 4, 4, 2, 3) == []


def test_top_left_causal_moves_the_diagonal_tile_last():
    """Square top-left causal, BM == BN: KV tile 1 of 4 visits Q tiles 2, 3 (full) and then 1 (the diagonal), per head."""
    assert visited(16, 16, WINDOW_TOPLEFT, WINDOW_TOPLEFT, 4, 4, 1, 1) == [
        (0, 2),
        (0, 3),
        (0, 1),
    ]
