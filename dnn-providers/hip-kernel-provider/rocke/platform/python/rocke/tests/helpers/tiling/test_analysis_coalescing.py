# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Standalone tests for the pure ANALYSIS layer (no matplotlib) -- the shared coalescing/vectorization backend.

Grounds the model on its own: the b128 cap, the fused-vs-scattered cache-line count, direction, and that it is
GENERIC (any distribution + any strides, never a fixed descriptor)."""

import pytest

from rocke.helpers.tiling.analysis import (
    addr_fn_from_strides, analyze_coalescing, assert_asm_backed, vector_transactions)
from rocke.helpers.tiling.transforms import interleave_idx


def _native_c():
    # interleaved f32 C accumulator (CRC recipe): 64 f32/lane, rectangular 16-M x 4-N per lane
    return {(L, mi * 16 + nj * 4 + r): (mi + 4 * (4 * (L // 16) + r), nj + 4 * (L % 16))
            for L in range(64) for mi in range(4) for nj in range(4) for r in range(4)}


def _shuffle(fwd, perm):
    return {(L, i): fwd[(L, perm[i])] for L in range(64) for i in range(len(perm))}


def test_addr_fn_from_strides_is_generic():
    a = addr_fn_from_strides((1, 64))          # M stride-1 (col-major)
    assert a(0, 0) == 0 and a(3, 0) == 3 and a(0, 1) == 64
    b = addr_fn_from_strides((64, 1))          # N stride-1 (row-major)
    assert b(0, 0) == 0 and b(0, 3) == 3 and b(1, 0) == 64


def test_b128_cap_from_vector_transactions():
    # a lane holding 8 contiguous f32 (stride-1) must split into TWO b128 transactions (4 f32 each), not one
    mp = {(0, r): (r, 0) for r in range(8)}
    ts, maxt = vector_transactions(mp, addr_fn_from_strides((1, 64)), dtype_bits=32)
    assert maxt == 2                            # 8 f32 / (b128 = 4 f32) = 2 transactions


def test_crc_m_contig_scatters_n_contig_coalesces():
    native = _native_c()
    m = analyze_coalescing(_shuffle(native, interleave_idx(1, 16, 64)), ("M", "N"), (1, 64), 32, direction="store", line_bytes=128)
    n = analyze_coalescing(_shuffle(native, interleave_idx(1, 4, 64)), ("M", "N"), (64, 1), 32, direction="store", line_bytes=128)
    assert not m.fully_coalesced and n.fully_coalesced          # M-contig against grain, N-contig with grain
    assert m.worst_lines == 4 * n.worst_lines                   # the 4x against-grain factor (lanes_N/lanes_M)
    assert m.per_instruction[0].vw_elems == 4                   # b128 = 4 f32 per lane


def test_labels_and_stride1_axis_by_name():
    # dims are carried so the report names the contiguous axis -- no M/N mix-up regardless of coord order
    native = _native_c()
    m = analyze_coalescing(_shuffle(native, interleave_idx(1, 16, 64)), ("M", "N"), (1, 64), 32, line_bytes=128)
    n = analyze_coalescing(_shuffle(native, interleave_idx(1, 4, 64)), ("M", "N"), (64, 1), 32, line_bytes=128)
    assert m.stride1_axis == "M" and n.stride1_axis == "N"
    assert m.dims == ("M", "N") and m.strides == (1, 64)


def test_operand_agnostic_A_and_B():
    # NOT just C tiles: an A operand (M,K) and a B operand (K,N) analyze the same way -- generic over labels
    # a simple lane owns 4 contiguous elements on its stride-1 axis -> one b128, fully coalesced
    a_dist = {(lane, r): (lane, r) for lane in range(64) for r in range(4)}   # coord (M,K), K stride-1
    a = analyze_coalescing(a_dist, ("M", "K"), (4, 1), 16, direction="load", line_bytes=128)  # f16 A load
    assert a.stride1_axis == "K" and a.per_instruction[0].vw_elems == 4       # 4 f16 contiguous
    b_dist = {(lane, r): (r, lane) for lane in range(64) for r in range(4)}   # coord (K,N), K stride-1
    b = analyze_coalescing(b_dist, ("K", "N"), (1, 4), 16, direction="load", line_bytes=128)
    assert b.stride1_axis == "K"


def test_direction_is_honored():
    # a load and a store of the same layout differ in transaction time order (order_by reg vs addr)
    native = _native_c()
    dist = _shuffle(native, interleave_idx(1, 16, 64))
    store = analyze_coalescing(dist, ("M", "N"), (1, 64), 32, direction="store", line_bytes=128)
    load = analyze_coalescing(dist, ("M", "N"), (1, 64), 32, direction="load", line_bytes=128)
    assert store.direction == "store" and load.direction == "load"
    assert len(store.per_instruction) == len(load.per_instruction)   # same #instructions, possibly re-ordered


def test_reconcile_flags_ideal_vs_achieved_gap():
    # the b128-ideal width the layout SUPPORTS vs what the compiler ACTUALLY emitted -- a gap is a SUSPECTED
    # BUG (this is exactly the signal that surfaced the C-store b64/b128 defect), never silently reconciled.
    native = _native_c()
    r = analyze_coalescing(_shuffle(native, interleave_idx(1, 4, 64)), ("M", "N"), (64, 1), 32,
                           direction="store", line_bytes=128)
    assert r.ideal_vw_elems == 4
    ok, _ = r.reconcile(4)
    assert ok                                                        # achieved == ideal -> consistent
    bad, note = r.reconcile(2)                                       # b64 emitted where b128 supported
    assert not bad and "SUSPECTED BUG" in note                      # under-shoot flagged, not dismissed
    over, _ = r.reconcile(8)
    assert not over                                                 # over-shoot is suspicious too


def test_asm_gate_raises_when_asm_disagrees():
    # the compiled ASM MUST back the model; a mismatch is FATAL (a bug in the viz OR the codegen), not a warn.
    native = _native_c()
    r = analyze_coalescing(_shuffle(native, interleave_idx(1, 4, 64)), ("M", "N"), (64, 1), 32,
                           direction="store", line_bytes=128)
    assert_asm_backed(r, 4)                                          # matches -> passes
    with pytest.raises(AssertionError, match="ASM does not back"):
        assert_asm_backed(r, 2)                                      # b64 where b128 supported -> test FAILS


def test_multiple_small_thread_tiles_fill_gaps_across_phases():
    # a lane owning SEVERAL small stride-1 patches (not one big run) -> several wave-wide instructions; each
    # phase is its own instruction and the gaps fill in across phases. The model must not merge them.
    # 4 lanes, each owns two separate 2-element stride-1 runs at disjoint N offsets (a "small tiles" fragment).
    dist = {}
    for lane in range(4):
        for r in range(2):
            dist[(lane, r)] = (lane, r)             # patch A: N in {0,1}
            dist[(lane, 2 + r)] = (lane, 8 + r)     # patch B: N in {8,9} -- a GAP between the two patches
    rep = analyze_coalescing(dist, ("M", "N"), (0, 1), 16, direction="store", line_bytes=128)
    assert len(rep.per_instruction) == 2           # two phases, one per small patch (not one merged run)
    assert all(i.vw_elems == 2 for i in rep.per_instruction)   # each patch a b32 pair, gaps preserved
