# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared building blocks for the gfx1250 wave32 WMMA attention kernels.

Both the tiled-2D prefill kernel (``attention_tiled_2d``) and the split-KV 3D
decode kernel (``attention_tiled_3d``) are wave32 WMMA ``16x16x32`` attention
forwards over a paged KV cache with the same per-tile structure:

    Q*K^T (WMMA)  ->  masked online softmax  ->  P staged in LDS
                  ->  V staged in LDS (fp8/bf16 dequant)  ->  P*V (WMMA)

This module factors out the pieces that are byte-identical between the two so
the kernel modules only express what is structurally different (tile-range
bounding, output destination, 2D-vs-segment grid).

The KV block lookup is passed in as a ``phys_block`` callable so the 2D kernel
(one paged block per 32-token tile) can reuse a single cached block id while the
3D kernel (block_size 16 -> two blocks per tile) does a per-token lookup —
without either kernel changing the emitted code.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ...core.ir import BF16, F32, FP8E4M3, I32, IRBuilder, Type, Value, VectorType
from ...helpers.attention import (
    dequant_fp8x8_to_dtype,
    wave_reduce_max,
    wave_reduce_sum,
)


def _wmma_spacing(b: IRBuilder, spacing: int) -> None:
    """Optional ``v_nop`` padding after a WMMA to mitigate the gfx1250
    dependent-WMMA co-execution hazard (occasional garbage accumulator at high
    occupancy). ``spacing == 0`` emits nothing."""
    if spacing > 0:
        b.inline_asm("\n".join(["v_nop"] * spacing), "", result_type=None)


WMMA_OP_ID = "wmma_gfx1250_f32_16x16x32_bf16"
WAVE = 32
BLOCK_M = 16
WMMA_N = 16
WMMA_K = 32
HEAD_SIZE = 64

# A ``phys_block`` maps a global key/token position to its paged physical block.
PhysBlockFn = Callable[[Value], Value]


def kv_storage_ir(kv_storage_dtype) -> Type:
    """IR element type for the KV cache (bf16 direct, fp8e4m3 dequantized)."""
    if kv_storage_dtype in (None, "bf16"):
        return BF16
    if kv_storage_dtype == "fp8e4m3":
        return FP8E4M3
    raise ValueError(f"unsupported kv_storage_dtype {kv_storage_dtype!r}")


def check_wmma_arch(arch: str) -> Tuple[bool, str]:
    """Return whether ``arch`` exposes the wave32 16x16x32 bf16 WMMA atom."""
    from ...core.arch import ArchTarget

    if arch != "gfx1250":
        return (
            False,
            f"gfx1250 WMMA attention only supports arch='gfx1250' (got {arch!r})",
        )
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    op = target.mma.by_op_id(WMMA_OP_ID)
    if op is None or op.family != "wmma" or op.wave_size != WAVE:
        return False, f"gfx1250 WMMA attention requires {WMMA_OP_ID} wave32 WMMA"
    return True, "supported"


def resolve_wmma(arch: str):
    """Return ``(op, a_layout, c_layout, a_frag_len, c_frag_len)`` for the atom."""
    from ...core.arch import ArchTarget

    op = ArchTarget.from_gfx(arch).mma.by_op_id(WMMA_OP_ID)
    return op, op.a_layout(), op.c_layout(), op.a_frag_len, op.c_frag_len


def load_kv16(
    b: IRBuilder, ptr: Value, base: Value, scale: Value, kv_dtype: Type, out_dtype: Type
) -> Value:
    """Load 16 contiguous head-dims as ``<16 x out_dtype>`` (bf16 direct, fp8 dequant)."""
    if kv_dtype is BF16:
        return b.global_load_vN(ptr, base, BF16, 16, align=16)
    lo = b.global_load_vN(ptr, base, FP8E4M3, 8, align=8)
    hi = b.global_load_vN(ptr, b.add(base, b.const_i32(8)), FP8E4M3, 8, align=8)
    return b.vec_concat(
        dequant_fp8x8_to_dtype(b, lo, scale, out_dtype),
        dequant_fp8x8_to_dtype(b, hi, scale, out_dtype),
    )


def load_q_frags(
    b: IRBuilder,
    query: Value,
    q_addr_row_base: Value,
    half_k: Value,
    q_valid: Value,
    *,
    head_size: int,
    a_frag: int,
    dtype: Type,
) -> List[Value]:
    """Load the per-lane Q (A) fragments, zeroing lanes whose row is invalid."""
    frags = []
    splat = b.vector_splat(q_valid, a_frag)
    for d in range(head_size // WMMA_K):
        q_addr = b.add(b.add(q_addr_row_base, b.const_i32(d * WMMA_K)), half_k)
        q_raw = b.global_load_vN(query, q_addr, dtype, a_frag, align=a_frag * 2)
        frags.append(b.vector_select(splat, q_raw, b.zero_vec(dtype, a_frag)))
    return frags


def compute_qk_scores(
    b: IRBuilder,
    q_frags: List[Value],
    key: Value,
    kv_desc,
    *,
    tile_base: Value,
    lane_row: Value,
    half_k: Value,
    kv_head_idx: Value,
    block_size: int,
    head_size: int,
    kv_dtype: Type,
    k_scale: Value,
    dtype: Type,
    c_frag: int,
    phys_block: PhysBlockFn,
    spacing: int = 0,
) -> List[Value]:
    """Q*K^T for the two 16-token N-subtiles of a 32-token tile -> [score0, score1]."""
    scores = []
    for nsub in range(2):
        score = b.zero_vec_f32(c_frag)
        k_pos = b.add(b.add(tile_base, b.const_i32(nsub * WMMA_N)), lane_row)
        pblk = phys_block(k_pos)
        token_in_block = b.mod(k_pos, b.const_i32(block_size))
        for d in range(head_size // WMMA_K):
            k_addr = kv_desc.offset(
                b,
                physical_block=pblk,
                token_in_block=token_in_block,
                kv_head=kv_head_idx,
                dim=b.add(b.const_i32(d * WMMA_K), half_k),
            )
            k_frag = load_kv16(b, key, k_addr, k_scale, kv_dtype, dtype)
            score = b.wmma_gfx1250_f32_16x16x32_bf16(q_frags[d], k_frag, score)
            _wmma_spacing(b, spacing)
        scores.append(score)
    return scores


def softmax_row_update(
    b: IRBuilder,
    m_prev: Value,
    l_prev: Value,
    srs: List[Value],
    *,
    neg_inf: Value,
    zero_f: Value,
    use_dpp: bool = False,
) -> Tuple[Value, Value, Value, List[Value]]:
    """One online-softmax step over the two N-subtiles of a row.

    Returns ``(m_new, l_new, alpha, [p0, p1])`` where ``alpha`` rescales the
    running accumulator and ``p0/p1`` are the unnormalized (base-2) probabilities.

    Fully masked subtiles (all ``neg_inf``) contribute zero probability mass,
    matching gfx950's ``fcmp ogt(max, -inf)`` guard so sentinel ``-1e30``
    semantics never produce ``exp2(0)`` mass on empty subtiles.
    """
    rm0 = wave_reduce_max(b, srs[0], wave_size=WAVE, lanes_per_row=16, use_dpp=use_dpp)
    rm1 = wave_reduce_max(b, srs[1], wave_size=WAVE, lanes_per_row=16, use_dpp=use_dpp)
    has0 = b.fcmp("ogt", rm0, neg_inf)
    has1 = b.fcmp("ogt", rm1, neg_inf)
    tile_has = b.lor(has0, has1)
    m_cand = b.fmax(rm0, rm1)
    m_new = b.select(tile_has, b.fmax(m_prev, m_cand), m_prev)
    alpha = b.exp2(b.fsub(m_prev, m_new))
    p0 = b.select(has0, b.exp2(b.fsub(srs[0], m_new)), zero_f)
    p1 = b.select(has1, b.exp2(b.fsub(srs[1], m_new)), zero_f)
    rs0 = wave_reduce_sum(b, p0, wave_size=WAVE, lanes_per_row=16, use_dpp=use_dpp)
    rs1 = wave_reduce_sum(b, p1, wave_size=WAVE, lanes_per_row=16, use_dpp=use_dpp)
    l_new = b.fadd(b.fmul(l_prev, alpha), b.fadd(rs0, rs1))
    return m_new, l_new, alpha, [p0, p1]


def stage_v_tile_transposed(
    b: IRBuilder,
    V_lds_T: Value,
    value: Value,
    kv_desc,
    *,
    kv_head_idx: Value,
    tile_base: Value,
    lane: Value,
    block_size: int,
    head_size: int,
    kv_dtype: Type,
    v_scale: Value,
    dtype: Type,
    phys_block: PhysBlockFn,
    buf_idx: Optional[Value] = None,
) -> None:
    """Stage the 32-token V tile into a **dim-major** ``V_lds_T`` (``[HD, T]``).

    Lane ``L`` still wide-loads token ``L``'s contiguous head-dim row from
    global (``head_size/8`` x vec8, fp8->bf16 dequant), but scatters it into
    column ``L`` of ``V_lds_T`` (``V_lds_T[d, L]``). This transposed layout lets
    ``compute_pv_wide`` read each lane's 16-element WMMA B fragment (16 tokens at
    a fixed head-dim) as **one wide ``ds_load``** instead of 16 scalar
    ``ds_load_u16`` — addressing the gfx1250 "no ds_read_tr" LDS-read bottleneck.
    The transpose cost lands on the store side, hidden behind the global-load
    latency.
    """
    v_global = b.add(tile_base, lane)
    vpblk = phys_block(v_global)
    v_tib = b.mod(v_global, b.const_i32(block_size))
    v_row_base = kv_desc.offset(
        b,
        physical_block=vpblk,
        token_in_block=v_tib,
        kv_head=kv_head_idx,
        dim=b.const_i32(0),
    )
    for dd in range(head_size // 8):
        if kv_dtype is BF16:
            v8 = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), BF16, 8, align=16
            )
        else:
            raw = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), FP8E4M3, 8, align=8
            )
            v8 = dequant_fp8x8_to_dtype(b, raw, v_scale, dtype)
        for i in range(8):
            row = b.const_i32(dd * 8 + i)
            idx = [row, lane] if buf_idx is None else [buf_idx, row, lane]
            b.smem_store_vN(V_lds_T, idx, b.vec_extract(v8, i), 1)


def compute_pv_wide(
    b: IRBuilder,
    P_lds: Value,
    V_lds_T: Value,
    accs: List[Value],
    *,
    a_map,
    lane: Value,
    lane_row: Value,
    a_frag: int,
    head_size: int,
    dtype: Type,
    v_extra_idx: Optional[Value] = None,
    p_extra_idx: Optional[Value] = None,
    spacing: int = 0,
) -> List[Value]:
    """P*V using **wide** LDS reads (one ``ds_load`` per fragment).

    Requires ``V_lds_T`` in dim-major ``[HD, T]`` layout (see
    :func:`stage_v_tile_transposed`) and ``P_lds`` in ``[BLOCK_M, T]``. Each
    lane's A (P) and B (V) WMMA fragments are 16 contiguous tokens at a fixed
    (row / head-dim), so they load as a single ``smem_load_vN(..., n=a_frag)``
    vector instead of ``a_frag`` scalar gathers.
    """
    # k0 = (lane // 16) * 16 — the lane's first K (token) slot; the a_frag
    # contiguous slots are k0 .. k0+a_frag-1 (matches a_map.coord(lane, j)[1]).
    # smem_load_vN caps at vec8, so read the a_frag fragment as ceil(a_frag/8)
    # wide chunks and concat — still ~8x fewer LDS ops than the scalar gather.
    CH = 8
    k0 = b.mul(b.div(lane, b.const_i32(16)), b.const_i32(16))
    col = b.mod(lane, b.const_i32(16))

    def _wide_frag(smem, row, extra):
        chunks = []
        for off in range(0, a_frag, CH):
            n = min(CH, a_frag - off)
            kk = b.add(k0, b.const_i32(off))
            if extra is None:
                chunks.append(b.smem_load_vN(smem, row, kk, dtype=dtype, n=n))
            else:
                chunks.append(b.smem_load_vN(smem, extra, row, kk, dtype=dtype, n=n))
        frag = chunks[0]
        for c in chunks[1:]:
            frag = b.vec_concat(frag, c)
        return frag

    p_a = _wide_frag(P_lds, lane_row, p_extra_idx)
    out = list(accs)
    for d in range(head_size // WMMA_N):
        d_col = b.add(b.const_i32(d * WMMA_N), col)
        v_b = _wide_frag(V_lds_T, d_col, v_extra_idx)
        out[d] = b.wmma_gfx1250_f32_16x16x32_bf16(p_a, v_b, out[d])
        _wmma_spacing(b, spacing)
    return out


def stage_v_tile(
    b: IRBuilder,
    V_lds: Value,
    value: Value,
    kv_desc,
    *,
    kv_head_idx: Value,
    tile_base: Value,
    lane: Value,
    block_size: int,
    head_size: int,
    kv_dtype: Type,
    v_scale: Value,
    dtype: Type,
    phys_block: PhysBlockFn,
) -> None:
    """Stage the 32-token V tile into ``V_lds`` (fp8->bf16 dequant), vectorised.

    Lane ``L`` (0..31) loads token ``L``'s contiguous head_size row as
    ``head_size/8`` x vec8 and stores it to ``V_lds[L]`` — one distinct token per
    lane (the wave is 32 lanes, the tile is 32 tokens). The caller must
    ``b.sync()`` afterwards before reading ``V_lds``.
    """
    v_global = b.add(tile_base, lane)
    vpblk = phys_block(v_global)
    v_tib = b.mod(v_global, b.const_i32(block_size))
    v_row_base = kv_desc.offset(
        b,
        physical_block=vpblk,
        token_in_block=v_tib,
        kv_head=kv_head_idx,
        dim=b.const_i32(0),
    )
    for dd in range(head_size // 8):
        if kv_dtype is BF16:
            v8 = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), BF16, 8, align=16
            )
        else:
            raw = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), FP8E4M3, 8, align=8
            )
            v8 = dequant_fp8x8_to_dtype(b, raw, v_scale, dtype)
        b.smem_store_vN(V_lds, [lane, b.const_i32(dd * 8)], v8, 8)


def stage_v_tile_buf(
    b: IRBuilder,
    V_lds: Value,
    buf_idx: Value,
    value: Value,
    kv_desc,
    *,
    kv_head_idx: Value,
    tile_base: Value,
    lane: Value,
    block_size: int,
    head_size: int,
    kv_dtype: Type,
    v_scale: Value,
    dtype: Type,
    phys_block: PhysBlockFn,
) -> None:
    """Like :func:`stage_v_tile` but stores under ``V_lds[buf_idx, token, dim]``."""
    v_global = b.add(tile_base, lane)
    vpblk = phys_block(v_global)
    v_tib = b.mod(v_global, b.const_i32(block_size))
    v_row_base = kv_desc.offset(
        b,
        physical_block=vpblk,
        token_in_block=v_tib,
        kv_head=kv_head_idx,
        dim=b.const_i32(0),
    )
    for dd in range(head_size // 8):
        if kv_dtype is BF16:
            v8 = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), BF16, 8, align=16
            )
        else:
            raw = b.global_load_vN(
                value, b.add(v_row_base, b.const_i32(dd * 8)), FP8E4M3, 8, align=8
            )
            v8 = dequant_fp8x8_to_dtype(b, raw, v_scale, dtype)
        b.smem_store_vN(V_lds, [buf_idx, lane, b.const_i32(dd * 8)], v8, 8)


def compute_pv(
    b: IRBuilder,
    P_lds: Value,
    V_lds: Value,
    accs: List[Value],
    *,
    a_map,
    c_map,
    lane: Value,
    lane_row: Value,
    col: Value,
    a_frag: int,
    c_frag: int,
    head_size: int,
    dtype: Type,
    v_extra_idx: Optional[Value] = None,
    p_extra_idx: Optional[Value] = None,
    spacing: int = 0,
) -> List[Value]:
    """P*V over the staged LDS tiles, accumulating into ``accs`` (per d-block).

    ``p_extra_idx`` / ``v_extra_idx`` prepend a leading LDS index (e.g. a
    per-wave slab in the cooperative multi-wave decode CTA); ``None`` keeps the
    original 2D ``[row, col]`` / 3D ``[buf, row, col]`` indexing.
    """
    p_a = b.zero_vec(dtype, a_frag)
    for j in range(a_frag):
        a_k = a_map.coord(b, lane, j)[1]
        if p_extra_idx is None:
            p_load = b.smem_load_vN(P_lds, lane_row, a_k, dtype=dtype, n=1)
        else:
            p_load = b.smem_load_vN(P_lds, p_extra_idx, lane_row, a_k, dtype=dtype, n=1)
        p_a = b.vec_insert(p_a, b.vec_extract(p_load, 0), j)
    return _compute_pv_inner(
        b,
        p_a,
        V_lds,
        accs,
        a_map=a_map,
        lane=lane,
        col=col,
        a_frag=a_frag,
        head_size=head_size,
        dtype=dtype,
        v_extra_idx=v_extra_idx,
        spacing=spacing,
    )


def _bpermute_vec8(b: IRBuilder, vec8: Value, src_lane: Value, dtype: Type) -> Value:
    """``ds_bpermute`` a ``<8 x dtype>`` (16-bit) fragment from ``src_lane``.

    Bitcasts to ``<4 x i32>``, gathers each dword from ``src_lane``, recombines.
    """
    src_addr = b.mul(src_lane, b.const_i32(4))
    i32v = b.bitcast(vec8, VectorType(I32, 4))
    out = b.zero_vec(I32, 4)
    for j in range(4):
        out = b.vec_insert(out, b.ds_bpermute(src_addr, b.vec_extract(i32v, j)), j)
    return b.bitcast(out, VectorType(dtype, 8))


def compute_pv_dstr(
    b: IRBuilder,
    P_lds: Value,
    V_lds: Value,
    accs: List[Value],
    *,
    a_map,
    lane: Value,
    lane_row: Value,
    a_frag: int,
    head_size: int,
    dtype: Type,
    v_extra_idx: Optional[Value] = None,
    p_extra_idx: Optional[Value] = None,
    spacing: int = 0,
) -> List[Value]:
    """P*V using the gfx1250 **hardware transpose-LDS read** (``ds_load_tr16_b128``).

    V is staged token-major ``V_lds[T, HD]`` (the natural layout). Per 16-dim
    WMMA-N block, two ``ds_load_tr16_b128`` reads (dims 0-7 / 8-15) transpose a
    16-token x 8-dim slab; because the wave32 transpose splits the two token
    halves across the ``lane^8`` pair, a single ``ds_bpermute`` stitches them
    into the K=32 WMMA B fragment (``a_map``: 16 tokens/lane at ``dim=lane%16``).
    This replaces the 64 scalar ``ds_load_u16`` gathers/tile with 8 ``ds_load_tr``
    + per-block permute -- the gfx950 ``ds_read_tr`` equivalent (decoded + GPU
    verified, see examples/gfx1250/attention probes).

    P (A operand) is built via the same scalar ``a_map`` gather as
    :func:`compute_pv` (P_lds is tiny). ``v_extra_idx`` / ``p_extra_idx`` prepend
    a leading LDS index (double-buffer / per-wave slab).
    """
    # --- A (P) operand: scalar a_map gather (same as compute_pv) ---
    p_a = b.zero_vec(dtype, a_frag)
    for j in range(a_frag):
        a_k = a_map.coord(b, lane, j)[1]
        if p_extra_idx is None:
            p_load = b.smem_load_vN(P_lds, lane_row, a_k, dtype=dtype, n=1)
        else:
            p_load = b.smem_load_vN(P_lds, p_extra_idx, lane_row, a_k, dtype=dtype, n=1)
        p_a = b.vec_insert(p_a, b.vec_extract(p_load, 0), j)

    # --- B (V) operand: ds_load_tr16_b128 + lane^8 stitch, per WMMA-N block ---
    lt8 = b.cmp_lt(b.mod(lane, b.const_i32(16)), b.const_i32(8))
    partner = b.xor(lane, b.const_i32(8))
    splat16 = b.vector_splat(lt8, 16)
    out = list(accs)
    for d in range(head_size // WMMA_N):
        c0 = d * WMMA_N
        if v_extra_idx is None:
            r0 = b.ds_read_tr16_b128(V_lds, lane, b.const_i32(c0), dtype=dtype)
            r1 = b.ds_read_tr16_b128(V_lds, lane, b.const_i32(c0 + 8), dtype=dtype)
        else:
            r0 = b.ds_read_tr16_b128(
                V_lds, v_extra_idx, lane, b.const_i32(c0), dtype=dtype
            )
            r1 = b.ds_read_tr16_b128(
                V_lds, v_extra_idx, lane, b.const_i32(c0 + 8), dtype=dtype
            )
        # bpermute r0 and r1 *separately* across the lane^8 pair (NOT the
        # per-lane-selected vector: ds_bpermute reads the partner lane's value,
        # and the partner holds the other read, so selecting before permuting
        # would gather the wrong half).
        p0 = _bpermute_vec8(b, r0, partner, dtype)
        p1 = _bpermute_vec8(b, r1, partner, dtype)
        v_b = b.vector_select(splat16, b.vec_concat(r0, p0), b.vec_concat(p1, r1))
        out[d] = b.wmma_gfx1250_f32_16x16x32_bf16(p_a, v_b, out[d])
        _wmma_spacing(b, spacing)
    return out


def compute_pv_from_probs(
    b: IRBuilder,
    ps0: List[Value],
    ps1: List[Value],
    V_lds: Value,
    accs: List[Value],
    *,
    a_map,
    c_map,
    lane: Value,
    col: Value,
    a_frag: int,
    c_frag: int,
    head_size: int,
    dtype: Type,
    v_extra_idx: Optional[Value] = None,
    spacing: int = 0,
) -> List[Value]:
    """P*V using register-resident unnormalized probs (drops ``P_lds`` round-trip).

    ``ps0[r]`` / ``ps1[r]`` are the two 16-wide N-subtiles for softmax slot ``r``,
    matching the staging order used by ``attention_tiled_{2,3}d`` when writing
    ``P_lds``.

    gfx1250 WMMA C and A layouts differ:

    * QK C: ``P[row=(lane//16)*8 + r, col=lane%16 (+ nsub*16)]``.
    * PV A: ``P[row=lane%16, col=(lane//16)*16 + j]``.

    The mapping is therefore a row/column transpose across lane halves and
    fragment slots. Use ``ds_bpermute`` with a runtime source-lane address so
    each A slot gathers the exact C scalar that the old LDS path would load.
    """
    p_a = b.zero_vec(dtype, a_frag)
    lane_half = b.div(lane, b.const_i32(16))
    lane_col = b.mod(lane, b.const_i32(16))
    row_half = b.div(lane_col, b.const_i32(8))
    row_reg = b.mod(lane_col, b.const_i32(8))
    for j in range(a_frag):
        # Source lane that owns P[row=lane_col, col=(lane_half*16+j)] in QK-C.
        src_lane = b.add(b.mul(row_half, b.const_i32(16)), b.const_i32(j))
        src_addr = b.mul(src_lane, b.const_i32(4))
        elt = b.const_f32(0.0)
        for r in range(8):
            p0_i = b.bitcast(ps0[r], I32)
            p1_i = b.bitcast(ps1[r], I32)
            from_p0 = b.bitcast(b.ds_bpermute(src_addr, p0_i), F32)
            from_p1 = b.bitcast(b.ds_bpermute(src_addr, p1_i), F32)
            from_col_half = b.select(
                b.cmp_eq(lane_half, b.const_i32(0)), from_p0, from_p1
            )
            elt = b.select(b.cmp_eq(row_reg, b.const_i32(r)), from_col_half, elt)
        p_a = b.vec_insert(p_a, b.cast_f32_to(elt, dtype), j)
    return _compute_pv_inner(
        b,
        p_a,
        V_lds,
        accs,
        a_map=a_map,
        lane=lane,
        col=col,
        a_frag=a_frag,
        head_size=head_size,
        dtype=dtype,
        v_extra_idx=v_extra_idx,
        spacing=spacing,
    )


def _compute_pv_inner(
    b: IRBuilder,
    p_a: Value,
    V_lds: Value,
    accs: List[Value],
    *,
    a_map,
    lane: Value,
    col: Value,
    a_frag: int,
    head_size: int,
    dtype: Type,
    v_extra_idx: Optional[Value],
    spacing: int,
) -> List[Value]:
    out = list(accs)
    for d in range(head_size // WMMA_N):
        d_col = b.add(b.const_i32(d * WMMA_N), col)
        v_b = b.zero_vec(dtype, a_frag)
        for j in range(a_frag):
            v_k = a_map.coord(b, lane, j)[1]
            if v_extra_idx is None:
                v_coords = (v_k, d_col)
            else:
                v_coords = (v_extra_idx, v_k, d_col)
            v_b = b.vec_insert(
                v_b,
                b.vec_extract(b.smem_load_vN(V_lds, *v_coords, dtype=dtype, n=1), 0),
                j,
            )
        out[d] = b.wmma_gfx1250_f32_16x16x32_bf16(p_a, v_b, out[d])
        _wmma_spacing(b, spacing)
    return out
