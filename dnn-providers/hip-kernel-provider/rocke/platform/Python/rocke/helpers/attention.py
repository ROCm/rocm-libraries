# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention-specific helper objects for unified paged attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from ..core.ir import BF8E5M2, FP8E4M3, IRBuilder, Type, Value


def next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (int(x) - 1).bit_length()


@dataclass(frozen=True)
class Attention2DConfig:
    BLOCK_M: int
    BLOCK_Q: int
    TILE_SIZE: int
    num_warps: int
    num_stages: int
    waves_per_eu: int = 2


@dataclass(frozen=True)
class Attention3DConfig:
    TILE_SIZE: int
    NUM_SEGMENTS_PER_SEQ: int
    num_warps: int
    num_stages: int
    waves_per_eu: int = 2


def select_2d_config(
    *,
    block_size: int,
    head_size: int,
    sliding_window: int,
    all_decode: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_queries_per_kv: int,
    num_2d_prgms: int,
) -> Attention2DConfig:
    """Mirror AITER's `select_2d_config` exactly."""
    block_m = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
    tile_size = 64
    max_num_stages_2d = 2 if head_size > 128 else 4
    if not all_decode:
        num_stages_2d = 1
        num_warps = 2
    else:
        num_stages_2d = 3
        num_warps = 2
        tile_size = block_size
    if max_seqlen_q >= 256:
        block_m = 128
        num_stages_2d = 1
        num_warps = 4
    block_q = block_m // num_queries_per_kv
    return Attention2DConfig(
        BLOCK_M=block_m,
        BLOCK_Q=block_q,
        TILE_SIZE=tile_size,
        num_warps=num_warps,
        num_stages=min(max_num_stages_2d, num_stages_2d),
    )


def use_2d_kernel(
    *,
    head_size: int,
    sliding_window: int,
    all_decode: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    target_num_prgms: int,
    num_2d_prgms: int,
) -> bool:
    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )


def select_3d_config(
    *,
    head_size: int,
    block_size: int,
    element_size: int,
    max_seqlen_k: int,
    target_num_prgms: int,
    num_2d_prgms: int,
) -> Tuple[Attention3DConfig, Attention3DConfig]:
    """Mirror AITER's `select_3d_config` exactly."""
    reduce_num_warps = 2
    attn_warps = 2
    tile_size = block_size
    num_segments = math.ceil(target_num_prgms / num_2d_prgms)
    num_segments = next_power_of_2(num_segments)
    num_segments = min(num_segments, 128)
    min_segments = 16 if tile_size <= 16 else 8
    num_segments = max(num_segments, min_segments)
    if num_segments == min_segments:
        reduce_num_warps = 1
    return (
        Attention3DConfig(tile_size, num_segments, attn_warps, 1),
        Attention3DConfig(tile_size, num_segments, reduce_num_warps, 1),
    )


@dataclass(frozen=True)
class PagedKvDescriptor:
    """Address helper for `[num_blocks, block_size, num_kv_heads, head]` KV."""

    block_size: int
    stride_0: int
    stride_1: int
    stride_2: int
    stride_3: int

    def offset(
        self,
        b: IRBuilder,
        physical_block: Value,
        token_in_block: Value,
        kv_head: Value,
        dim: Value,
    ) -> Value:
        off = b.mul(physical_block, b.const_i32(self.stride_0))
        off = b.add(off, b.mul(token_in_block, b.const_i32(self.stride_1)))
        off = b.add(off, b.mul(kv_head, b.const_i32(self.stride_2)))
        off = b.add(off, b.mul(dim, b.const_i32(self.stride_3)))
        return off

    def block_base_from_table(
        self,
        b: IRBuilder,
        *,
        block_table: Value,
        seq_idx: Value,
        tile_idx: Value,
        block_table_stride: Value,
        kv_head: Value,
    ) -> Value:
        """Base offset for one logical KV tile through a page table.

        This centralizes the common paged-attention transform:

        ``(seq_idx, tile_idx, kv_head) -> physical_block -> base_offset``.

        ``stride_*`` may be in elements or bytes; the method preserves
        that unit. Attention async-DMA paths pass byte strides so the
        returned offset can be used directly as a buffer-load byte
        offset.
        """
        physical_block = b.global_load_i32(
            block_table,
            b.add(b.mul(seq_idx, block_table_stride), tile_idx),
        )
        return self.offset(
            b,
            physical_block=physical_block,
            token_in_block=b.const_i32(0),
            kv_head=kv_head,
            dim=b.const_i32(0),
        )

    def linear_half_voff(
        self,
        b: IRBuilder,
        linear_half: Value,
        *,
        head_size: int,
    ) -> Value:
        """``(token_in_tile, head_dim_in_halves)`` byte offset within a tile.

        Given a per-thread half index inside one ``[T, HD]`` KV tile
        (e.g. ``tid * 8 + call * THREADS * 8``), compute the byte
        offset relative to the tile base. This collapses the two-step
        ``(linear_half // HD, linear_half % HD)`` arithmetic the async
        DMA loaders previously inlined.
        """
        c_hd = b.const_i32(head_size)
        token = b.div(linear_half, c_hd)
        dim_bytes = b.mul(b.mod(linear_half, c_hd), b.const_i32(self.stride_3))
        return b.add(b.mul(token, b.const_i32(self.stride_1)), dim_bytes)


@dataclass(frozen=True)
class OnlineSoftmaxState:
    """Scalar online softmax state for one logical row.

    `m` is the running row max, `l_sum` is the running denominator
    (sum of `exp2(s - m)` over keys seen so far), and `acc` is the
    running value-weighted accumulator. Mirrors the standard
    "online softmax" formulation used by FA-2 and CK Tile's
    `block_tile_reduce` helpers.
    """

    m: Value
    l_sum: Value
    acc: Value

    def update(self, b: IRBuilder, score: Value, value: Value) -> "OnlineSoftmaxState":
        new_m = b.fmax(self.m, score)
        alpha = b.exp2(b.fsub(self.m, new_m))
        p = b.exp2(b.fsub(score, new_m))
        new_l_sum = b.fadd(b.fmul(self.l_sum, alpha), p)
        new_acc = b.fadd(b.fmul(self.acc, alpha), b.fmul(p, value))
        return OnlineSoftmaxState(new_m, new_l_sum, new_acc)

    def normalize(self, b: IRBuilder) -> Value:
        return b.fmul(self.acc, b.rcp(self.l_sum))


def causal_mask(
    b: IRBuilder, key_pos: Value, context_len: Value, query_pos: Value
) -> Value:
    return b.cmp_le(key_pos, b.add(context_len, query_pos))


def sliding_window_mask(
    b: IRBuilder,
    key_pos: Value,
    context_len: Value,
    query_pos: Value,
    sliding_window: int,
) -> Value:
    # context_len + query_pos - key_pos < sliding_window
    dist = b.sub(b.add(context_len, query_pos), key_pos)
    return b.cmp_lt(dist, b.const_i32(sliding_window))


def apply_attention_mask(
    b: IRBuilder,
    score_log2: Value,
    *,
    mask_mode: str,
    k_idx: Value,
    query_pos: Value,
    sliding_window: int = 0,
    context_len: Value = None,
    neg_inf: Value = None,
) -> Value:
    """Apply one of CK Tile's standard attention mask modes to a score.

    Maps the ``mask_mode`` string (one of ``"none"`` / ``"causal"`` /
    ``"sliding_window"``) to the right :func:`causal_mask` /
    :func:`sliding_window_mask` predicate and uses :meth:`select` to
    force masked-out positions to ``-inf`` so the softmax exponential
    collapses to zero.

    ``query_pos`` is the row's query position (relative to the start
    of the sequence for varlen, global token id for self-attention).
    ``context_len`` (typically ``0`` for fresh prefill, the prior
    cache length for paged-KV decode) shifts the mask boundary; pass
    ``None`` to default to ``b.const_i32(0)``.

    The helper is a no-op for ``mask_mode == "none"``: the score is
    returned unchanged. Composes with the warp-distributed FMHA body
    and the MFMA-tiled body alike -- both pass the *post-reduction*
    score and expect a single i1 predicate per (q_pos, k_idx) pair.
    """
    if mask_mode == "none":
        return score_log2
    if neg_inf is None:
        neg_inf = b.const_f32(-1e30)
    if context_len is None:
        context_len = b.const_i32(0)
    if mask_mode == "causal":
        keep = causal_mask(b, k_idx, context_len, query_pos)
    elif mask_mode == "sliding_window":
        keep = sliding_window_mask(
            b,
            k_idx,
            context_len,
            query_pos,
            sliding_window,
        )
    else:
        raise ValueError(f"unknown mask_mode {mask_mode!r}")
    return b.select(keep, score_log2, neg_inf)


def apply_softcap_scalar(b: IRBuilder, score: Value, softcap: Value) -> Value:
    """softcap * tanh(score / softcap)."""
    return b.fmul(softcap, b.tanh(b.fdiv(score, softcap)))


# ---------------------------------------------------------------------------
# Cross-lane reductions (CK Tile ``block_tile_reduce_xor_sync`` pattern)
# ---------------------------------------------------------------------------


def warp_xor_reduce_max(b: IRBuilder, v: Value, stages: int = 4) -> Value:
    """Wave64 16-lane butterfly max reduction via ``ds_swizzle_b32``.

    Reduces ``v`` across lanes whose ``lane % 16`` differ but
    ``lane / 16`` is fixed (each group of 16 lanes that share the same
    MFMA ``m_row_group``). After ``stages`` (default 4 for 16-lane
    reduction), every lane in the group holds the max of the 16
    inputs.

    The XOR sequence uses masks ``1, 2, 4, 8`` (all ``< 32``), so each
    stage stays inside one 32-lane half and
    :meth:`IRBuilder.warp_shuffle_xor` lowers it to a single
    ``ds_swizzle_b32`` SWAP-mode op (NOT ``ds_bpermute`` — that path is
    only taken for the cross-half ``mask >= 32`` case). This matches CK
    Tile's ``block_tile_reduce_xor_sync``. This helper used to live in
    :mod:`rocke.instances.gfx950.attention_tiled_2d` as a private
    function; promoting it makes the same reduction available to the
    3D segment kernel, the future MFMA-based norm kernels, and any
    other op that needs an in-warp 16-lane butterfly.
    """
    cur = v
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fmax(cur, remote)
    return cur


def warp_xor_reduce_sum(b: IRBuilder, v: Value, stages: int = 4) -> Value:
    """Wave64 16-lane butterfly sum reduction via ``ds_swizzle_b32``.

    See :func:`warp_xor_reduce_max` for the lane-XOR rationale (masks
    ``1, 2, 4, 8`` lower to ``ds_swizzle_b32``, not ``ds_bpermute``);
    the only difference is the combiner is ``fadd`` instead of ``fmax``.
    """
    cur = v
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fadd(cur, remote)
    return cur


def warp_xor_reduce_max_32lane(b: IRBuilder, v: Value) -> Value:
    """Reduce ``v`` across one 32-lane half of a wave64 with XOR stages.

    This is the reduction shape used by gfx950
    ``mfma_f32_32x32x16_*`` output tiles. Per CK Tile's
    ``WarpGemmAttributeMfmaImpl*F32M32N32K16`` traits, a row of the
    32x32 accumulator is wholly contained within one 32-lane half:

    - lanes 0..31 own the even 4-row groups
      ``{0-3, 8-11, 16-19, 24-27}``
    - lanes 32..63 own the odd 4-row groups
      ``{4-7, 12-15, 20-23, 28-31}``

    So the row-reduce needs a 5-stage intra-half butterfly
    (xor masks 1,2,4,8,16), not the old 16-lane 4-stage pattern.
    ``IRBuilder.warp_shuffle_xor`` lowers these intra-32 masks to
    ``ds_swizzle_b32`` SWAP mode, not ``ds_bpermute``.
    """
    cur = v
    for k in range(5):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fmax(cur, remote)
    return cur


def warp_xor_reduce_sum_32lane(b: IRBuilder, v: Value) -> Value:
    """Sum-reduction sibling of :func:`warp_xor_reduce_max_32lane`."""
    cur = v
    for k in range(5):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fadd(cur, remote)
    return cur


def wave64_reduce_max(b: IRBuilder, v: Value) -> Value:
    """Reduce ``v`` across all 64 lanes of a wave with an XOR butterfly.

    Six XOR stages (masks ``1, 2, 4, 8, 16, 32``). The first five stay
    inside one 32-lane half (``ds_swizzle_b32``); the last (mask 32) is
    a cross-half swap, which :meth:`IRBuilder.warp_shuffle_xor` lowers
    to ``ds_bpermute``. After the butterfly every lane in the wave
    holds the max of all 64 inputs. Used by the split-KV reduce kernel
    to fold the per-segment partial maxima that each lane accumulated
    over its strided segment subset.
    """
    cur = v
    for k in range(6):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fmax(cur, remote)
    return cur


def wave64_reduce_sum(b: IRBuilder, v: Value) -> Value:
    """Sum-reduction sibling of :func:`wave64_reduce_max` (``fadd``)."""
    cur = v
    for k in range(6):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fadd(cur, remote)
    return cur


# ---------------------------------------------------------------------------
# Arch-parameterized wave online-softmax row reduction (wave32 + wave64)
# ---------------------------------------------------------------------------
#
# The FMHA online-softmax needs a *row* reduction: every lane that holds a
# column of one score / probability row XOR-butterflies its partial across the
# lanes that share that row so each lane ends with the row max / row sum.
#
# The two backends differ only in how many lanes a row spans, never in the
# butterfly itself:
#
#   * wave64 MFMA 16x16x* : a C-row is contained in a 16-lane group
#     (``lane // 16`` fixed) -> 4 XOR stages (masks 1,2,4,8). This is the
#     long-standing CDNA path (``warp_xor_reduce_max/sum`` default stages=4).
#   * wave32 WMMA 16x16x16 : a C-row spans the 16 lanes of one wave32 half
#     (``lane // 16`` fixed) -> 4 XOR stages (masks 1,2,4,8) -- the masks
#     stay inside the 32-lane half and lower to ``ds_swizzle`` (see
#     ``instances/gfx1151/wmma_fmha_fwd.py::_half_reduce``).
#
# Both reduce one 16-lane row in 4 stages, so the wave32 path is *not* a
# different algorithm -- only the wave_size context differs. These helpers give
# the unify phase one entry point parameterized by ``wave_size`` whose
# wave64 default reproduces the existing CDNA reduction byte-for-byte (same XOR
# mask sequence, same combiner, same order), so routing CDNA through them does
# not perturb numerics.


def wave_reduce_stages(wave_size: int = 64, lanes_per_row: int = 16) -> int:
    """Number of XOR butterfly stages to reduce a row across ``lanes_per_row``.

    ``stages = log2(lanes_per_row)``. ``lanes_per_row`` must be a power of two
    and must not exceed ``wave_size`` (the row cannot span more lanes than the
    wave has). For the standard 16x16 MFMA / WMMA tile a row spans 16 lanes on
    both wave64 and wave32, giving 4 stages.
    """
    if lanes_per_row <= 0 or (lanes_per_row & (lanes_per_row - 1)) != 0:
        raise ValueError(f"lanes_per_row must be a power of two, got {lanes_per_row}")
    if lanes_per_row > wave_size:
        raise ValueError(
            f"lanes_per_row ({lanes_per_row}) cannot exceed wave_size ({wave_size})"
        )
    return lanes_per_row.bit_length() - 1


def wave_reduce_max(
    b: IRBuilder,
    v: Value,
    *,
    wave_size: int = 64,
    lanes_per_row: int = 16,
    use_dpp: bool = False,
) -> Value:
    """Arch-parameterized row-max reduction for the online softmax.

    Reduces ``v`` across the ``lanes_per_row`` lanes that share one tile row
    via an XOR butterfly, working identically on wave64 (CDNA / MFMA) and
    wave32 (gfx1151 / WMMA). After the reduction every lane in the row group
    holds the row max.

    The wave64 default (``wave_size=64, lanes_per_row=16``) emits the exact
    4-stage XOR sequence (masks 1,2,4,8) of :func:`warp_xor_reduce_max`, so
    the long-standing CDNA path is reproduced byte-for-byte. The wave32 WMMA
    softmax passes ``wave_size=32`` (still ``lanes_per_row=16``) and gets the
    same 4-stage butterfly, matching the hand-written wave32 ``_half_reduce``.

    The XOR masks (``< lanes_per_row``) always stay inside one wave half on
    wave32, so :meth:`IRBuilder.warp_shuffle_xor` lowers them to the in-half
    swizzle on both wave sizes.
    """
    stages = wave_reduce_stages(wave_size, lanes_per_row)
    cur = v
    if use_dpp:
        for k in range(stages):
            cur = b.vop2_f32_dpp_xor(cur, 1 << k, "v_max_f32")
        return cur
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fmax(cur, remote)
    return cur


def wave_reduce_sum(
    b: IRBuilder,
    v: Value,
    *,
    wave_size: int = 64,
    lanes_per_row: int = 16,
    use_dpp: bool = False,
) -> Value:
    """Arch-parameterized row-sum reduction; ``fadd`` sibling of
    :func:`wave_reduce_max`.

    Same lane geometry, stage count, default-wave64-byte-identical guarantee,
    and wave32 compatibility as :func:`wave_reduce_max`; the only difference is
    the combiner is ``fadd`` (the online-softmax denominator accumulation).
    """
    stages = wave_reduce_stages(wave_size, lanes_per_row)
    cur = v
    if use_dpp:
        for k in range(stages):
            cur = b.vop2_f32_dpp_xor(cur, 1 << k, "v_add_f32")
        return cur
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fadd(cur, remote)
    return cur


def mfma_32x32x16_c_row(b: IRBuilder, lane: Value, elem_idx: int) -> Value:
    """Return the MFMA-local output row for a ``32x32x16`` C element.

    ``elem_idx`` is the element number inside the per-lane ``<16 x f32>``
    accumulator. The formula mirrors CK Tile's distribution encoding:

    ``row = (elem_idx // 4) * 8 + (lane // 32) * 4 + (elem_idx % 4)``.

    This is intentionally a helper rather than open-coded arithmetic: the
    32x32x16 migration needs to use the same row mapping for QK mask,
    softmax state, P register layout, PV accumulation, and epilogue.
    """
    if not (0 <= elem_idx < 16):
        raise ValueError(f"mfma_32x32x16 elem_idx must be 0..15, got {elem_idx}")
    lane_half = b.div(lane, b.const_i32(32))
    return b.add(
        b.add(b.const_i32((elem_idx // 4) * 8), b.mul(lane_half, b.const_i32(4))),
        b.const_i32(elem_idx % 4),
    )


def mfma_32x32x16_c_col(b: IRBuilder, lane: Value, n_tile32: int = 0) -> Value:
    """Return the MFMA-local output col for ``32x32x16`` C elements.

    Every element in a lane's ``<16 x f32>`` accumulator shares the same
    column. ``n_tile32`` is the 32-column tile index within the current
    KV tile, so for T=64 the two QK N-tiles use ``n_tile32`` 0 and 1.
    """
    return b.add(b.const_i32(n_tile32 * 32), b.mod(lane, b.const_i32(32)))


# ---------------------------------------------------------------------------
# Softcap (log2-domain)
# ---------------------------------------------------------------------------


def apply_softcap_log2(b: IRBuilder, score_log2: Value, softcap: Value) -> Value:
    """``softcap * tanh(score_natural / softcap)`` computed via exp2 only.

    Given a log2-domain score (i.e. the natural-domain score already
    multiplied by ``log2(e)``), return the natural-domain softcapped
    value. The closed form avoids ``math.tanh`` (which the AMDGPU
    backend does not lower) by computing

    .. code-block:: text

    Sdiv = score_log2 / softcap
    p1 = exp2(Sdiv) = e^( score_natural / softcap)
    p2 = exp2(-Sdiv) = e^(-score_natural / softcap)
    out = softcap * (p1 - p2) / (p1 + p2)
    """
    sdiv = b.fdiv(score_log2, softcap)
    p1 = b.exp2(sdiv)
    p2 = b.exp2(b.fneg(sdiv))
    return b.fmul(softcap, b.fmul(b.fsub(p1, p2), b.rcp(b.fadd(p1, p2))))


def safe_inv_l(b: IRBuilder, denom: Value) -> Value:
    """Reciprocal of the online-softmax denominator with a zero guard.

    When an entire Q tile is masked off (sparse jenga / VSA all-masked
    rows), the softmax denominator ``denom`` is zero and ``rcp(0) = +inf``
    would poison the output (``inf * 0 -> NaN``). This forces
    ``inv_l = 0`` for the zero case so the normalized contribution
    evaluates to ``acc * 0 == 0`` (the intended "no attention" output).

    Emits the three ops ``fcmp oeq denom, 0 -> rcp denom -> select`` in
    that order, matching the inline guard the MFMA / WMMA epilogues use.
    """
    zero_mask = b.fcmp("oeq", denom, b.const_f32(0.0))
    inv_l_raw = b.rcp(denom)
    return b.select(zero_mask, b.const_f32(0.0), inv_l_raw)


# ---------------------------------------------------------------------------
# MFMA dtype dispatch (small but shared across every tiled attention kernel)
# ---------------------------------------------------------------------------


def mfma_16x16x16_for_dtype(
    b: IRBuilder, dtype: Type, a: Value, bv: Value, c: Value
) -> Value:
    """Dispatch ``mfma_f32_16x16x16_<dtype>`` for fp16 / bf16."""
    if dtype.name == "f16":
        return b.mfma_f32_16x16x16_f16(a, bv, c)
    if dtype.name == "bf16":
        return b.mfma_f32_16x16x16_bf16(a, bv, c)
    raise ValueError(f"unsupported MFMA 16x16x16 dtype {dtype.name}")


def mfma_16x16x32_for_dtype(
    b: IRBuilder, dtype: Type, a: Value, bv: Value, c: Value
) -> Value:
    """Dispatch ``mfma_f32_16x16x32_<dtype>`` for fp16 / bf16."""
    if dtype.name == "f16":
        return b.mfma_f32_16x16x32_f16(a, bv, c)
    if dtype.name == "bf16":
        return b.mfma_f32_16x16x32_bf16(a, bv, c)
    raise ValueError(f"unsupported MFMA 16x16x32 dtype {dtype.name}")


def mfma_32x32x8_for_dtype(
    b: IRBuilder, dtype: Type, a: Value, bv: Value, c: Value
) -> Value:
    """Dispatch ``mfma_f32_32x32x8_<dtype>`` for fp16.

    The 32x32x8 atom is the canonical wide-K f16 MFMA fragment that exists
    on gfx942 (CDNA3) -- unlike 32x32x16, which is gfx950-only. Per-lane
    operand types:
      - A: ``<4 x half>``  (M=32 x K=8 / 64 lanes)
      - B: ``<4 x half>``  (K=8 x N=32 / 64 lanes)
      - C/D: ``<16 x float>`` (M=32 x N=32 / 64 lanes)

    The C output lane layout is **identical** to the 32x32x16 atom (only K
    per atom differs), so the shared ``_C32_DIST`` / ``_mfma_32x32_c_*``
    distribution drives both. Both fp16 and bf16 are gfx942-legal here:
    ``mfma_f32_32x32x8_f16`` and ``mfma_f32_32x32x8_bf16`` (the ``.1k``
    intrinsic) both select on CDNA3. (The wider K=16 bf16 atom
    ``mfma_f32_32x32x16_bf16`` is CDNA4/gfx950-only -- the gfx942 backend
    ``Cannot select`` it -- so the gfx942 wide-bf16 flash path uses THIS K=8
    atom, not 32x32x16.)
    """
    if dtype.name == "f16":
        return b.mfma_f32_32x32x8_f16(a, bv, c)
    if dtype.name == "bf16":
        return b.mfma_f32_32x32x8_bf16(a, bv, c)
    raise ValueError(f"unsupported MFMA 32x32x8 dtype {dtype.name} (fp16/bf16 only)")


def mfma_32x32x16_for_dtype(
    b: IRBuilder, dtype: Type, a: Value, bv: Value, c: Value
) -> Value:
    """Dispatch ``mfma_f32_32x32x16_<dtype>`` for fp16 / bf16.

    Per-lane operand types:
      - A: ``<8 x bfloat>`` or ``<8 x half>`` (M=32 × K=16 / 64 lanes)
      - B: ``<8 x bfloat>`` or ``<8 x half>`` (K=16 × N=32 / 64 lanes)
      - C/D: ``<16 x float>`` (M=32 × N=32 / 64 lanes)

    Output lane layout (per CK Tile
    ``WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K16`` traits):
      - col = L % 32                                   (one column per lane)
      - mlane = L / 32                                 (which 32-lane half)
      - per-thread element ``t`` (0..15):
          row = (t // 4) * 8 + mlane * 4 + (t % 4)
        i.e. 4 outer "quad-row blocks" × 4 inner rows per block
      - Lanes [0..31] own rows {0-3, 8-11, 16-19, 24-27}
      - Lanes [32..63] own rows {4-7, 12-15, 20-23, 28-31}

    Critical for softmax: **no row is shared across the two 32-lane
    halves**, so per-row reduction only needs intra-half reduce — NO
    ``permlane32_swap`` required for max/sum over N=32. (For N>32 we
    combine multiple MFMA N-tiles' values in-lane via ``v_max3``
    before the cross-lane reduce.)
    """
    if dtype.name == "f16":
        return b.mfma_f32_32x32x16_f16(a, bv, c)
    if dtype.name == "bf16":
        return b.mfma_f32_32x32x16_bf16(a, bv, c)
    if dtype.name == "fp8e4m3":
        # Native fp8 x fp8 -> f32 (same per-lane operand layout as bf16:
        # 8 fp8 / lane for A and B, 16 f32 / lane for C). Used by the
        # native-fp8 QK path to skip the fp8->bf16 dequant entirely.
        return b.mfma_f32_32x32x16_fp8(a, bv, c)
    raise ValueError(f"unsupported MFMA 32x32x16 dtype {dtype.name}")


# ---------------------------------------------------------------------------
# Binary search on ``cu_q``
# ---------------------------------------------------------------------------


def binary_search_seq_idx(
    b: IRBuilder,
    cu_q: Value,
    q_block_global_idx: Value,
    num_seqs: Value,
    *,
    block_q: int,
    iterations: int,
    per_token: bool = False,
) -> Value:
    """Triton-style binary search for the seq_idx for this q_block.

    Mirrors ``aiter.ops.triton.attention.unified_attention``'s
    ``find_seq_idx``. By default (``per_token=False``) the loop
    invariant is ``cu_q[i] // BLOCK_Q + i <= target`` (i.e. the
    cumulative Q-block count up to sequence ``i``). The caller
    specializes ``iterations`` from the known problem batch size;
    32 is a safe fallback for unspecialized tests.

    P49: ``per_token=True`` switches to per-token mode
    (``cu_q[s] <= q_token < cu_q[s+1]``) used by ``fmha_paged_prefill``,
    ``fmha_appendkv``, and ``attention_unified``'s scalar oracle.
    The Q dimension passed in is interpreted as a token index (not a
    block index) and the BLOCK_Q divisor / iterator-add is dropped so
    the search compares raw token ranges. Mirrors AITER's
    ``find_seq_idx(use_q_block_mode=False)``.
    """
    bq = b.const_i32(block_q)
    loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(iterations),
        b.const_i32(1),
        [("left", b.const_i32(0)), ("right", num_seqs)],
        iv_name="bs_i",
    )
    with loop as (_iv, (left, right)):
        done = b.cmp_ge(left, right)
        mid = b.div(b.add(left, right), b.const_i32(2))
        val = b.global_load_i32(cu_q, mid)
        if per_token:
            # ``mid_val = cu_q[mid]`` -- find the largest s with
            # ``cu_q[s] <= q_token``.
            mid_val = val
        else:
            mid_val = b.add(b.div(val, bq), mid)
        le = b.cmp_le(mid_val, q_block_global_idx)
        nl = b.select(le, b.add(mid, b.const_i32(1)), left)
        nr = b.select(le, right, mid)
        b.scf_yield(b.select(done, left, nl), b.select(done, right, nr))
    return b.sub(loop.results[0], b.const_i32(1))


def dequant_fp8x8_to_dtype(
    b: IRBuilder,
    fp8_vec: Value,
    scale: Value,
    dtype: Type,
) -> Value:
    """In-register dequant of ``<8 x fp8e4m3>`` to a packed ``<8 x dtype>``.

    Splits the 8 fp8 inputs into two ``<4 x fp8>`` quads, runs the packed
    ``cvt_pk_f32_fp8x4`` on each, multiplies every f32 lane by ``scale``,
    casts to ``dtype`` and re-packs into a ``<8 x dtype>`` ready to feed the
    bf16/fp16 MFMA.

    **Invariant (correctness-critical):** the scale is applied as an
    *unfused* explicit ``fmul`` after ``cvt_pk_f32_fp8`` -- NOT via the fused
    ``cvt_scalef32_pk_f32_fp8``, which uses an E8M0-only scale and silently
    truncates non-power-of-two scales. Keeping the multiply explicit
    preserves correctness for arbitrary (non-pow2) ``k_scale``/``v_scale``.

    This is the promotion of the byte-identical dequant chain that the 2D
    tiled QK path (``_read_k8_mfma_operand`` and the inline 16x16 K_FP8_MFMA
    branch) and the 3D segment loader (``_issue_fp8_dequant_loads``) each
    emitted independently. ``b.vec_pack`` / ``b.cvt_pk_f32_fp8x4`` /
    ``b.fmul`` / ``b.cast_f32_to`` are emitted in the same order the inline
    sites used, so the lowered IR is unchanged.
    """
    lo_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4)], FP8E4M3)
    hi_fp8 = b.vec_pack([b.vec_extract(fp8_vec, i) for i in range(4, 8)], FP8E4M3)
    lo_f32 = b.cvt_pk_f32_fp8x4(lo_fp8)
    hi_f32 = b.cvt_pk_f32_fp8x4(hi_fp8)
    deq = [
        b.cast_f32_to(b.fmul(b.vec_extract(lo_f32, i), scale), dtype) for i in range(4)
    ] + [
        b.cast_f32_to(b.fmul(b.vec_extract(hi_f32, i), scale), dtype) for i in range(4)
    ]
    return b.vec_pack(deq, dtype)


def dequant_bf8x8_to_dtype(
    b: IRBuilder,
    bf8_vec: Value,
    scale: Value,
    dtype: Type,
) -> Value:
    """In-register dequant of ``<8 x bf8e5m2>`` to ``<8 x dtype>``."""
    lo_bf8 = b.vec_pack([b.vec_extract(bf8_vec, i) for i in range(4)], BF8E5M2)
    hi_bf8 = b.vec_pack([b.vec_extract(bf8_vec, i) for i in range(4, 8)], BF8E5M2)
    lo_f32 = b.cvt_pk_f32_bf8x4(lo_bf8)
    hi_f32 = b.cvt_pk_f32_bf8x4(hi_bf8)
    deq = [
        b.cast_f32_to(b.fmul(b.vec_extract(lo_f32, i), scale), dtype) for i in range(4)
    ] + [
        b.cast_f32_to(b.fmul(b.vec_extract(hi_f32, i), scale), dtype) for i in range(4)
    ]
    return b.vec_pack(deq, dtype)


def pv32_v_load_paired(
    b: IRBuilder,
    *,
    V_lds: Value,
    v_buf: Value,
    n: int,
    k: int,
    lane_half32: Value,
    lane_col32: Value,
    dtype: Type,
) -> Value:
    """Promoted 32x32x16 PV V-load (P50): two paired ``ds_read_tr16_b64``
    + concat → ``<8 x dtype>`` per lane.

    Reference inlined version at ``instances/attention_tiled_2d.py:
    2610-2666`` (TRANSPOSED_HALF_LOCAL_PV branch). Same lane-layout
    arithmetic, just lifted into a callable so the future P12 / P47
    swap to ``ds_read_b128_tr_b16`` is a one-line edit at every call
    site.

    Inputs:

    * ``V_lds`` — LDS allocation token holding the V tile.
    * ``v_buf`` — i32 selector for the active V buffer (ping-pong /
      double-buffer index).
    * ``n`` / ``k`` — Python-int N-tile and K-tile indices for this
      atom's slot.
    * ``lane_half32`` / ``lane_col32`` — i32 SSA values produced by
      the kernel's lane-decode (``lane_half32 in {0, 1}``,
      ``lane_col32 in [0, 32)``).

    Output: per-lane ``<8 x dtype>`` ready to feed
    ``mfma_f32_32x32x16_<dtype>`` as the A operand of the transposed
    PV.
    """
    col_group16 = b.mul(b.div(lane_col32, b.const_i32(16)), b.const_i32(16))
    tr_col32 = b.add(
        col_group16,
        b.mul(b.mod(lane_col32, b.const_i32(4)), b.const_i32(4)),
    )
    tr_row_base32 = b.add(
        b.add(
            b.const_i32(k * 16),
            b.mul(lane_half32, b.const_i32(4)),
        ),
        b.mod(b.div(lane_col32, b.const_i32(4)), b.const_i32(4)),
    )
    A_r0 = b.ds_read_tr16_b64(
        V_lds,
        v_buf,
        tr_row_base32,
        b.add(b.const_i32(n * 32), tr_col32),
        dtype=dtype,
    )
    A_r1 = b.ds_read_tr16_b64(
        V_lds,
        v_buf,
        b.add(tr_row_base32, b.const_i32(8)),
        b.add(b.const_i32(n * 32), tr_col32),
        dtype=dtype,
    )
    return b.vec_concat(A_r0, A_r1)
