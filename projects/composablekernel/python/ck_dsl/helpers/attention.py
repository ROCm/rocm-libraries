# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention-specific helper objects for unified paged attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from ..core.ir import IRBuilder, Type, Value


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


def apply_softcap_scalar(b: IRBuilder, score: Value, softcap: Value) -> Value:
    """softcap * tanh(score / softcap)."""
    return b.fmul(softcap, b.tanh(b.fdiv(score, softcap)))


# ---------------------------------------------------------------------------
# Cross-lane reductions (CK Tile ``block_tile_reduce_xor_sync`` pattern)
# ---------------------------------------------------------------------------


def warp_xor_reduce_max(b: IRBuilder, v: Value, stages: int = 4) -> Value:
    """Wave64 16-lane butterfly max reduction via ``ds_bpermute``.

    Reduces ``v`` across lanes whose ``lane % 16`` differ but
    ``lane / 16`` is fixed (each group of 16 lanes that share the same
    MFMA ``m_row_group``). After ``stages`` (default 4 for 16-lane
    reduction), every lane in the group holds the max of the 16
    inputs.

    The XOR sequence ``addr = (lane ^ 2^k) << 2`` for ``k in 0..stages-1``
    matches CK Tile's ``block_tile_reduce_xor_sync``. This helper used
    to live in :mod:`ck_dsl.instances.attention_tiled_2d` as a private
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
    """Wave64 16-lane butterfly sum reduction via ``ds_bpermute``.

    See :func:`warp_xor_reduce_max` for the lane-XOR rationale; the
    only difference is the combiner is ``fadd`` instead of ``fmax``.
    """
    cur = v
    for k in range(stages):
        remote = b.warp_shuffle_xor(cur, 1 << k)
        cur = b.fadd(cur, remote)
    return cur


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
        p1   = exp2(Sdiv)           = e^( score_natural / softcap)
        p2   = exp2(-Sdiv)          = e^(-score_natural / softcap)
        out  = softcap * (p1 - p2) / (p1 + p2)
    """
    sdiv = b.fdiv(score_log2, softcap)
    p1 = b.exp2(sdiv)
    p2 = b.exp2(b.fneg(sdiv))
    return b.fmul(softcap, b.fmul(b.fsub(p1, p2), b.rcp(b.fadd(p1, p2))))


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
) -> Value:
    """Triton-style binary search for the seq_idx for this q_block.

    Mirrors ``aiter.ops.triton.attention.unified_attention``'s
    ``find_seq_idx(use_q_block_mode=True)`` -- the loop invariant is
    ``cu_q[i] // BLOCK_Q + i <= target`` (i.e. the cumulative Q-block
    count up to sequence ``i``). The caller specializes ``iterations``
    from the known problem batch size; 32 is a safe fallback for
    unspecialized tests.
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
        mid_val = b.add(b.div(val, bq), mid)
        le = b.cmp_le(mid_val, q_block_global_idx)
        nl = b.select(le, b.add(mid, b.const_i32(1)), left)
        nr = b.select(le, right, mid)
        b.scf_yield(b.select(done, left, nl), b.select(done, right, nr))
    return b.sub(loop.results[0], b.const_i32(1))
