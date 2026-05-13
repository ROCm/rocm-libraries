# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention-specific helper objects for unified paged attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

from ..core.ir import IRBuilder, Value


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
