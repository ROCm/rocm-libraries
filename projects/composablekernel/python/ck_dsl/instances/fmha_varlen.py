# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Variable-length FMHA forward (CK Tile ``01_fmha`` varlen parity).

Packs ``B`` sequences of arbitrary lengths into one flat ``(total_q,
H, D)`` tensor and uses cumulative-sequence-length arrays
(``cu_seqlens_q`` and ``cu_seqlens_k``) to address into them.

Layout::

 Q: (total_q, num_query_heads, head_size)
 K: (total_k, num_kv_heads, head_size)
 V: (total_k, num_kv_heads, head_size)
 O: (total_q, num_query_heads, head_size)
 cu_seqlens_q: (B + 1,) i32, exclusive prefix sum
 cu_seqlens_k: (B + 1,) i32, exclusive prefix sum

The kernel uses :class:`FmhaKernelBuilder` to declare params, decode
grid coords, and build the signature so this file stays focused on
the varlen-specific logic (per-q-token sequence lookup + per-sequence
K-base offset).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec


__all__ = [
    "FmhaFwdVarlenSpec",
    "build_fmha_fwd_varlen",
    "fmha_fwd_varlen_grid",
    "fmha_fwd_varlen_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaFwdVarlenSpec:
    """One concrete FMHA-fwd-varlen configuration."""

    common: FmhaCommonSpec
    max_seqlen_q: int
    max_seqlen_k: int
    batch: int
    name: str = "ck_dsl_fmha_fwd_varlen"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.max_seqlen_q}",
            f"K{self.max_seqlen_k}",
            f"B{self.batch}",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaFwdVarlenSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.batch <= 0:
        return False, f"batch must be > 0 (got {spec.batch})"
    if spec.max_seqlen_q <= 0 or spec.max_seqlen_k <= 0:
        return False, (
            f"max_seqlen_q / max_seqlen_k must be > 0 "
            f"(got {spec.max_seqlen_q}, {spec.max_seqlen_k})"
        )
    # The MFMA path requires per-sequence seqlen_q and seqlen_k to be
    # multiples of the BLOCK_M / BLOCK_K tile so a single CTA's tile
    # stays within one sequence and the K-loop covers whole tiles.
    if spec.max_seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA varlen needs max_seqlen_q ({spec.max_seqlen_q}) "
            f"to be a multiple of BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.max_seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"MFMA varlen needs max_seqlen_k ({spec.max_seqlen_k}) "
            f"to be a multiple of BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA varlen needs head_size % 16 == 0 (got {spec.common.shape.head_size})"
        )
    return True, "ok"


def build_fmha_fwd_varlen(spec: FmhaFwdVarlenSpec):
    """Varlen FMHA forward kernel (MFMA-tiled body).

    Grid: ``(total_q / BLOCK_M, num_query_heads, 1)``. Each CTA handles
    one ``BLOCK_M`` Q-row tile -- which must fall entirely within one
    sequence (per-sequence seqlen_q is required to be a multiple of
    BLOCK_M by ``is_valid_spec``).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fmha_fwd_varlen spec: {why}")

    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    # MFMA: one wave64 warp per CTA.
    kb.block_size(64)
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("cu_seqlens_q", dtype="i32")
    kb.add_ptr("cu_seqlens_k", dtype="i32")
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("total_q", "i32")
    kb.add_scalar("batch", "i32")
    kb.add_strides("q", "k", "v", "o")
    kb.decode_grid()

    b = kb.builder
    cu_seqlens_q = kb.ptr("cu_seqlens_q")
    cu_seqlens_k = kb.ptr("cu_seqlens_k")
    # ``q_token`` reuses block_id_x; semantically a tile index here.
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    scale_log2 = kb.scalar("scale_log2")

    # The first global Q row this CTA owns.
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))

    # Find which sequence this tile belongs to via linear scan of
    # cu_seqlens_q. Since the tile is BLOCK_M-aligned and per-sequence
    # seqlen_q is BLOCK_M-aligned (validated), all 16 rows fall in the
    # same sequence.
    seq_idx_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.batch),
        b.const_i32(1),
        [("seq", b.const_i32(0))],
        iv_name="bs_i",
    )
    with seq_idx_loop as (iv, (seq,)):
        cuq_next = b.global_load_i32(cu_seqlens_q, b.add(iv, b.const_i32(1)))
        is_in_seq = b.cmp_lt(q_tile_base, cuq_next)
        b.scf_yield(b.select(is_in_seq, seq, b.add(seq, b.const_i32(1))))
    seq_idx = seq_idx_loop.results[0]

    cuq_base = b.global_load_i32(cu_seqlens_q, seq_idx)
    local_q_tile = b.sub(q_tile_base, cuq_base)
    cuk_base = b.global_load_i32(cu_seqlens_k, seq_idx)
    cuk_next = b.global_load_i32(cu_seqlens_k, b.add(seq_idx, b.const_i32(1)))
    seqlen_k = b.sub(cuk_next, cuk_base)

    k_token_offset = b.mul(cuk_base, kb.stride_token("k"))
    v_token_offset = b.mul(cuk_base, kb.stride_token("v"))

    causal_ctx = b.const_i32(0) if s.mask_mode in ("causal", "sliding_window") else None

    mfma_attention_fwd_inner_body(
        b,
        Q=kb.tensor("Q"),
        K=kb.tensor("K"),
        V=kb.tensor("V"),
        O=kb.tensor("O"),
        head_size=s.head_size,
        seqlen_k=seqlen_k,
        q_tile_base=q_tile_base,
        # Within-sequence Q position for the causal mask check.
        q_pos_base=local_q_tile,
        head_idx=head_idx,
        kv_head_idx=kv_head_idx,
        stride_q_token=kb.stride_token("q"),
        stride_q_head=kb.stride_head("q"),
        stride_k_token=kb.stride_token("k"),
        stride_k_head=kb.stride_head("k"),
        stride_v_token=kb.stride_token("v"),
        stride_v_head=kb.stride_head("v"),
        stride_o_token=kb.stride_token("o"),
        stride_o_head=kb.stride_head("o"),
        scale_log2=scale_log2,
        dtype=s.dtype,
        mask_mode=s.mask_mode,
        sliding_window=s.sliding_window,
        causal_ctx_offset=causal_ctx,
        k_token_offset_elems=k_token_offset,
        v_token_offset_elems=v_token_offset,
    )
    b.ret()
    return kb.kernel


def fmha_fwd_varlen_grid(
    spec: FmhaFwdVarlenSpec, *, total_q: int
) -> Tuple[int, int, int]:
    """MFMA varlen grid: one CTA per Q-row tile (16 rows) per head."""
    return (
        total_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        1,
    )


def fmha_fwd_varlen_signature(spec: FmhaFwdVarlenSpec):
    """Construct the same signature shape FmhaKernelBuilder generates."""
    # Build a throw-away builder just to query the signature shape.
    # This keeps the signature contract single-sourced from the
    # builder so the spec and the build function can't drift.
    kb = FmhaKernelBuilder("ck_dsl_fmha_fwd_varlen_sig_probe", spec.common)
    kb.add_tensor("Q")
    kb.add_tensor("K")
    kb.add_tensor("V")
    kb.add_tensor("O")
    kb.add_ptr("cu_seqlens_q", dtype="i32")
    kb.add_ptr("cu_seqlens_k", dtype="i32")
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("total_q", "i32")
    kb.add_scalar("batch", "i32")
    kb.add_strides("q", "k", "v", "o")
    return kb.signature()
