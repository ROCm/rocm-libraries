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

from ...helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ...helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from .fmha_arch import validate_fmha_mfma_atom


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
    name: str = "rocke_fmha_fwd_varlen"

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


def is_valid_spec(spec: FmhaFwdVarlenSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # MFMA-tiled body uses the narrow f16/bf16 16x16x16 atom; reject
    # arches whose catalog lacks it before lowering reaches comgr.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
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


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the varlen FMHA-fwd kernel ABI (shared between build + sig).

    Single-sources the param list so the build path and the signature
    probe can't drift. The ``readonly`` / ``writeonly`` alias hints
    affect only the emitted param attributes (not the signature shape,
    which :meth:`FmhaKernelBuilder.signature` derives from the param
    *order*), so the same declaration is reused verbatim by both.
    """
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


def build_fmha_fwd_varlen(spec: FmhaFwdVarlenSpec, arch: str = "gfx950"):
    """Varlen FMHA forward kernel (MFMA-tiled body).

    Grid: ``(total_q / BLOCK_M, num_query_heads, 1)``. Each CTA handles
    one ``BLOCK_M`` Q-row tile -- which must fall entirely within one
    sequence (per-sequence seqlen_q is required to be a multiple of
    BLOCK_M by ``is_valid_spec``).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_fwd_varlen spec: {why}")

    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    # MFMA: one wave64 warp per CTA.
    kb.block_size(64)
    _declare_params(kb)
    kb.decode_grid()

    b = kb.builder
    cu_seqlens_q = kb.ptr("cu_seqlens_q")
    cu_seqlens_k = kb.ptr("cu_seqlens_k")
    # ``q_token`` reuses block_id_x; semantically a tile index here.
    q_tile_idx = kb.q_token
    # ``head_idx`` and ``kv_head_idx`` come from ``block_id_y`` and a
    # const-divisor div; both are wave-uniform. Pinning them as SGPR
    # also turns the address math inside
    # ``mfma_attention_fwd_inner_body`` into scalar-ALU ops
    # (s_mul_i32 / s_add_i32) instead of v_ ops, matching the CK
    # Tile ``amd_wave_read_first_lane(...)`` pattern in
    # ``include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp``.
    head_idx = b.to_sgpr_u32(kb.head_idx)
    kv_head_idx = b.to_sgpr_u32(kb.kv_head_idx)
    scale_log2 = kb.scalar("scale_log2")

    # The first global Q row this CTA owns; wave-uniform (depends only
    # on block_id_x), so it lifts to SGPR cleanly.
    q_tile_base = b.to_sgpr_u32(b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M)))

    # Find which sequence this tile belongs to via Python-unrolled
    # scan of cu_seqlens_q. ``spec.batch`` is a compile-time constant
    # so the unrolled chain becomes ``B`` compares + ``B`` cmovs --
    # no scf.for backedge, no induction-variable update. Since the
    # tile is BLOCK_M-aligned and per-sequence seqlen_q is
    # BLOCK_M-aligned (validated), all 16 rows fall in the same
    # sequence and the result is wave-uniform (input is block_id_x).
    # ``to_sgpr_u32`` parks ``seq_idx`` and the derived addresses in
    # scalar registers so the downstream global loads pick them up as
    # ``s_buffer_load_dword``.
    seq = b.const_i32(0)
    for i in range(spec.batch):
        cuq_next = b.global_load_i32(cu_seqlens_q, b.const_i32(i + 1))
        is_in_seq = b.cmp_lt(q_tile_base, cuq_next)
        seq = b.select(is_in_seq, seq, b.add(seq, b.const_i32(1)))
    seq_idx = b.to_sgpr_u32(seq)

    cuq_base = b.to_sgpr_u32(b.global_load_i32(cu_seqlens_q, seq_idx))
    local_q_tile = b.to_sgpr_u32(b.sub(q_tile_base, cuq_base))
    cuk_base = b.to_sgpr_u32(b.global_load_i32(cu_seqlens_k, seq_idx))
    cuk_next = b.global_load_i32(cu_seqlens_k, b.add(seq_idx, b.const_i32(1)))
    seqlen_k = b.to_sgpr_u32(b.sub(cuk_next, cuk_base))

    k_token_offset = b.to_sgpr_u32(b.mul(cuk_base, kb.stride_token("k")))
    v_token_offset = b.to_sgpr_u32(b.mul(cuk_base, kb.stride_token("v")))

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
        arch=arch,
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
    kb = FmhaKernelBuilder("rocke_fmha_fwd_varlen_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()
