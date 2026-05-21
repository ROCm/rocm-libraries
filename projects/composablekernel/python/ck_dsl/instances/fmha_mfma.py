# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled FMHA forward (production attention kernel).

This is the **first attention kernel** consuming the production
``mfma_f32_16x16x16_f16`` atom directly. One CTA = one wave64 warp
handles a 16-row Q tile across the full K span via QK + softmax +
PV MFMA chain. The helper :func:`mfma_attention_fwd_inner_body`
factors the QK / softmax / PV pipeline; this module's
:func:`build_fmha_fwd_mfma` is the thin spec→kernel wrapper.

Grid layout: ``(seqlen_q // BLOCK_M, num_query_heads, batch)``. The
spec's ``seqlen_q`` must be a multiple of ``BLOCK_M = 16``;
``head_size`` must be a multiple of ``16`` (all standard FMHA head
sizes -- 64 / 128 / 256 -- qualify); ``seqlen_k`` must be a multiple
of ``BLOCK_K = 16``.

CK Tile parity context: the standard CK Tile ``01_fmha`` fwd kernel
uses the same atom shape + LDS-staging skeleton; v1 here ships the
single-warp variant (no multi-warp BLOCK_M, no async DMA), which is
already ~16x denser in FLOPs than the warp-scalar body and forms
the spec surface the multi-warp + cshuffle hoist consumes verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import KernelDef
from ..helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec


__all__ = [
    "FmhaMfmaSpec",
    "build_fmha_fwd_mfma",
    "fmha_fwd_mfma_grid",
    "fmha_fwd_mfma_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaMfmaSpec:
    """One MFMA-tiled FMHA forward configuration.

    ``seqlen_q`` and ``seqlen_k`` are compile-time so the spec can
    pre-size the launch grid; runtime variability (e.g. varlen) lifts
    via a derived spec that overrides the grid computation.
    """

    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "ck_dsl_fmha_fwd_mfma"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaMfmaSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    s = spec.common.shape
    if s.head_size % 16 != 0:
        return False, (f"MFMA FMHA needs head_size % 16 == 0 (got {s.head_size})")
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"seqlen_q ({spec.seqlen_q}) must be a multiple of BLOCK_M "
            f"({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"seqlen_k ({spec.seqlen_k}) must be a multiple of BLOCK_K "
            f"({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.dtype != "f16":
        return False, (
            f"MFMA FMHA v1 ships f16 only; bf16 lands once the bf16 "
            f"MfmaAtom factory is exposed (got {spec.common.dtype})"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """The MFMA FMHA kernel ABI (shared between build + sig)."""
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def build_fmha_fwd_mfma(spec: FmhaMfmaSpec) -> KernelDef:
    """MFMA-tiled FMHA forward kernel.

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, batch)``. Each
    CTA owns one ``(q_tile, head, batch)`` triple.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fmha_mfma spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)  # one wave64 warp per CTA
    _declare_params(kb)
    kb.decode_grid(has_batch_axis=True)
    b = kb.builder

    seqlen_q = kb.scalar("seqlen_q")
    seqlen_k = kb.scalar("seqlen_k")
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = kb.batch_idx

    # block_id_x is the Q-tile index; q_tile_base is its first Q row.
    q_tile_idx = kb.q_token  # reuses block_id_x; semantically a tile, not a token
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))

    # For batched inputs, the per-batch offset gets absorbed into
    # k_token_offset / v_token_offset, q_tile_base, and the O offset.
    # Q / O are batched along the first dim with stride
    # ``seqlen_q * stride_q_token``.
    q_batch_offset = b.mul(b.mul(batch_idx, seqlen_q), kb.stride_token("q"))
    o_batch_offset = b.mul(b.mul(batch_idx, seqlen_q), kb.stride_token("o"))
    k_batch_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("k"))
    v_batch_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("v"))

    # Stride math: q_token's offset = (q_tile_base * stride_q_token +
    # batch * seqlen_q * stride_q_token + head * stride_q_head).
    # The helper does ``q_row * stride_q_token`` so we need to absorb
    # the batch offset into q_tile_base (effectively shifting it by
    # batch * seqlen_q). Easiest: pre-shift q_tile_base in elements.
    # Same trick for K / V via ``k_token_offset_elems``.

    causal_ctx = b.const_i32(0)  # self-attention: no cache offset

    mfma_attention_fwd_inner_body(
        b,
        Q=kb.tensor("Q"),
        K=kb.tensor("K"),
        V=kb.tensor("V"),
        O=kb.tensor("O"),
        head_size=s.head_size,
        seqlen_k=seqlen_k,
        q_tile_base=b.add(q_tile_base, b.mul(batch_idx, seqlen_q)),
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
        scale_log2=kb.scalar("scale_log2"),
        dtype=s.dtype,
        mask_mode=s.mask_mode,
        sliding_window=s.sliding_window,
        causal_ctx_offset=causal_ctx,
        k_token_offset_elems=k_batch_offset,
        v_token_offset_elems=v_batch_offset,
    )
    # Unused offsets q/o_batch_offset are absorbed into q_tile_base
    # and stride math above; reference them so lint stays happy.
    _ = (q_batch_offset, o_batch_offset)
    b.ret()
    return kb.kernel


def fmha_fwd_mfma_grid(spec: FmhaMfmaSpec, *, batch: int) -> Tuple[int, int, int]:
    q_tiles = spec.seqlen_q // MFMA_ATTN_BLOCK_M
    return (q_tiles, spec.common.shape.num_query_heads, batch)


def fmha_fwd_mfma_signature(spec: FmhaMfmaSpec):
    kb = FmhaKernelBuilder("ck_dsl_fmha_fwd_mfma_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()
