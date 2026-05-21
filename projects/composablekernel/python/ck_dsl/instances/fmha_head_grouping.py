# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GQA / MQA head-grouped FMHA forward (CK Tile ``01_fmha`` head_grouping).

The pure multi-head case is ``num_query_heads == num_kv_heads``; the
head-grouping variants share one K/V head across multiple Q heads:

* **MQA (multi-query)** -- ``num_kv_heads == 1``.
* **GQA (grouped-query)** -- ``num_kv_heads in (2, 4, 8, ...)``.

The mapping is the same for both: each Q head ``hq in [0, HQ)``
reads its K/V head ``hk = hq // (HQ / HK)``. This is what
:class:`FmhaKernelBuilder.decode_grid` does for free.
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
    "FmhaFwdHeadGroupingSpec",
    "build_fmha_fwd_head_grouping",
    "fmha_fwd_head_grouping_grid",
    "fmha_fwd_head_grouping_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaFwdHeadGroupingSpec:
    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "ck_dsl_fmha_fwd_head_grouping"

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

    @property
    def is_mqa(self) -> bool:
        return self.common.shape.num_kv_heads == 1

    @property
    def is_gqa(self) -> bool:
        return self.common.shape.num_query_heads > self.common.shape.num_kv_heads > 1

    @property
    def grouping_label(self) -> str:
        return "mqa" if self.is_mqa else ("gqa" if self.is_gqa else "mha")


def is_valid_spec(spec: FmhaFwdHeadGroupingSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA head_grouping needs seqlen_q ({spec.seqlen_q}) to be a "
            f"multiple of BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"MFMA head_grouping needs seqlen_k ({spec.seqlen_k}) to be a "
            f"multiple of BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA head_grouping needs head_size % 16 == 0 "
            f"(got {spec.common.shape.head_size})"
        )
    s = spec.common.shape
    if s.num_query_heads <= s.num_kv_heads:
        return False, (
            f"head_grouping spec requires HQ > HK; got "
            f"HQ={s.num_query_heads}, HK={s.num_kv_heads}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the head_grouping kernel ABI (shared between build + sig)."""
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def build_fmha_fwd_head_grouping(spec: FmhaFwdHeadGroupingSpec):
    """GQA / MQA FMHA forward kernel (MFMA-tiled body).

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, batch)``. Each CTA
    handles one ``BLOCK_M = 16`` Q-row tile for one head + batch.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid head_grouping spec: {why}")
    s = spec.common
    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    # One wave64 warp per CTA -- MFMA is per-wave.
    kb.block_size(64)
    _declare_params(kb)
    kb.decode_grid(has_batch_axis=True)
    b = kb.builder

    seqlen_q = kb.scalar("seqlen_q")
    seqlen_k = kb.scalar("seqlen_k")
    # The grid's block_id_x is the Q-tile index (semantically a tile,
    # not a token); ``kb.q_token`` reuses the same axis.
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = kb.batch_idx

    # Batch offsets fold into k_token_offset / v_token_offset (added
    # inside the helper's row-base math) and into q_tile_base.
    k_token_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("k"))
    v_token_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("v"))
    q_tile_local = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    q_tile_base = b.add(q_tile_local, b.mul(batch_idx, seqlen_q))

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
        # Within-batch Q position drives the causal mask check (the
        # global Q index would shift across batches and confuse the
        # mask boundary).
        q_pos_base=q_tile_local,
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
        k_token_offset_elems=k_token_offset,
        v_token_offset_elems=v_token_offset,
    )
    b.ret()
    return kb.kernel


def fmha_fwd_head_grouping_grid(
    spec: FmhaFwdHeadGroupingSpec, *, batch: int
) -> Tuple[int, int, int]:
    # MFMA path: one CTA per (q_tile, head, batch) -- each CTA covers
    # ``BLOCK_M`` Q rows.
    return (
        spec.seqlen_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        batch,
    )


def fmha_fwd_head_grouping_signature(spec: FmhaFwdHeadGroupingSpec):
    kb = FmhaKernelBuilder(
        "ck_dsl_fmha_fwd_head_grouping_sig_probe",
        spec.common,
    )
    _declare_params(kb)
    return kb.signature()
