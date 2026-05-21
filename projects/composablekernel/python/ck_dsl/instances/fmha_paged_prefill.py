# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged-KV prefill FMHA forward (CK Tile ``01_fmha pagedkv_prefill``).

Differs from the regular varlen FMHA forward by the K / V layout:
instead of a contiguous ``(total_k, HK, D)`` tensor, K / V live in a
paged cache ``(num_blocks, block_size, HK, D)`` and the per-sequence
``block_table[seq_idx, :]`` indirects each logical K-position to its
physical block. The kernel performs the **full block-table
indirection** (not just block-table[0]) so sequences spanning multiple
non-contiguous physical blocks work correctly.

This kernel is for **prefill** (the first forward pass over a long
prompt); the split-KV variant for single-token decode lives in
:mod:`fmha_splitkv_decode`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import KernelDef
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE, fmha_warp_fwd_inner_body


__all__ = [
    "FmhaFwdPagedPrefillSpec",
    "build_fmha_fwd_paged_prefill",
    "fmha_fwd_paged_prefill_grid",
    "fmha_fwd_paged_prefill_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaFwdPagedPrefillSpec:
    common: FmhaCommonSpec
    page_block_size: int
    max_blocks_per_seq: int
    batch: int
    name: str = "ck_dsl_fmha_fwd_paged_prefill"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"PG{self.page_block_size}",
            f"B{self.batch}",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaFwdPagedPrefillSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.batch <= 0:
        return False, f"batch must be > 0 (got {spec.batch})"
    if (
        spec.page_block_size <= 0
        or (spec.page_block_size & (spec.page_block_size - 1)) != 0
    ):
        return False, (
            f"page_block_size must be a positive power of two "
            f"(got {spec.page_block_size})"
        )
    if spec.max_blocks_per_seq <= 0:
        return False, (
            f"max_blocks_per_seq must be > 0 (got {spec.max_blocks_per_seq})"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the paged-prefill kernel ABI."""
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K_cache", readonly=True)
    kb.add_tensor("V_cache", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("block_table", dtype="i32", readonly=True)
    kb.add_ptr("cu_seqlens_q", dtype="i32", readonly=True)
    kb.add_ptr("seqlens_k", dtype="i32", readonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("total_q", "i32")
    kb.add_scalar("batch", "i32")
    kb.add_strides("q")
    # K cache uses (block, page, kv_head) addressing -- its own stride
    # set, not the standard "stride_k_token / stride_k_head" pair.
    kb.add_scalar("stride_block", "i32")
    kb.add_scalar("stride_page", "i32")
    kb.add_scalar("stride_kv_head", "i32")
    kb.add_scalar("stride_v_block", "i32")
    kb.add_scalar("stride_v_page", "i32")
    kb.add_scalar("stride_v_kv_head", "i32")
    kb.add_strides("o")
    kb.add_scalar("block_table_stride", "i32")


def build_fmha_fwd_paged_prefill(spec: FmhaFwdPagedPrefillSpec) -> KernelDef:
    """Paged-KV prefill kernel with full block_table indirection."""
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fmha_fwd_paged_prefill spec: {why}")
    s = spec.common
    head_size = s.shape.head_size

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(WARP_SIZE)
    _declare_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K_cache = kb.tensor("K_cache")
    V_cache = kb.tensor("V_cache")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    block_table = kb.ptr("block_table")
    cu_seqlens_q = kb.ptr("cu_seqlens_q")
    seqlens_k_ptr = kb.ptr("seqlens_k")
    scale_log2 = kb.scalar("scale_log2")
    q_token = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    block_table_stride = kb.scalar("block_table_stride")

    # Per-q_token sequence lookup -- linear scan.
    seq_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.batch),
        b.const_i32(1),
        [("seq", b.const_i32(0))],
        iv_name="bs_i",
    )
    with seq_loop as (iv, (seq,)):
        cuq_next = b.global_load_i32(cu_seqlens_q, b.add(iv, b.const_i32(1)))
        is_in = b.cmp_lt(q_token, cuq_next)
        b.scf_yield(b.select(is_in, seq, b.add(seq, b.const_i32(1))))
    seq_idx = seq_loop.results[0]
    cuq_base = b.global_load_i32(cu_seqlens_q, seq_idx)
    local_q = b.sub(q_token, cuq_base)
    seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx)

    # Per-k_idx paged-KV indirection (full block_table walk).
    block_table_row_base = b.mul(seq_idx, block_table_stride)
    c_page_block = b.const_i32(spec.page_block_size)
    stride_block = kb.scalar("stride_block")
    stride_page = kb.scalar("stride_page")
    stride_kv_head = kb.scalar("stride_kv_head")
    stride_v_block = kb.scalar("stride_v_block")
    stride_v_page = kb.scalar("stride_v_page")
    stride_v_kv_head = kb.scalar("stride_v_kv_head")

    def _paged_row(stride_blk, stride_pg, stride_h):
        def _row(b, k_idx):
            block_idx_in_seq = b.div(k_idx, c_page_block)
            page_in_block = b.mod(k_idx, c_page_block)
            block_id = b.global_load_i32(
                block_table,
                b.add(block_table_row_base, block_idx_in_seq),
            )
            return b.add(
                b.add(
                    b.mul(block_id, stride_blk),
                    b.mul(page_in_block, stride_pg),
                ),
                b.mul(kv_head_idx, stride_h),
            )

        return _row

    causal_ctx = local_q if s.mask_mode in ("causal", "sliding_window") else None
    fmha_warp_fwd_inner_body(
        b,
        Q=Q,
        K=K_cache,
        V=V_cache,
        O=O,
        head_size=head_size,
        seqlen_k=seqlen_k,
        q_token=q_token,
        head_idx=head_idx,
        kv_head_idx=kv_head_idx,
        stride_q_token=kb.stride_token("q"),
        stride_q_head=kb.stride_head("q"),
        # Unused (the row-base callbacks compute K/V offsets directly),
        # but the warp body's API still requires them.
        stride_k_token=stride_page,
        stride_k_head=stride_kv_head,
        stride_v_token=stride_v_page,
        stride_v_head=stride_v_kv_head,
        stride_o_token=kb.stride_token("o"),
        stride_o_head=kb.stride_head("o"),
        scale_log2=scale_log2,
        dtype=s.dtype,
        mask_mode=s.mask_mode,
        sliding_window=s.sliding_window,
        causal_ctx_len=causal_ctx,
        k_row_base_fn=_paged_row(stride_block, stride_page, stride_kv_head),
        v_row_base_fn=_paged_row(stride_v_block, stride_v_page, stride_v_kv_head),
    )
    b.ret()
    return kb.kernel


def fmha_fwd_paged_prefill_grid(
    spec: FmhaFwdPagedPrefillSpec, *, total_q: int
) -> Tuple[int, int, int]:
    return (total_q, spec.common.shape.num_query_heads, 1)


def fmha_fwd_paged_prefill_signature(spec: FmhaFwdPagedPrefillSpec):
    kb = FmhaKernelBuilder(
        "ck_dsl_fmha_fwd_paged_prefill_sig_probe",
        spec.common,
    )
    _declare_params(kb)
    return kb.signature()
