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

import math
from dataclasses import dataclass
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE, fmha_warp_fwd_inner_body
from .fmha_arch import validate_fmha_mfma_atom


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
    name: str = "rocke_fmha_fwd_paged_prefill"
    # P67: when ``use_mfma_body=True``, the kernel swaps the
    # warp-distributed ``fmha_warp_fwd_inner_body`` for
    # :func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`.
    # Requires ``total_q % MFMA_ATTN_BLOCK_M == 0`` and
    # ``head_size % MFMA_ATTN_BLOCK_K == 0``; the helper accepts the
    # paged-row callback so the page-table indirection plumbs in
    # without a body change. ~10-30× speedup on production paged-
    # prefill workloads (long-context, batched).
    use_mfma_body: bool = False

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


def is_valid_spec(
    spec: FmhaFwdPagedPrefillSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # The MFMA-tiled body (use_mfma_body=True) runs the narrow f16/bf16
    # 16x16x16 atom; reject arches whose catalog lacks it. The default
    # warp-distributed body uses scalar FMA and is arch-neutral, but the
    # atom is on both supported arches so we validate uniformly.
    if spec.use_mfma_body:
        ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
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


def build_fmha_fwd_paged_prefill(
    spec: FmhaFwdPagedPrefillSpec, arch: str = "gfx950"
) -> KernelDef:
    """Paged-KV prefill kernel with full block_table indirection."""
    ok, why = is_valid_spec(spec, arch)
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

    # Per-q_token sequence lookup -- ceil(log2(batch+1))-iter binary
    # search instead of the previous O(batch) linear scan. Invariant
    # mirrors AITER's ``find_seq_idx(use_q_block_mode=False)`` in
    # ``aiter.ops.triton.attention.unified_attention``: we look for
    # the unique ``s`` such that ``cu_q[s] <= q_token < cu_q[s+1]``.
    # The shared ``binary_search_seq_idx`` helper bakes in a
    # ``cu_q[mid]/BLOCK_Q + mid`` invariant for the q-block-mode
    # paths (varlen MFMA, tiled 2D / 3D), which is different from the
    # per-token mode the paged-prefill kernel needs -- so we keep the
    # inline form here instead of routing through the helper.
    bs_iters = max(1, int(math.ceil(math.log2(spec.batch + 1))))
    bs_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(bs_iters),
        b.const_i32(1),
        [
            ("bs_left", b.const_i32(0)),
            ("bs_right", b.const_i32(spec.batch)),
        ],
        iv_name="bs_i",
    )
    with bs_loop as (_iv, (left, right)):
        done = b.cmp_ge(left, right)
        mid = b.div(b.add(left, right), b.const_i32(2))
        cuq_next_mid = b.global_load_i32(cu_seqlens_q, b.add(mid, b.const_i32(1)))
        go_right = b.cmp_le(cuq_next_mid, q_token)
        nl = b.select(go_right, b.add(mid, b.const_i32(1)), left)
        nr = b.select(go_right, right, mid)
        b.scf_yield(b.select(done, left, nl), b.select(done, right, nr))
    seq_idx = bs_loop.results[0]
    cuq_base = b.global_load_i32(cu_seqlens_q, seq_idx)
    local_q = b.sub(q_token, cuq_base)
    seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx)

    # Per-k_idx paged-KV indirection (full block_table walk). Because
    # ``page_block_size`` is validated as a positive power of two,
    # we replace the (k_idx / PG) and (k_idx % PG) integer ops with
    # an ``lshr`` + ``land`` pair -- one ALU op each instead of the
    # full divider sequence. This is the same lowering CK Tile uses
    # in ``BlockKvCacheCoordinator`` when the block_size constant is
    # a power of two (``coord = (lane * pack + idx) >> log2(BS)``).
    block_table_row_base = b.mul(seq_idx, block_table_stride)
    pg = int(spec.page_block_size)
    assert (pg & (pg - 1)) == 0, "page_block_size must be a power of two (validated)"
    pg_log2 = pg.bit_length() - 1
    pg_mask = pg - 1
    c_pg_log2 = b.const_i32(pg_log2)
    c_pg_mask = b.const_i32(pg_mask)
    stride_block = kb.scalar("stride_block")
    stride_page = kb.scalar("stride_page")
    stride_kv_head = kb.scalar("stride_kv_head")
    stride_v_block = kb.scalar("stride_v_block")
    stride_v_page = kb.scalar("stride_v_page")
    stride_v_kv_head = kb.scalar("stride_v_kv_head")

    def _paged_row(stride_blk, stride_pg, stride_h):
        # ``kv_head_idx * stride_h`` is invariant across the K-loop the
        # warp body drives this callback over; hoist it to the closure
        # scope (built once per kernel) so the per-k_idx callback only
        # emits the block_id load + the two block/page muls. The DSL
        # optimizer does not move ops across the scf.for boundary
        # (core/passes.py runs CSE/DCE within a block, no LICM), so a
        # term left inside ``_row`` would be re-emitted every K-iter.
        head_off = b.mul(kv_head_idx, stride_h)

        def _row(b, k_idx):
            block_idx_in_seq = b.lshr(k_idx, c_pg_log2)
            page_in_block = b.land(k_idx, c_pg_mask)
            block_id = b.global_load_i32(
                block_table,
                b.add(block_table_row_base, block_idx_in_seq),
            )
            return b.add(
                b.add(
                    b.mul(block_id, stride_blk),
                    b.mul(page_in_block, stride_pg),
                ),
                head_off,
            )

        return _row

    causal_ctx = local_q if s.mask_mode in ("causal", "sliding_window") else None
    if spec.use_mfma_body:
        # P67: MFMA-tiled body. The host launcher must use the
        # corresponding MFMA grid (``ceil_div(total_q,
        # MFMA_ATTN_BLOCK_M)`` instead of the per-token grid). The
        # paged-row callbacks plumb in via the helper's
        # ``k_row_base_fn`` / ``v_row_base_fn`` signature; everything
        # else is the same as the warp-distributed path.
        from ...helpers.mfma_attention import (
            MFMA_ATTN_BLOCK_M,
            mfma_attention_fwd_inner_body,
        )

        q_tile_base = b.mul(q_token, b.const_i32(MFMA_ATTN_BLOCK_M))
        mfma_attention_fwd_inner_body(
            b,
            Q=Q,
            K=K_cache,
            V=V_cache,
            O=O,
            head_size=head_size,
            seqlen_k=seqlen_k,
            q_tile_base=q_tile_base,
            head_idx=head_idx,
            kv_head_idx=kv_head_idx,
            q_pos_base=local_q,
            stride_q_token=kb.stride_token("q"),
            stride_q_head=kb.stride_head("q"),
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
            causal_ctx_offset=causal_ctx,
            k_row_base_fn=_paged_row(stride_block, stride_page, stride_kv_head),
            v_row_base_fn=_paged_row(stride_v_block, stride_v_page, stride_v_kv_head),
            arch=arch,
        )
    else:
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
        "rocke_fmha_fwd_paged_prefill_sig_probe",
        spec.common,
    )
    _declare_params(kb)
    return kb.signature()
