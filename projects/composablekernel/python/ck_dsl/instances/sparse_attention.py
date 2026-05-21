# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sparse attention forward kernels (CK Tile ``50_sparse_attn`` parity).

Two CK Tile sparse-attention configurations:

* **Jenga block-sparse** (``build_jenga_sparse_attention``) -- the
  caller pre-builds a ``MaskBitmap[q_block, k_block]`` i8 array
  (``1`` = attend, ``0`` = skip). Each K position's contribution is
  gated by a runtime check of the mask byte for its enclosing K-block.

* **VSA (variable-size attention)** (``build_vsa_sparse_attention``)
  -- each ``q_block`` has its own LUT ``BlockLut[q_block, slot]`` of
  length ``BlockCount[q_block]`` pointing at the K-blocks it should
  attend to. A small per-K-position lookup checks whether the
  current K-block index appears in the LUT.

Both kernels reuse :func:`fmha_warp_fwd_inner_body` (warp-distributed
head-dim, butterfly reductions, no LDS state) and gate the per-K
softmax update via ``extra_mask_predicate``. The runtime predicate
forces the score for masked positions to ``-inf`` so the softmax
exponential collapses to zero -- mathematically identical to skipping
the position, slightly more bandwidth (the K row still gets loaded)
but compatible with the dense K-loop's state-threading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import I8, IRBuilder, KernelDef
from ..helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec


__all__ = [
    "JengaSparseSpec",
    "VsaSparseSpec",
    "build_jenga_sparse_attention",
    "build_vsa_sparse_attention",
    "is_valid_jenga_spec",
    "is_valid_vsa_spec",
    "jenga_sparse_attention_grid",
    "jenga_sparse_attention_signature",
    "vsa_sparse_attention_grid",
    "vsa_sparse_attention_signature",
]


@dataclass(frozen=True)
class JengaSparseSpec:
    """One Jenga block-sparse attention configuration."""

    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    block_q: int = 1
    block_k: int = 64
    name: str = "ck_dsl_jenga_sparse_attn"

    @property
    def num_q_blocks(self) -> int:
        return (self.seqlen_q + self.block_q - 1) // self.block_q

    @property
    def num_k_blocks(self) -> int:
        return (self.seqlen_k + self.block_k - 1) // self.block_k

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
            f"BQ{self.block_q}",
            f"BK{self.block_k}",
        )


@dataclass(frozen=True)
class VsaSparseSpec:
    """One variable-size sparse attention configuration."""

    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    block_q: int = 1
    block_k: int = 64
    max_blocks_per_q: int = 32
    name: str = "ck_dsl_vsa_sparse_attn"

    @property
    def num_q_blocks(self) -> int:
        return (self.seqlen_q + self.block_q - 1) // self.block_q

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
            f"BQ{self.block_q}",
            f"BK{self.block_k}",
            f"MB{self.max_blocks_per_q}",
        )


def is_valid_jenga_spec(spec: JengaSparseSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.block_q <= 0 or spec.block_k <= 0:
        return False, (
            f"block_q / block_k must be > 0 (got {spec.block_q}, {spec.block_k})"
        )
    if spec.seqlen_k % spec.block_k != 0:
        return False, (
            f"seqlen_k ({spec.seqlen_k}) must be divisible by block_k ({spec.block_k})"
        )
    # MFMA-tiled body constraints.
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA jenga sparse needs seqlen_q ({spec.seqlen_q}) "
            f"divisible by BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"MFMA jenga sparse needs seqlen_k ({spec.seqlen_k}) "
            f"divisible by BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.block_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"sparsity block_k ({spec.block_k}) must be a multiple of "
            f"MFMA BLOCK_K ({MFMA_ATTN_BLOCK_K}) so each sparsity block "
            f"covers a whole number of MFMA K-tiles"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA jenga sparse needs head_size % 16 == 0 "
            f"(got {spec.common.shape.head_size})"
        )
    return True, "ok"


def is_valid_vsa_spec(spec: VsaSparseSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.max_blocks_per_q <= 0:
        return False, (f"max_blocks_per_q must be > 0 (got {spec.max_blocks_per_q})")
    if spec.seqlen_k % spec.block_k != 0:
        return False, (
            f"seqlen_k ({spec.seqlen_k}) must be divisible by block_k ({spec.block_k})"
        )
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA VSA needs seqlen_q ({spec.seqlen_q}) divisible by "
            f"BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"MFMA VSA needs seqlen_k ({spec.seqlen_k}) divisible by "
            f"BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.block_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"VSA block_k ({spec.block_k}) must be a multiple of MFMA "
            f"BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA VSA needs head_size % 16 == 0 (got {spec.common.shape.head_size})"
        )
    return True, "ok"


def _declare_jenga_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("mask", dtype="i8", readonly=True, align=1)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def _declare_vsa_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("block_lut", dtype="i32", readonly=True)
    kb.add_ptr("block_count", dtype="i32", readonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def build_jenga_sparse_attention(spec: JengaSparseSpec) -> KernelDef:
    """Jenga block-sparse forward kernel (MFMA-tiled).

    Per-K-tile mask gate via ``MaskBitmap[q_block, k_block]``: when the
    bitmap byte for the enclosing sparsity block is zero, the whole
    16-position K-tile is force-masked to ``-inf`` (no softmax
    contribution, no PV contribution). The K loads still fire (no
    block-level skip in the MFMA inner; that's a v2 hoist with
    scf.if over the K-tile).
    """
    ok, why = is_valid_jenga_spec(spec)
    if not ok:
        raise ValueError(f"invalid jenga_sparse spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)
    _declare_jenga_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    mask = kb.ptr("mask")
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k_arg = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    # The Q tile's first row determines its sparsity-q-block.
    q_block_idx = b.div(q_tile_base, b.const_i32(spec.block_q))
    mask_row_base = b.mul(q_block_idx, b.const_i32(spec.num_k_blocks))

    # Each MFMA K-tile = ``MFMA_ATTN_BLOCK_K`` K positions = one
    # ``block_k / MFMA_ATTN_BLOCK_K``-th of one sparsity block.
    tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K
    c_tpbk = b.const_i32(tiles_per_block_k)

    def _jenga_tile_predicate(b: IRBuilder, kt):
        """``MaskBitmap[q_block, kt // tiles_per_block_k] != 0``."""
        k_block_idx = b.div(kt, c_tpbk)
        mask_off = b.add(mask_row_base, k_block_idx)
        mask_byte = b.global_load(mask, mask_off, I8)
        zero_i8 = b._op(  # noqa: SLF001 - i8 const factory
            "arith.constant",
            result_types=[mask_byte.type],
            attrs={"value": 0, "ity": "i8"},
            result_name_hint="cz",
        ).result
        return b.cmp_ne(mask_byte, zero_i8)

    mfma_attention_fwd_inner_body(
        b,
        Q=Q,
        K=K,
        V=V,
        O=O,
        head_size=s.shape.head_size,
        seqlen_k=seqlen_k_arg,
        q_tile_base=q_tile_base,
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
        mask_mode="none",
        extra_mask_predicate=_jenga_tile_predicate,
    )
    b.ret()
    return kb.kernel


def build_vsa_sparse_attention(spec: VsaSparseSpec) -> KernelDef:
    """Variable-size sparse attention forward kernel (MFMA-tiled).

    The VSA LUT lookup happens once per MFMA K-tile (at
    ``MFMA_ATTN_BLOCK_K`` granularity); when the K-tile's enclosing
    sparsity block isn't in ``BlockLut[q_block]``, the K-tile is
    force-masked to ``-inf``.
    """
    ok, why = is_valid_vsa_spec(spec)
    if not ok:
        raise ValueError(f"invalid vsa_sparse spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)
    _declare_vsa_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    block_lut = kb.ptr("block_lut")
    block_count = kb.ptr("block_count")
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k_arg = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    q_block_idx = b.div(q_tile_base, b.const_i32(spec.block_q))
    lut_row_base = b.mul(q_block_idx, b.const_i32(spec.max_blocks_per_q))
    block_count_v = b.global_load_i32(block_count, q_block_idx)
    tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K
    c_tpbk = b.const_i32(tiles_per_block_k)

    def _vsa_tile_predicate(b: IRBuilder, kt):
        """Tile-level VSA predicate -- checks whether
        ``kt // tiles_per_block_k`` appears in the q_block's LUT.
        """
        k_block_idx = b.div(kt, c_tpbk)
        slot_loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(spec.max_blocks_per_q),
            b.const_i32(1),
            [("found", b.const_i32(0))],
            iv_name="vsa_slot",
        )
        with slot_loop as (slot, (found,)):
            in_range = b.cmp_lt(slot, block_count_v)
            slot_off = b.add(lut_row_base, slot)
            lut_val = b.global_load_i32(block_lut, slot_off)
            hit_i1 = b.land(in_range, b.cmp_eq(lut_val, k_block_idx))
            hit_i32 = b.select(hit_i1, b.const_i32(1), b.const_i32(0))
            b.scf_yield(b.lor(found, hit_i32))
        return b.cmp_ne(slot_loop.results[0], b.const_i32(0))

    mfma_attention_fwd_inner_body(
        b,
        Q=Q,
        K=K,
        V=V,
        O=O,
        head_size=s.shape.head_size,
        seqlen_k=seqlen_k_arg,
        q_tile_base=q_tile_base,
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
        mask_mode="none",
        extra_mask_predicate=_vsa_tile_predicate,
    )
    b.ret()
    return kb.kernel


def jenga_sparse_attention_grid(spec: JengaSparseSpec) -> Tuple[int, int, int]:
    return (
        spec.seqlen_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        1,
    )


def vsa_sparse_attention_grid(spec: VsaSparseSpec) -> Tuple[int, int, int]:
    return (
        spec.seqlen_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        1,
    )


def jenga_sparse_attention_signature(spec: JengaSparseSpec):
    kb = FmhaKernelBuilder("jenga_sig_probe", spec.common)
    _declare_jenga_params(kb)
    return kb.signature()


def vsa_sparse_attention_signature(spec: VsaSparseSpec):
    kb = FmhaKernelBuilder("vsa_sig_probe", spec.common)
    _declare_vsa_params(kb)
    return kb.signature()
