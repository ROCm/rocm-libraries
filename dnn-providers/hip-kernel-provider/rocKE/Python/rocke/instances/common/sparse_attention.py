# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sparse attention forward kernels (CK Tile ``50_sparse_attn`` parity).

Two CK Tile sparse-attention configurations:

* **Jenga block-sparse** (``build_jenga_sparse_attention``) -- the
  caller pre-builds a ``MaskBitmap[q_block, k_block]`` i8 array
  (``1`` = attend, ``0`` = skip). Each K-tile's contribution is gated
  by the bitmap byte for its enclosing sparsity K-block.

* **VSA (variable-size attention)** (``build_vsa_sparse_attention``)
  -- each ``q_block`` has its own LUT ``BlockLut[q_block, slot]`` of
  length ``BlockCount[q_block]`` pointing at the K-blocks it should
  attend to. A small per-K-tile lookup checks whether the current
  K-block index appears in the LUT.

Both kernels reuse :func:`mfma_attention_fwd_inner_body` (MFMA-tiled
QK -> softmax -> PV) and gate the per-K-tile softmax update via
``extra_mask_predicate``. The runtime predicate forces the score for
masked K-tiles to ``-inf`` so the softmax exponential collapses to
zero -- mathematically identical to skipping the position, slightly
more bandwidth (the K row still gets loaded) but compatible with the
dense K-loop's state-threading. A v2 hoist could short-circuit the
whole MFMA tile via an outer ``scf.if`` on ``keep_tile``; CK Tile's
``BlockFmhaPipelineQRKSVSAsyncJenga`` (jenga reference) uses exactly
that pattern.

Both predicates are powered by an **LDS-staged mask bitmap**:

* The jenga kernel cooperatively loads ``Mask[q_block, :]`` into LDS
  once per CTA, replacing the per-K-tile global mask byte load.
* The VSA kernel allocates an LDS i8 bitmap and scatters the per-LUT
  entries into it (``bitmap[lut_val] = 1``), replacing the original
  per-K-tile O(``max_blocks_per_q``) global LUT scan with a single
  LDS byte read. This matches the spirit of CK Tile's
  ``block_relation_onehot`` pattern (jenga reference) and the LUT
  walk in the VSA reference, lifted into LDS so the predicate is
  scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import I8, IRBuilder, KernelDef, Value
from ...helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ...helpers.spec import kernel_name_join
from ...helpers.transforms import calculate_magic_numbers, do_magic_division
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from .fmha_arch import validate_fmha_mfma_atom


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


_BLOCK_SIZE = 64  # one wave64 per CTA (matches mfma_attention helper)


def _magic_div(b: IRBuilder, dividend: Value, divisor: int) -> Value:
    """``dividend // divisor`` via CK Tile's magic mul-hi division.

    Used for the sparsity-block index decode (``q_tile_base // block_q``
    and ``kt // tiles_per_block_k``). The divisor is a compile-time
    constant, so ``(multiplier, shift)`` fold to immediates and the cost is
    one ``v_mul_hi_u32`` + add + shift instead of the AMDGPU integer
    divider. The dividend is a non-negative i32 (a Q-tile base or an MFMA
    K-tile index), inside the magic sequence's 31-bit unsigned range, so
    the unsigned magic quotient equals the ``b.div`` it replaces (device
    lowering of ``merge_v2_magic_division``).
    """
    mult, shift = calculate_magic_numbers(divisor)
    return do_magic_division(b, dividend, mult, shift)


@dataclass(frozen=True)
class JengaSparseSpec:
    """One Jenga block-sparse attention configuration."""

    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    block_q: int = 1
    block_k: int = 64
    name: str = "rocke_jenga_sparse_attn"

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
    name: str = "rocke_vsa_sparse_attn"
    # P89: when ``max_blocks_per_q <= wave_size = 64``, the scatter
    # pass can use a single uniform ``wave_ballot`` + LDS write per
    # chunk instead of the per-chunk ``scf.if``. Defaults to True
    # because most VSA workloads stay below the wave cap; falls back
    # to the legacy chunked form when ``max_blocks_per_q > 64``.
    use_wave_ballot_scatter: bool = True

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
            f"MB{self.max_blocks_per_q}",
        )


def is_valid_jenga_spec(
    spec: JengaSparseSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # MFMA-tiled body uses the narrow f16/bf16 16x16x16 atom.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
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


def is_valid_vsa_spec(spec: VsaSparseSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # MFMA-tiled body uses the narrow f16/bf16 16x16x16 atom.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
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


# ---------------------------------------------------------------------
# LDS bitmap primitives (jenga + VSA share these idioms)
# ---------------------------------------------------------------------


def _const_i8(b: IRBuilder, value: int) -> Value:
    """Build an i8 constant via ``arith.constant`` (no scalar i8 factory)."""
    return b._op(  # noqa: SLF001 - the i8 const factory lives at the op layer
        "arith.constant",
        result_types=[I8],
        attrs={"value": int(value), "ity": "i8"},
        result_name_hint="ci8",
    ).result


def _cooperative_iter(
    b: IRBuilder,
    *,
    tid: Value,
    total: int,
    body,
) -> None:
    """Run ``body(slot)`` for each ``slot in [0, total)`` distributed across
    the wave64. For ``total > 64`` we issue a chained ``chunk * 64`` walk;
    in-range checks are elided when a chunk is fully covered. Used to
    cooperatively load/zero/scatter LDS arrays.
    """
    if total <= 0:
        return
    for chunk in range((total + _BLOCK_SIZE - 1) // _BLOCK_SIZE):
        base = chunk * _BLOCK_SIZE
        slot = tid if base == 0 else b.add(tid, b.const_i32(base))
        if base + _BLOCK_SIZE <= total:
            body(slot)
        else:
            in_range = b.cmp_lt(slot, b.const_i32(total))
            with b.scf_if(in_range):
                body(slot)


def _stage_jenga_mask_to_lds(
    b: IRBuilder,
    *,
    mask_global: Value,
    mask_row_base: Value,
    num_k_blocks: int,
    tid: Value,
) -> Value:
    """Cooperatively copy one Q-block's mask row into LDS.

    Returns the LDS allocation handle. After the call (and the
    accompanying ``b.sync()``) the bitmap can be queried in O(1) per
    K-tile from the predicate body.

    Reference: CK Tile's ``BlockFmhaPipelineQRKSVSAsyncJenga`` (jenga
    reference at ``include/ck_tile/ops/sparse_attn/pipeline/
    block_fmha_pipeline_qr_ks_vs_async_jenga.hpp:260-268``) does the
    same one-shot LDS stage via ``amd_direct_load_global_to_lds`` of
    ``block_relation_onehot_ptr``; this is the synchronous-load DSL
    analog.
    """
    mask_lds = b.smem_alloc(I8, [num_k_blocks], name_hint="jenga_mask")

    def _body(slot: Value) -> None:
        mask_off = b.add(mask_row_base, slot)
        mask_byte = b.global_load(mask_global, mask_off, I8)
        b.smem_store_vN(mask_lds, [slot], mask_byte, 1)

    _cooperative_iter(b, tid=tid, total=num_k_blocks, body=_body)
    return mask_lds


def _stage_vsa_bitmap_to_lds(
    b: IRBuilder,
    *,
    block_lut: Value,
    block_count: Value,
    q_block_idx: Value,
    lut_row_base: Value,
    num_k_blocks: int,
    max_blocks_per_q: int,
    tid: Value,
) -> Value:
    """Build the per-(q_block) K-attend bitmap in LDS.

    Three cooperative passes:

    1. Allocate ``num_k_blocks`` i8 slots in LDS.
    2. Zero them (one byte per lane, looped if ``num_k_blocks > 64``).
    3. For each lane ``l in [0, max_blocks_per_q)`` with ``l <
       block_count[q_block]``, scatter ``bitmap[lut[q_block, l]] = 1``.

    Concurrent writes from multiple lanes to the same slot are safe
    because they all store ``1`` (idempotent). After the final
    ``b.sync()`` the predicate is one LDS byte read.

    Wins vs the previous per-K-tile global scan:
      * One global LUT load per lane per CTA -> from O(``seqlen_k /
        block_k`` * ``max_blocks_per_q``) per CTA down to
        O(``max_blocks_per_q`` / wave + ``seqlen_k / block_k``).
      * One LDS read per K-tile predicate -> from
        O(``max_blocks_per_q``) global loads + a chained ``scf.for``
        per-tile down to O(1) per tile.

    Reference: CK Tile's ``BlockFmhaPipelineQRKSVSAsyncVsa`` (VSA
    reference at ``include/ck_tile/ops/sparse_attn/pipeline/
    block_fmha_pipeline_qr_ks_vs_async_vsa.hpp:203, 332``) drives the
    K-loop directly off ``kv_block_idx_ptr[i_total_loops]``, which is
    a strictly stronger (tile-skipping) form. The LDS-bitmap mask
    here is the predicate-gated DSL analog that re-uses
    ``extra_mask_predicate`` without restructuring the K-loop.
    """
    bitmap_lds = b.smem_alloc(I8, [num_k_blocks], name_hint="vsa_bitmap")
    zero_i8 = _const_i8(b, 0)
    one_i8 = _const_i8(b, 1)

    # Pass 1: zero the bitmap. With ``num_k_blocks`` typically <= 256
    # this is a few VALU ops on a single wave.
    def _zero_body(slot: Value) -> None:
        b.smem_store_vN(bitmap_lds, [slot], zero_i8, 1)

    _cooperative_iter(b, tid=tid, total=num_k_blocks, body=_zero_body)
    b.sync()

    # Pass 2: scatter LUT-pointed slots. ``block_count[q_block]`` is a
    # single i32 load and is wave-uniform.
    block_count_v = b.global_load_i32(block_count, q_block_idx)

    def _scatter_body(slot: Value) -> None:
        in_range = b.cmp_lt(slot, block_count_v)
        with b.scf_if(in_range):
            slot_off = b.add(lut_row_base, slot)
            lut_val = b.global_load_i32(block_lut, slot_off)
            b.smem_store_vN(bitmap_lds, [lut_val], one_i8, 1)

    # The static cooperative iter handles ``max_blocks_per_q > 64`` via
    # chunked walks; the per-lane in-range guard handles
    # ``slot >= block_count_v``.
    for chunk in range((max_blocks_per_q + _BLOCK_SIZE - 1) // _BLOCK_SIZE):
        base = chunk * _BLOCK_SIZE
        slot = tid if base == 0 else b.add(tid, b.const_i32(base))
        # We still need a static-range check for the last partial chunk
        # so out-of-range lanes do not even issue the global LUT load.
        if base + _BLOCK_SIZE <= max_blocks_per_q:
            _scatter_body(slot)
        else:
            with b.scf_if(b.cmp_lt(slot, b.const_i32(max_blocks_per_q))):
                _scatter_body(slot)
    return bitmap_lds


def _lds_bitmap_predicate(
    b: IRBuilder,
    bitmap_lds: Value,
    k_block_idx: Value,
) -> Value:
    """``bitmap_lds[k_block_idx] != 0`` -- one LDS byte read + i8 cmp."""
    loaded = b.smem_load_vN(bitmap_lds, k_block_idx, dtype=I8, n=1)
    byte_v = b.vec_extract(loaded, 0)
    return b.cmp_ne(byte_v, _const_i8(b, 0))


# ---------------------------------------------------------------------
# Jenga: build kernel
# ---------------------------------------------------------------------


def build_jenga_sparse_attention(
    spec: JengaSparseSpec, arch: str = "gfx950"
) -> KernelDef:
    """Jenga block-sparse forward kernel (MFMA-tiled).

    Per-K-tile mask gate via ``MaskBitmap[q_block, k_block]``: when the
    bitmap byte for the enclosing sparsity block is zero, the whole
    16-position K-tile is force-masked to ``-inf`` (no softmax
    contribution, no PV contribution). The K loads still fire (no
    block-level skip in the MFMA inner -- a v2 hoist with ``scf.if``
    over the K-tile that matches the CK Tile reference's
    ``block_relation_onehot`` skip is a follow-on).

    The mask bitmap is staged to LDS once per CTA so the predicate
    body is a single LDS byte read (~5 cycle ``ds_read_u8``) instead
    of a per-K-tile global byte load (~hundreds of cycles on a cold
    L1 line).
    """
    ok, why = is_valid_jenga_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid jenga_sparse spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(_BLOCK_SIZE)
    _declare_jenga_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation
    mask = kb.ptr("mask")
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k_arg = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    # The Q tile's first row determines its sparsity-q-block.
    q_block_idx = _magic_div(b, q_tile_base, spec.block_q)
    mask_row_base = b.mul(q_block_idx, b.const_i32(spec.num_k_blocks))

    # Stage the per-Q-block mask row into LDS once. The K-tile predicate
    # then reads from LDS instead of replaying a global load per tile.
    tid = b.thread_id_x()
    mask_lds = _stage_jenga_mask_to_lds(
        b,
        mask_global=mask,
        mask_row_base=mask_row_base,
        num_k_blocks=spec.num_k_blocks,
        tid=tid,
    )
    b.sync()

    # Each MFMA K-tile = ``MFMA_ATTN_BLOCK_K`` K positions = one
    # ``block_k / MFMA_ATTN_BLOCK_K``-th of one sparsity block.
    tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K

    def _jenga_tile_predicate(b: IRBuilder, kt):
        """``MaskBitmap[q_block, kt // tiles_per_block_k] != 0`` from LDS."""
        k_block_idx = _magic_div(b, kt, tiles_per_block_k)
        return _lds_bitmap_predicate(b, mask_lds, k_block_idx)

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
        arch=arch,
    )
    b.ret()
    return kb.kernel


# ---------------------------------------------------------------------
# VSA: build kernel
# ---------------------------------------------------------------------


def build_vsa_sparse_attention(spec: VsaSparseSpec, arch: str = "gfx950") -> KernelDef:
    """Variable-size sparse attention forward kernel (MFMA-tiled).

    The VSA LUT lookup is **pre-computed** into an LDS bitmap once per
    CTA: each lane reads one LUT slot (if in range), and stores ``1``
    at ``bitmap[lut[q_block, slot]]``. Concurrent writes are safe
    (idempotent stores of ``1``). The predicate then becomes a single
    LDS byte read per K-tile.

    This replaces the v1 ``scf.for`` slot scan that, for every K-tile,
    walked all ``max_blocks_per_q`` slots in global memory looking for
    a hit. The new path issues O(``max_blocks_per_q``) global LUT
    loads once at the top of the CTA and O(1) LDS reads per K-tile,
    instead of O(``max_blocks_per_q``) global loads + a chained
    ``scf.for`` for *every* MFMA K-tile in the dense loop.
    """
    ok, why = is_valid_vsa_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid vsa_sparse spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(_BLOCK_SIZE)
    _declare_vsa_params(kb)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation
    block_lut = kb.ptr("block_lut")
    block_count = kb.ptr("block_count")
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k_arg = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    q_block_idx = _magic_div(b, q_tile_base, spec.block_q)
    lut_row_base = b.mul(q_block_idx, b.const_i32(spec.max_blocks_per_q))

    tid = b.thread_id_x()
    bitmap_lds = _stage_vsa_bitmap_to_lds(
        b,
        block_lut=block_lut,
        block_count=block_count,
        q_block_idx=q_block_idx,
        lut_row_base=lut_row_base,
        num_k_blocks=spec.num_k_blocks,
        max_blocks_per_q=spec.max_blocks_per_q,
        tid=tid,
    )
    b.sync()

    tiles_per_block_k = spec.block_k // MFMA_ATTN_BLOCK_K

    def _vsa_tile_predicate(b: IRBuilder, kt):
        """Tile-level VSA predicate via the LDS-staged bitmap.

        ``kt`` is the MFMA K-tile index; the sparsity-block index is
        ``kt // tiles_per_block_k``. After the staging step above the
        whole "is this block in the LUT?" question collapses to one
        LDS byte read + i8 compare.
        """
        k_block_idx = _magic_div(b, kt, tiles_per_block_k)
        return _lds_bitmap_predicate(b, bitmap_lds, k_block_idx)

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
        arch=arch,
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
