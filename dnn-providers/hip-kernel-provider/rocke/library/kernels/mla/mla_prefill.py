# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MLA prefill forward kernel — compressed-KV + decoupled RoPE, causal, bf16.

Op: ``mla_prefill_fwd``

Supported architectures: gfx942 and gfx950.  bf16 only (fp8 is Phase 2).

## Design (DESIGN.md §11.4 — CK (192, 128) path)

MLA geometry (canonical for DeepSeek/GLM-5/Kimi-K2):
  d_nope = 128,  d_rope = 64,  d_qk = 192,  d_v = 128,  r_KV = 512

This kernel implements the **separate-expansion** path from DESIGN.md §11.4:

    Prior GEMMs (done outside this kernel, e.g. by PyTorch or a fused kernel):
        q          = q_latent @ W_UQ^T       [total_q, H_q, 192]  (nope+rope)
        K_nope     = c_KV @ W_UK_K^T         [total_kv, 192]
        K_exp      = concat(K_nope, K_rope)  [total_kv, 192]  → packed into paged KV
        V_exp_pad  = pad(c_KV @ W_UV^T, 0→64) [total_kv, 192]  V padded to 192, zeros in [128:]

    This kernel:
        Flash prefill attention: q @ K_exp^T → softmax → @ V_exp_pad → out[:, :, :128]

This is a thin wrapper over ``build_fmha_fwd_paged_prefill`` with:
    head_size = 192 (d_qk = d_nope + d_rope)
    mask_mode = "causal"
    use_mfma_body = True
    H_k = 1 (MLA always has a single KV head)

The V padding trick: K_exp and V_exp_pad share the same paged KV layout
``[num_blocks, block_size, 1, 192]``.  The last 64 columns of V_exp_pad are
zero, so they do not contribute to the weighted sum.  The caller truncates
the output to ``[:, :, :128]`` after the kernel returns.

Grid: ``(H_q, total_q // 16, 1)`` for ``use_mfma_body=True``.
Block: ``(64, 1, 1)``.

H_q parametrized: 128 (DeepSeek-R1, GLM-5) or 64 (Kimi-K2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from rocke.core.ir import KernelDef
from rocke.helpers.spec import kernel_name_join


__all__ = [
    "MlaPrefillSpec",
    "build_mla_prefill_fwd",
    "is_valid_mla_prefill_spec",
    "mla_prefill_fwd_grid",
    "mla_prefill_fwd_signature",
    # MFMA tiled variant (faster; uses build_unified_attention_2d_tiled)
    "MlaPrefillMfmaSpec",
    "build_mla_prefill_mfma_fwd",
    "is_valid_mla_prefill_mfma_spec",
    "mla_prefill_mfma_grid",
    "mla_prefill_mfma_signature",
    # v2: purpose-built MLA kernel — inner H_q loop, K/V shared across heads
    "build_mla_prefill_mfma_fwd_v2",
    "mla_prefill_mfma_v2_grid",
    "mla_prefill_mfma_v2_signature",
]

# ---------------------------------------------------------------------------
# MLA geometry constants
# ---------------------------------------------------------------------------
D_NOPE: int = 128
D_ROPE: int = 64
D_QK: int = D_NOPE + D_ROPE   # = 192  — effective head_size for QK attention
D_V: int = 128                 # actual V head dim; output truncated to this
R_KV: int = 512                # kv_lora_rank

# Kernel head_size: must be power-of-two-dividable by WARP_SIZE=64 for warp body.
# We pad K_exp from 192 to 256 (zeros in [192:256]) so ept=256/64=4 (vectorizable).
# V_exp is padded from 128 to 256 (zeros in [128:256]); output truncated to [:128].
# The zeros pad does not affect attention scores or output values.
KERNEL_HEAD_SIZE: int = 256


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MlaPrefillSpec:
    """Compile-time MLA prefill configuration.

    Args:
        num_query_heads: H_q — 128 for DeepSeek-R1/GLM-5, 64 for Kimi-K2.
        block_size:      Paged-KV block size.  16 is the canonical MLA value.
        batch:           Maximum number of sequences in one call.
        name:            Kernel name prefix.
    """

    num_query_heads: int
    block_size: int = 16
    batch: int = 1
    name: str = "rocke_mla_prefill_fwd"

    def __post_init__(self) -> None:
        if self.num_query_heads not in (64, 128):
            raise ValueError(
                f"MlaPrefillSpec.num_query_heads must be 64 or 128; "
                f"got {self.num_query_heads}"
            )
        if self.block_size not in (16, 32, 64):
            raise ValueError(
                f"MlaPrefillSpec.block_size must be 16/32/64; got {self.block_size}"
            )

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"Hq{self.num_query_heads}",
            f"Bs{self.block_size}",
            "bf16",
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_mla_prefill_spec(
    spec: MlaPrefillSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``."""
    from kernels.common.fmha_paged_prefill import FmhaFwdPagedPrefillSpec, is_valid_spec as _ok
    from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape

    if arch not in ("gfx942", "gfx950"):
        return False, f"MLA prefill bf16 requires gfx942 or gfx950; got {arch!r}"

    inner = FmhaFwdPagedPrefillSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(
                head_size=KERNEL_HEAD_SIZE,
                num_query_heads=spec.num_query_heads,
                num_kv_heads=1,
                block_size_k=spec.block_size,
            ),
            dtype="bf16",
            mask_mode="causal",
        ),
        page_block_size=spec.block_size,
        max_blocks_per_seq=8192,
        batch=spec.batch,
        use_mfma_body=False,
    )
    return _ok(inner, arch)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_mla_prefill_fwd(
    spec: MlaPrefillSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the MLA prefill forward kernel.

    Emits a ``fmha_fwd_paged_prefill`` kernel with:
      - ``head_size = 192`` (d_qk = d_nope + d_rope)
      - ``num_kv_heads = 1`` (MLA's single latent KV head)
      - ``mask_mode = "causal"``
      - ``use_mfma_body = True`` (MFMA-tiled body, ~10-30× vs warp-scalar)

    The kernel reads:
      - Q:       ``[total_q, H_q, 192]`` bf16  — expanded query (W_UQ applied prior)
      - K_cache: ``[num_blocks, block_size, 1, 192]`` bf16 — K_exp = [K_nope ‖ K_rope]
      - V_cache: ``[num_blocks, block_size, 1, 192]`` bf16 — V_exp padded to 192 (last 64 = 0)
      - block_table, cu_seqlens_q, seqlens_k: standard paged-prefill ABI

    The output is ``[total_q, H_q, 192]``.  The caller truncates to
    ``[:, :, :128]`` to recover the true d_V=128 output.

    Args:
        spec: Compile-time parameters.
        arch: ``"gfx942"`` or ``"gfx950"``.

    Returns:
        A :class:`~rocke.core.ir.KernelDef` ready for compilation.
    """
    ok, why = is_valid_mla_prefill_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid MlaPrefillSpec: {why}")

    from kernels.common.fmha_paged_prefill import (
        FmhaFwdPagedPrefillSpec,
        build_fmha_fwd_paged_prefill,
    )
    from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape

    inner_spec = FmhaFwdPagedPrefillSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(
                head_size=KERNEL_HEAD_SIZE,
                num_query_heads=spec.num_query_heads,
                num_kv_heads=1,
                block_size_k=spec.block_size,
            ),
            dtype="bf16",
            mask_mode="causal",
        ),
        page_block_size=spec.block_size,
        max_blocks_per_seq=8192,
        batch=spec.batch,
        name=spec.kernel_name(),
        use_mfma_body=False,
    )
    return build_fmha_fwd_paged_prefill(inner_spec, arch)


# ---------------------------------------------------------------------------
# Grid and signature
# ---------------------------------------------------------------------------


def mla_prefill_fwd_grid(
    spec: MlaPrefillSpec, *, total_q: int, batch: int = 1
) -> Tuple[int, int, int]:
    """Return the launch grid: ``(total_q, H_q, 1)``.

    One CTA per (q_token, head) pair — matches warp-distributed paged-prefill body.
    """
    return (total_q, spec.num_query_heads, 1)


def mla_prefill_fwd_signature(spec: MlaPrefillSpec) -> List[dict]:
    """Return the kernel ABI as ``[{"name": ..., "type": ...}, ...]`` dicts."""
    from kernels.common.fmha_paged_prefill import (
        FmhaFwdPagedPrefillSpec,
        fmha_fwd_paged_prefill_signature,
    )
    from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape

    inner_spec = FmhaFwdPagedPrefillSpec(
        common=FmhaCommonSpec(
            shape=FmhaShape(
                head_size=KERNEL_HEAD_SIZE,
                num_query_heads=spec.num_query_heads,
                num_kv_heads=1,
                block_size_k=spec.block_size,
            ),
            dtype="bf16",
            mask_mode="causal",
        ),
        page_block_size=spec.block_size,
        max_blocks_per_seq=8192,
        batch=spec.batch,
        use_mfma_body=False,
    )
    return fmha_fwd_paged_prefill_signature(inner_spec)


# ===========================================================================
# MFMA tiled variant — uses build_unified_attention_2d_tiled (H_q=H_k=1)
# ===========================================================================
#
# Strategy: compile one kernel for (H_q=1, H_k=1, head_size=256, causal).
# At runtime call it H_q times, slicing Q and O by head index each time.
# K_exp / V_exp are shared across all heads (shape [num_blocks, bs, 1, 256]).
#
# This gives the full tiled-2D performance (ring, MFMA 32x32, LDS staging)
# without needing a GQA-ratio-aware kernel for H_q=128/H_k=1.
#
# ABI: same as build_unified_attention_2d_tiled with H_q=H_k=1.
# Signature produced by _attn_signature(include_bt_stride=True).


@dataclass(frozen=True)
class MlaPrefillMfmaSpec:
    """Compile-time config for the MFMA tiled MLA prefill kernel.

    Args:
        num_query_heads: H_q — 128 (DeepSeek-R1/GLM-5) or 64 (Kimi-K2).
        block_size:      Paged-KV block size (16 canonical).
        batch:           Maximum number of sequences per call (for v2 binary
                         search iteration count; does not affect v1).
        name:            Kernel name prefix.
    """

    num_query_heads: int
    block_size: int = 16
    batch: int = 1
    name: str = "rocke_mla_prefill_mfma"

    def __post_init__(self) -> None:
        if self.num_query_heads not in (64, 128):
            raise ValueError(
                f"MlaPrefillMfmaSpec.num_query_heads must be 64 or 128; "
                f"got {self.num_query_heads}"
            )
        if self.block_size not in (16, 32, 64):
            raise ValueError(
                f"MlaPrefillMfmaSpec.block_size must be 16/32/64; got {self.block_size}"
            )

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"Hq{self.num_query_heads}",
            f"Bs{self.block_size}",
            "bf16",
        )


def is_valid_mla_prefill_mfma_spec(
    spec: MlaPrefillMfmaSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the MFMA tiled variant."""
    if arch not in ("gfx942", "gfx950"):
        return False, f"MLA prefill mfma requires gfx942 or gfx950; got {arch!r}"
    from kernels import UnifiedAttentionProblem, supports_native_unified_attention_tiled
    from kernels.common import attention_unified as au
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = arch
    try:
        p = UnifiedAttentionProblem(
            total_q=spec.block_size * 8,  # small probe
            num_seqs=1,
            num_query_heads=1, num_kv_heads=1,
            head_size=KERNEL_HEAD_SIZE,
            block_size=spec.block_size,
            max_seqlen_q=spec.block_size * 8,
            max_seqlen_k=spec.block_size * 8,
            dtype="bf16",
        )
        return supports_native_unified_attention_tiled(p)
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch


def build_mla_prefill_mfma_fwd(
    spec: MlaPrefillMfmaSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the MFMA tiled MLA prefill kernel (H_q=H_k=1 per launch).

    Emits a ``build_unified_attention_2d_tiled`` kernel with:
      - head_size = KERNEL_HEAD_SIZE = 256
      - H_q = H_k = 1  (one head per launch; caller issues H_q parallel launches)
      - mask_mode = "causal"

    K/V cache is NOT replicated: shape ``[nb, bs, 1, 256]`` is shared across
    all H_q launches.  Each launch processes Q for one head via a contiguous
    ``[sq, 256]`` slice.  H_q launches are dispatched on separate HIP streams
    so they execute in parallel (limited by GPU CU availability).

    Grid per launch: ``(1, ceil(sq/tile)+1, 1)``.
    Block: ``(64, 1, 1)``.
    """
    ok, why = is_valid_mla_prefill_mfma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid MlaPrefillMfmaSpec: {why}")

    from kernels.common import attention_unified as au
    from kernels.common.attention_unified import (
        _tiled_2d_impl,
        _tiled_spec_from_problem,
        UnifiedAttentionProblem,
    )

    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = arch
    au._2D_LAUNCH_META.clear()
    try:
        p = UnifiedAttentionProblem(
            total_q=8192,
            num_seqs=1,
            num_query_heads=1, num_kv_heads=1,
            head_size=KERNEL_HEAD_SIZE,
            block_size=spec.block_size,
            max_seqlen_q=8192,
            max_seqlen_k=8192,
            dtype="bf16",
        )
        tiled_spec = _tiled_spec_from_problem(p)
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()

    _, build_fn, _ = _tiled_2d_impl(arch)
    kdef = build_fn(tiled_spec, arch=arch)
    kdef.name = spec.kernel_name() + "_mfma_h1"
    return kdef


def mla_prefill_mfma_grid(
    spec: MlaPrefillMfmaSpec, *, total_q: int, batch: int = 1
) -> Tuple[int, int, int]:
    """Return the per-head launch grid ``(1, q_blocks+1, 1)``.

    Caller dispatches this grid H_q times (one per head) on separate streams.
    """
    from kernels.common import attention_unified as au
    from kernels.common.attention_unified import (
        _get_2d_launch_meta,
        _tiled_cache_key,
        UnifiedAttentionProblem,
    )
    old_arch = au._RESOLVED_ATTENTION_ARCH
    try:
        p = UnifiedAttentionProblem(
            total_q=total_q, num_seqs=1,
            num_query_heads=1, num_kv_heads=1,
            head_size=KERNEL_HEAD_SIZE,
            block_size=spec.block_size,
            max_seqlen_q=total_q, max_seqlen_k=total_q,
            dtype="bf16",
        )
        meta = _get_2d_launch_meta(p, _tiled_cache_key(p))
        return meta.grid
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


def mla_prefill_mfma_signature(spec: MlaPrefillMfmaSpec) -> List[dict]:
    """Return the MFMA kernel ABI as ``[{"name": ..., "type": ...}, ...]``."""
    from kernels.common.attention_unified import _attn_signature
    sig = _attn_signature("bf16", include_bt_stride=True, include_qq_bias_stride=True)
    return sig


# ===========================================================================
# v2: purpose-built MLA MFMA kernel — inner H_q loop, K/V read once
# ===========================================================================
#
# Correct architecture for MLA (H_k=1):
#   Grid: (1, ceil(total_q / BLOCK_M), 1)
#   Block: (64, 1, 1)  — one wave64 per CTA
#
#   Per CTA:
#     q_tile_idx  = block_id_y  ← tile index, not token index
#     q_tile_base = q_tile_idx * BLOCK_M  ← first Q row this CTA owns
#     kv_head_idx = 0  (H_k=1 always)
#
#     binary search → seq_idx → cuq_base, local_q_base, seqlen_k
#
#     scf_for h in [0, H_q):
#       pass q_ptr + h * stride_q_head as Q,
#       pass out_ptr + h * stride_o_head as O,
#       with head_idx = h, kv_head_idx = 0,
#       call mfma_attention_fwd_inner_body(K=K_shared, V=V_shared, ...)
#
# This eliminates:
#   - Any K/V replication (K_cache/V_cache are never duplicated)
#   - The causal-mask bug (local_q_base is the true token offset, not tile)
#   - The OOM from H_q-replicated q_all / out_all tensors
#
# K/V data stays hot in L1/LDS across the inner H_q loop, giving
# significant bandwidth savings vs the old per-head-launch approach.


def build_mla_prefill_mfma_fwd_v2(
    spec: MlaPrefillMfmaSpec, arch: str = "gfx950"
) -> "KernelDef":
    """Build the v2 MLA prefill MFMA kernel with an inner H_q loop.

    Grid: ``(1, ceil(total_q / BLOCK_M), 1)`` — one CTA per Q tile.
    Block: ``(64, 1, 1)`` — one wave64 per CTA.

    Each CTA processes ``BLOCK_M = 16`` Q rows through all H_q heads.
    K_cache and V_cache (shape ``[nb, bs, 1, D_QK]``) are read once per
    CTA and shared across all H_q iterations of the inner loop.

    The inner head loop is a ``scf_for`` (runtime loop, not Python
    unroll) so H_q=128 generates compact IR rather than 128 copies of
    the flash-attention body.

    Args:
        spec: Compile-time MLA config (num_query_heads, block_size).
        arch: ``"gfx942"`` or ``"gfx950"``.

    Returns:
        A :class:`~rocke.core.ir.KernelDef` ready for compilation.
    """
    ok, why = is_valid_mla_prefill_mfma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid MlaPrefillMfmaSpec for v2: {why}")

    from rocke.core.ir import IRBuilder, PtrType, I32, F32, BF16
    from rocke.helpers.mfma_attention import (
        MFMA_ATTN_BLOCK_M,
        mfma_attention_fwd_inner_body,
    )
    from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape, FmhaKernelBuilder

    hq = spec.num_query_heads
    pg = spec.block_size
    assert (pg & (pg - 1)) == 0, "block_size must be power of two"
    pg_log2 = pg.bit_length() - 1
    pg_mask = pg - 1

    # Build via FmhaKernelBuilder for consistent ABI (strides, param
    # naming) — but we override the grid decode manually since the grid
    # convention differs from the standard per-token layout.
    common = FmhaCommonSpec(
        shape=FmhaShape(
            head_size=KERNEL_HEAD_SIZE,
            num_query_heads=hq,
            num_kv_heads=1,
            block_size_k=pg,
        ),
        dtype="bf16",
        mask_mode="causal",
    )
    kb = FmhaKernelBuilder(spec.kernel_name() + "_v2", common)
    kb.block_size(64)

    # ABI: same tensors/pointers as fmha_fwd_paged_prefill
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
    kb.add_scalar("stride_block", "i32")
    kb.add_scalar("stride_page", "i32")
    kb.add_scalar("stride_kv_head", "i32")
    kb.add_scalar("stride_v_block", "i32")
    kb.add_scalar("stride_v_page", "i32")
    kb.add_scalar("stride_v_kv_head", "i32")
    kb.add_strides("o")
    kb.add_scalar("block_table_stride", "i32")
    # num_query_heads as a runtime param so the same kernel binary
    # works for any H_q (even though it is compile-time in the spec).
    kb.add_scalar("num_query_heads", "i32")

    b = kb.builder

    Q = kb.tensor("Q")
    K_cache = kb.tensor("K_cache")
    V_cache = kb.tensor("V_cache")
    O = kb.tensor("O")  # noqa: E741
    block_table = kb.ptr("block_table")
    cu_seqlens_q = kb.ptr("cu_seqlens_q")
    seqlens_k_ptr = kb.ptr("seqlens_k")
    scale_log2 = kb.scalar("scale_log2")
    block_table_stride = kb.scalar("block_table_stride")
    num_query_heads = kb.scalar("num_query_heads")

    stride_block = kb.scalar("stride_block")
    stride_page = kb.scalar("stride_page")
    stride_kv_head = kb.scalar("stride_kv_head")
    stride_v_block = kb.scalar("stride_v_block")
    stride_v_page = kb.scalar("stride_v_page")
    stride_v_kv_head = kb.scalar("stride_v_kv_head")

    # Grid: block_id_y = Q-tile index; block_id_x / z unused.
    q_tile_idx = b.to_sgpr_u32(b.block_id_y())
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))

    # kv_head_idx is always 0 for MLA.
    kv_head_idx = b.const_i32(0)

    # Binary search: find which sequence this q-tile belongs to.
    # We compare q_tile_base (first row of the tile) against cu_seqlens_q.
    # Convention: cu_seqlens_q[s] <= q_tile_base < cu_seqlens_q[s+1].
    batch_val = kb.scalar("batch")
    bs_iters = max(1, int(math.ceil(math.log2(spec.batch + 1))))
    bs_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(bs_iters),
        b.const_i32(1),
        [
            ("bs_left", b.const_i32(0)),
            ("bs_right", batch_val),
        ],
        iv_name="bs_i",
    )
    with bs_loop as (_iv, (left, right)):
        done = b.cmp_ge(left, right)
        mid = b.div(b.add(left, right), b.const_i32(2))
        cuq_next_mid = b.global_load_i32(cu_seqlens_q, b.add(mid, b.const_i32(1)))
        go_right = b.cmp_le(cuq_next_mid, q_tile_base)
        nl = b.select(go_right, b.add(mid, b.const_i32(1)), left)
        nr = b.select(go_right, right, mid)
        b.scf_yield(b.select(done, left, nl), b.select(done, right, nr))

    seq_idx = bs_loop.results[0]
    cuq_base = b.global_load_i32(cu_seqlens_q, seq_idx)
    # local_q_base: the token offset of this tile's first row within its sequence.
    # Used as causal_ctx_offset so mask k <= local_q + row_in_tile is correct.
    local_q_base = b.sub(q_tile_base, cuq_base)
    seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx)

    # Paged-KV row-base callbacks — kv_head_idx=0 is hoisted.
    block_table_row_base = b.mul(seq_idx, block_table_stride)
    c_pg_log2 = b.const_i32(pg_log2)
    c_pg_mask = b.const_i32(pg_mask)

    def _paged_k_row(b, k_idx):
        block_idx = b.lshr(k_idx, c_pg_log2)
        page_in_block = b.land(k_idx, c_pg_mask)
        block_id = b.global_load_i32(
            block_table,
            b.add(block_table_row_base, block_idx),
        )
        return b.add(
            b.mul(block_id, stride_block),
            b.mul(page_in_block, stride_page),
        )
        # kv_head=0, so stride_kv_head * 0 = 0; omit the multiply.

    def _paged_v_row(b, k_idx):
        block_idx = b.lshr(k_idx, c_pg_log2)
        page_in_block = b.land(k_idx, c_pg_mask)
        block_id = b.global_load_i32(
            block_table,
            b.add(block_table_row_base, block_idx),
        )
        return b.add(
            b.mul(block_id, stride_v_block),
            b.mul(page_in_block, stride_v_page),
        )

    # Inner loop over H_q heads using scf_for (avoids 128x IR unroll).
    # For each head h:
    #   Q pointer offset: h * stride_q_head elements from Q base
    #   O pointer offset: h * stride_o_head elements from O base
    # We pass head_idx=h and stride_q_head/stride_o_head to the inner body;
    # the body computes: q_addr = (q_tile_base + m_in_atom)*stride_q_token + h*stride_q_head
    hloop = b.scf_for(
        b.const_i32(0),
        num_query_heads,
        b.const_i32(1),
        iv_name="hq",
    )
    with hloop as hq_idx:
        mfma_attention_fwd_inner_body(
            b,
            Q=Q,
            K=K_cache,
            V=V_cache,
            O=O,
            head_size=KERNEL_HEAD_SIZE,
            seqlen_k=seqlen_k,
            q_tile_base=q_tile_base,
            # q_pos_base: position used for the causal mask check.
            # We want: k_idx <= local_q_base + row_in_tile.
            # apply_attention_mask computes: k_idx <= causal_ctx_offset + query_pos
            # where query_pos = q_pos_base + m_blk*4 + r.
            # Setting q_pos_base=local_q_base and causal_ctx_offset=0 gives:
            #   query_pos = local_q_base + row_in_tile  ✓
            q_pos_base=local_q_base,
            head_idx=hq_idx,
            kv_head_idx=kv_head_idx,
            stride_q_token=kb.stride_token("q"),
            stride_q_head=kb.stride_head("q"),
            # stride_k/v_token: the row-base callbacks already return the physical
            # element offset (block*stride_block + page*stride_page), so these
            # strides are unused when k_row_base_fn is set. Pass stride_page as
            # a safe fallback for any dense code paths.
            stride_k_token=stride_page,
            stride_k_head=stride_kv_head,
            stride_v_token=stride_v_page,
            stride_v_head=stride_v_kv_head,
            stride_o_token=kb.stride_token("o"),
            stride_o_head=kb.stride_head("o"),
            scale_log2=scale_log2,
            dtype="bf16",
            mask_mode="causal",
            causal_ctx_offset=b.const_i32(0),
            k_row_base_fn=_paged_k_row,
            v_row_base_fn=_paged_v_row,
            arch=arch,
        )

    b.ret()
    return kb.kernel


def mla_prefill_mfma_v2_grid(
    spec: MlaPrefillMfmaSpec, *, total_q: int, batch: int = 1
) -> Tuple[int, int, int]:
    """Return the v2 launch grid: ``(1, ceil(total_q / BLOCK_M), 1)``.

    One CTA per Q tile; each CTA iterates over all H_q heads internally.
    """
    from rocke.helpers.mfma_attention import MFMA_ATTN_BLOCK_M

    q_tiles = (total_q + MFMA_ATTN_BLOCK_M - 1) // MFMA_ATTN_BLOCK_M
    return (1, q_tiles, 1)


def mla_prefill_mfma_v2_signature(spec: MlaPrefillMfmaSpec) -> List[dict]:
    """Return the v2 kernel ABI as ``[{"name": ..., "type": ...}, ...]``."""
    # Build a throwaway kernel just to extract the signature.
    from rocke.helpers.mfma_attention import MFMA_ATTN_BLOCK_M
    from kernels.common._fmha_common import FmhaCommonSpec, FmhaShape, FmhaKernelBuilder

    common = FmhaCommonSpec(
        shape=FmhaShape(
            head_size=KERNEL_HEAD_SIZE,
            num_query_heads=spec.num_query_heads,
            num_kv_heads=1,
            block_size_k=spec.block_size,
        ),
        dtype="bf16",
        mask_mode="causal",
    )
    kb = FmhaKernelBuilder("_v2_sig_probe", common)
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
    kb.add_scalar("stride_block", "i32")
    kb.add_scalar("stride_page", "i32")
    kb.add_scalar("stride_kv_head", "i32")
    kb.add_scalar("stride_v_block", "i32")
    kb.add_scalar("stride_v_page", "i32")
    kb.add_scalar("stride_v_kv_head", "i32")
    kb.add_strides("o")
    kb.add_scalar("block_table_stride", "i32")
    kb.add_scalar("num_query_heads", "i32")
    return kb.signature()
