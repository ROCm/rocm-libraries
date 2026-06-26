# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sage attention forward (CK Tile ``49_sageattention`` parity).

Extends FMHA-fwd-fp8 with per-block / per-head Q and K scales that
compensate for the dynamic range loss when Q and K live in
fp8 / int8 / int4 storage. The pipeline (per K-block)::

    K_dequant = dequant_codebook[K_quant[k_block, :]] * k_block_scale
    score_log2 = (Q[q_token, :] . K_dequant) * Q_scale * K_scale
    ... (online softmax as in fmha_fwd) ...
    V_dequant = dequant_codebook[V_quant[k_block, :]] * v_block_scale
    acc += p * V_dequant

The four CK Tile Sage variants share one entry-point, parameterised by
``quant_mode``:

* ``"fp16_bf16"``  -- baseline (no QK quant; pipeline validation).
* ``"fp8_bf16"``   -- Q in activation dtype; K/V in fp8e4m3 with per-block scales.
* ``"i8_fp8_bf16"`` -- K/V stored as i8; codebook re-materialises fp32 values.
* ``"i4_fp8_bf16"`` -- K/V as packed i4; 16-entry codebook + per-block scale.

Two physical kernels back the four modes:

* The **MFMA-tiled body** (``mfma_attention_fwd_inner_body``) is the
  fast path for ``fp16_bf16`` and ``fp8_bf16`` when the spec's
  ``seqlen_q`` / ``seqlen_k`` / scale-blocks align with the MFMA
  geometry (``BLOCK_M = BLOCK_K = 16``). One wave64 warp processes
  one ``BLOCK_M = 16`` Q-row tile per CTA. Q+K per-block scales fold
  into the QK score via ``extra_score_transform``; ``q_scale`` is
  loaded once per CTA and pre-multiplied into ``scale_log2`` so the
  per-K-tile path only adds the K-block scale. K/V load + dequant for
  fp8 K/V is handled inside the MFMA helper.

* The **warp-distributed body** is the universal fallback (one warp
  per ``(q_token, head)`` row). It supports all four quant modes and
  is the only path for ``i8_fp8_bf16`` / ``i4_fp8_bf16`` (the MFMA
  helper has no native i8 / i4 dequant path) and for ``fp16_bf16`` /
  ``fp8_bf16`` specs whose ``seqlen_q`` is below ``BLOCK_M`` or
  unaligned. The body's per-quant-mode improvements vs v1:

  - Vectorised K/V byte slice load via ``global_load_vN`` when each
    lane owns ``>= 2`` consecutive elements (``EPT >= 2``). Saves
    ``EPT - 1`` VMEM transactions per K-iter for fp8/i8 paths and
    for the fp16/bf16 path on head_size > 64.
  - Direct f32-codebook lookup for the i8 / i4 paths -- the v1
    chain ``i8 → codebook f32 → cvt_f32_to_fp8 → cvt_fp8_to_f32``
    becomes ``i8 → codebook f32`` (two ``llvm.amdgcn.cvt.*.fp8``
    fewer per element).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import (
    BF8E5M2,
    F32,
    FP8E4M3,
    I32,
    I8,
    IRBuilder,
    KernelDef,
    Value,
)
from ...helpers.i4_dequant import unpack_i4_byte_to_pair_i32
from ...helpers.io import io_ir_type, load_scalar_as_f32
from ...helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ...helpers.qk_scale import (
    QkScaleSpec,
    apply_qk_scales,
    load_k_scale_for_block,
    load_q_scale_for_block,
)
from ...helpers.spec import kernel_name_join
from ...helpers.transforms import calculate_magic_numbers, do_magic_division
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE, fmha_warp_fwd_inner_body
from .fmha_arch import validate_fmha_mfma_atom


__all__ = [
    "SageAttentionSpec",
    "SageQuantMode",
    "build_sage_attention",
    "is_valid_spec",
    "sage_attention_grid",
    "sage_attention_signature",
]


SageQuantMode = Literal["fp16_bf16", "fp8_bf16", "i8_fp8_bf16", "i4_fp8_bf16"]

_MFMA_QUANT_MODES = ("fp16_bf16", "fp8_bf16")
_CODEBOOK_QUANT_MODES = ("i8_fp8_bf16", "i4_fp8_bf16")


def _magic_div(b: IRBuilder, dividend: Value, divisor: int) -> Value:
    """``dividend // divisor`` via CK Tile's magic mul-hi division.

    Used for the per-block scale-index decode (``pos // scale_block``) and
    the i4 packed-byte offset (``lane_d_base // 2``). The divisor is a
    compile-time constant so ``(multiplier, shift)`` fold to immediates and
    the runtime cost is one ``v_mul_hi_u32`` + add + shift instead of the
    AMDGPU integer divider. The dividend is a non-negative i32 (a Q/K
    position or a lane-derived offset), inside the magic sequence's 31-bit
    unsigned range, so the unsigned magic quotient equals the ``b.div`` it
    replaces (device lowering of ``merge_v2_magic_division``).
    """
    mult, shift = calculate_magic_numbers(divisor)
    return do_magic_division(b, dividend, mult, shift)


@dataclass(frozen=True)
class SageAttentionSpec:
    """One concrete Sage attention configuration."""

    common: FmhaCommonSpec
    quant_mode: SageQuantMode
    q_scale: QkScaleSpec
    k_scale: QkScaleSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "rocke_sage_attention"
    # P90: per-block scale outer loop. When ``True`` and ``scale_block
    # > BLOCK_K * 4`` (typical), the K-loop is wrapped in an outer
    # loop over ``(seqlen_k / scale_block)`` scale blocks, each
    # hosting ``scale_block / BLOCK_K`` K-tiles. ``k_scale`` is
    # loaded once per outer iteration instead of every K-tile.
    # Saves a per-iter f32 scale load when scale_block is large.
    use_outer_scale_loop: bool = False

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            self.quant_mode,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"Q{self.seqlen_q}",
            f"K{self.seqlen_k}",
        )


def _mfma_dimensions_ok(spec: SageAttentionSpec) -> bool:
    """True iff the spec satisfies the MFMA-tiled body's geometry rules.

    All four conditions must hold:

    * ``seqlen_q % BLOCK_M == 0`` (CTA owns one BLOCK_M Q-tile)
    * ``seqlen_k % BLOCK_K == 0`` (helper K-loop trip count is integer)
    * ``head_size % 16 == 0`` (MFMA atom k-dim divides head_dim)
    * each per_block scale's ``scale_block`` is a multiple of the
      corresponding MFMA tile dim (so all rows of a Q-tile share one
      Q-scale and each K-tile uses one K-scale)
    """
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False
    if spec.common.shape.head_size % MFMA_ATTN_BLOCK_M != 0:
        return False
    if (
        spec.q_scale.layout == "per_block"
        and spec.q_scale.scale_block % MFMA_ATTN_BLOCK_M != 0
    ):
        return False
    if (
        spec.k_scale.layout == "per_block"
        and spec.k_scale.scale_block % MFMA_ATTN_BLOCK_K != 0
    ):
        return False
    return True


def _uses_mfma_path(spec: SageAttentionSpec) -> bool:
    """True iff the spec routes to ``mfma_attention_fwd_inner_body``.

    The MFMA path is the fast path for fp16/fp8 K-V; the warp body is
    the universal fallback (and the only path for i8/i4).
    """
    return spec.quant_mode in _MFMA_QUANT_MODES and _mfma_dimensions_ok(spec)


def is_valid_spec(spec: SageAttentionSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # Both the MFMA fast path and the warp fallback ultimately run the
    # narrow f16/bf16 16x16x16 QK/PV atom (the fp8 K/V path dequants on
    # load and falls back to the f16 atom; native_fp8_path is not used
    # here). Validate it against the target catalog so an arch lacking
    # the atom rejects cleanly before comgr.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
    if not ok:
        return False, why
    if spec.quant_mode not in (
        "fp16_bf16",
        "fp8_bf16",
        "i8_fp8_bf16",
        "i4_fp8_bf16",
    ):
        return False, f"unknown quant_mode {spec.quant_mode!r}"
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    if spec.quant_mode == "i4_fp8_bf16" and (spec.common.shape.head_size % 2) != 0:
        return False, (
            f"i4 sage requires head_size divisible by 2 "
            f"(got {spec.common.shape.head_size})"
        )
    # The warp body is the universal fallback so the spec must satisfy
    # *its* head_size constraint (head_size % WARP_SIZE == 0).
    if spec.common.shape.head_size % WARP_SIZE != 0:
        return False, (
            f"sage warp body needs head_size % {WARP_SIZE} == 0 "
            f"(got {spec.common.shape.head_size})"
        )
    return True, "ok"


def _kv_pointee_for_quant_mode(quant_mode: str, dtype: str):
    if quant_mode == "fp16_bf16":
        return io_ir_type(dtype)
    if quant_mode == "fp8_bf16":
        return FP8E4M3
    return I8


def _kv_dtype_str(quant_mode: str, dtype: str) -> str:
    """Map ``quant_mode`` to the ABI dtype string for K / V tensors."""
    if quant_mode == "fp16_bf16":
        return dtype
    if quant_mode == "fp8_bf16":
        return "fp8e4m3"
    return "i8"


def _declare_params(kb: FmhaKernelBuilder, spec) -> None:
    """Declare the sage attention kernel ABI (shared between build + sig)."""
    kv_dtype = _kv_dtype_str(spec.quant_mode, kb.common.dtype)
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", dtype=kv_dtype, readonly=True, align=8)
    kb.add_tensor("V", dtype=kv_dtype, readonly=True, align=8)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_ptr("q_scale", dtype="f32", readonly=True)
    kb.add_ptr("k_scale", dtype="f32", readonly=True)
    if spec.quant_mode in _CODEBOOK_QUANT_MODES:
        kb.add_ptr("codebook_k", dtype="f32", readonly=True)
        kb.add_ptr("codebook_v", dtype="f32", readonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


# ---------------------------------------------------------------------
# Codebook-dequant primitives (i8 / i4 paths)
# ---------------------------------------------------------------------
#
# Both i8 and i4 modes pack a signed integer into the K/V cache and
# dequant via a small f32 codebook (256 entries for i8, 16 for i4).
# The original (v1) lowering went:
#
#     i8  → sext(i32) → +128  → load f32 → cvt_f32_to_fp8 → cvt_fp8_to_f32
#     i4  → unpack    → +8    → load f32 → cvt_f32_to_fp8 → cvt_fp8_to_f32
#
# but the two-cvt round-trip discards the codebook's full f32
# precision and emits two ``llvm.amdgcn.cvt.fp8.f32`` /
# ``llvm.amdgcn.cvt.f32.fp8`` per element. The codebook value *is*
# already f32, so the helpers below produce f32 directly and skip the
# round-trip. CK Tile's ``BlockSageAttentionPipelineQRKSVS`` performs
# the QK GEMM in fp8/i8 and dequants the accumulator (a different
# pipeline). For the DSL warp body we accumulate in f32 throughout, so
# f32-direct codebook dequant is the natural match.
#
# Reference paths:
#   CK Tile :: include/ck_tile/ops/sageattention/pipeline/block_sageattn_pipeline_qr_ks_vs.hpp:341-373
#       — k_descale loaded per K-iter (BLOCKSCALE) / per-warp / per-thread.
#   helpers/codebook.py::codebook_lookup_i8_to_fp8 — the round-trip we drop.
#   helpers/i4_dequant.py::unpack_i4_byte_to_pair_i32 — re-used here.


# Codebook table sizes (one f32 entry per quantised level).
_CODEBOOK_I8_ENTRIES = 256  # i8 in [-128, 127] -> idx = i8 + 128
_CODEBOOK_I4_ENTRIES = 16  # i4 in [-8, 7] -> idx = i4 + 8


def _stage_codebook_to_lds(
    b: IRBuilder, cb_global: Value, *, n_entries: int, tid: Value, name_hint: str
) -> Value:
    """Cooperatively copy the constant dequant codebook into LDS once
    per CTA.

    The codebook is loop-invariant across the entire K-loop (256 f32 for
    i8, 16 f32 for i4). Reading it from global memory on *every* K-tile
    (``_codebook_*_to_f32`` did ``global_load_f32`` per dequanted
    element) put a VMEM dependency on the hot path. Staging it into LDS
    once turns each per-iter dequant lookup into a ~5-cycle ``ds_read_b32``
    instead of an L1-cached (but still VMEM-issue-slot-consuming) global
    load -- the same "constant table -> LDS" lever the sparse mask path
    already uses (see ``sparse_attention._stage_jenga_mask_to_lds``).

    Returns the LDS handle; the caller must ``b.sync()`` before reading.
    """
    cb_lds = b.smem_alloc(F32, [n_entries], name_hint=name_hint)
    # One wave64 stages the table; loop when n_entries > WARP_SIZE.
    for base in range(0, n_entries, WARP_SIZE):
        slot = tid if base == 0 else b.add(tid, b.const_i32(base))
        if base + WARP_SIZE <= n_entries:
            val = b.global_load_f32(cb_global, slot)
            b.smem_store_vN_f32(cb_lds, [slot], val, 1)
        else:
            with b.scf_if(b.cmp_lt(slot, b.const_i32(n_entries))):
                val = b.global_load_f32(cb_global, slot)
                b.smem_store_vN_f32(cb_lds, [slot], val, 1)
    return cb_lds


def _codebook_lds_lookup_f32(b: IRBuilder, cb_lds: Value, idx: Value) -> Value:
    """One f32 codebook entry from the LDS-staged table (``ds_read_b32``)."""
    return b.vec_extract(b.smem_load_vN_f32(cb_lds, idx, n=1), 0)


def _codebook_i8_to_f32(b: IRBuilder, cb_ptr: Value, i8_value: Value) -> Value:
    """One i8 byte → f32 via codebook (skips fp8 round-trip).

    ``cb_ptr`` is either a global pointer (legacy) or an LDS handle
    (when the codebook has been staged into LDS by
    :func:`_stage_codebook_to_lds`); the load primitive is chosen from
    the handle's type so both paths share one call site.
    """
    i32 = b.sext(i8_value, I32)
    idx = b.add(i32, b.const_i32(128))
    if _is_lds_handle(cb_ptr):
        return _codebook_lds_lookup_f32(b, cb_ptr, idx)
    return b.global_load_f32(cb_ptr, idx)


def _codebook_i4_pair_to_f32(
    b: IRBuilder, cb_ptr: Value, packed_byte_i8: Value
) -> Tuple[Value, Value]:
    """One packed-i4 byte → two f32 values via codebook (skips fp8 round-trip)."""
    lo_i32, hi_i32 = unpack_i4_byte_to_pair_i32(b, packed_byte_i8)
    c8 = b.const_i32(8)
    if _is_lds_handle(cb_ptr):
        lo = _codebook_lds_lookup_f32(b, cb_ptr, b.add(lo_i32, c8))
        hi = _codebook_lds_lookup_f32(b, cb_ptr, b.add(hi_i32, c8))
        return lo, hi
    lo = b.global_load_f32(cb_ptr, b.add(lo_i32, c8))
    hi = b.global_load_f32(cb_ptr, b.add(hi_i32, c8))
    return lo, hi


def _is_lds_handle(v: Value) -> bool:
    """True iff ``v`` is an LDS (smem) allocation handle rather than a
    global pointer -- used to dispatch the codebook lookup primitive."""
    return type(v.type).__name__ == "SmemType"


def _vectorised_byte_slice(
    b: IRBuilder, KV, base, lane_d_base, ept: int, elem_ty
) -> list[Value]:
    """Load this lane's ``ept`` consecutive elements (single byte or
    16-bit) as one vector and return a Python list of per-element
    :class:`Value`.

    For ``ept == 1`` the scalar load path is used (vector loads require
    ``n in {2, 4, 8, 16}``); for ``ept >= 2`` we issue a single
    ``global_load_vN`` and extract per-element. The compiler folds this
    into one VMEM transaction (``buffer_load_dword`` for fp8 ept=4,
    ``buffer_load_dwordx2`` for fp8 ept=8, etc.).

    ``global_load_vN`` accepts ``{f16, bf16, fp8e4m3, bf8e5m2}`` only;
    for ``i8`` storage (sage int variant) we fall back to scalar
    loads. AMDGPU still folds consecutive per-lane ``buffer_load_ubyte``
    into ``buffer_load_dword`` when the lane addresses are contiguous,
    so the dynamic VMEM transaction count is the same in practice.
    """
    addr_base = b.add(base, lane_d_base)
    if ept == 1:
        return [b.global_load(KV, addr_base, elem_ty)]
    # ``global_load_vN`` rejects ``i8``; route those through scalar
    # loads. f16/bf16 vN requires n in {2, 4, 8}; fp8/bf8 vN allows
    # n in {2, 4, 8, 16}.
    is_f16_like = elem_ty.name in ("f16", "bf16")
    is_fp8_like = elem_ty.name in ("fp8e4m3", "bf8e5m2")
    if (is_f16_like and ept in (2, 4, 8)) or (is_fp8_like and ept in (2, 4, 8, 16)):
        elem_bytes = 2 if is_f16_like else 1
        vec = b.global_load_vN(KV, addr_base, elem_ty, ept, align=ept * elem_bytes)
        return [b.vec_extract(vec, k) for k in range(ept)]
    # Scalar fallback for i8 / non-power-of-two ept (head_size=192 ⇒
    # ept=3). The backend coalesces contiguous byte loads into a
    # single dword load.
    return [
        b.global_load(KV, b.add(addr_base, b.const_i32(k)), elem_ty) for k in range(ept)
    ]


def _load_kv_lane_f32(
    b: IRBuilder,
    *,
    KV: Value,
    base: Value,
    lane_d_base: Value,
    ept: int,
    quant_mode: str,
    cb_ptr,
    kv_ty,
    dtype: str,
) -> list[Value]:
    """Load this lane's full ``ept`` head-dim K (or V) slice as f32.

    The shared front-end for the warp body's K and V dequant chains.

    * ``fp16_bf16`` / ``fp8_bf16`` -- vector load + per-element f32 cast.
    * ``i8_fp8_bf16`` -- vector byte load + direct f32 codebook lookup
      (no fp8 round-trip).

    For the i4 path, the caller handles the packed-byte structure
    directly via :func:`_codebook_i4_pair_to_f32` and does not call
    this helper.
    """
    if quant_mode in ("fp16_bf16", "fp8_bf16"):
        elems = _vectorised_byte_slice(b, KV, base, lane_d_base, ept, kv_ty)
        if quant_mode == "fp16_bf16":
            return [b.cast_to_f32(e) for e in elems]
        return [b.cvt_fp8_to_f32(e) for e in elems]
    if quant_mode == "i8_fp8_bf16":
        bytes_ = _vectorised_byte_slice(b, KV, base, lane_d_base, ept, I8)
        return [_codebook_i8_to_f32(b, cb_ptr, byt) for byt in bytes_]
    raise ValueError(f"unsupported quant_mode for lane load: {quant_mode!r}")


def _load_q_lane_f32(
    b: IRBuilder, Q, q_row_base, lane_d_base, ept: int, dtype: str, kv_ty
) -> list[Value]:
    """Load this lane's ``ept`` Q elements as f32 (vectorised when ept >= 2)."""
    addr_base = b.add(q_row_base, lane_d_base)
    elem_bytes = 2
    if ept in (2, 4, 8):
        vec = b.global_load_vN(Q, addr_base, kv_ty, ept, align=ept * elem_bytes)
        return [b.cast_to_f32(b.vec_extract(vec, k)) for k in range(ept)]
    return [
        load_scalar_as_f32(b, Q, b.add(addr_base, b.const_i32(k)), dtype=dtype)
        for k in range(ept)
    ]


# ---------------------------------------------------------------------
# MFMA-tiled path (fp16_bf16 / fp8_bf16)
# ---------------------------------------------------------------------


def _build_sage_mfma(spec: SageAttentionSpec, arch: str = "gfx950") -> KernelDef:
    """MFMA-tiled sage forward for fp16 / fp8 K-V storage.

    Wraps :func:`mfma_attention_fwd_inner_body` with sage's per-block
    Q+K scale application. The Q-scale is loaded once per CTA and
    pre-multiplied into ``scale_log2`` (constant across the K-loop).
    The K-scale is applied per K-tile inside
    ``extra_score_transform``; when both scales are per_head we fold
    them all into ``scale_log2`` and drop the transform.

    Reference: ``fmha_fwd_fp8.py`` uses the same helper for the
    non-sage fp8 path; this builder is the sage delta on top.
    """
    s = spec.common
    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)
    _declare_params(kb, spec)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation
    q_scale_ptr = kb.ptr("q_scale")
    k_scale_ptr = kb.ptr("k_scale")
    scale_log2_raw = kb.scalar("scale_log2")
    seqlen_k_arg = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))
    batch_idx = b.const_i32(0)

    # Pre-load the Q-scale once per CTA. With ``scale_block %
    # BLOCK_M == 0`` (enforced in is_valid_spec) the same scale covers
    # every row of the BLOCK_M Q-tile.
    if spec.q_scale.layout == "per_block":
        q_block_idx = _magic_div(b, q_tile_base, spec.q_scale.scale_block)
    else:
        q_block_idx = b.const_i32(0)
    q_scale_v = load_q_scale_for_block(
        b,
        q_scale_ptr,
        spec=spec.q_scale,
        batch_idx=batch_idx,
        head_idx=head_idx,
        q_block_idx=q_block_idx,
    )

    # For per_head K-scale we also fold it into ``scale_log2`` so the
    # K-loop sees only ``score * scale_log2`` (one fmul instead of
    # three). For per_block K-scale we keep a per-K-tile loader in
    # ``extra_score_transform``.
    if spec.k_scale.layout == "per_head":
        k_scale_const = load_k_scale_for_block(
            b,
            k_scale_ptr,
            spec=spec.k_scale,
            batch_idx=batch_idx,
            head_idx=kv_head_idx,
            k_block_idx=b.const_i32(0),
        )
        scale_log2 = b.fmul(scale_log2_raw, b.fmul(q_scale_v, k_scale_const))
        extra_score_transform = None
    else:
        scale_log2 = b.fmul(scale_log2_raw, q_scale_v)

        c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)

        def _k_block_scale_transform(
            b: IRBuilder, score: Value, kt: Value, _row_in_atom: int
        ) -> Value:
            """Apply ``k_scale[k_block_idx(kt)]`` to one per-lane score.

            CK Tile reference: BLOCKSCALE branch at
            ``block_sageattn_pipeline_qr_ks_vs.hpp:341`` --
            ``k_descale = k_descale_ptr[(seqlen_k_start + i_total_loops * kN0) / kBlockScaleSizeK]``.
            """
            k_pos = b.mul(kt, c_block_k)
            k_block_idx = _magic_div(b, k_pos, spec.k_scale.scale_block)
            k_scale_v = load_k_scale_for_block(
                b,
                k_scale_ptr,
                spec=spec.k_scale,
                batch_idx=batch_idx,
                head_idx=kv_head_idx,
                k_block_idx=k_block_idx,
            )
            return b.fmul(score, k_scale_v)

        extra_score_transform = _k_block_scale_transform

    causal_ctx = b.const_i32(0) if s.mask_mode in ("causal", "sliding_window") else None
    kv_dtype = (
        "fp8e4m3" if spec.quant_mode == "fp8_bf16" else None
    )  # fp16_bf16: no KV dequant on load

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
        mask_mode=s.mask_mode,
        sliding_window=s.sliding_window,
        causal_ctx_offset=causal_ctx,
        kv_dtype=kv_dtype,
        extra_score_transform=extra_score_transform,
        arch=arch,
    )
    b.ret()
    return kb.kernel


# ---------------------------------------------------------------------
# Codebook warp-distributed path (i8 / i4)
# ---------------------------------------------------------------------


def _build_sage_warp(spec: SageAttentionSpec) -> KernelDef:
    """Warp-distributed sage forward (universal fallback).

    One wave64 warp per ``(q_token, head)`` row. Lanes distribute the
    head_dim axis, the QK dot reduces via the warp-shuffle butterfly,
    and the online softmax state plus the per-lane accumulator slice
    live in registers across the K-loop via ``scf.for`` iter_args.

    The online-softmax loop + epilogue are delegated to the shared
    :func:`fmha_warp_fwd_inner_body`; sage only supplies:

    1. A **dequant K/V lane loader** (``kv_lane_loader`` seam) that
       reads this lane's head-dim slice and dequantises it to f32:
       vectorised byte slice + f32 cast for fp16/fp8 (via
       :func:`_load_kv_lane_f32`), direct f32-codebook lookup for the
       int variants (i8 via :func:`_load_kv_lane_f32`, i4 via
       :func:`_codebook_i4_pair_to_f32`) -- no fp8 round-trip.
    2. A **Q lane loader** (``q_lane_loader`` seam) that reads this
       lane's Q slice as f32 (vectorised when ``EPT`` admits a
       power-of-two ``global_load_vN``).
    3. The **per-block Q+K scale** as an ``extra_score_transform``:
       ``q_scale`` is loaded once per CTA (q_token is wave-uniform);
       ``k_scale`` is reloaded per K-iter for per_block. Both fold into
       the score via :func:`apply_qk_scales` after the QK reduction and
       before the softmax update -- the same point sage applied them in
       the open-coded loop.

    The shared body keeps the per-slot scalar accumulator iter_args and
    scalar epilogue stores (the ``kv_lane_loader`` list path), so the
    int loaders fully own their packed-byte addressing and the dequant
    arithmetic is numerically identical to the previous open-coded body.
    """
    s = spec.common
    dtype = s.dtype
    H = s.shape.head_size
    block_size = WARP_SIZE
    kv_ty = _kv_pointee_for_quant_mode(spec.quant_mode, dtype)
    q_pointee = io_ir_type(dtype)

    if spec.quant_mode == "i4_fp8_bf16":
        # Two head-dim slots per lane (one byte = two nibbles); requires
        # head_size = 2 * warp_size i.e. >= 128.
        if H < 2 * WARP_SIZE:
            raise ValueError(
                f"i4 sage requires head_size >= {2 * WARP_SIZE} so each "
                f"lane owns one packed byte (two nibbles); got {H}"
            )
        if H % (2 * WARP_SIZE) != 0:
            raise ValueError(
                f"i4 sage requires head_size % {2 * WARP_SIZE} == 0; got {H}"
            )
        ept_pairs = H // (2 * WARP_SIZE)
        if ept_pairs != 1:
            raise ValueError(
                "i4 sage v1 supports head_size == 128 (one byte per lane); "
                f"got head_size={H} which would need {ept_pairs} bytes/lane"
            )

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(block_size)
    _declare_params(kb, spec)
    kb.decode_grid()
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation
    q_scale_ptr = kb.ptr("q_scale")
    k_scale_ptr = kb.ptr("k_scale")
    cb_k = kb.ptr("codebook_k") if spec.quant_mode in _CODEBOOK_QUANT_MODES else None
    cb_v = kb.ptr("codebook_v") if spec.quant_mode in _CODEBOOK_QUANT_MODES else None
    scale_log2 = kb.scalar("scale_log2")
    seqlen_k = kb.scalar("seqlen_k")
    q_token = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = b.const_i32(0)
    tid = b.thread_id_x()

    # Stage the constant dequant codebook(s) into LDS once per CTA so the
    # per-K-tile dequant lookups become ``ds_read_b32`` instead of a hot-path
    # ``global_load_f32`` (one VMEM load per dequanted element, every K-iter).
    # The table is loop-invariant; this removes a VMEM dependency from the
    # K-loop's critical path. fp16/fp8 modes have no codebook (cb_k is None).
    if cb_k is not None:
        cb_entries = (
            _CODEBOOK_I4_ENTRIES
            if spec.quant_mode == "i4_fp8_bf16"
            else _CODEBOOK_I8_ENTRIES
        )
        cb_k = _stage_codebook_to_lds(
            b, cb_k, n_entries=cb_entries, tid=tid, name_hint="sage_cb_k"
        )
        cb_v = _stage_codebook_to_lds(
            b, cb_v, n_entries=cb_entries, tid=tid, name_hint="sage_cb_v"
        )
        b.sync()

    # Per-Q-block scale (loaded once for the whole CTA -- q_token is
    # the same across the wave).
    q_block_idx = (
        _magic_div(b, q_token, spec.q_scale.scale_block)
        if spec.q_scale.layout == "per_block"
        else b.const_i32(0)
    )
    q_scale_v = load_q_scale_for_block(
        b,
        q_scale_ptr,
        spec=spec.q_scale,
        batch_idx=batch_idx,
        head_idx=head_idx,
        q_block_idx=q_block_idx,
    )

    is_i4 = spec.quant_mode == "i4_fp8_bf16"

    def _q_lane_loader(b: IRBuilder, q_row_base, lane_d_base, ept) -> list[Value]:
        """This lane's ``ept`` Q elements as f32 (vectorised when ept>=2)."""
        return _load_q_lane_f32(b, Q, q_row_base, lane_d_base, ept, dtype, q_pointee)

    def _kv_lane_loader(b: IRBuilder, k_idx, k_row_base, v_row_base, lane_d_base, ept):
        """This lane's K and V head-dim slices, dequantised to f32.

        i4: one packed byte per lane -> two nibbles -> two f32 (direct
        codebook). All other modes: vectorised byte slice + f32 dequant
        via :func:`_load_kv_lane_f32`.
        """
        if is_i4:
            byte_off = _magic_div(b, lane_d_base, 2)  # = tid
            packed_k = b.global_load(K, b.add(k_row_base, byte_off), I8)
            lo_k, hi_k = _codebook_i4_pair_to_f32(b, cb_k, packed_k)
            packed_v = b.global_load(V, b.add(v_row_base, byte_off), I8)
            v_lo, v_hi = _codebook_i4_pair_to_f32(b, cb_v, packed_v)
            return [lo_k, hi_k], [v_lo, v_hi]
        k_lane = _load_kv_lane_f32(
            b,
            KV=K,
            base=k_row_base,
            lane_d_base=lane_d_base,
            ept=ept,
            quant_mode=spec.quant_mode,
            cb_ptr=cb_k,
            kv_ty=kv_ty,
            dtype=dtype,
        )
        v_lane = _load_kv_lane_f32(
            b,
            KV=V,
            base=v_row_base,
            lane_d_base=lane_d_base,
            ept=ept,
            quant_mode=spec.quant_mode,
            cb_ptr=cb_v,
            kv_ty=kv_ty,
            dtype=dtype,
        )
        return k_lane, v_lane

    def _qk_scale_transform(b: IRBuilder, score_log2: Value, k_idx: Value) -> Value:
        """Apply ``q_scale * k_scale`` to the post-reduction score.

        Mirrors the open-coded ``apply_qk_scales`` call: ``q_scale`` is
        the wave-uniform per-Q-block scale; ``k_scale`` is reloaded per
        K-iter for per_block layout.
        """
        k_block_idx = (
            _magic_div(b, k_idx, spec.k_scale.scale_block)
            if spec.k_scale.layout == "per_block"
            else b.const_i32(0)
        )
        k_scale_v = load_k_scale_for_block(
            b,
            k_scale_ptr,
            spec=spec.k_scale,
            batch_idx=batch_idx,
            head_idx=kv_head_idx,
            k_block_idx=k_block_idx,
        )
        return apply_qk_scales(b, score_log2, q_scale=q_scale_v, k_scale=k_scale_v)

    causal_ctx = q_token if s.mask_mode in ("causal", "sliding_window") else None

    fmha_warp_fwd_inner_body(
        b,
        Q=Q,
        K=K,
        V=V,
        O=O,
        head_size=H,
        seqlen_k=seqlen_k,
        q_token=q_token,
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
        dtype=dtype,
        mask_mode=s.mask_mode,
        sliding_window=s.sliding_window,
        causal_ctx_len=causal_ctx,
        extra_score_transform=_qk_scale_transform,
        kv_lane_loader=_kv_lane_loader,
        q_lane_loader=_q_lane_loader,
    )

    b.ret()
    return kb.kernel


# ---------------------------------------------------------------------
# Public builders + grid + signature
# ---------------------------------------------------------------------


def build_sage_attention(spec: SageAttentionSpec, arch: str = "gfx950") -> KernelDef:
    """Sage attention forward; picks the MFMA fast path when the spec
    aligns and falls back to the warp body otherwise.

    * **MFMA fast path** (fp16_bf16 / fp8_bf16 with
      :func:`_mfma_dimensions_ok`): one wave64 warp per ``BLOCK_M``
      Q-tile, vectorised K/V loads, hardware MFMA chain. Q+K scales
      fold into ``scale_log2`` / ``extra_score_transform``.
    * **Warp body fallback** (all other cases including i8/i4 and
      small / unaligned fp16/fp8 specs): one wave64 warp per
      ``(q_token, head)`` row with vectorised lane loads and direct
      f32-codebook dequant for the int variants.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid sage_attention spec: {why}")
    if _uses_mfma_path(spec):
        return _build_sage_mfma(spec, arch)
    return _build_sage_warp(spec)


def sage_attention_grid(spec: SageAttentionSpec) -> Tuple[int, int, int]:
    """Launch grid for :func:`build_sage_attention`.

    * MFMA path: ``(seqlen_q / BLOCK_M, num_query_heads, 1)`` -- one
      CTA per ``BLOCK_M = 16``-row Q tile.
    * Warp body path: ``(seqlen_q, num_query_heads, 1)`` -- one CTA
      per ``(q_token, head)`` row.
    """
    if _uses_mfma_path(spec):
        return (
            spec.seqlen_q // MFMA_ATTN_BLOCK_M,
            spec.common.shape.num_query_heads,
            1,
        )
    return (spec.seqlen_q, spec.common.shape.num_query_heads, 1)


def sage_attention_signature(spec: SageAttentionSpec):
    kb = FmhaKernelBuilder("rocke_sage_attention_sig_probe", spec.common)
    _declare_params(kb, spec)
    return kb.signature()


_BF8E5M2 = BF8E5M2  # noqa: F841 - re-export anchor
_F32 = F32  # noqa: F841 - re-export anchor
