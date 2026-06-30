# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FP8 FMHA forward (CK Tile ``01_fmha`` fp8 parity).

K and V are stored in fp8e4m3 (or bf8e5m2) and dequantised on load
with per-tensor scales; the warp-distributed body promotes them to
f32 for the QK / PV math and emits the cshuffle to the activation
dtype at the end. The MFMA-tiled variant (follow-on) consumes the
same spec + uses ``mfma_f32_16x16x32_fp8`` to do the dequant inside
the atom rather than at the f32-promotion step.

The output ``O`` is stored back in the **activation dtype** (f16 /
bf16), not in fp8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from ...core.ir import KernelDef
from ...helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ...helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec


__all__ = [
    "FmhaFwdFp8Spec",
    "build_fmha_fwd_fp8",
    "fmha_fwd_fp8_grid",
    "fmha_fwd_fp8_signature",
    "is_valid_spec",
]


KvFp8DType = Literal["fp8e4m3", "bf8e5m2"]


@dataclass(frozen=True)
class FmhaFwdFp8Spec:
    common: FmhaCommonSpec
    kv_dtype: KvFp8DType = "fp8e4m3"
    seqlen_q: int = 1
    seqlen_k: int = 0
    # G3: by default the K/V bytes are interpreted as OCP fp8
    # (e4m3fn / e5m2), which matches gfx950 / gfx11 hardware decode but
    # NOT the gfx942 (gfx9_mfma) native e4m3fnuz / e5m2fnuz decode. Set
    # this True only when the host has quantised K/V to the fnuz format
    # AND the reference is fnuz-correct, which acknowledges running on a
    # gfx9_mfma target. This flag does not change the emitted IR (the
    # ``cvt.f32.fp8`` intrinsic is the same); it only gates the
    # arch-validity check so OCP bytes are not silently mis-decoded.
    fp8_fnuz: bool = False
    # gfx950 occupancy hint (``"amdgpu-waves-per-eu"``). The fp8 MFMA
    # body is VGPR-bound: with the LLVM default the kernel allocates
    # ~138 arch-VGPR + ~42 AGPR (the AGPR copies are accumulator
    # spill into the matrix-ACC bank), capping occupancy at 12
    # waves/CU. Setting waves-per-eu=4 forces the register allocator
    # to a 128-VGPR / 0-AGPR budget (no AGPR<->VGPR copies on the
    # accumulator-touching path -- the ``use_agpr_alloc_zero`` effect
    # from the runbook §12.1.G) and lifts occupancy to 16 waves/CU
    # with no spill. Measured ~12% faster on a (HD=128, HQ=HK=8,
    # Q=512, K=2048) gfx950 shape (178 -> 157 us, warmup15/iters50,
    # median of 5). ``None`` keeps the LLVM heuristic.
    waves_per_eu: Optional[int] = 4
    name: str = "rocke_fmha_fwd_fp8"

    def kernel_name(self) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            self.kv_dtype,
            f"Q{self.seqlen_q}",
            self.common.mask_mode,
        )


# CDNA generations whose ``cvt.f32.fp8`` / ``cvt.f32.bf8`` intrinsics
# decode the fp8 byte as the **fnuz** variant (e4m3fnuz / e5m2fnuz, the
# MI200/MI300 native format) rather than the OCP ``e4m3fn`` / ``e5m2``
# format that gfx950 (MI350) and gfx11 use. ``ArchTarget.target_family``
# is the SSOT discriminator: gfx942/gfx940/gfx941 all share the
# ``gfx9_mfma`` family, gfx950 has its own ``gfx950`` family.
_FNUZ_FP8_TARGET_FAMILIES = frozenset({"gfx9_mfma"})


def is_valid_spec(spec: FmhaFwdFp8Spec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one FP8 FMHA fwd config on ``arch``.

    The default build path (``native_fp8_path=False`` in the shared
    helper) dequantises fp8e4m3 / bf8e5m2 K/V on load via the packed
    ``cvt_pk_f32_fp8x4`` / ``cvt_pk_f32_bf8x4`` instructions and then
    runs the f16 ``mfma_f32_16x16x16_f16`` atom for QK / PV. The legacy
    f16 atom exists on gfx942 and gfx950, but the fp8 *decode* differs:

    **G3 -- fnuz vs OCP fp8.** The ``llvm.amdgcn.cvt.f32.fp8`` intrinsic
    this dequant path emits is identical IR on gfx942 and gfx950, but the
    hardware interprets the same byte differently. gfx942 (and the rest
    of the ``gfx9_mfma`` generation) decodes it as **e4m3fnuz** (no
    inf/nan, bias 8); gfx950 / gfx11 decode it as OCP **e4m3fn** (bias 7,
    with inf/nan). A host buffer quantised to OCP fp8 therefore produces
    silently-wrong attention output on gfx942. Rather than emit a kernel
    that mis-decodes its K/V, this predicate rejects the OCP-fp8 path on
    ``gfx9_mfma`` targets with a structured reason. Callers that have
    genuinely fnuz-quantised K/V (and a fnuz reference) can opt in via
    ``spec.fp8_fnuz=True``, which acknowledges the gfx942 native format.

    The architecture facts (legal f16 MFMA atom for the dequant-to-f16
    chain, per-WG LDS capacity, fp8 decode family) are sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch -- or one
    that ever drops the legacy f16 atom -- is rejected with a structured
    reason rather than crashing comgr (or returning wrong numbers) at
    lower time. gfx950 behavior / atom selection is unchanged.

    Note: the native fp8 MFMA atom (``mfma_f32_16x16x32_fp8``) lift is a
    follow-on; once it lands as the build default this predicate must
    gate it via ``target.mma.has_shape(..., 16, 16, 32)`` for the fp8
    combo, which is the gfx950-only path.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.kv_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            f"kv_dtype must be 'fp8e4m3' or 'bf8e5m2', got {spec.kv_dtype!r}"
        )

    # G3: OCP-fp8 K/V on a fnuz-decoding target is silently wrong. Reject
    # unless the caller declares fnuz-quantised inputs (``fp8_fnuz=True``).
    if target.target_family in _FNUZ_FP8_TARGET_FAMILIES and not spec.fp8_fnuz:
        return False, (
            f"fp8 K/V on {arch} (target_family={target.target_family!r}) "
            f"decodes via the native e4m3fnuz/e5m2fnuz format, not OCP "
            f"e4m3fn/e5m2; the default {spec.kv_dtype} path assumes OCP "
            f"bytes and would silently mis-decode K/V. Quantise K/V to "
            f"fnuz and set FmhaFwdFp8Spec(fp8_fnuz=True), or run the "
            f"OCP-fp8 attention on gfx950 / gfx11."
        )
    if spec.seqlen_q <= 0:
        return False, f"seqlen_q must be > 0 (got {spec.seqlen_q})"
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA fp8 attention needs seqlen_q ({spec.seqlen_q}) to "
            f"be a multiple of BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA fp8 attention needs head_size % 16 == 0 "
            f"(got {spec.common.shape.head_size})"
        )

    # The dequant-on-load path emits the f16 16x16x16 atom; require it
    # on the target catalog (gfx942 / gfx950 both carry it).
    if not target.supports_dtype_combo("f16", "f16", "fp32"):
        return False, f"unsupported f16 MFMA dtype combo on {arch}"
    if not target.mma.has_shape(
        a_dtype="f16",
        b_dtype="f16",
        c_dtype="fp32",
        m=MFMA_ATTN_BLOCK_M,
        n=MFMA_ATTN_BLOCK_M,
        k=MFMA_ATTN_BLOCK_M,
    ):
        return False, (
            f"unsupported f16 warp_tile "
            f"({MFMA_ATTN_BLOCK_M},{MFMA_ATTN_BLOCK_M},{MFMA_ATTN_BLOCK_M}) "
            f"on {arch}"
        )

    # LDS budget: one BLOCK_M x BLOCK_K f16 P-staging buffer.
    bytes_lds = MFMA_ATTN_BLOCK_M * MFMA_ATTN_BLOCK_M * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder, spec: FmhaFwdFp8Spec) -> None:
    """Declare the FP8 FMHA kernel ABI (shared between build + sig)."""
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", dtype=spec.kv_dtype, readonly=True, align=8)
    kb.add_tensor("V", dtype=spec.kv_dtype, readonly=True, align=8)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("k_scale", "f32")
    kb.add_scalar("v_scale", "f32")
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def build_fmha_fwd_fp8(spec: FmhaFwdFp8Spec, arch: str = "gfx950") -> KernelDef:
    """FP8 FMHA forward kernel (MFMA-tiled body, fp8 K/V dequant on load).

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, 1)``. Each CTA handles
    one ``BLOCK_M = 16`` Q-row tile. K/V are loaded as fp8 / bf8 bytes
    and dequantised to f16 on the load path before the f16 MFMA chain.

    The kernel currently passes per-tensor ``k_scale`` and ``v_scale``
    parameters but uses them in the inline dequant path. The native
    fp8 MFMA atom (``mfma_f32_16x16x32_fp8``) lift will subsume the
    explicit dequant once the atom-input lane layout is wired through
    the shared helper.

    ``arch`` selects the MMA catalog: the dequant-on-load body emits the
    legacy ``mfma_f32_16x16x16_f16`` atom for the QK / PV chain, which
    exists on both gfx942 and gfx950, so the kernel is arch-polymorphic.
    ``arch`` is threaded through :func:`is_valid_spec` (pre-build catalog
    check) and into the shared helper so the finally-selected atom is
    validated against the target before it reaches comgr. On the default
    ``arch="gfx950"`` the emitted IR is unchanged.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_fwd_fp8 spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)
    _declare_params(kb, spec)
    kb.decode_grid()
    b = kb.builder
    # Occupancy hint: forces a lower VGPR budget + zero AGPR (see the
    # ``waves_per_eu`` field doc). Emitted as ``"amdgpu-waves-per-eu"``.
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    # Per-tensor K and V dequant scales. ``k_scale`` is folded into
    # ``scale_log2`` (the QK result lives in log2 score space, so a
    # constant K-side scale becomes a constant pre-softmax multiplier).
    # ``v_scale`` is passed to the helper and applied at the epilogue.
    k_scale = kb.scalar("k_scale")
    v_scale = kb.scalar("v_scale")
    scale_log2_raw = kb.scalar("scale_log2")
    scale_log2 = b.fmul(scale_log2_raw, k_scale)
    seqlen_k = kb.scalar("seqlen_k")
    q_tile_idx = kb.q_token
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    q_tile_base = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))

    causal_ctx = b.const_i32(0) if s.mask_mode in ("causal", "sliding_window") else None

    mfma_attention_fwd_inner_body(
        b,
        Q=Q,
        K=K,
        V=V,
        O=O,
        head_size=s.head_size,
        seqlen_k=seqlen_k,
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
        # fp8 / bf8 K/V dequant on load.
        kv_dtype=spec.kv_dtype,
        v_scale=v_scale,
        arch=arch,
    )
    b.ret()
    return kb.kernel


def fmha_fwd_fp8_grid(spec: FmhaFwdFp8Spec) -> Tuple[int, int, int]:
    """MFMA fp8 grid: one CTA per Q-row tile (16 rows) per head."""
    return (
        spec.seqlen_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        1,
    )


def fmha_fwd_fp8_signature(spec: FmhaFwdFp8Spec):
    kb = FmhaKernelBuilder("rocke_fmha_fwd_fp8_sig_probe", spec.common)
    _declare_params(kb, spec)
    return kb.signature()
