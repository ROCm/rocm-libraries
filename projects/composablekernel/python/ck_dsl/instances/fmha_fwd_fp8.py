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
from typing import Literal, Tuple

from ..core.ir import KernelDef
from ..helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ..helpers.spec import kernel_name_join
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
    name: str = "ck_dsl_fmha_fwd_fp8"

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


def is_valid_spec(spec: FmhaFwdFp8Spec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    if spec.kv_dtype not in ("fp8e4m3", "bf8e5m2"):
        return False, (
            f"kv_dtype must be 'fp8e4m3' or 'bf8e5m2', got {spec.kv_dtype!r}"
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


def build_fmha_fwd_fp8(spec: FmhaFwdFp8Spec) -> KernelDef:
    """FP8 FMHA forward kernel (MFMA-tiled body, fp8 K/V dequant on load).

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, 1)``. Each CTA handles
    one ``BLOCK_M = 16`` Q-row tile. K/V are loaded as fp8 / bf8 bytes
    and dequantised to f16 on the load path before the f16 MFMA chain.

    The kernel currently passes per-tensor ``k_scale`` and ``v_scale``
    parameters but uses them in the inline dequant path. The native
    fp8 MFMA atom (``mfma_f32_16x16x32_fp8``) lift will subsume the
    explicit dequant once the atom-input lane layout is wired through
    the shared helper.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid fmha_fwd_fp8 spec: {why}")
    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(64)
    _declare_params(kb, spec)
    kb.decode_grid()
    b = kb.builder

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
    kb = FmhaKernelBuilder("ck_dsl_fmha_fwd_fp8_sig_probe", spec.common)
    _declare_params(kb, spec)
    return kb.signature()
