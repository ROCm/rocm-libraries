# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Free-stride SDPA forward layout shell (hipDNN requirement 2 / SH-01).

The net-new MFMA-body forward attention kernel that reads Q/K/V from
arbitrary memory layouts instead of assuming one fixed contiguous order.
It is the generic host ("shell") that the other hipDNN attention features
(scale, per-sequence lengths / padding, mask variants, bias) plug into on
gfx942/gfx950.

Target layouts (the requirement)::

 4D [B, H, S, D], both BSHD (tokens-major) and BHSD (heads-major)
 general runtime strides on the B / H / S axes
 packed QKV read as strided views into one buffer

Design: this file is an ADDRESSING shell only. The attention math (QK^T,
softmax, PV, epilogue) is reused verbatim from
``rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`` -- the shell
just computes the per-tile base offsets and feeds them in, exactly like
``fmha_varlen.py`` does for the packed/cu_seqlens case.

Addressing is layout-general. The body reads Q/K/V/O through runtime
``(head, token)`` strides, and this shell rebases each tensor's BASE POINTER to
the CTA's batch slice (``batch_idx * stride_*_batch``) before calling the body.
Batch, head and token offsets are therefore all runtime, so BSHD, BHSD, general
(non-contiguous) B/H/S strides and packed QKV are all expressed by the stride
params and base pointers the launcher passes, with nothing baked. The head dim
is assumed innermost with unit stride (the vectorized head-dim loads). The
batch rebase is safe because the body loads through raw pointers and never
sizes a buffer span from the original base.

Python-only for now and OUTSIDE the byte-identity gate (a C++ port of the
addressing is deferred): the shared inner body is reused UNCHANGED, so no
byte-identity obligation is added to it. On-GPU numeric verification across the
layouts is still pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import I64
from rocke.helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from rocke.helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from .fmha_arch import validate_fmha_mfma_atom


__all__ = [
    "FmhaSdpaFwdSpec",
    "build_fmha_sdpa_fwd",
    "fmha_sdpa_fwd_grid",
    "fmha_sdpa_fwd_signature",
    "is_valid_spec",
]

# Named contiguous layouts the launcher can derive strides for (see
# _sdpa_fwd_strides.contiguous_strides). General and packed strides are passed
# by the caller; the kernel itself is layout-agnostic (strides are runtime).
_SUPPORTED_LAYOUTS = ("bhsd", "bshd")

# Element size (bytes) for the per-batch base-pointer rebase.
_ELEM_BYTES = {"f16": 2, "fp16": 2, "bf16": 2}


@dataclass(frozen=True)
class FmhaSdpaFwdSpec:
    """One concrete free-stride SDPA-fwd configuration.

    Problem shape (required): ``common`` (carries head/dtype/mask knobs),
    ``batch``, ``seqlen_q``, ``seqlen_k``. Every non-shape field is
    defaulted so existing descriptors/packs stay valid when fields are
    added (platform AGENTS.md "a new spec field is defaulted").

    ``layout`` is the physical order of the 4D [B, H, S, D] tensors;
    ``packed_qkv`` marks Q/K/V as strided views into one buffer. Both
    default to the contiguous-BHSD baseline the scaffold implements.
    """

    common: FmhaCommonSpec
    batch: int
    seqlen_q: int
    seqlen_k: int
    layout: str = "bhsd"
    packed_qkv: bool = False
    name: str = "rocke_fmha_sdpa_fwd"

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
            f"B{self.batch}",
            self.layout,
            "packed" if self.packed_qkv else "unpacked",
            self.common.mask_mode,
        )


def is_valid_spec(spec: FmhaSdpaFwdSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # MFMA-tiled body uses the narrow f16/bf16 16x16x16 atom; reject arches
    # whose catalog lacks it before lowering reaches comgr.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
    if not ok:
        return False, why
    if spec.batch <= 0:
        return False, f"batch must be > 0 (got {spec.batch})"
    if spec.seqlen_q <= 0 or spec.seqlen_k <= 0:
        return False, (
            f"seqlen_q / seqlen_k must be > 0 (got {spec.seqlen_q}, {spec.seqlen_k})"
        )
    # Same tile-multiple requirement as the varlen body: a CTA owns one
    # BLOCK_M Q-row tile and the K-loop covers whole BLOCK_K tiles. Chris's
    # arbitrary-length tail handling (ticket 8) lifts this later.
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"MFMA sdpa needs seqlen_q ({spec.seqlen_q}) to be a multiple of "
            f"BLOCK_M ({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"MFMA sdpa needs seqlen_k ({spec.seqlen_k}) to be a multiple of "
            f"BLOCK_K ({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.shape.head_size % 16 != 0:
        return False, (
            f"MFMA sdpa needs head_size % 16 == 0 (got {spec.common.shape.head_size})"
        )
    # The kernel is layout-agnostic (runtime strides + a per-batch pointer
    # rebase), so packed QKV and either contiguous order build the same kernel;
    # only the launcher's stride derivation is named-layout specific.
    if spec.layout not in _SUPPORTED_LAYOUTS:
        return False, (
            f"layout {spec.layout!r} is not a named contiguous order; "
            f"supported: {_SUPPORTED_LAYOUTS} (general strides are caller-provided)"
        )
    if spec.common.dtype not in _ELEM_BYTES:
        return False, (
            f"dtype {spec.common.dtype!r} not supported; "
            f"one of {sorted(_ELEM_BYTES)}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """Declare the SDPA-fwd kernel ABI (shared between build + signature).

    Single-sources the param list so the build path and the signature
    probe can't drift. ``add_strides`` gives the per-tensor token/head
    strides; the batch strides are the net-new free-stride params this
    shell adds on top (K/V are read by the inner body's offset hooks;
    Q/O become load-bearing in Phase 3, see module docstring).
    """
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_strides("q", "k", "v", "o")
    kb.add_scalar("stride_q_batch", "i32")
    kb.add_scalar("stride_k_batch", "i32")
    kb.add_scalar("stride_v_batch", "i32")
    kb.add_scalar("stride_o_batch", "i32")


def build_fmha_sdpa_fwd(spec: FmhaSdpaFwdSpec, arch: str = "gfx950"):
    """Free-stride SDPA forward kernel (MFMA-tiled body).

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, batch)``. Each CTA handles
    one ``BLOCK_M`` Q-row tile of one ``(head, batch)`` slice.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_sdpa_fwd spec: {why}")

    s = spec.common

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    # MFMA: one wave64 warp per CTA.
    kb.block_size(64)
    _declare_params(kb)
    # has_batch_axis=True binds batch_idx from block_id_z; GQA/MQA div
    # derives kv_head_idx from the shape.
    kb.decode_grid(has_batch_axis=True)

    b = kb.builder
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = kb.batch_idx
    scale_log2 = kb.scalar("scale_log2")

    # Within-sequence Q position of this CTA's tile. It addresses Q rows AND is
    # the causal-mask position; batch is carried by the rebased pointer below,
    # so this stays the within-sequence row. Wave-uniform (block_id_x only).
    q_pos_base = b.to_sgpr_u32(b.mul(kb.q_token, b.const_i32(MFMA_ATTN_BLOCK_M)))

    # Rebase each tensor's base pointer to this CTA's batch slice:
    # ptr + batch_idx * stride_*_batch (elements) * elem_bytes. The body then
    # addresses head and token relative to the rebased pointer, so batch, head
    # and token offsets are all runtime and any layout works. Safe because the
    # body loads through raw pointers (no span is sized from the original base).
    elem = _ELEM_BYTES[s.dtype]

    def _batch_base(name):
        off = b.mul(
            b.zext(batch_idx, I64), b.zext(kb.scalar(f"stride_{name}_batch"), I64)
        )
        return b.global_ptr_add(kb.tensor(name.upper()), b.mul(off, b.const_i64(elem)))

    q_base, k_base, v_base, o_base = (_batch_base(n) for n in ("q", "k", "v", "o"))

    # Dense: every sequence has the same length, known at build time.
    seqlen_k = b.const_i32(spec.seqlen_k)

    causal_ctx = b.const_i32(0) if s.mask_mode in ("causal", "sliding_window") else None

    mfma_attention_fwd_inner_body(
        b,
        Q=q_base,
        K=k_base,
        V=v_base,
        O=o_base,
        head_size=s.head_size,
        seqlen_k=seqlen_k,
        q_tile_base=q_pos_base,
        q_pos_base=q_pos_base,
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
        arch=arch,
    )
    b.ret()
    return kb.kernel


def fmha_sdpa_fwd_grid(spec: FmhaSdpaFwdSpec) -> Tuple[int, int, int]:
    """MFMA SDPA grid: one CTA per Q-row tile (BLOCK_M rows) per head per batch."""
    return (
        spec.seqlen_q // MFMA_ATTN_BLOCK_M,
        spec.common.shape.num_query_heads,
        spec.batch,
    )


def fmha_sdpa_fwd_signature(spec: FmhaSdpaFwdSpec):
    """Construct the same signature shape FmhaKernelBuilder generates."""
    # Throw-away builder to single-source the signature from _declare_params,
    # so the spec and the build function can't drift.
    kb = FmhaKernelBuilder("rocke_fmha_sdpa_fwd_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()
