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

SCAFFOLD STATUS (Phase 1-2): dense contiguous BHSD only. Batch is threaded
through the inner body's existing K/V offset hooks (general-stride correct)
and, for Q/O, through a contiguous fold into ``q_tile_base`` (see the
PHASE-3 notes below). The following are declared in the spec but NOT yet
implemented and are rejected by ``is_valid_spec``:

* BSHD order and general (non-contiguous) B/H/S strides,
* packed-QKV strided views.

PHASE-3 (general strides): the inner body has K/V offset hooks
(``k_token_offset_elems`` / ``v_token_offset_elems``) but no Q/O offset
hook, so a general-stride Q/O batch offset needs the shared body to gain a
``q_base_offset_elems`` / ``o_base_offset_elems`` param pair (mirrored in
the C++ engine, re-blessed golden). Until then Q/O batch addressing relies
on the contiguous fold, which assumes ``stride_q_batch`` is a whole number
of ``stride_q_token`` units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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

# Layouts the shell is designed to serve. Only the contiguous BHSD baseline
# is implemented in the Phase 1-2 scaffold; the rest gate in is_valid_spec.
_SUPPORTED_LAYOUTS = ("bhsd",)
_PLANNED_LAYOUTS = ("bshd",)


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
    # Scaffold gates: only contiguous BHSD, unpacked, is implemented.
    if spec.layout not in _SUPPORTED_LAYOUTS:
        planned = spec.layout in _PLANNED_LAYOUTS
        return False, (
            f"layout {spec.layout!r} not yet implemented"
            + (" (planned, Phase 3)" if planned else "")
            + f"; supported: {_SUPPORTED_LAYOUTS}"
        )
    if spec.packed_qkv:
        return False, "packed_qkv not yet implemented (Phase 3)"
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

    # Within-sequence Q position of this CTA's tile (used for the causal
    # mask check). Wave-uniform: depends only on block_id_x.
    q_pos_base = b.to_sgpr_u32(b.mul(kb.q_token, b.const_i32(MFMA_ATTN_BLOCK_M)))

    # Q/O batch addressing (CONTIGUOUS fold, Phase 3 extension point):
    # per-batch Q holds num_query_heads * seqlen_q token-stride units, so
    # fold the batch offset into the addressing base. This is exact only
    # for contiguous BHSD (stride_q_batch == num_query_heads*seqlen_q *
    # stride_q_token). General strides need a q/o offset hook on the body.
    q_batch_tokens = b.mul(
        batch_idx, b.const_i32(s.shape.num_query_heads * spec.seqlen_q)
    )
    q_tile_base = b.to_sgpr_u32(b.add(q_pos_base, q_batch_tokens))

    # K/V batch addressing via the inner body's existing element-offset
    # hooks. batch_idx * stride_*_batch is an absolute element offset, so
    # this is general-stride correct (unlike the Q/O fold above).
    k_batch_offset = b.to_sgpr_u32(b.mul(batch_idx, kb.scalar("stride_k_batch")))
    v_batch_offset = b.to_sgpr_u32(b.mul(batch_idx, kb.scalar("stride_v_batch")))

    # Dense: every sequence has the same length, known at build time.
    seqlen_k = b.const_i32(spec.seqlen_k)

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
        k_token_offset_elems=k_batch_offset,
        v_token_offset_elems=v_batch_offset,
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
