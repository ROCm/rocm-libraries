# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tiled FMHA forward (production attention kernel; unified MMA body).

This is the **first attention kernel** consuming the production
``16x16x16`` f16 contract atom directly. One CTA = one wave handles a
16-row Q tile across the full K span via the QK + softmax + PV matmul
chain. The single body is arch-polymorphic: the ``16x16x16`` atom
resolves to ``mfma_f32_16x16x16_f16`` on CDNA wave64 (gfx942/gfx950)
and ``wmma_f32_16x16x16_f16`` on RDNA wave32 (gfx1151).

The module / symbol keep the historical ``_mfma`` suffix for caller
stability (it predates the MFMA/WMMA unification), but
:func:`build_fmha_fwd_mfma` emits MFMA *or* WMMA depending on ``arch``;
it is not CDNA-only. The helper :func:`mfma_attention_fwd_inner_body`
factors the QK / softmax / PV pipeline and dispatches on the target
wave size; this module's :func:`build_fmha_fwd_mfma` is the thin
spec→kernel wrapper.

Grid layout: ``(seqlen_q // BLOCK_M, num_query_heads, batch)``. The
spec's ``seqlen_q`` must be a multiple of ``BLOCK_M = 16``;
``head_size`` must be a multiple of ``16`` (all standard FMHA head
sizes -- 64 / 128 / 256 -- qualify); ``seqlen_k`` must be a multiple
of ``BLOCK_K = 16``.

CK Tile parity context: the standard CK Tile ``01_fmha`` fwd kernel
uses the same atom shape + LDS-staging skeleton; v1 here ships the
single-warp variant (no multi-warp BLOCK_M, no async DMA), which is
already ~16x denser in FLOPs than the warp-scalar body and forms
the spec surface the multi-warp + cshuffle hoist consumes verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.mfma_attention import (
    MFMA_ATTN_BLOCK_K,
    MFMA_ATTN_BLOCK_M,
    mfma_attention_fwd_inner_body,
)
from ...helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec


__all__ = [
    "FmhaMfmaSpec",
    "build_fmha_fwd_mfma",
    "fmha_fwd_mfma_grid",
    "fmha_fwd_mfma_signature",
    "is_valid_spec",
]


@dataclass(frozen=True)
class FmhaMfmaSpec:
    """One MFMA-tiled FMHA forward configuration.

    ``seqlen_q`` and ``seqlen_k`` are compile-time so the spec can
    pre-size the launch grid; runtime variability (e.g. varlen) lifts
    via a derived spec that overrides the grid computation.
    """

    common: FmhaCommonSpec
    seqlen_q: int
    seqlen_k: int
    name: str = "rocke_fmha_fwd_mfma"

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
            self.common.mask_mode,
        )


def _mma_family(arch: str) -> str:
    """MMA atom family for ``arch``: ``"wmma"`` on the RDNA wave32 targets
    (gfx11xx), ``"mma"`` (MFMA) on CDNA.

    The single body in :func:`build_fmha_fwd_mfma` selects the QK/PV atom from
    the target catalog using this family, so the same spec resolves to an MFMA
    atom on gfx942/gfx950 and a WMMA atom on gfx1151 -- the attention analogue
    of the unified GEMM's family selection.
    """
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def is_valid_spec(spec: FmhaMfmaSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one tiled FMHA fwd config on ``arch``.

    The body runs the canonical small-tile ``16x16x16`` f16 atom
    (BLOCK_M = BLOCK_K = 16). On CDNA (gfx942 / gfx950) that is the
    ``mfma_f32_16x16x16_f16`` MFMA atom; on the RDNA wave32 targets
    (gfx1151) it is the ``wmma_f32_16x16x16_f16`` WMMA atom. The same
    spec is thus arch-polymorphic across both families -- the matmul,
    online-softmax reduction, and P fragment re-layout are driven off the
    contract's per-arch ``MmaOp`` (see
    :func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`).
    The architecture facts (legal atom for the f16 ⊗ f16 → f32 combo,
    per-WG LDS capacity, wave size) are sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch -- or an arch
    that ever drops the 16x16x16 atom -- is rejected with a structured
    reason rather than crashing comgr at lower time. gfx950 / gfx942
    behaviour and atom selection are unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    family = _mma_family(arch)

    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    s = spec.common.shape
    if s.head_size % 16 != 0:
        return False, (f"tiled FMHA needs head_size % 16 == 0 (got {s.head_size})")
    if spec.seqlen_q % MFMA_ATTN_BLOCK_M != 0:
        return False, (
            f"seqlen_q ({spec.seqlen_q}) must be a multiple of BLOCK_M "
            f"({MFMA_ATTN_BLOCK_M})"
        )
    if spec.seqlen_k % MFMA_ATTN_BLOCK_K != 0:
        return False, (
            f"seqlen_k ({spec.seqlen_k}) must be a multiple of BLOCK_K "
            f"({MFMA_ATTN_BLOCK_K})"
        )
    if spec.common.dtype != "f16":
        return False, (
            f"tiled FMHA v1 ships f16 only; bf16 lands once the bf16 "
            f"atom factory is exposed (got {spec.common.dtype})"
        )

    # The QK / PV chain is the f16 16x16x16 atom; require it on the
    # target catalog (MFMA on CDNA, WMMA on RDNA) so an arch missing the
    # atom is rejected cleanly.
    if not target.supports_dtype_combo("f16", "f16", "fp32", family=family):
        return False, f"unsupported f16 {family} dtype combo on {arch}"
    if not target.mma.has_shape(
        family=family,
        a_dtype="f16",
        b_dtype="f16",
        c_dtype="fp32",
        m=MFMA_ATTN_BLOCK_M,
        n=MFMA_ATTN_BLOCK_M,
        k=MFMA_ATTN_BLOCK_K,
    ):
        return False, (
            f"unsupported f16 {family} warp_tile "
            f"({MFMA_ATTN_BLOCK_M},{MFMA_ATTN_BLOCK_M},{MFMA_ATTN_BLOCK_K}) "
            f"on {arch}"
        )

    # LDS budget: one BLOCK_M x BLOCK_K f16 P-staging buffer.
    bytes_lds = MFMA_ATTN_BLOCK_M * MFMA_ATTN_BLOCK_K * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(kb: FmhaKernelBuilder) -> None:
    """The MFMA FMHA kernel ABI (shared between build + sig)."""
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("seqlen_q", "i32")
    kb.add_scalar("seqlen_k", "i32")
    kb.add_strides("q", "k", "v", "o")


def build_fmha_fwd_mfma(spec: FmhaMfmaSpec, arch: str = "gfx950") -> KernelDef:
    """Tiled FMHA forward kernel (one unified body, MFMA on CDNA / WMMA on RDNA).

    Grid: ``(seqlen_q / BLOCK_M, num_query_heads, batch)``. Each CTA owns one
    ``(q_tile, head, batch)`` triple and one wave (wave64 on CDNA, wave32 on
    RDNA).

    ``arch`` selects the QK/PV matmul atom from the contract catalog: the f16
    ``16x16x16`` atom resolves to ``mfma_f32_16x16x16_f16`` on gfx942/gfx950 and
    ``wmma_f32_16x16x16_f16`` on gfx1151. The single common inner body
    (:func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`)
    dispatches on the target wave size and drives the matmul, online-softmax
    reduction, and P fragment re-layout off that op's contract layout maps. On
    the default ``arch="gfx950"`` the emitted IR is byte-for-byte unchanged from
    the pre-unification MFMA kernel; ``arch="gfx1151"`` produces the WMMA
    variant. An arch-illegal config is rejected by :func:`is_valid_spec` before
    any IR is emitted.

    Per-batch causal masking (G4): the masked query position is the
    *within-batch* row (``q_tile_local``), passed to the inner body as
    ``q_pos_base``, while the global-batched row (``q_tile_local +
    batch_idx * seqlen_q``) addresses Q / O in memory via ``q_tile_base``.
    This makes the causal / sliding-window mask correct for ``batch > 1``
    (each batch's row 0 is causal position 0, not ``batch_idx * seqlen_q``),
    matching the gfx1151 adapter
    :func:`rocke.instances.gfx1151.wmma_fmha_fwd.build_wmma_fmha_fwd`.

    ``q_pos_base`` is only threaded for the masked modes
    (``causal`` / ``sliding_window``); for ``mask_mode == "none"`` the
    masked position is unused, so it is left unset and the emitted CDNA IR
    is byte-for-byte unchanged from the pre-G4 kernel (the default
    ``q_pos_base = q_tile_base`` path). The masked-mode IR does change
    (the within-batch position replaces the global-batched one in the mask
    predicate) -- that is the bug fix; the masked path was numerically
    wrong for ``batch > 1`` before.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid fmha_mfma spec: {why}")
    s = spec.common

    # One wave per CTA: wave64 on CDNA (MFMA), wave32 on RDNA (WMMA). The block
    # size is sourced from the target so the launch geometry matches the atom
    # family the body selects.
    from ...core.arch import ArchTarget

    wave_size = ArchTarget.from_gfx(arch).wave_size

    kb = FmhaKernelBuilder(spec.kernel_name(), s)
    kb.block_size(wave_size)  # one wave per CTA (64 on CDNA, 32 on RDNA)
    _declare_params(kb)
    kb.decode_grid(has_batch_axis=True)
    b = kb.builder

    seqlen_q = kb.scalar("seqlen_q")
    seqlen_k = kb.scalar("seqlen_k")
    head_idx = kb.head_idx
    kv_head_idx = kb.kv_head_idx
    batch_idx = kb.batch_idx

    # block_id_x is the Q-tile index; q_tile_local is its first Q row
    # within the current batch.
    q_tile_idx = kb.q_token  # reuses block_id_x; semantically a tile, not a token
    q_tile_local = b.mul(q_tile_idx, b.const_i32(MFMA_ATTN_BLOCK_M))

    # Per-batch shift in rows (for Q and O). The helper multiplies
    # the row index by stride_q_token / stride_o_token internally, so
    # we only need to pass the row offset -- the previous duplicate
    # ``q_batch_offset = batch_idx * seqlen_q * stride_q_token``
    # multiply was dead and has been removed (the helper does that
    # same multiplication once via ``q_row * stride_q_token``). For
    # K / V we still need byte / element offsets because the helper
    # treats ``k_token_offset_elems`` as an additive element offset
    # rather than as a row index.
    batch_row_q = b.mul(batch_idx, seqlen_q)
    k_batch_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("k"))
    v_batch_offset = b.mul(b.mul(batch_idx, seqlen_k), kb.stride_token("v"))

    causal_ctx = b.const_i32(0)  # self-attention: no cache offset

    # G4: the masked query position is the within-batch row so causal /
    # sliding-window masking is correct for ``batch > 1`` -- batch b's row 0
    # is causal position 0, not ``b * seqlen_q``. Only thread it for the
    # masked modes; for ``mask_mode == "none"`` the position is unused and
    # leaving ``q_pos_base`` unset keeps the CDNA IR byte-identical (the
    # default ``q_pos_base = q_tile_base`` path).
    masked = s.mask_mode in ("causal", "sliding_window")
    q_pos_base = q_tile_local if masked else None

    mfma_attention_fwd_inner_body(
        b,
        Q=kb.tensor("Q"),
        K=kb.tensor("K"),
        V=kb.tensor("V"),
        O=kb.tensor("O"),
        head_size=s.head_size,
        seqlen_k=seqlen_k,
        # q_tile_base = local Q row + per-batch row shift. The helper
        # does ``(q_tile_base + m_in_atom) * stride_q_token`` for both
        # Q and O, which is the equivalent of folding
        # ``batch_idx * seqlen_q * stride_{q,o}_token`` into the row
        # index up front. Saves one multiply per axis vs the previous
        # ``b.mul(b.mul(batch_idx, seqlen_q), stride_{q,o}_token)``
        # which the helper then ignored.
        q_tile_base=b.add(q_tile_local, batch_row_q),
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
        scale_log2=kb.scalar("scale_log2"),
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


def fmha_fwd_mfma_grid(spec: FmhaMfmaSpec, *, batch: int) -> Tuple[int, int, int]:
    q_tiles = spec.seqlen_q // MFMA_ATTN_BLOCK_M
    return (q_tiles, spec.common.shape.num_query_heads, batch)


def fmha_fwd_mfma_signature(spec: FmhaMfmaSpec):
    kb = FmhaKernelBuilder("rocke_fmha_fwd_mfma_sig_probe", spec.common)
    _declare_params(kb)
    return kb.signature()
