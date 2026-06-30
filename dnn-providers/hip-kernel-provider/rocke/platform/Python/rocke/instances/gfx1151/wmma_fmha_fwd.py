# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WMMA FMHA forward for gfx1151 (RDNA3.5 / Strix Halo) — the first RDNA attention.

RDNA has no MFMA, so the QK^T and PV matmuls are built around the gfx11
``wmma_f32_16x16x16_f16`` instruction with a **wave32** thread mapping.

**Unification status (folded).** The wave32 QK -> online-softmax -> PV loop is
no longer hand-written here: it now lives in the *single* common FMHA-forward
inner body :func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`,
which dispatches to the WMMA wave32 path on an RDNA target and the MFMA wave64
path on CDNA off the per-arch ``MmaOp`` selected from the contract catalog --
the attention analogue of the unified ``gemm_universal``. This module is now a
thin **adapter**: it owns the gfx1151 kernel ABI, the ``(seqlen_q // 16,
num_query_heads, batch)`` grid decode, and the per-batch pointer arithmetic, and
hands the rest to the common body. The CDNA MFMA attention path stays
byte-for-byte identical to before the unification.

Everything physical about the WMMA fragments — which lane holds which
``(row, k)`` / ``(k, col)`` / ``(row, col)`` element — is read inside the common
body from the MMA contract's verified gfx1151 layout maps (``op.a_layout()`` /
``op.b_layout()`` / ``op.c_layout()`` on the ``wmma_f32_16x16x16_f16``
``MmaOp``), and the matmul itself is emitted through the target-neutral
``b.mma(op, a, b, c)``. The wave size and the reduction stage count come from
the contract so the kernel never hard-codes wave32 magic numbers.

Algorithm (one wave32 per ``(q_tile, head, batch)``; BLOCK_M = BLOCK_K = 16):

  * Grid ``(seqlen_q // 16, num_query_heads, batch)``; one wave owns 16 Q rows.
  * **QK^T**: ``S[q,k] = sum_d Q[q,d] * K[k,d]``. WMMA computes ``A @ B^T`` with
    A row-major ``M×K`` and B row-major ``N×K``; mapping A=Q (q-rows × d) and
    B=K (k-rows × d) gives exactly ``Q @ K^T``. ``head_size // 16`` WMMA steps
    accumulate the ``<8 x f32>`` score fragment.
  * **Online softmax** over the score fragment. In the accumulator layout each
    lane ``l`` owns one k-column (``l % 16``) and 8 q-rows (slot ``i`` →
    ``row 2*i + l // 16``). A per-q-row reduction over the 16 k-columns is a
    butterfly across the 16 lanes of one wave32 half (xor masks 1,2,4,8). The
    running ``m`` (row max) / ``l`` (row sum) state and the PV accumulator are
    carried through the K-loop as ``scf.for`` iter-args, exactly as the MFMA
    body does, but sized to the wave32 fragment.
  * **P staging**: the softmax probabilities live in the *accumulator* layout
    (lane = k-col), but the PV matmul needs them in the *A-operand* layout
    (lane = q-row, the 16 k-values as the fragment). We round-trip P through a
    16×16 LDS tile to transpose the distribution, mirroring the LDS P-staging
    in the MFMA body.
  * **PV**: ``O[q,d] = sum_k P[q,k] * V[k,d]``. WMMA's ``A @ B^T`` needs B in
    ``N×K`` = ``d×k`` layout, i.e. the B fragment for d-column ``c`` is the
    V-*column* ``V[k, c]`` for k = 0..15 — a strided gather of V. ``head_size
    // 16`` N-tiles of d are produced, each a ``<8 x f32>`` accumulator.
  * **Epilogue**: ``O[q,d] = acc[q,d] / l[q]`` (with the zero-denominator guard
    the MFMA body uses for fully-masked rows), truncated to f16 and scattered
    to the accumulator's ``(row, col)`` coordinates.

No async DMA, no multi-tile-per-wave: correctness-first, like the gfx1151 WMMA
GEMM it is modelled on. Tuning (LDS K/V staging, ping-pong) is a follow-on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F16, F32, I32, IRBuilder, KernelDef, PtrType

__all__ = [
    "WmmaFmhaFwdSpec",
    "build_wmma_fmha_fwd",
    "wmma_fmha_fwd_grid",
    "is_valid_spec",
]

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"  # RDNA3/3.5 (gfx11) atom
_WMMA_OP_ID_GFX12 = "wmma_gfx12_f32_16x16x16_f16"  # RDNA4 (gfx12) split-K atom
_BLOCK_M = 16  # Q rows per wave per CTA
_BLOCK_K = 16  # K positions per K-tile (WMMA N dim of QK^T)


def _wmma_op_id_for_arch(arch: str) -> str:
    """The f16 WMMA attention atom op_id for ``arch``: the RDNA4 split-K atom on
    gfx1201, else the RDNA3/3.5 cross-half-duplicated atom. Mirrors
    :func:`rocke.helpers.mfma_attention._wmma_attn_op_id`."""
    return _WMMA_OP_ID_GFX12 if arch == "gfx1201" else _WMMA_OP_ID


@dataclass(frozen=True)
class WmmaFmhaFwdSpec:
    """One gfx1151 WMMA FMHA forward configuration.

    ``head_size`` must be a multiple of 16 (the WMMA K/N tile); standard FMHA
    head sizes 64 / 128 / 256 qualify. ``seqlen_q`` / ``seqlen_k`` are runtime
    kernel args (the grid is sized from ``seqlen_q`` at launch), so the spec
    only carries the compile-time tile facts.
    """

    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0  # 0 -> equal to num_query_heads (MHA)
    dtype: str = "fp16"
    mask_mode: str = "none"  # "none" | "causal"
    sliding_window: int = 0
    # Opt lever (see examples/gfx1151/attention case study): staging the K-tile's
    # V rows through LDS for the PV B-operand cuts global loads ~3.3x but is a
    # consistent 1.5-1.8x *regression* on gfx1151 -- the PV B-operand is an
    # inherently column-strided V[k, d_col] gather, so LDS only relocates the
    # strided reads while adding a barrier this single-wave-per-CTA kernel has no
    # occupancy to hide, whereas the baseline gather stays cache-resident.
    # Default off (the measured winner); kept togglable for the A/B study.
    v_lds_stage: bool = False
    name: str = "rocke_wmma_fmha_fwd"

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "f16"):
            raise ValueError(
                f"WmmaFmhaFwdSpec currently supports fp16 only, got {self.dtype!r}"
            )
        if self.head_size % 16 != 0:
            raise ValueError(
                f"head_size must be a multiple of 16, got {self.head_size}"
            )
        if self.mask_mode not in ("none", "causal"):
            raise ValueError(
                f"WMMA FMHA supports mask_mode 'none'/'causal', got {self.mask_mode!r}"
            )

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        # One wave per block; resolved from the contract at build time too.
        return 32

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            "wmma16x16x16",
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            "fp16",
            self.mask_mode,
            "vlds" if self.v_lds_stage else "vgather",
        )


def is_valid_spec(spec: WmmaFmhaFwdSpec, arch: str = "gfx1151") -> Tuple[bool, str]:
    """Return ``(ok, reason)``. The WMMA 16x16x16 f16 atom must exist on ``arch``
    and the target must be wave32 (WMMA is an RDNA/gfx11 instruction)."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    op_id = _wmma_op_id_for_arch(arch)
    op = target.mma.by_op_id(op_id)
    if op is None or op.family != "wmma":
        return False, (
            f"WMMA {op_id} atom absent on {arch} "
            f"(WMMA is an RDNA gfx11/gfx12 instruction; this kernel needs a "
            f"wave32 RDNA target)"
        )
    if target.wave_size != op.wave_size:
        return False, (
            f"arch wave size {target.wave_size} != WMMA atom wave size "
            f"{op.wave_size} on {arch}"
        )
    if spec.head_size % 16 != 0:
        return False, f"head_size must be a multiple of 16 (got {spec.head_size})"
    # LDS budget: one 16x16 f16 P-staging tile, plus (when V-LDS staging is on)
    # one 16 x head_size f16 V tile.
    bytes_lds = _BLOCK_M * _BLOCK_K * 2
    if spec.v_lds_stage:
        bytes_lds += _BLOCK_M * spec.head_size * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(b: IRBuilder):
    """Kernel ABI (shared between build + grid helpers).

    Dense self-attention layout: Q/K/V/O are ``[seqlen, num_heads, head_size]``
    row-major within a batch; the batch axis is the grid Z dim and is folded in
    via ``seqlen`` * the batch index. Strides are passed explicitly so the same
    kernel serves both MHA and GQA (kv_head stride differs).
    """
    Q = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    K = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    V = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    out_ptr = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )
    scale_log2 = b.param("scale_log2", F32)
    seqlen_q = b.param("seqlen_q", I32)
    seqlen_k = b.param("seqlen_k", I32)
    # Element strides (row-major): token (seq) and head strides for each tensor.
    stride_q_token = b.param("stride_q_token", I32)
    stride_q_head = b.param("stride_q_head", I32)
    stride_k_token = b.param("stride_k_token", I32)
    stride_k_head = b.param("stride_k_head", I32)
    stride_v_token = b.param("stride_v_token", I32)
    stride_v_head = b.param("stride_v_head", I32)
    stride_o_token = b.param("stride_o_token", I32)
    stride_o_head = b.param("stride_o_head", I32)
    return {
        "Q": Q,
        "K": K,
        "V": V,
        "O": out_ptr,
        "scale_log2": scale_log2,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "stride_q_token": stride_q_token,
        "stride_q_head": stride_q_head,
        "stride_k_token": stride_k_token,
        "stride_k_head": stride_k_head,
        "stride_v_token": stride_v_token,
        "stride_v_head": stride_v_head,
        "stride_o_token": stride_o_token,
        "stride_o_head": stride_o_head,
    }


def build_wmma_fmha_fwd(spec: WmmaFmhaFwdSpec, arch: str = "gfx1151") -> KernelDef:
    """Build the gfx1151 WMMA FMHA forward ``KernelDef``.

    Grid: ``(seqlen_q // 16, num_query_heads, batch)``. ``arch`` selects the
    WMMA atom from the contract catalog; on a non-gfx11 arch the atom is absent
    and the build is rejected by :func:`is_valid_spec` before any IR is emitted.

    Folded into the unified forward: this is now a thin adapter over the single
    common FMHA-forward inner body
    (:func:`rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`), which
    dispatches to the WMMA wave32 path on an RDNA target and the MFMA wave64
    path on CDNA. The wave32 QK/PV matmuls, online-softmax reduction, and P
    fragment re-layout are all driven off the ``wmma_f32_16x16x16_f16``
    ``MmaOp`` layout maps inside that body -- there is no longer a second
    hand-written WMMA attention loop. This adapter only supplies the gfx1151
    kernel ABI / grid decode and the per-batch pointer arithmetic.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wmma_fmha_fwd spec: {why}")

    from ...core.arch import ArchTarget
    from ...helpers.mfma_attention import mfma_attention_fwd_inner_body

    target = ArchTarget.from_gfx(arch)
    wave = target.wave_size  # 32 for WMMA

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    p = _declare_params(b)

    c16 = b.const_i32(16)

    q_tile = b.block_id_x()  # Q-tile index (16 rows)
    head = b.block_id_y()  # query head
    batch = b.block_id_z()  # batch index

    # GQA: kv head = query head // (num_query_heads // kv_heads).
    qh = spec.num_query_heads
    kvh = spec.kv_heads
    if kvh == qh:
        kv_head = head
    else:
        kv_head = b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]

    # Per-batch row shift for Q/O (rows) and additive element offset for K/V,
    # matching the CDNA MFMA forward wrapper: the inner body multiplies the row
    # index by stride_{q,o}_token internally, so Q/O only need the row offset;
    # K/V take an additive element offset.
    q_row0 = b.mul(q_tile, c16)  # first Q row of this tile
    batch_row_q = b.mul(batch, seqlen_q)  # batch shift in Q rows
    batch_off_k = b.mul(b.mul(batch, seqlen_k), p["stride_k_token"])
    batch_off_v = b.mul(b.mul(batch, seqlen_k), p["stride_v_token"])

    mfma_attention_fwd_inner_body(
        b,
        Q=p["Q"],
        K=p["K"],
        V=p["V"],
        O=p["O"],
        head_size=spec.head_size,
        seqlen_k=seqlen_k,
        # Global Q/O row index folds the batch shift in; the within-batch q
        # position used by the mask is q_pos_base = q_row0.
        q_tile_base=b.add(q_row0, batch_row_q),
        head_idx=head,
        kv_head_idx=kv_head,
        q_pos_base=q_row0,
        stride_q_token=p["stride_q_token"],
        stride_q_head=p["stride_q_head"],
        stride_k_token=p["stride_k_token"],
        stride_k_head=p["stride_k_head"],
        stride_v_token=p["stride_v_token"],
        stride_v_head=p["stride_v_head"],
        stride_o_token=p["stride_o_token"],
        stride_o_head=p["stride_o_head"],
        scale_log2=p["scale_log2"],
        dtype="f16",
        mask_mode=spec.mask_mode,
        sliding_window=spec.sliding_window,
        causal_ctx_offset=b.const_i32(0),
        k_token_offset_elems=batch_off_k,
        v_token_offset_elems=batch_off_v,
        wmma_v_lds_stage=spec.v_lds_stage,
        arch=arch,
    )
    b.ret()
    return b.kernel


def wmma_fmha_fwd_grid(spec: WmmaFmhaFwdSpec, *, seqlen_q: int, batch: int):
    """Launch grid ``(seqlen_q // 16, num_query_heads, batch)``."""
    if seqlen_q % _BLOCK_M != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {_BLOCK_M}")
    return (seqlen_q // _BLOCK_M, spec.num_query_heads, batch)
