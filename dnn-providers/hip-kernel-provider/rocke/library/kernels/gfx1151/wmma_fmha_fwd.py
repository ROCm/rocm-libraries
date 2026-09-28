# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP16/BF16 WMMA attention forward for gfx1151.

RDNA has no MFMA, so the QK^T and PV matmuls are built around the gfx11
``wmma_f32_16x16x16_f16`` / ``wmma_f32_16x16x16_bf16`` with a **wave32** mapping.

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
``op.b_layout()`` / ``op.c_layout()`` on the dtype-matched WMMA
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

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType

__all__ = [
    "WmmaFmhaFwdSpec",
    "build_wmma_fmha_fwd",
    "wmma_fmha_fwd_grid",
    "is_valid_spec",
]

_BLOCK_M = 16  # Q rows per wave per CTA
_BLOCK_K = 16  # K positions per K-tile (WMMA N dim of QK^T)


def _wmma_op_id_for_arch(arch: str, dtype: str) -> str:
    """Select the dtype-matched WMMA atom, including the gfx1201 split-K layout."""
    elem = "bf16" if dtype == "bf16" else "f16"
    prefix = "wmma_gfx12" if arch == "gfx1201" else "wmma"
    return f"{prefix}_f32_16x16x16_{elem}"


@dataclass(frozen=True)
class WmmaFmhaFwdSpec:
    """One gfx1151 WMMA FMHA forward configuration.

    ``head_size`` must be a multiple of 16 (the WMMA K/N tile); standard FMHA
    head sizes 64 / 128 / 256 qualify. For GQA, ``num_query_heads`` must be a
    multiple of ``num_kv_heads`` so every query head maps to a valid KV head.
    ``seqlen_q`` / ``seqlen_k`` are runtime kernel args (the grid is sized from
    ``seqlen_q`` at launch), so the spec only carries the compile-time tile
    facts.

    ``causal_bottom_right=True`` shifts the causal diagonal by
    ``seqlen_k - seqlen_q``. It requires ``mask_mode="causal"`` and returns
    zero for a query row with no visible keys.
    """

    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0  # 0 -> equal to num_query_heads (MHA)
    dtype: str = "fp16"
    mask_mode: str = "none"  # "none" | "causal"
    sliding_window: int = 0
    # Optional V staging through LDS; benchmark it per shape and dtype.
    v_lds_stage: bool = False
    name: str = "rocke_wmma_fmha_fwd"
    causal_bottom_right: bool = False
    query_tail: bool = False
    kv_tail: bool = False
    use_softcap: bool = False
    use_sinks: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in ("fp16", "f16", "bf16"):
            raise ValueError(
                f"WmmaFmhaFwdSpec supports fp16/bf16, got {self.dtype!r}"
            )
        if self.head_size % 16 != 0:
            raise ValueError(
                f"head_size must be a multiple of 16, got {self.head_size}"
            )
        if self.mask_mode not in ("none", "causal"):
            raise ValueError(
                f"WMMA FMHA supports mask_mode 'none'/'causal', got {self.mask_mode!r}"
            )
        if self.causal_bottom_right and self.mask_mode != "causal":
            raise ValueError("bottom-right alignment requires causal masking")
        if self.sliding_window < 0 or (self.sliding_window and self.mask_mode != "causal"):
            raise ValueError("sliding-window attention requires a nonnegative width and causal masking")
        if self.num_kv_heads and self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be a multiple of num_kv_heads for GQA "
                f"(got num_query_heads={self.num_query_heads}, "
                f"num_kv_heads={self.num_kv_heads}); otherwise a query head "
                "would map to an out-of-range KV head"
            )

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        # One wave per block; resolved from the contract at build time too.
        return 32

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            "wmma16x16x16",
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            "bf16" if self.dtype == "bf16" else "fp16",
            "causal_br" if self.causal_bottom_right else self.mask_mode,
            "vlds" if self.v_lds_stage else "vgather",
            f"sw{self.sliding_window}" if self.sliding_window else "",
            flags={
                "qtail": self.query_tail, "kvtail": self.kv_tail,
                "softcap": self.use_softcap, "sinks": self.use_sinks,
                "alibi": self.use_alibi, "qqbias": self.use_qq_bias,
            },
        )


def is_valid_spec(spec: WmmaFmhaFwdSpec, arch: str = "gfx1151") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a dtype-matched WMMA atom on a wave32 target."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    op_id = _wmma_op_id_for_arch(arch, spec.dtype)
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
    # One 16-bit P-staging tile, plus an optional 16-bit V tile.
    bytes_lds = _BLOCK_M * _BLOCK_K * 2
    if spec.v_lds_stage:
        bytes_lds += _BLOCK_M * spec.head_size * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(b: IRBuilder, spec: WmmaFmhaFwdSpec):
    """Kernel ABI (shared between build + grid helpers).

    Dense self-attention layout: Q/K/V/O are ``[seqlen, num_heads, head_size]``
    row-major within a batch; the batch axis is the grid Z dim and is folded in
    via ``seqlen`` * the batch index. Strides are passed explicitly so the same
    kernel serves both MHA and GQA (kv_head stride differs).
    """
    elem = BF16 if spec.dtype == "bf16" else F16
    Q = b.param("Q", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    K = b.param("K", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    V = b.param("V", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    out_ptr = b.param(
        "O", PtrType(elem, "global"), noalias=True, writeonly=True, align=16
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
    params = {
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
    if spec.use_softcap:
        params["softcap"] = b.param("softcap", F32)
    if spec.use_sinks:
        params["sink_ptr"] = b.param("sink_ptr", PtrType(elem, "global"), readonly=True, align=2)
    if spec.use_alibi:
        params["alibi_slopes_ptr"] = b.param(
            "alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4,
        )
    if spec.use_qq_bias:
        params["qq_bias_ptr"] = b.param(
            "qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4,
        )
        params["qq_bias_rows"] = b.param("qq_bias_rows", I32)
        params["qq_bias_cols"] = b.param("qq_bias_cols", I32)
        params["qq_bias_stride"] = b.param("qq_bias_stride", I32)
    return params


def _score_features(b, spec, params, head, context):
    if not (spec.use_softcap or spec.use_sinks or spec.use_alibi or spec.use_qq_bias):
        return None, None
    log2e = b.const_f32(1.4426950408889634)
    cap = b.fmul(params["softcap"], log2e) if spec.use_softcap else None
    sink = None
    if spec.use_sinks:
        elem = BF16 if spec.dtype == "bf16" else F16
        sink = b.fmul(b.cast_to_f32(b.global_load(params["sink_ptr"], head, elem, align=2)), log2e)
    slope = None
    if spec.use_alibi:
        slope = b.fmul(b.global_load(params["alibi_slopes_ptr"], head, F32, align=4), log2e)
    zero_f = b.const_f32(0.0) if spec.use_qq_bias else None
    zero_i = b.const_i32(0) if spec.use_qq_bias else None

    def transform(builder, score, _kt, _row, query_pos, key_pos):
        if cap is not None:
            score = builder.fmul(cap, builder.tanh(builder.fdiv(score, cap)))
        relative_k = builder.sub(key_pos, context) if slope is not None or spec.use_qq_bias else None
        if slope is not None:
            score = builder.fadd(score, builder.fmul(slope, builder.sitofp_f32(relative_k)))
        if spec.use_qq_bias:
            q_ok = builder.cmp_lt(query_pos, params["qq_bias_rows"])
            k_lo = builder.cmp_ge(relative_k, zero_i)
            k_hi = builder.cmp_lt(relative_k, params["qq_bias_cols"])
            keep = builder.land(q_ok, builder.land(k_lo, k_hi))
            index = builder.add(builder.mul(query_pos, params["qq_bias_stride"]), relative_k)
            bias = builder.masked_global_load(params["qq_bias_ptr"], index, keep, zero_f, F32, align=4)
            score = builder.fadd(score, builder.fmul(bias, log2e))
        return score

    return transform if spec.use_softcap or spec.use_alibi or spec.use_qq_bias else None, sink


def _window_tiles(b, width, query_start, query_length, key_length, context, tile):
    if not width:
        return None, None
    zero = b.const_i32(0)
    one = b.const_i32(1)
    last = b.const_i32(15)
    lower = b.sub(b.add(query_start, context), b.const_i32(width - 1))
    lower = b.select(b.cmp_lt(lower, zero), zero, lower)
    q_last = b.add(query_start, last)
    q_limit = b.sub(query_length, one)
    q_last = b.select(b.cmp_gt(q_last, q_limit), q_limit, q_last)
    upper = b.add(b.add(q_last, context), one)
    upper = b.select(b.cmp_gt(upper, key_length), key_length, upper)
    upper = b.select(b.cmp_lt(upper, zero), zero, upper)
    start = b.div(lower, tile)
    stop = b.div(b.add(upper, last), tile)
    return start, stop


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

    from rocke.core.arch import ArchTarget
    from rocke.helpers.mfma_attention import mfma_attention_fwd_inner_body

    target = ArchTarget.from_gfx(arch)
    wave = target.wave_size  # 32 for WMMA

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave
    p = _declare_params(b, spec)

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
    q_global = b.add(q_row0, batch_row_q)
    context = b.sub(seqlen_k, seqlen_q) if spec.causal_bottom_right else b.const_i32(0)
    strict = (
        spec.causal_bottom_right or spec.dtype == "bf16" or spec.kv_tail
        or spec.sliding_window or spec.use_softcap or spec.use_sinks
        or spec.use_alibi or spec.use_qq_bias
    )
    masked = b.const_f32(float("-inf")) if strict else None
    score_transform, sink = _score_features(b, spec, p, head, context)
    tile_start, tile_stop = _window_tiles(
        b, spec.sliding_window, q_row0, seqlen_q, seqlen_k, context, c16,
    )

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
        q_tile_base=q_global,
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
        dtype="bf16" if spec.dtype == "bf16" else "f16",
        mask_mode=spec.mask_mode,
        sliding_window=spec.sliding_window,
        causal_ctx_offset=context,
        mask_neg_inf=masked,
        k_token_offset_elems=batch_off_k,
        v_token_offset_elems=batch_off_v,
        wmma_v_lds_stage=spec.v_lds_stage,
        arch=arch,
        wmma_seqlen_q=seqlen_q if spec.query_tail else None,
        wmma_kv_tail=spec.kv_tail,
        extra_score_transform=score_transform,
        sink_log2=sink,
        k_tile_start=tile_start,
        k_tile_stop=tile_stop,
    )
    b.ret()
    return b.kernel


def wmma_fmha_fwd_grid(spec: WmmaFmhaFwdSpec, *, seqlen_q: int, batch: int):
    """Cover all query tiles; partial tiles require the query-tail specialization."""
    if not spec.query_tail and seqlen_q % _BLOCK_M != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {_BLOCK_M}")
    return ((seqlen_q + _BLOCK_M - 1) // _BLOCK_M, spec.num_query_heads, batch)
