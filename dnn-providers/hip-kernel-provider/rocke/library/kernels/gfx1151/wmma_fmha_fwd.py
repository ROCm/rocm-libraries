# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FP16/BF16 WMMA attention forward for gfx1151.

RDNA has no MFMA, so the QK^T and PV matmuls are built around the gfx11
``wmma_f32_16x16x16_f16`` / ``wmma_f32_16x16x16_bf16`` with a **wave32** mapping.

The wave32 QK -> online-softmax -> PV loop lives in the common FMHA-forward inner
bodies of :mod:`rocke.helpers.mfma_attention`
(:func:`~rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body`, and
``wmma_swapqk_fwd_inner_body`` for the transposed-QK specialization). Those bodies
read the lane layout from the per-arch MMA contract and emit the matmul through
the target-neutral ``b.mma``. This module is a thin **adapter**: it owns the
gfx1151 kernel ABI, the ``(ceil(seqlen_q / q_rows_per_cta), num_query_heads,
batch * value_tiles)`` grid decode, and the per-batch pointer arithmetic, and hands
the rest to the common body.

Algorithm (per ``(q_tile, head, batch)``; BLOCK_M = BLOCK_K = 16; one or two
wave32 per CTA, see ``num_waves``):

  * **QK^T**: ``S[q,k] = sum_d Q[q,d] * K[k,d]``. WMMA computes ``A @ B^T`` with
    A row-major ``M×K`` and B row-major ``N×K``; mapping A=Q (q-rows × d) and
    B=K (k-rows × d) gives exactly ``Q @ K^T``. ``head_size // 16`` WMMA steps
    accumulate the ``<8 x f32>`` score fragment.
  * **Score features** (all optional, applied to the f32 scores before the
    softmax): causal masks (top-left or bottom-right), sliding window, softcap,
    ALiBi, FP32 QQ-bias and per-head sinks. Masked scores are ``-inf`` and a
    fully masked row writes zero through the zero-denominator guard.
  * **Online softmax** over the score fragment. In the accumulator layout each
    lane ``l`` owns one k-column (``l % 16``) and 8 q-rows (slot ``i`` →
    ``row 2*i + l // 16``). A per-q-row reduction over the 16 k-columns is a
    butterfly across the 16 lanes of one wave32 half (xor masks 1,2,4,8). The
    running ``m`` (row max) / ``l`` (row sum) state and the PV accumulator are
    carried through the K-loop as ``scf.for`` iter-args.
  * **P staging**: the softmax probabilities live in the *accumulator* layout
    (lane = k-col), but the PV matmul needs them in the *A-operand* layout
    (lane = q-row, the 16 k-values as the fragment). P is round-tripped through a
    16×16 LDS tile to transpose the distribution.
  * **PV**: ``O[q,d] = sum_k P[q,k] * V[k,d]``. WMMA's ``A @ B^T`` needs B in
    ``N×K`` = ``d×k`` layout, i.e. the B fragment for d-column ``c`` is the
    V-*column* ``V[k, c]`` for k = 0..15 — a strided gather of V, optionally
    staged through LDS (``v_lds_stage``). ``head_size // 16`` N-tiles of d are
    produced, each a ``<8 x f32>`` accumulator; ``value_tile_size`` splits them
    across CTAs.
  * **Epilogue**: ``O[q,d] = acc[q,d] / l[q]``, converted to the Q dtype
    (fp16 or bf16) and scattered to the accumulator's ``(row, col)`` coordinates.

Inputs may be dense ``[batch, seq, head, dim]``, ragged (``cu_seqlens``) or paged
KV, optionally with OCP-E4M3FN KV storage. The FP16 aligned dense D64/D128 case
has a transposed-QK specialization. Which spec to use for a request is decided by
``dispatch.attention.gfx1151``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import BF16, F16, FP8E4M3, F32, I32, IRBuilder, KernelDef, PtrType

__all__ = [
    "WmmaFmhaFwdSpec",
    "build_wmma_fmha_fwd",
    "wmma_fmha_fwd_grid",
    "wmma_fmha_fwd_signature",
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
    """One runtime-shape-generic gfx1151 WMMA FMHA forward configuration.

    ``head_size`` and the optional distinct ``v_head_size`` are the only
    problem dimensions specialized into the code object; both are multiples of
    16 in ``[16, 256]``. Batch, sequence lengths, query/KV head counts, tensor
    strides, bottom-right alignment, LSE selection, and window bounds are
    kernel arguments so one AOT object serves all matching runtime shapes.
    ``mask_mode`` selects unmasked, causal, or arbitrary diagonal-band masking;
    the latter consumes runtime ``window_left``/``window_right``.
    """

    head_size: int
    dtype: str = "fp16"
    mask_mode: str = "none"  # "none" | "causal" | "window"
    # Optional V staging through LDS; benchmark it per shape and dtype.
    v_lds_stage: bool = False
    name: str = "rocke_wmma_fmha_fwd"
    query_tail: bool = False
    kv_tail: bool = False
    use_softcap: bool = False
    use_sinks: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False
    layout: str = "dense"  # "dense" | "ragged" | "paged"
    page_block_size: int = 0
    kv_dtype: str = ""  # "" -> Q dtype; "fp8e4m3" -> OCP E4M3FN bytes
    transposed_qk: bool = False
    block_n: int = 32
    num_waves: int = 1
    scheduler_strategy: str | None = None
    value_tile_size: int = 0  # 0 => full head; otherwise one output-column tile per CTA
    # Standard path only: bound the K loop at the causal diagonal so fully
    # masked key tiles are never loaded. The transposed path always does this.
    causal_tile_skip: bool = False
    # V/output head width when it differs from the Q/K ``head_size``; 0 => equal.
    v_head_size: int = 0
    # Dense additive attention bias ``[B|1, H|1, Sq|1, Sk]`` added to the scores
    # (after softcap/alibi/qq_bias). Key dim is unit-stride; the B/H/Q strides are
    # runtime args (0 broadcasts). ``bias_dtype`` is "f32" or "q" (the Q dtype).
    use_attn_bias: bool = False
    bias_dtype: str = "f32"

    def __post_init__(self) -> None:
        from rocke.core.codegen_policy import normalize_scheduler_strategy

        object.__setattr__(
            self,
            "scheduler_strategy",
            normalize_scheduler_strategy(self.scheduler_strategy),
        )
        if self.dtype not in ("fp16", "f16", "bf16"):
            raise ValueError(f"WmmaFmhaFwdSpec supports fp16/bf16, got {self.dtype!r}")
        if self.kv_dtype not in ("", "fp8e4m3"):
            raise ValueError("KV storage must match Q or use OCP fp8e4m3")
        if not 16 <= self.head_size <= 256 or self.head_size % 16:
            raise ValueError(
                f"head_size must be a multiple of 16 in [16, 256], got {self.head_size}"
            )
        if self.v_head_size < 0 or self.v_head_size > 256 or self.v_head_size % 16:
            raise ValueError(
                "v_head_size must be zero or a multiple of 16 in [16, 256], "
                f"got {self.v_head_size}"
            )
        if self.bias_dtype not in ("f32", "q"):
            raise ValueError(
                f"bias_dtype must be 'f32' or 'q', got {self.bias_dtype!r}"
            )
        if self.v_head_size == self.head_size:
            raise ValueError("v_head_size must be 0 when it equals head_size")
        if self.v_head_size and self.transposed_qk:
            raise ValueError("v_head_size is not supported with transposed_qk")
        if self.value_tile_size and (
            self.value_tile_size < 16
            or self.value_tile_size % 16
            or self.value_tile_size >= self.v_dim
            or self.v_dim % self.value_tile_size
            or self.transposed_qk
        ):
            raise ValueError(
                "value_tile_size must be a proper multiple-of-16 divisor of the V head on the standard WMMA path"
            )
        if self.mask_mode not in ("none", "causal", "window"):
            raise ValueError(
                "WMMA FMHA supports mask_mode 'none'/'causal'/'window', "
                f"got {self.mask_mode!r}"
            )
        if self.causal_tile_skip and (self.mask_mode != "causal" or self.transposed_qk):
            raise ValueError(
                "causal_tile_skip requires causal masking on the standard "
                "(non-transposed) path"
            )
        if self.layout not in ("dense", "ragged", "paged"):
            raise ValueError(f"unsupported attention layout {self.layout!r}")
        if self.layout == "paged":
            if self.page_block_size <= 0 or self.page_block_size & (
                self.page_block_size - 1
            ):
                raise ValueError(
                    "paged attention requires a positive power-of-two page size"
                )
        elif self.page_block_size:
            raise ValueError("page_block_size is only valid for paged attention")
        if self.transposed_qk:
            if self.dtype not in ("fp16", "f16", "bf16") or self.head_size not in (
                64,
                128,
            ):
                raise ValueError("transposed QK supports FP16/BF16 D64/D128")
            if self.block_n not in (32, 64) or self.num_waves not in (1, 2):
                raise ValueError(
                    "transposed QK requires block_n 32/64 and one or two waves"
                )
            if (
                self.layout != "dense"
                or self.kv_dtype
                or self.query_tail
                or self.kv_tail
                or self.v_lds_stage
                or self.mask_mode == "window"
                or self.use_softcap
                or self.use_sinks
                or self.use_alibi
                or self.use_qq_bias
                or self.use_attn_bias
            ):
                raise ValueError(
                    "transposed QK requires aligned dense inputs without windows "
                    "or extra score features"
                )
        elif self.block_n != 32 or self.num_waves != 1:
            raise ValueError("block_n and num_waves are transposed-QK options")

    @property
    def v_dim(self) -> int:
        return self.v_head_size or self.head_size

    @property
    def value_tiles(self) -> int:
        return self.v_dim // self.value_tile_size if self.value_tile_size else 1

    @property
    def block_size(self) -> int:
        return 32 * (self.num_waves if self.transposed_qk else 1)

    @property
    def q_rows_per_cta(self) -> int:
        return _BLOCK_M * (self.num_waves if self.transposed_qk else 1)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            "wmma_swapqk" if self.transposed_qk else "wmma16x16x16",
            f"H{self.head_size}",
            "bf16" if self.dtype == "bf16" else "fp16",
            self.mask_mode,
            "vlds" if self.v_lds_stage else "vgather",
            self.layout if self.layout != "dense" else "",
            f"bs{self.page_block_size}" if self.layout == "paged" else "",
            f"kv{self.kv_dtype}" if self.kv_dtype else "",
            f"bn{self.block_n}" if self.transposed_qk else "",
            f"w{self.num_waves}" if self.transposed_qk else "",
            (
                "sched_" + self.scheduler_strategy.replace("-", "_")
                if self.scheduler_strategy
                else ""
            ),
            f"dv{self.value_tile_size}" if self.value_tile_size else "",
            f"vh{self.v_head_size}" if self.v_head_size else "",
            f"abias_{self.bias_dtype}" if self.use_attn_bias else "",
            flags={
                "qtail": self.query_tail or self.layout != "dense",
                "kvtail": self.kv_tail or self.layout != "dense",
                "softcap": self.use_softcap,
                "sinks": self.use_sinks,
                "alibi": self.use_alibi,
                "qqbias": self.use_qq_bias,
                "cskip": self.causal_tile_skip,
            },
        )


def is_valid_spec(spec: WmmaFmhaFwdSpec, arch: str = "gfx1151") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a dtype-matched WMMA atom on a wave32 target."""
    from rocke.core.arch import ArchTarget

    if spec.transposed_qk and arch != "gfx1151":
        return False, "transposed QK requires gfx1151"
    if spec.value_tile_size and arch != "gfx1151":
        return False, "output-column tiling requires gfx1151"
    if spec.v_head_size and arch != "gfx1151":
        return False, "a distinct V head size requires gfx1151"

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
        bytes_lds += _BLOCK_M * (spec.value_tile_size or spec.v_dim) * 2
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )
    return True, "ok"


def _declare_params(b: IRBuilder, spec: WmmaFmhaFwdSpec):
    """Declare the fixed runtime-shape ABI plus layout-specific metadata."""
    elem = BF16 if spec.dtype == "bf16" else F16
    kv_elem = FP8E4M3 if spec.kv_dtype else elem
    Q = b.param("Q", PtrType(elem, "global"), noalias=True, readonly=True, align=2)
    K = b.param(
        "K",
        PtrType(kv_elem, "global"),
        noalias=True,
        readonly=True,
        align=1 if spec.kv_dtype else 2,
    )
    V = b.param(
        "V",
        PtrType(kv_elem, "global"),
        noalias=True,
        readonly=True,
        align=1 if spec.kv_dtype else 2,
    )
    out_ptr = b.param(
        "O", PtrType(elem, "global"), noalias=True, writeonly=True, align=2
    )
    lse_ptr = b.param("LSE", PtrType(F32, "global"), writeonly=True, align=4)
    params = {
        "Q": Q,
        "K": K,
        "V": V,
        "O": out_ptr,
        "LSE": lse_ptr,
        "scale_log2": b.param("scale_log2", F32),
        "seqlen_q": b.param("seqlen_q", I32),
        "seqlen_k": b.param("seqlen_k", I32),
        "num_query_heads": b.param("num_query_heads", I32),
        "num_kv_heads": b.param("num_kv_heads", I32),
        "bottom_right": b.param("bottom_right", I32),
        "window_left": b.param("window_left", I32),
        "window_right": b.param("window_right", I32),
        "write_lse": b.param("write_lse", I32),
    }
    for tensor in ("q", "k", "v", "o"):
        params[f"stride_{tensor}_batch"] = b.param(f"stride_{tensor}_batch", I32)
        params[f"stride_{tensor}_token"] = b.param(f"stride_{tensor}_token", I32)
        params[f"stride_{tensor}_head"] = b.param(f"stride_{tensor}_head", I32)
    params["stride_lse_batch"] = b.param("stride_lse_batch", I32)
    params["stride_lse_token"] = b.param("stride_lse_token", I32)
    params["stride_lse_head"] = b.param("stride_lse_head", I32)
    if spec.use_softcap:
        params["softcap"] = b.param("softcap", F32)
    if spec.use_sinks:
        params["sink_ptr"] = b.param(
            "sink_ptr", PtrType(elem, "global"), readonly=True, align=2
        )
    if spec.use_alibi:
        params["alibi_slopes_ptr"] = b.param(
            "alibi_slopes_ptr",
            PtrType(F32, "global"),
            readonly=True,
            align=4,
        )
    if spec.use_qq_bias:
        params["qq_bias_ptr"] = b.param(
            "qq_bias_ptr",
            PtrType(F32, "global"),
            readonly=True,
            align=4,
        )
        params["qq_bias_rows"] = b.param("qq_bias_rows", I32)
        params["qq_bias_cols"] = b.param("qq_bias_cols", I32)
        params["qq_bias_stride"] = b.param("qq_bias_stride", I32)
    if spec.layout != "dense":
        params["cu_seqlens_q"] = b.param(
            "cu_seqlens_q",
            PtrType(I32, "global"),
            readonly=True,
            align=4,
        )
        if spec.layout == "ragged":
            params["cu_seqlens_k"] = b.param(
                "cu_seqlens_k",
                PtrType(I32, "global"),
                readonly=True,
                align=4,
            )
        else:
            params["seqused_k"] = b.param(
                "seqused_k",
                PtrType(I32, "global"),
                readonly=True,
                align=4,
            )
            params["block_table"] = b.param(
                "block_table",
                PtrType(I32, "global"),
                readonly=True,
                align=4,
            )
            params["block_table_stride"] = b.param("block_table_stride", I32)
            params["stride_k_block"] = b.param("stride_k_block", I32)
            params["stride_v_block"] = b.param("stride_v_block", I32)
    if spec.kv_dtype:
        params["k_scale"] = b.param("k_scale", F32)
        params["v_scale"] = b.param("v_scale", F32)
    if spec.use_attn_bias:
        bias_elem = F32 if spec.bias_dtype == "f32" else _q_elem(spec)
        params["attn_bias_ptr"] = b.param(
            "attn_bias_ptr",
            PtrType(bias_elem, "global"),
            noalias=True,
            readonly=True,
            align=4 if spec.bias_dtype == "f32" else 2,
        )
        params["bias_stride_b"] = b.param("bias_stride_b", I32)
        params["bias_stride_h"] = b.param("bias_stride_h", I32)
        params["bias_stride_q"] = b.param("bias_stride_q", I32)
    return params


def _q_elem(spec):
    return BF16 if spec.dtype == "bf16" else F16


def _score_features(b, spec, params, head, context, batch, seqlen_q, seqlen_k, masked):
    windowed = spec.mask_mode == "window"
    if not (
        windowed
        or spec.use_softcap
        or spec.use_sinks
        or spec.use_alibi
        or spec.use_qq_bias
        or spec.use_attn_bias
    ):
        return None, None
    log2e = b.const_f32(1.4426950408889634)
    cap = b.fmul(params["softcap"], log2e) if spec.use_softcap else None
    sink = None
    if spec.use_sinks:
        elem = BF16 if spec.dtype == "bf16" else F16
        sink = b.fmul(
            b.cast_to_f32(b.global_load(params["sink_ptr"], head, elem, align=2)),
            log2e,
        )
    slope = None
    if spec.use_alibi:
        slope = b.fmul(
            b.global_load(params["alibi_slopes_ptr"], head, F32, align=4), log2e
        )
    zero_f = b.const_f32(0.0) if spec.use_qq_bias else None
    zero_i = b.const_i32(0) if spec.use_qq_bias or windowed else None
    bias_base = bias_zero = bias_elem = None
    if spec.use_attn_bias:
        bias_elem = F32 if spec.bias_dtype == "f32" else _q_elem(spec)
        bias_zero = b.const_f32(0.0)
        bias_base = b.add(
            b.mul(batch, params["bias_stride_b"]),
            b.mul(head, params["bias_stride_h"]),
        )

    def transform(builder, score, _kt, _row, query_pos, key_pos):
        if cap is not None:
            score = builder.fmul(cap, builder.tanh(builder.fdiv(score, cap)))
        relative_k = (
            builder.sub(key_pos, context)
            if slope is not None or spec.use_qq_bias
            else None
        )
        if slope is not None:
            score = builder.fadd(
                score, builder.fmul(slope, builder.sitofp_f32(relative_k))
            )
        if spec.use_qq_bias:
            q_ok = builder.cmp_lt(query_pos, params["qq_bias_rows"])
            k_lo = builder.cmp_ge(relative_k, zero_i)
            k_hi = builder.cmp_lt(relative_k, params["qq_bias_cols"])
            keep = builder.land(q_ok, builder.land(k_lo, k_hi))
            index = builder.add(
                builder.mul(query_pos, params["qq_bias_stride"]), relative_k
            )
            bias = builder.masked_global_load(
                params["qq_bias_ptr"], index, keep, zero_f, F32, align=4
            )
            score = builder.fadd(score, builder.fmul(bias, log2e))
        if windowed:
            center = builder.add(query_pos, context)
            left = params["window_left"]
            right = params["window_right"]
            keep_left = builder.lor(
                builder.cmp_lt(left, zero_i),
                builder.cmp_ge(key_pos, builder.sub(center, left)),
            )
            keep_right = builder.lor(
                builder.cmp_lt(right, zero_i),
                builder.cmp_le(key_pos, builder.add(center, right)),
            )
            score = builder.select(builder.land(keep_left, keep_right), score, masked)
        if spec.use_attn_bias:
            keep = builder.land(
                builder.cmp_lt(query_pos, seqlen_q), builder.cmp_lt(key_pos, seqlen_k)
            )
            index = builder.add(
                builder.add(bias_base, builder.mul(query_pos, params["bias_stride_q"])),
                key_pos,
            )
            if spec.bias_dtype == "f32":
                bias = builder.masked_global_load(
                    params["attn_bias_ptr"], index, keep, bias_zero, F32, align=4
                )
            else:
                safe = builder.select(keep, index, builder.const_i32(0))
                loaded = builder.cast_to_f32(
                    builder.global_load(
                        params["attn_bias_ptr"], safe, bias_elem, align=2
                    )
                )
                bias = builder.select(keep, loaded, bias_zero)
            score = builder.fadd(score, builder.fmul(bias, log2e))
        return score

    return (
        transform
        if windowed
        or spec.use_softcap
        or spec.use_alibi
        or spec.use_qq_bias
        or spec.use_attn_bias
        else None
    ), sink


def _window_tiles(
    b,
    left,
    right,
    query_start,
    query_length,
    key_length,
    context,
    tile,
):
    zero = b.const_i32(0)
    one = b.const_i32(1)
    last = b.const_i32(15)
    first_center = b.add(query_start, context)
    lower = b.sub(first_center, left)
    lower = b.select(b.cmp_lt(left, zero), zero, lower)
    lower = b.select(b.cmp_lt(lower, zero), zero, lower)
    q_last = b.add(query_start, last)
    q_limit = b.sub(query_length, one)
    q_last = b.select(b.cmp_gt(q_last, q_limit), q_limit, q_last)
    upper = b.add(b.add(q_last, context), b.add(right, one))
    upper = b.select(b.cmp_lt(right, zero), key_length, upper)
    upper = b.select(b.cmp_gt(upper, key_length), key_length, upper)
    upper = b.select(b.cmp_lt(upper, zero), zero, upper)
    start = b.div(lower, tile)
    stop = b.div(b.add(upper, last), tile)
    return start, stop


def _paged_rows(b, spec, params, batch, kv_head):
    page_log2 = b.const_i32(spec.page_block_size.bit_length() - 1)
    page_mask = b.const_i32(spec.page_block_size - 1)
    table_row = b.mul(batch, params["block_table_stride"])

    def rows(stride_block, stride_token, stride_head):
        head_offset = b.mul(kv_head, stride_head)

        def row(b, token):
            logical_block = b.lshr(token, page_log2)
            page_token = b.land(token, page_mask)
            physical_block = b.global_load_i32(
                params["block_table"],
                b.add(table_row, logical_block),
            )
            return b.add(
                b.add(
                    b.mul(physical_block, stride_block), b.mul(page_token, stride_token)
                ),
                head_offset,
            )

        return row

    return (
        rows(
            params["stride_k_block"], params["stride_k_token"], params["stride_k_head"]
        ),
        rows(
            params["stride_v_block"], params["stride_v_token"], params["stride_v_head"]
        ),
    )


def build_wmma_fmha_fwd(spec: WmmaFmhaFwdSpec, arch: str = "gfx1151") -> KernelDef:
    """Build one runtime-shape-generic gfx1151 WMMA FMHA forward kernel."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid wmma_fmha_fwd spec: {why}")

    from rocke.core.arch import ArchTarget
    from rocke.core.codegen_policy import CodegenPolicy, apply_codegen_policy
    from rocke.helpers.mfma_attention import mfma_attention_fwd_inner_body

    target = ArchTarget.from_gfx(arch)
    wave = target.wave_size
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = wave * (
        spec.num_waves if spec.transposed_qk else 1
    )
    apply_codegen_policy(
        b.kernel, CodegenPolicy(scheduler_strategy=spec.scheduler_strategy)
    )
    p = _declare_params(b, spec)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    q_tile = b.block_id_x()
    head = b.block_id_y()
    batch = b.block_id_z()
    value_offset = None
    if spec.value_tile_size:
        batch_tile = batch
        tiles = b.const_i32(spec.value_tiles)
        batch = b.div(batch_tile, tiles)
        value_tile = b.mod(batch_tile, tiles)
        value_offset = b.mul(value_tile, b.const_i32(spec.value_tile_size))

    # Runtime GQA: one AOT object serves every integral Hq/Hkv ratio.
    group_size = b.div(p["num_query_heads"], p["num_kv_heads"])
    kv_head = b.div(head, group_size)
    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    q_step = b.const_i32(spec.q_rows_per_cta) if spec.transposed_qk else c16
    q_row0 = b.mul(q_tile, q_step)

    Q, K, V, O, LSE = p["Q"], p["K"], p["V"], p["O"], p["LSE"]
    if spec.layout == "dense":
        q_elem_bytes = b.const_i32(2)
        kv_elem_bytes = b.const_i32(1 if spec.kv_dtype else 2)
        lse_elem_bytes = b.const_i32(4)

        def batch_ptr(pointer, stride, elem_bytes):
            offset = b.mul(b.mul(batch, stride), elem_bytes)
            return b.global_ptr_add(pointer, offset)

        Q = batch_ptr(Q, p["stride_q_batch"], q_elem_bytes)
        K = batch_ptr(K, p["stride_k_batch"], kv_elem_bytes)
        V = batch_ptr(V, p["stride_v_batch"], kv_elem_bytes)
        O = batch_ptr(O, p["stride_o_batch"], q_elem_bytes)
        LSE = batch_ptr(LSE, p["stride_lse_batch"], lse_elem_bytes)
        q_global = q_row0
        batch_off_k = batch_off_v = c0
    else:
        next_batch = b.add(batch, b.const_i32(1))
        batch_row_q = b.global_load_i32(p["cu_seqlens_q"], batch)
        q_end = b.global_load_i32(p["cu_seqlens_q"], next_batch)
        seqlen_q = b.sub(q_end, batch_row_q)
        with b.scf_if(b.cmp_ge(q_row0, seqlen_q)):
            b.ret()
        if spec.layout == "ragged":
            k_start = b.global_load_i32(p["cu_seqlens_k"], batch)
            k_end = b.global_load_i32(p["cu_seqlens_k"], next_batch)
            seqlen_k = b.sub(k_end, k_start)
            batch_off_k = b.mul(k_start, p["stride_k_token"])
            batch_off_v = b.mul(k_start, p["stride_v_token"])
        else:
            seqlen_k = b.global_load_i32(p["seqused_k"], batch)
            batch_off_k = batch_off_v = c0
        q_global = b.add(q_row0, batch_row_q)

    context = b.select(b.cmp_ne(p["bottom_right"], c0), b.sub(seqlen_k, seqlen_q), c0)
    strict = (
        spec.mask_mode != "none"
        or spec.dtype == "bf16"
        or spec.kv_tail
        or spec.causal_tile_skip
        or spec.use_softcap
        or spec.use_sinks
        or spec.use_alibi
        or spec.use_qq_bias
        or spec.use_attn_bias
        or spec.layout != "dense"
    )
    masked = b.const_f32(float("-inf")) if strict else None
    score_transform, sink = _score_features(
        b, spec, p, head, context, batch, seqlen_q, seqlen_k, masked
    )
    tile_start = tile_stop = None
    if spec.mask_mode == "window":
        tile_start, tile_stop = _window_tiles(
            b,
            p["window_left"],
            p["window_right"],
            q_row0,
            seqlen_q,
            seqlen_k,
            context,
            c16,
        )
    elif spec.causal_tile_skip:
        tile_start, tile_stop = _window_tiles(
            b,
            b.const_i32(-1),
            c0,
            q_row0,
            seqlen_q,
            seqlen_k,
            context,
            c16,
        )
    k_row, v_row = (None, None)
    if spec.layout == "paged":
        k_row, v_row = _paged_rows(b, spec, p, batch, kv_head)

    common_lse = {
        "lse": LSE,
        "write_lse": p["write_lse"],
        "stride_lse_token": p["stride_lse_token"],
        "stride_lse_head": p["stride_lse_head"],
    }
    if spec.transposed_qk:
        from rocke.helpers.wmma_swapqk import wmma_swapqk_fwd_inner_body

        wmma_swapqk_fwd_inner_body(
            b,
            Q=Q,
            K=K,
            V=V,
            O=O,
            head_size=spec.head_size,
            dtype="bf16" if spec.dtype == "bf16" else "f16",
            seqlen_k=seqlen_k,
            q_tile_base=q_global,
            q_pos_base=q_row0,
            head_idx=head,
            kv_head_idx=kv_head,
            stride_q_token=p["stride_q_token"],
            stride_q_head=p["stride_q_head"],
            stride_k_token=p["stride_k_token"],
            stride_k_head=p["stride_k_head"],
            stride_v_token=p["stride_v_token"],
            stride_v_head=p["stride_v_head"],
            stride_o_token=p["stride_o_token"],
            stride_o_head=p["stride_o_head"],
            scale_log2=p["scale_log2"],
            k_token_offset_elems=batch_off_k,
            v_token_offset_elems=batch_off_v,
            mask_mode=spec.mask_mode,
            block_n=spec.block_n,
            n_waves=spec.num_waves,
            arch=arch,
            causal_ctx_offset=context if spec.mask_mode == "causal" else None,
            mask_neg_inf=masked,
            **common_lse,
        )
        b.ret()
        return b.kernel

    mfma_attention_fwd_inner_body(
        b,
        Q=Q,
        K=K,
        V=V,
        O=O,
        head_size=spec.head_size,
        seqlen_k=seqlen_k,
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
        mask_mode="causal" if spec.mask_mode == "causal" else "none",
        sliding_window=0,
        causal_ctx_offset=context,
        mask_neg_inf=masked,
        k_token_offset_elems=batch_off_k,
        v_token_offset_elems=batch_off_v,
        wmma_v_lds_stage=spec.v_lds_stage,
        arch=arch,
        wmma_seqlen_q=seqlen_q if spec.query_tail or spec.layout != "dense" else None,
        wmma_kv_tail=spec.kv_tail or spec.layout != "dense",
        wmma_value_tile_size=spec.value_tile_size,
        wmma_value_offset=value_offset,
        wmma_v_head_size=spec.v_head_size,
        extra_score_transform=score_transform,
        sink_log2=sink,
        k_tile_start=tile_start,
        k_tile_stop=tile_stop,
        k_row_base_fn=k_row,
        v_row_base_fn=v_row,
        kv_dtype=spec.kv_dtype or None,
        k_scale=p.get("k_scale"),
        v_scale=p.get("v_scale"),
        **common_lse,
    )
    b.ret()
    return b.kernel


def wmma_fmha_fwd_grid(
    spec: WmmaFmhaFwdSpec, *, seqlen_q: int, num_query_heads: int, batch: int
):
    """Cover one runtime query/head/batch problem."""
    block_m = spec.q_rows_per_cta
    if spec.layout == "dense" and not spec.query_tail and seqlen_q % block_m != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {block_m}")
    if num_query_heads <= 0:
        raise ValueError("num_query_heads must be positive")
    if spec.value_tile_size and not 0 <= batch <= 0x7FFFFFFF // spec.value_tiles:
        raise ValueError("batch/output-tile grid does not fit I32")
    return (
        (seqlen_q + block_m - 1) // block_m,
        num_query_heads,
        batch * spec.value_tiles,
    )


def wmma_fmha_fwd_signature(spec: WmmaFmhaFwdSpec):
    """Return the specialized ABI without emitting the attention body."""
    b = IRBuilder(spec.kernel_name())
    _declare_params(b, spec)
    return tuple(
        {"name": param.name, "type": param.type.name} for param in b.kernel.params
    )
