# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-KV decode FMHA forward (CK Tile ``01_fmha`` splitkv).

Decoding from a long KV cache with a small batch is bandwidth-limited;
splitting the K dimension across many CTAs (each handling one
K-segment) and then reducing the per-segment ``(m, l, acc)`` triples
lifts occupancy to fully saturate the SMs.

Two-launch pipeline:

1. ``build_fmha_fwd_splitkv_decode_segment`` -- one CTA per
   ``(seq_idx, head_idx, segment_idx)``. Each CTA walks its slice
   of the K cache and emits the per-segment ``(m, l, acc)`` to the
   workspace.
2. ``build_fmha_fwd_splitkv_decode_reduce`` -- combine the
   per-segment triples back into the final ``O = acc/l`` using the
   online-softmax merge rule.

Both kernels use the warp-distributed body (one warp per CTA, lane
distributes the head_dim) so no LDS state and no thread-redundant
work. The production tiled split-KV path lives in
:mod:`rocke.instances.gfx950.attention_tiled_3d`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import IRBuilder, KernelDef, Value
from ...helpers.attention import (
    apply_attention_mask,
    warp_xor_reduce_sum,
)
from ...helpers.io import (
    load_lane_slice_f32,
    pack_f32_to,
    store_scalar_from_f32,
    store_vec,
)
from ...helpers.spec import kernel_name_join
from ...helpers.transforms import calculate_magic_numbers, do_magic_division
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE
from .fmha_arch import validate_fmha_mfma_atom


__all__ = [
    "FmhaFwdSplitKvDecodeSpec",
    "build_fmha_fwd_splitkv_decode_segment",
    "build_fmha_fwd_splitkv_decode_reduce",
    "fmha_fwd_splitkv_decode_segment_grid",
    "fmha_fwd_splitkv_decode_segment_signature",
    "fmha_fwd_splitkv_decode_reduce_grid",
    "fmha_fwd_splitkv_decode_reduce_signature",
    "is_valid_spec",
]


# Power-of-two vector widths the DSL's ``global_load_vN`` / packed
# stores understand. EPT == 1 (head_size=64) and EPT == 3 (head_size=
# 192) fall back to per-element scalar paths so every supported head
# size keeps working.
_VEC_WIDTHS = (2, 4, 8)


def _store_lane_slice_f32_packed(
    b: IRBuilder,
    ptr: Value,
    row_base: Value,
    lane_d_base: Value,
    values_f32: list[Value],
    *,
    dtype: str,
    ept: int,
) -> None:
    """One packed ``global_store_vN`` for the final O tile when EPT
    is a supported vector width; per-element scalar stores otherwise.

    For EPT in {2, 4, 8} we pack the per-lane f32 outputs through
    :func:`pack_f32_to` (which lowers to a single ``vec_pack`` after
    each value's ``cast_f32_to`` trunc to the target dtype) and emit
    one ``memref.global_store_vN`` -- AMDGPU lowers this to a single
    ``buffer_store_dword{x2,x4}`` for the standard 4/8/16-byte
    payloads.
    """
    if ept in _VEC_WIDTHS:
        packed = pack_f32_to(b, values_f32, dtype=dtype)
        store_vec(b, ptr, b.add(row_base, lane_d_base), packed, n=ept)
        return
    for k in range(ept):
        store_scalar_from_f32(
            b,
            ptr,
            b.add(row_base, b.add(lane_d_base, b.const_i32(k))),
            values_f32[k],
            dtype=dtype,
        )


@dataclass(frozen=True)
class FmhaFwdSplitKvDecodeSpec:
    common: FmhaCommonSpec
    batch: int
    num_segments: int
    name: str = "rocke_fmha_fwd_splitkv_decode"
    # P68: GQA-aware MFMA grid. When True, the segment kernel groups
    # GQA heads into a BLOCK_M tile (BLOCK_Q = BLOCK_M /
    # num_queries_per_kv Q-token rows per CTA) and runs the MFMA
    # body on the (Q-token-bunch, kv_head) plane. Requires the spec's
    # ``num_queries_per_kv`` field to evenly divide ``BLOCK_M = 16``.
    use_mfma_body: bool = False
    # P71: per-head sliding-window tile pruning. When True, the
    # segment kernel computes ``first_allowed_key = context_len +
    # qpos_lo - SLIDING_WINDOW + 1`` and short-circuits K-tiles
    # outside the band. Requires ``common.sliding_window > 0``.
    prune_sliding_window: bool = False

    def kernel_name(self, phase: str) -> str:
        s = self.common.shape
        return kernel_name_join(
            self.name,
            phase,
            f"H{s.head_size}",
            f"HQ{s.num_query_heads}",
            f"HK{s.num_kv_heads}",
            self.common.dtype,
            f"B{self.batch}",
            f"S{self.num_segments}",
        )


def is_valid_spec(
    spec: FmhaFwdSplitKvDecodeSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
    if not ok:
        return False, why
    # Both segment + reduce kernels use the warp-distributed scalar body
    # (no MFMA), but validate the dtype's narrow atom against the target
    # catalog so the predicate stays arch-aware and consistent with the
    # sibling FMHA instances. The 16x16x16 f16/bf16 atom is on both
    # gfx942 and gfx950, so valid dtypes pass on both.
    ok, why = validate_fmha_mfma_atom(spec.common.dtype, arch)
    if not ok:
        return False, why
    if spec.batch <= 0:
        return False, f"batch must be > 0 (got {spec.batch})"
    if spec.num_segments not in (1, 2, 4, 8, 16, 32, 64, 128):
        return False, (f"num_segments {spec.num_segments} not in {{1, 2, ..., 128}}")
    return True, "ok"


def _declare_segment_params(kb: FmhaKernelBuilder) -> None:
    kb.add_tensor("Q", readonly=True)
    kb.add_tensor("K", readonly=True)
    kb.add_tensor("V", readonly=True)
    kb.add_ptr("seqlens_k", dtype="i32", readonly=True)
    kb.add_ptr("ws_m", dtype="f32", readonly=False)
    kb.add_ptr("ws_l", dtype="f32", readonly=False)
    kb.add_ptr("ws_acc", dtype="f32", readonly=False)
    kb.add_scalar("scale_log2", "f32")
    kb.add_scalar("batch", "i32")
    # Q is (batch, head, dim) for decode -- only the seq stride matters,
    # but use the standard token/head pair so the builder's row_base
    # helpers do the math.
    kb.add_scalar("stride_q_seq", "i32")
    kb.add_scalar("stride_q_head", "i32")
    kb.add_strides("k", "v")


def _declare_reduce_params(kb: FmhaKernelBuilder) -> None:
    kb.add_ptr("ws_m", dtype="f32", readonly=True)
    kb.add_ptr("ws_l", dtype="f32", readonly=True)
    kb.add_ptr("ws_acc", dtype="f32", readonly=True)
    kb.add_tensor("O", readonly=False, writeonly=True)
    kb.add_scalar("batch", "i32")
    kb.add_scalar("stride_o_seq", "i32")
    kb.add_scalar("stride_o_head", "i32")


def build_fmha_fwd_splitkv_decode_segment(
    spec: FmhaFwdSplitKvDecodeSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Segment kernel: one CTA per ``(seq_idx, head_idx, segment_idx)``."""
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid splitkv_decode spec: {why}")
    s = spec.common
    H = s.shape.head_size
    if H % WARP_SIZE != 0:
        raise ValueError(
            f"splitkv_decode warp body needs H % {WARP_SIZE} == 0; got {H}"
        )
    ept = H // WARP_SIZE

    kb = FmhaKernelBuilder(spec.kernel_name("seg"), s)
    kb.block_size(WARP_SIZE)
    _declare_segment_params(kb)
    b = kb.builder

    Q = kb.tensor("Q")
    K = kb.tensor("K")
    V = kb.tensor("V")
    seqlens_k_ptr = kb.ptr("seqlens_k")
    ws_m = kb.ptr("ws_m")
    ws_l = kb.ptr("ws_l")
    ws_acc = kb.ptr("ws_acc")
    scale_log2 = kb.scalar("scale_log2")
    stride_q_seq = kb.scalar("stride_q_seq")
    stride_q_head = kb.scalar("stride_q_head")

    seq_idx = b.block_id_x()
    head_idx = b.block_id_y()
    segment_idx = b.block_id_z()
    nqkv = s.shape.num_queries_per_kv
    # GQA head -> kv-head map: ``kv_head = head // num_queries_per_kv``.
    # ``num_queries_per_kv`` need not be a power of two (e.g. 6 in some
    # GQA layouts), so the pow-2 right-shift fast path is unavailable.
    # Instead of a hardware integer divide, emit CK Tile's magic-division
    # mul-hi sequence (``do_magic_division``; the device lowering of
    # ``merge_v2_magic_division``): the divisor is a compile-time constant
    # so ``(multiplier, shift)`` fold to immediates and the runtime cost is
    # one ``v_mul_hi_u32`` + add + shift instead of the ~20-cycle integer
    # divider. ``head_idx`` is a grid axis (0..num_query_heads-1), well
    # inside the documented 31-bit unsigned range of the magic sequence.
    nqkv_mult, nqkv_shift = calculate_magic_numbers(nqkv)
    kv_head_idx = do_magic_division(b, head_idx, nqkv_mult, nqkv_shift)

    # ``num_segments`` is validated as a positive power of two
    # ({1, 2, 4, ..., 128}); replace ``seqlen_k / NUM_SEG`` with a
    # right-shift -- one VALU op instead of the integer divider's
    # ~20-cycle Newton-Raphson sequence on AMDGPU.
    num_seg = int(spec.num_segments)
    assert (num_seg & (num_seg - 1)) == 0, "num_segments must be a power of two"
    num_seg_log2 = num_seg.bit_length() - 1
    seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx)
    seg_len_base = b.lshr(seqlen_k, b.const_i32(num_seg_log2))
    seg_start = b.mul(segment_idx, seg_len_base)
    seg_end_raw = b.add(seg_start, seg_len_base)
    seg_end = b.select(
        b.cmp_ge(b.add(segment_idx, b.const_i32(1)), b.const_i32(num_seg)),
        seqlen_k,
        seg_end_raw,
    )

    q_row = b.add(b.mul(seq_idx, stride_q_seq), b.mul(head_idx, stride_q_head))
    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))

    # Hoist the loop-invariant ``kv_head_idx * stride_head + lane_d_base``
    # K/V row offsets out of the K-loop. The DSL optimizer does not run
    # LICM (core/passes.py: CSE/DCE only, no code motion across the scf.for
    # boundary), so without this the inner loop recomputes the head term
    # and the lane offset on every K-iteration -- the dominant SALU cost
    # in the warp-scalar body (probe_isa_inspect showed salu >> valu). The
    # per-iter address math collapses to one ``k_idx * stride_token + base``
    # add. ``lane_d_base`` is folded in too so the load helper's
    # ``row_base + lane_d_base`` add is also hoisted.
    kv_head_k_off = b.add(b.mul(kv_head_idx, kb.stride_head("k")), lane_d_base)
    kv_head_v_off = b.add(b.mul(kv_head_idx, kb.stride_head("v")), lane_d_base)
    stride_k_tok = kb.stride_token("k")
    stride_v_tok = kb.stride_token("v")

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # One vectorised global load for this lane's Q slice (constant
    # across the K-loop, lives in registers thereafter).
    q_lane = load_lane_slice_f32(b, Q, q_row, lane_d_base, dtype=s.dtype, ept=ept)

    iter_args = [("m", neg_inf), ("l", zero_f)]
    for k in range(ept):
        iter_args.append((f"a{k}", zero_f))

    k_loop = b.scf_for_iter(
        seg_start,
        seg_end,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="k_idx",
    )
    with k_loop as (k_idx, state_vals):
        m, l = state_vals[0], state_vals[1]  # noqa: E741 - online-softmax (m,l) state
        acc_iter = state_vals[2:]
        # Address math hoisted: only ``k_idx * stride_token`` varies per
        # iter; the head + lane terms are folded into ``kv_head_*_off``
        # outside the loop. ``load_lane_slice_f32`` takes a zero lane
        # offset because the lane term is already baked into the base.
        k_row = b.add(b.mul(k_idx, stride_k_tok), kv_head_k_off)
        v_row = b.add(b.mul(k_idx, stride_v_tok), kv_head_v_off)
        # Vectorised per-lane K / V loads. Same pattern AITER uses
        # for the decode KV stream: one ``tl.load(... mask=...)`` for
        # the whole HEAD_SIZE_PADDED slice per K position (the K-loop
        # there has the same shape -- iterate over keys, accumulate
        # online softmax in registers).
        k_lane = load_lane_slice_f32(
            b, K, k_row, b.const_i32(0), dtype=s.dtype, ept=ept
        )
        v_lane = load_lane_slice_f32(
            b, V, v_row, b.const_i32(0), dtype=s.dtype, ept=ept
        )
        partial = zero_f
        for k in range(ept):
            # fma => guaranteed single v_fma_f32 per term (one rounding,
            # one issue slot) for the per-lane QK dot partial.
            partial = b.fma(q_lane[k], k_lane[k], partial)
        dot = warp_xor_reduce_sum(b, partial, stages=6)
        score_log2 = b.fmul(dot, scale_log2)
        # Decode-time causal: query is one new token at position seqlen_k - 1.
        # Mask isn't typically needed during a single-token decode; preserved
        # here for ABI consistency with the forward variants.
        score_log2 = apply_attention_mask(
            b,
            score_log2,
            mask_mode=s.mask_mode,
            k_idx=k_idx,
            query_pos=b.const_i32(0),
            sliding_window=s.sliding_window,
        )

        m_new = b.fmax(m, score_log2)
        alpha = b.exp2(b.fsub(m, m_new))
        p = b.exp2(b.fsub(score_log2, m_new))
        l_new = b.fma(l, alpha, p)
        new_yields = [m_new, l_new]
        for k in range(ept):
            # acc' = acc*alpha + p*v  => one rescale mul + one fma
            # (single rounding on the p*v term, one fewer issue slot).
            new_yields.append(b.fma(p, v_lane[k], b.fmul(acc_iter[k], alpha)))
        b.scf_yield(*new_yields)

    m_final = k_loop.results[0]
    l_final = k_loop.results[1]
    acc_final = list(k_loop.results[2:])

    # ``seg_stride = num_query_heads * batch`` and ``H`` (head_size)
    # are compile-time constants. The workspace index expression
    # ``(seg * NUM_QH * BATCH + seq * NUM_QH + head)`` doesn't
    # simplify further (NUM_QH is not always a power of two when
    # GQA factors are used), but ``ws_idx * H`` does -- head_size
    # is in {32, 64, 128, 192, 256} and the ones the warp body
    # supports (H % WARP_SIZE == 0) are {64, 128, 192, 256}. For the
    # power-of-two cases we collapse the multiply to a left-shift.
    seg_stride = b.mul(
        b.const_i32(s.shape.num_query_heads),
        b.const_i32(spec.batch),
    )
    ws_idx = b.add(
        b.mul(segment_idx, seg_stride),
        b.add(b.mul(seq_idx, b.const_i32(s.shape.num_query_heads)), head_idx),
    )
    if (H & (H - 1)) == 0:
        ws_idx_acc_base = b.shl(ws_idx, b.const_i32(H.bit_length() - 1))
    else:
        ws_idx_acc_base = b.mul(ws_idx, b.const_i32(H))
    is_lead = b.cmp_eq(tid, b.const_i32(0))
    with b.scf_if(is_lead):
        b.global_store(ws_m, ws_idx, m_final, align=4)
        b.global_store(ws_l, ws_idx, l_final, align=4)
    # ``ws_acc`` is f32 and the DSL's ``global_store_vN`` only covers
    # 16-bit vector payloads on the global-memory surface (no
    # ``buffer_store_dwordxN`` wrapper for f32 is exposed today). The
    # per-lane scalar stores already coalesce into a single VMEM
    # transaction per cache line through the AMDGPU memory controller
    # -- this is the same shape AITER's ``segm_output`` writeback uses
    # in the 3D segment kernel of
    # ``aiter.ops.triton.attention.unified_attention``.
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        b.global_store(
            ws_acc,
            b.add(ws_idx_acc_base, d),
            acc_final[k],
            align=4,
        )

    b.ret()
    return kb.kernel


def build_fmha_fwd_splitkv_decode_reduce(
    spec: FmhaFwdSplitKvDecodeSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Reduce kernel: combine per-segment ``(m, l, acc)`` into ``O``."""
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid splitkv_decode spec: {why}")
    s = spec.common
    H = s.shape.head_size
    if H % WARP_SIZE != 0:
        raise ValueError(f"splitkv_decode reduce needs H % {WARP_SIZE} == 0; got {H}")
    ept = H // WARP_SIZE

    kb = FmhaKernelBuilder(spec.kernel_name("reduce"), s)
    kb.block_size(WARP_SIZE)
    _declare_reduce_params(kb)
    b = kb.builder

    ws_m = kb.ptr("ws_m")
    ws_l = kb.ptr("ws_l")
    ws_acc = kb.ptr("ws_acc")
    O = kb.tensor("O")  # noqa: E741 - standard attention notation (Q,K,V,O)
    stride_o_seq = kb.scalar("stride_o_seq")
    stride_o_head = kb.scalar("stride_o_head")

    seq_idx = b.block_id_x()
    head_idx = b.block_id_y()
    seg_stride = b.mul(
        b.const_i32(s.shape.num_query_heads),
        b.const_i32(spec.batch),
    )
    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))

    # Match the AITER ``reduce_segments`` (and CK DSL
    # ``attention_tiled_3d.build_unified_attention_reduce_tiled``)
    # numerically-stable two-pass form: pass 1 finds the overall max
    # across segments, pass 2 sums ``l_seg * select(m_seg > -inf,
    # exp2(m_seg - overall_max), 0)`` to get the global denominator,
    # pass 3 (per-lane d slot) computes the rescaled acc and writes
    # ``O[d] = select(overall_expsum == 0, 0, acc[d] * rcp(...))``.
    # The previous streaming online-softmax merge worked for the
    # generic case but produced NaN whenever every segment for one
    # (seq, head) was masked off (l == 0 -> rcp(0) = +inf -> 0 * inf
    # = NaN at the trunc-to-f16 step).
    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)

    base_ml = b.add(
        b.mul(seq_idx, b.const_i32(s.shape.num_query_heads)),
        head_idx,
    )

    # Pass 1: overall_max.
    mx_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("mx", neg_inf)],
        iv_name="s_mx",
    )
    with mx_loop as (sv, (mx,)):
        ws_idx = b.add(b.mul(sv, seg_stride), base_ml)
        ms = b.global_load_f32(ws_m, ws_idx)
        b.scf_yield(b.fmax(mx, ms))
    overall_max = mx_loop.results[0]

    # Pass 2: overall_expsum (NaN-safe factor for masked-off segments).
    sum_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("den", zero_f)],
        iv_name="s_sum",
    )
    with sum_loop as (sv, (den,)):
        ws_idx = b.add(b.mul(sv, seg_stride), base_ml)
        ms = b.global_load_f32(ws_m, ws_idx)
        ls = b.global_load_f32(ws_l, ws_idx)
        ms_finite = b.fcmp("ogt", ms, neg_inf)
        factor_raw = b.exp2(b.fsub(ms, overall_max))
        factor = b.select(ms_finite, factor_raw, zero_f)
        b.scf_yield(b.fadd(den, b.fmul(ls, factor)))
    overall_expsum = sum_loop.results[0]
    safe_expsum = b.fcmp("oeq", overall_expsum, zero_f)
    inv_l = b.select(safe_expsum, zero_f, b.rcp(overall_expsum))

    # Pass 3: per-lane reduce + normalise + write. The per-segment
    # acc loads stay scalar f32 because the DSL's ``global_load_vN``
    # surface doesn't expose dword-vector loads for f32 today; the
    # per-lane fan-out is small (EPT = head_size / 64) and the loop
    # collapses naturally across the segment dimension.
    if H % WARP_SIZE == 0 and (H & (H - 1)) == 0:
        shift = H.bit_length() - 1
        ws_idx_acc_base_fn = lambda sv: b.shl(  # noqa: E731 - inline helper for clarity
            b.add(b.mul(sv, seg_stride), base_ml), b.const_i32(shift)
        )
    else:
        ws_idx_acc_base_fn = lambda sv: b.mul(  # noqa: E731
            b.add(b.mul(sv, seg_stride), base_ml), b.const_i32(H)
        )

    acc_per_lane = []
    for k in range(ept):
        acc_loop = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(spec.num_segments),
            b.const_i32(1),
            [(f"ac{k}", zero_f)],
            iv_name=f"s_acc{k}",
        )
        with acc_loop as (sv, (ac,)):
            ws_idx = b.add(b.mul(sv, seg_stride), base_ml)
            ms = b.global_load_f32(ws_m, ws_idx)
            ms_finite = b.fcmp("ogt", ms, neg_inf)
            factor_raw = b.exp2(b.fsub(ms, overall_max))
            factor = b.select(ms_finite, factor_raw, zero_f)
            d = b.add(lane_d_base, b.const_i32(k))
            ov = b.global_load_f32(ws_acc, b.add(ws_idx_acc_base_fn(sv), d))
            b.scf_yield(b.fadd(ac, b.fmul(ov, factor)))
        acc_per_lane.append(b.fmul(acc_loop.results[0], inv_l))

    o_row = b.add(b.mul(seq_idx, stride_o_seq), b.mul(head_idx, stride_o_head))
    # Final O write: one ``buffer_store_dwordxN`` for EPT in {2, 4, 8}
    # (after the per-lane f32 list is trunc-cast to the target dtype
    # and packed into one vector); per-element scalar stores for the
    # corner cases. Mirrors AITER's final ``tl.store(out_ptr, acc)``
    # vector write at the bottom of ``reduce_segments``.
    _store_lane_slice_f32_packed(
        b, O, o_row, lane_d_base, acc_per_lane, dtype=s.dtype, ept=ept
    )

    b.ret()
    return kb.kernel


def fmha_fwd_splitkv_decode_segment_grid(
    spec: FmhaFwdSplitKvDecodeSpec,
) -> Tuple[int, int, int]:
    return (spec.batch, spec.common.shape.num_query_heads, spec.num_segments)


def fmha_fwd_splitkv_decode_reduce_grid(
    spec: FmhaFwdSplitKvDecodeSpec,
) -> Tuple[int, int, int]:
    return (spec.batch, spec.common.shape.num_query_heads, 1)


def fmha_fwd_splitkv_decode_segment_signature(spec: FmhaFwdSplitKvDecodeSpec):
    kb = FmhaKernelBuilder(
        "rocke_fmha_fwd_splitkv_decode_seg_sig_probe",
        spec.common,
    )
    _declare_segment_params(kb)
    return kb.signature()


def fmha_fwd_splitkv_decode_reduce_signature(spec: FmhaFwdSplitKvDecodeSpec):
    kb = FmhaKernelBuilder(
        "rocke_fmha_fwd_splitkv_decode_reduce_sig_probe",
        spec.common,
    )
    _declare_reduce_params(kb)
    return kb.signature()
