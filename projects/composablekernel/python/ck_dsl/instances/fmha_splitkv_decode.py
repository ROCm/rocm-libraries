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
:mod:`ck_dsl.instances.attention_tiled_3d`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import KernelDef
from ..helpers.attention import (
    OnlineSoftmaxState,
    apply_attention_mask,
    warp_xor_reduce_sum,
)
from ..helpers.io import load_scalar_as_f32, store_scalar_from_f32
from ..helpers.spec import kernel_name_join
from ._fmha_common import FmhaCommonSpec, FmhaKernelBuilder, validate_common_spec
from ._fmha_warp_body import WARP_SIZE


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


@dataclass(frozen=True)
class FmhaFwdSplitKvDecodeSpec:
    common: FmhaCommonSpec
    batch: int
    num_segments: int
    name: str = "ck_dsl_fmha_fwd_splitkv_decode"

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


def is_valid_spec(spec: FmhaFwdSplitKvDecodeSpec) -> Tuple[bool, str]:
    ok, why = validate_common_spec(spec.common)
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
) -> KernelDef:
    """Segment kernel: one CTA per ``(seq_idx, head_idx, segment_idx)``."""
    ok, why = is_valid_spec(spec)
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
    kv_head_idx = b.div(head_idx, b.const_i32(s.shape.num_queries_per_kv))

    seqlen_k = b.global_load_i32(seqlens_k_ptr, seq_idx)
    seg_len_base = b.div(seqlen_k, b.const_i32(spec.num_segments))
    seg_start = b.mul(segment_idx, seg_len_base)
    seg_end_raw = b.add(seg_start, seg_len_base)
    seg_end = b.select(
        b.cmp_ge(b.add(segment_idx, b.const_i32(1)), b.const_i32(spec.num_segments)),
        seqlen_k,
        seg_end_raw,
    )

    q_row = b.add(b.mul(seq_idx, stride_q_seq), b.mul(head_idx, stride_q_head))
    tid = b.thread_id_x()
    lane_d_base = b.mul(tid, b.const_i32(ept))

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    q_lane = []
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        q_lane.append(load_scalar_as_f32(b, Q, b.add(q_row, d), dtype=s.dtype))

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
        k_row = b.add(
            b.mul(k_idx, kb.stride_token("k")),
            b.mul(kv_head_idx, kb.stride_head("k")),
        )
        v_row = b.add(
            b.mul(k_idx, kb.stride_token("v")),
            b.mul(kv_head_idx, kb.stride_head("v")),
        )
        partial = zero_f
        v_lane = []
        for k in range(ept):
            d = b.add(lane_d_base, b.const_i32(k))
            kd = load_scalar_as_f32(b, K, b.add(k_row, d), dtype=s.dtype)
            partial = b.fadd(partial, b.fmul(q_lane[k], kd))
            vd = load_scalar_as_f32(b, V, b.add(v_row, d), dtype=s.dtype)
            v_lane.append(vd)
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
        l_new = b.fadd(b.fmul(l, alpha), p)
        new_yields = [m_new, l_new]
        for k in range(ept):
            new_yields.append(b.fadd(b.fmul(acc_iter[k], alpha), b.fmul(p, v_lane[k])))
        b.scf_yield(*new_yields)

    m_final = k_loop.results[0]
    l_final = k_loop.results[1]
    acc_final = list(k_loop.results[2:])

    seg_stride = b.mul(
        b.const_i32(s.shape.num_query_heads),
        b.const_i32(spec.batch),
    )
    ws_idx = b.add(
        b.mul(segment_idx, seg_stride),
        b.add(b.mul(seq_idx, b.const_i32(s.shape.num_query_heads)), head_idx),
    )
    ws_idx_acc_base = b.mul(ws_idx, b.const_i32(H))
    is_lead = b.cmp_eq(tid, b.const_i32(0))
    with b.scf_if(is_lead):
        b.global_store(ws_m, ws_idx, m_final, align=4)
        b.global_store(ws_l, ws_idx, l_final, align=4)
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
) -> KernelDef:
    """Reduce kernel: combine per-segment ``(m, l, acc)`` into ``O``."""
    ok, why = is_valid_spec(spec)
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

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    iter_args = [("m", neg_inf), ("l", zero_f)]
    for k in range(ept):
        iter_args.append((f"a{k}", zero_f))
    seg_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="seg",
    )
    with seg_loop as (seg, state_vals):
        m, l = state_vals[0], state_vals[1]  # noqa: E741 - online-softmax (m,l) state
        acc_iter = state_vals[2:]
        ws_idx = b.add(
            b.mul(seg, seg_stride),
            b.add(
                b.mul(seq_idx, b.const_i32(s.shape.num_query_heads)),
                head_idx,
            ),
        )
        ws_idx_acc_base = b.mul(ws_idx, b.const_i32(H))
        m_seg = b.global_load_f32(ws_m, ws_idx)
        l_seg = b.global_load_f32(ws_l, ws_idx)
        m_new = b.fmax(m, m_seg)
        alpha = b.exp2(b.fsub(m, m_new))
        beta = b.exp2(b.fsub(m_seg, m_new))
        l_new = b.fadd(b.fmul(l, alpha), b.fmul(l_seg, beta))
        new_yields = [m_new, l_new]
        for k in range(ept):
            d = b.add(lane_d_base, b.const_i32(k))
            ad_seg = b.global_load_f32(ws_acc, b.add(ws_idx_acc_base, d))
            new_yields.append(b.fadd(b.fmul(acc_iter[k], alpha), b.fmul(ad_seg, beta)))
        b.scf_yield(*new_yields)

    l_final = seg_loop.results[1]
    inv_l = b.rcp(l_final)
    acc_final = list(seg_loop.results[2:])
    o_row = b.add(b.mul(seq_idx, stride_o_seq), b.mul(head_idx, stride_o_head))
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        store_scalar_from_f32(
            b,
            O,
            b.add(o_row, d),
            b.fmul(acc_final[k], inv_l),
            dtype=s.dtype,
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
        "ck_dsl_fmha_fwd_splitkv_decode_seg_sig_probe",
        spec.common,
    )
    _declare_segment_params(kb)
    return kb.signature()


def fmha_fwd_splitkv_decode_reduce_signature(spec: FmhaFwdSplitKvDecodeSpec):
    kb = FmhaKernelBuilder(
        "ck_dsl_fmha_fwd_splitkv_decode_reduce_sig_probe",
        spec.common,
    )
    _declare_reduce_params(kb)
    return kb.signature()


_OnlineSoftmaxState = OnlineSoftmaxState  # noqa: F841 - re-export anchor
