# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Warp-distributed FMHA forward inner-body.

This replaces the scalar-per-CTA placeholder that lived in
``_fmha_common.fmha_fwd_inner_body``. Key design choices:

* **One CTA per (q_token, head_idx)**. Grid axes match the previous
  layout so the variant kernels (varlen, paged-prefill, head_grouping,
  sage, sparse) keep their grids unchanged.
* **One warp = ``warp_size`` (64) threads per CTA.** Lane ``t`` owns
  the head-dim slice ``[t * EPT, (t+1) * EPT)`` where
  ``EPT = head_size / warp_size``. For the standard head sizes
  (64/128/256) EPT is 1/2/4.
* **All per-CTA state lives in registers** -- the online-softmax
  scalars ``m`` and ``l`` plus the per-lane accumulator slice of
  ``EPT`` f32 values are threaded through the K-loop as
  :func:`scf_for_iter` ``iter_args``. The old code's LDS scratch
  for ``acc`` / ``state`` is gone; LDS is back to being optional
  bandwidth optimisation, not the only correctness path.
* **The QK dot product reduces across the warp via the existing
  ``warp_xor_reduce_sum`` butterfly** (6 stages for wave64). Every
  lane gets the same dot product after the reduction, so the
  softmax update is identical on every lane and the loop-carried
  ``(m, l)`` stays consistent without any extra synchronisation.
* **No thread redundancy.** Each lane does ``EPT`` element loads
  per K-step, ``EPT`` multiplies for the QK partial dot, ``EPT``
  V loads, and ``EPT`` accumulator FMAs.

What this is NOT:

* MFMA. The QK and PV matmuls still go through scalar FMUL / FADD
  rather than ``mfma_f32_16x16x16_f16``. Lifting to MFMA is a v2
  follow-on that requires the proper 16x16-tile lane layout and
  cshuffle epilogue; the existing :mod:`attention_tiled_2d` kernel
  is the template. The shared spec surface here is compatible with
  that lift -- only the inner body changes.
* Multi-warp. ``num_warps > 1`` would let the CTA cover multiple
  query rows or split the K-loop; that's a v2 follow-on too.

When this body is used:

* ``head_size`` must be divisible by ``warp_size`` (64). Standard
  values 64 / 128 / 256 all qualify.
* The launcher must use ``block=(warp_size, 1, 1)`` -- the kernel's
  IR has no thread-local-id usage beyond the head-dim slice math.
"""

from __future__ import annotations

from typing import Optional

from ..core.ir import IRBuilder, Value
from ..helpers.attention import (
    causal_mask,
    sliding_window_mask,
    warp_xor_reduce_sum,
)
from ..helpers.io import load_scalar_as_f32, store_scalar_from_f32


__all__ = ["WARP_SIZE", "fmha_warp_fwd_inner_body"]


WARP_SIZE = 64


def fmha_warp_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_token: Value,
    head_idx: Value,
    kv_head_idx: Value,
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    dtype: str,
    mask_mode: str = "none",
    sliding_window: int = 0,
    causal_ctx_len: Optional[Value] = None,
    k_token_offset_elems: Optional[Value] = None,
    v_token_offset_elems: Optional[Value] = None,
    extra_score_transform=None,
    extra_mask_predicate=None,
    k_row_base_fn=None,
    v_row_base_fn=None,
) -> None:
    """One warp's worth of FMHA forward for one ``(q_token, head)`` row.

    ``extra_score_transform`` (if set) is a callable
    ``(b, score_log2, k_idx) -> score_log2`` invoked after the
    QK reduction and before the softmax update. Used by the sage
    attention path to apply per-block Q+K scales.

    ``extra_mask_predicate`` (if set) is a callable
    ``(b, k_idx) -> i1`` invoked before the softmax update. When the
    predicate is false, the score is forced to ``-inf`` (i.e. that
    key position is masked out). Used by the sparse-attention paths
    (jenga block-sparse, VSA indirect-LUT) to skip non-attended K
    positions without restructuring the K-loop.

    ``k_row_base_fn`` / ``v_row_base_fn`` (if set) are callables
    ``(b, k_idx) -> i32`` returning the linear element offset for the
    *row base* (everything except the head_dim slot) of K / V at
    logical ``k_idx``. The default is dense linear addressing
    ``k_idx * stride_k_token + kv_head * stride_k_head + k_token_off``.
    Override for paged-KV (where ``k_idx`` indirects through a
    block_table) or any non-linear K addressing.
    """
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"fmha_warp_fwd_inner_body dtype {dtype!r} not supported")
    if head_size % WARP_SIZE != 0:
        raise ValueError(
            f"fmha_warp_fwd_inner_body needs head_size % {WARP_SIZE} == 0; "
            f"got head_size={head_size}"
        )
    ept = head_size // WARP_SIZE

    tid = b.thread_id_x()
    c_ept = b.const_i32(ept)
    lane_d_base = b.mul(tid, c_ept)  # tid * EPT

    q_row_base = b.add(b.mul(q_token, stride_q_token), b.mul(head_idx, stride_q_head))
    o_row_base = b.add(b.mul(q_token, stride_o_token), b.mul(head_idx, stride_o_head))

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    # Pre-load this lane's Q slice (EPT scalars; held in registers).
    q_lane = []
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        q_lane.append(load_scalar_as_f32(b, Q, b.add(q_row_base, d), dtype=dtype))

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    iter_args = [("m", neg_inf), ("l", zero_f)]
    for k in range(ept):
        iter_args.append((f"a{k}", zero_f))

    k_loop = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="k_idx",
    )
    with k_loop as (k_idx, state_vals):
        m, l = state_vals[0], state_vals[1]  # noqa: E741 - online-softmax (m,l) state
        acc_iter = state_vals[2:]

        if k_row_base_fn is not None:
            k_row_base = k_row_base_fn(b, k_idx)
        else:
            k_row_base = b.add(
                b.add(
                    b.mul(k_idx, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )
        if v_row_base_fn is not None:
            v_row_base = v_row_base_fn(b, k_idx)
        else:
            v_row_base = b.add(
                b.add(
                    b.mul(k_idx, stride_v_token),
                    b.mul(kv_head_idx, stride_v_head),
                ),
                v_off,
            )

        # Per-lane partial dot + V load (only the lane's EPT slice).
        partial = b.const_f32(0.0)
        v_lane = []
        for k in range(ept):
            d = b.add(lane_d_base, b.const_i32(k))
            kd = load_scalar_as_f32(
                b,
                K,
                b.add(k_row_base, d),
                dtype=dtype,
            )
            partial = b.fadd(partial, b.fmul(q_lane[k], kd))
            vd = load_scalar_as_f32(
                b,
                V,
                b.add(v_row_base, d),
                dtype=dtype,
            )
            v_lane.append(vd)

        # Warp-wide butterfly reduce: 64-lane wave64 sum = 6 stages.
        dot = warp_xor_reduce_sum(b, partial, stages=6)

        score_log2 = b.fmul(dot, scale_log2)

        if extra_score_transform is not None:
            score_log2 = extra_score_transform(b, score_log2, k_idx)

        if extra_mask_predicate is not None:
            keep_extra = extra_mask_predicate(b, k_idx)
            score_log2 = b.select(keep_extra, score_log2, neg_inf)

        if mask_mode == "causal" and causal_ctx_len is not None:
            keep = causal_mask(b, k_idx, b.const_i32(0), causal_ctx_len)
            score_log2 = b.select(keep, score_log2, neg_inf)
        elif mask_mode == "sliding_window" and causal_ctx_len is not None:
            keep = sliding_window_mask(
                b,
                k_idx,
                b.const_i32(0),
                causal_ctx_len,
                sliding_window,
            )
            score_log2 = b.select(keep, score_log2, neg_inf)

        m_new = b.fmax(m, score_log2)
        alpha = b.exp2(b.fsub(m, m_new))
        p = b.exp2(b.fsub(score_log2, m_new))
        l_new = b.fadd(b.fmul(l, alpha), p)

        new_yields = [m_new, l_new]
        for k in range(ept):
            new_yields.append(b.fadd(b.fmul(acc_iter[k], alpha), b.fmul(p, v_lane[k])))
        b.scf_yield(*new_yields)

    # Pull loop results.
    results = k_loop.results
    l_final = results[1]
    acc_final = list(results[2:])
    inv_l = b.rcp(l_final)
    for k in range(ept):
        d = b.add(lane_d_base, b.const_i32(k))
        out_f32 = b.fmul(acc_final[k], inv_l)
        store_scalar_from_f32(
            b,
            O,
            b.add(o_row_base, d),
            out_f32,
            dtype=dtype,
        )
