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
* **No thread redundancy.** Each lane reads its own ``EPT``-element
  slice of K / V per K-step (one coalesced vmem transaction for
  EPT >= 2, one scalar load for EPT == 1) and contributes ``EPT``
  multiplies to the per-lane QK partial dot.
* **Vector primitives on the per-lane slice.** For EPT >= 2 the
  per-lane Q is pre-loaded as a single ``<EPT x f16>`` vector
  (promoted to ``<EPT x f32>``), the K and V loads are a single
  ``global_load_vN`` each, the QK partial dot is one
  ``vector_mul`` + ``vector_sum``, the PV accumulator is a single
  ``<EPT x f32>`` iter_arg updated with ``vector_mul`` /
  ``vector_add``, and the epilogue does one ``global_store_vN``.
  Mirrors the per-lane-slice pattern in CK Tile's
  ``BlockFmhaPipelineQRKSVS`` Q-row tile (registers, not LDS).

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

from ...core.ir import F32, IRBuilder, Value
from ...helpers.attention import (
    causal_mask,
    sliding_window_mask,
    warp_xor_reduce_sum,
)
from ...helpers.io import (
    io_ir_type,
    load_scalar_as_f32,
    load_vec_as_f32,
    store_scalar_from_f32,
)


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
    kv_lane_loader=None,
    q_lane_loader=None,
) -> None:
    """One warp's worth of FMHA forward for one ``(q_token, head)`` row.

    ``extra_score_transform`` (if set) is a callable
    ``(b, score_log2, k_idx) -> score_log2`` invoked after the
    QK reduction and before the softmax update. Used by the sage
    attention path to apply per-block Q+K scales.

    ``kv_lane_loader`` (if set) overrides the per-lane K/V slice load.
    It is a callable ``(b, k_idx, k_row_base, v_row_base, lane_d_base,
    ept) -> (k_f32_list, v_f32_list)`` returning two Python lists of
    ``ept`` f32 :class:`Value` (this lane's head-dim slice of K and V,
    already dequantised to f32). The default (``None``) reads the slice
    directly from ``K`` / ``V`` via :func:`load_scalar_as_f32` /
    :func:`load_vec_as_f32`. Sage attention supplies a loader that
    dequantises fp8 / i8 / i4 codebook K/V. When set, the body uses the
    generalised list path (per-slot scalar iter_args + scalar epilogue
    stores) regardless of ``ept`` so any per-lane addressing the loader
    owns (e.g. packed-i4 bytes) is respected.

    ``q_lane_loader`` (if set) overrides the per-lane Q slice load. It is
    a callable ``(b, q_row_base, lane_d_base, ept) -> q_f32_list``
    returning a Python list of ``ept`` f32 :class:`Value`. Only consulted
    when ``kv_lane_loader`` is also set (the list path). Defaults to the
    built-in Q load.

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

    dtype_ir = io_ir_type(dtype)
    tid = b.thread_id_x()
    c_ept = b.const_i32(ept)
    lane_d_base = b.mul(tid, c_ept)  # tid * EPT

    q_row_base = b.add(b.mul(q_token, stride_q_token), b.mul(head_idx, stride_q_head))
    o_row_base = b.add(b.mul(q_token, stride_o_token), b.mul(head_idx, stride_o_head))

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # The lane-local head-dim slice base address. Used by both the
    # scalar (ept==1) and vector (ept>=2) variants below.
    q_lane_addr = b.add(q_row_base, lane_d_base)
    o_lane_addr = b.add(o_row_base, lane_d_base)

    # Build the K / V row base computation as a closure so the two
    # body variants share the addressing logic verbatim. Mirrors
    # CK Tile's ``make_tile_window(..., k_dram_window)`` move pattern.
    def _row_bases(k_idx: Value):
        if k_row_base_fn is not None:
            kbase = k_row_base_fn(b, k_idx)
        else:
            kbase = b.add(
                b.add(
                    b.mul(k_idx, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )
        if v_row_base_fn is not None:
            vbase = v_row_base_fn(b, k_idx)
        else:
            vbase = b.add(
                b.add(
                    b.mul(k_idx, stride_v_token),
                    b.mul(kv_head_idx, stride_v_head),
                ),
                v_off,
            )
        return kbase, vbase

    def _apply_mask_and_score(score_log2: Value, k_idx: Value) -> Value:
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
        return score_log2

    if kv_lane_loader is not None:
        # ------------------------------------------------------------------
        # Generalised list path (custom per-lane loaders).
        #
        # The K/V (and optionally Q) slices come from a caller-supplied
        # dequant loader returning ``ept`` f32 values per lane. The QK
        # partial dot, online-softmax update and PV accumulator are kept
        # in the scalar-per-slot shape (``ept`` accumulator iter_args +
        # ``ept`` scalar epilogue stores) so the loader fully owns the
        # K/V addressing (e.g. packed-i4 bytes that do not map linearly
        # onto ``lane_d_base``). Used by the sage attention warp body.
        # ------------------------------------------------------------------
        if q_lane_loader is not None:
            q_f32_list = q_lane_loader(b, q_row_base, lane_d_base, ept)
        else:
            q_f32_list = load_vec_as_f32(b, Q, q_lane_addr, dtype=dtype, n=ept)

        iter_args = [("m", neg_inf), ("l", zero_f)]
        for slot in range(ept):
            iter_args.append((f"a{slot}", zero_f))

        k_loop = b.scf_for_iter(
            b.const_i32(0),
            seqlen_k,
            b.const_i32(1),
            iter_args=iter_args,
            iv_name="k_idx",
        )
        with k_loop as (k_idx, state_vals):
            m, lse = state_vals[0], state_vals[1]
            acc_iter = state_vals[2:]
            k_row_base, v_row_base = _row_bases(k_idx)

            k_f32_list, v_f32_list = kv_lane_loader(
                b, k_idx, k_row_base, v_row_base, lane_d_base, ept
            )

            partial = b.const_f32(0.0)
            for slot in range(ept):
                partial = b.fadd(partial, b.fmul(q_f32_list[slot], k_f32_list[slot]))

            dot = warp_xor_reduce_sum(b, partial, stages=6)
            score_log2 = _apply_mask_and_score(b.fmul(dot, scale_log2), k_idx)

            m_new = b.fmax(m, score_log2)
            alpha = b.exp2(b.fsub(m, m_new))
            p = b.exp2(b.fsub(score_log2, m_new))
            lse_new = b.fadd(b.fmul(lse, alpha), p)

            new_yields = [m_new, lse_new]
            for slot in range(ept):
                new_yields.append(
                    b.fadd(
                        b.fmul(acc_iter[slot], alpha),
                        b.fmul(p, v_f32_list[slot]),
                    )
                )
            b.scf_yield(*new_yields)

        results = k_loop.results
        l_final = results[1]
        acc_final = list(results[2:])
        inv_l = b.rcp(l_final)
        for slot in range(ept):
            store_scalar_from_f32(
                b,
                O,
                b.add(o_lane_addr, b.const_i32(slot)),
                b.fmul(acc_final[slot], inv_l),
                dtype=dtype,
            )
        return

    if ept == 1:
        # Scalar lane path (head_size == warp_size == 64). The vector
        # primitives reject n=1 and there is nothing to vectorise
        # inside a one-element lane slice; this branch keeps the
        # original per-element load/store sequence so the lowering is
        # unchanged for the most common test config.
        q_scalar = load_scalar_as_f32(b, Q, q_lane_addr, dtype=dtype)

        k_loop = b.scf_for_iter(
            b.const_i32(0),
            seqlen_k,
            b.const_i32(1),
            iter_args=[("m", neg_inf), ("l", zero_f), ("a0", zero_f)],
            iv_name="k_idx",
        )
        with k_loop as (k_idx, state_vals):
            m, lse, acc0 = state_vals[0], state_vals[1], state_vals[2]
            k_row_base, v_row_base = _row_bases(k_idx)

            kd = load_scalar_as_f32(b, K, b.add(k_row_base, lane_d_base), dtype=dtype)
            vd = load_scalar_as_f32(b, V, b.add(v_row_base, lane_d_base), dtype=dtype)
            partial = b.fmul(q_scalar, kd)

            # Warp-wide butterfly reduce: 64-lane wave64 sum = 6 stages.
            dot = warp_xor_reduce_sum(b, partial, stages=6)
            score_log2 = _apply_mask_and_score(b.fmul(dot, scale_log2), k_idx)

            m_new = b.fmax(m, score_log2)
            alpha = b.exp2(b.fsub(m, m_new))
            p = b.exp2(b.fsub(score_log2, m_new))
            lse_new = b.fadd(b.fmul(lse, alpha), p)
            new_acc0 = b.fadd(b.fmul(acc0, alpha), b.fmul(p, vd))
            b.scf_yield(m_new, lse_new, new_acc0)

        results = k_loop.results
        l_final = results[1]
        acc0_final = results[2]
        out_f32 = b.fmul(acc0_final, b.rcp(l_final))
        store_scalar_from_f32(b, O, o_lane_addr, out_f32, dtype=dtype)
        return

    # ------------------------------------------------------------------
    # Vector lane path (ept in {2, 4}, head_size in {128, 256}).
    #
    # One coalesced vmem load per lane per Q/K/V (vs ept loads in the
    # scalar version), and one coalesced vmem store per lane for O.
    # The PV accumulator is carried as a single <ept x f32> iter_arg
    # instead of ept scalar iter_args, halving / quartering the
    # iter_arg count and giving the scheduler a single live SSA value
    # for the per-lane partial output.
    # ------------------------------------------------------------------
    q_f32_list = load_vec_as_f32(b, Q, q_lane_addr, dtype=dtype, n=ept)
    q_vec_f32 = b.vec_pack(q_f32_list, F32)

    k_loop = b.scf_for_iter(
        b.const_i32(0),
        seqlen_k,
        b.const_i32(1),
        iter_args=[
            ("m", neg_inf),
            ("l", zero_f),
            ("acc", b.zero_vec_f32(ept)),
        ],
        iv_name="k_idx",
    )
    with k_loop as (k_idx, state_vals):
        m, lse, acc_v = state_vals[0], state_vals[1], state_vals[2]
        k_row_base, v_row_base = _row_bases(k_idx)

        k_lane_addr = b.add(k_row_base, lane_d_base)
        k_f32_list = load_vec_as_f32(b, K, k_lane_addr, dtype=dtype, n=ept)
        k_vec_f32 = b.vec_pack(k_f32_list, F32)

        v_lane_addr = b.add(v_row_base, lane_d_base)
        v_f32_list = load_vec_as_f32(b, V, v_lane_addr, dtype=dtype, n=ept)
        v_vec_f32 = b.vec_pack(v_f32_list, F32)

        # Per-lane partial QK dot via vector ops:
        #   partial = sum_i ( q_vec[i] * k_vec[i] )
        # lowers to one vector.fmul + a tree of fadds (identical to
        # the scalar accumulation but with the per-lane dependence
        # chain made explicit to the scheduler).
        partial = b.vector_sum(b.vector_mul(q_vec_f32, k_vec_f32))

        # Warp-wide butterfly reduce: 64-lane wave64 sum = 6 stages.
        dot = warp_xor_reduce_sum(b, partial, stages=6)
        score_log2 = _apply_mask_and_score(b.fmul(dot, scale_log2), k_idx)

        m_new = b.fmax(m, score_log2)
        alpha = b.exp2(b.fsub(m, m_new))
        p = b.exp2(b.fsub(score_log2, m_new))
        lse_new = b.fadd(b.fmul(lse, alpha), p)

        # PV accumulator update: acc = acc * alpha + p * v.
        # The splats reduce to a single VGPR broadcast each on AMDGPU
        # and the vector_add(vector_mul(...), vector_mul(...)) chain
        # fuses to two v_fma_f32 sequences.
        alpha_v = b.vector_splat(alpha, ept)
        p_v = b.vector_splat(p, ept)
        new_acc_v = b.vector_add(
            b.vector_mul(acc_v, alpha_v),
            b.vector_mul(p_v, v_vec_f32),
        )
        b.scf_yield(m_new, lse_new, new_acc_v)

    # Epilogue: O[lane] = acc / l in target dtype.
    # One vector.fmul + one vector.trunc_f32_to + one
    # memref.global_store_vN -- a single 16/32-bit-vector vmem store
    # vs ept scalar stores in the previous body.
    results = k_loop.results
    l_final = results[1]
    acc_final_v = results[2]

    inv_l = b.rcp(l_final)
    inv_l_v = b.vector_splat(inv_l, ept)
    out_v_f32 = b.vector_mul(acc_final_v, inv_l_v)
    out_v_dtype = b.vec_cast_f32_to(out_v_f32, dtype_ir)
    b.global_store_vN(O, o_lane_addr, out_v_dtype, ept, align=ept * 2)
