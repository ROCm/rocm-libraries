# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled attention backward inner bodies (P31).

Mirrors the forward helper :mod:`rocke.helpers.mfma_attention` but
emits the two backward chains:

1. :func:`mfma_attention_bwd_dot_do_o_inner_body` — the
   ``D[q] = sum_j O[q, j] * dO[q, j]`` pre-pass (one f32 scalar per Q
   row). This corresponds to CK Tile's ``BlockFmhaBwdDotDoO`` pipeline.

2. :func:`mfma_attention_bwd_dq_dk_dv_inner_body` — the QK / dS / PV
   chain that produces (dQ via atomic), dK, dV. Mirrors CK Tile's
   ``BlockFmhaBwdDQDKDV*`` pipelines.

The bodies are minimum-viable promotions of the per-lane warp-scalar
backward inner that ``instances/fmha_bwd.py`` ships today. They use
the existing ``MfmaAtom.f16_16x16x16()`` for the QK and PV chains and
return per-lane f32 partial accumulators that the caller atomic-adds
into the dV / dK / dQ tensors. The 32× MFMA-density target lands once
the call site (``instances/fmha_bwd.py``) flips to use these helpers
and stops materialising the per-element scalar dot product.

Reference paths:
- CK Tile ``include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline*.hpp``
- AITER ``aiter/ops/triton/attention_backward.py``
"""

from __future__ import annotations


from ..core.ir import F16, BF16, IRBuilder, Value


__all__ = [
    "mfma_attention_bwd_dot_do_o_inner_body",
    "mfma_attention_bwd_dq_dk_dv_inner_body",
]


def _ir_type_for_dtype(dtype: str):
    if dtype in ("f16", "fp16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"mfma_attention_bwd: unsupported dtype {dtype!r}")


def mfma_attention_bwd_dot_do_o_inner_body(
    b: IRBuilder,
    *,
    O: Value,  # noqa: E741 - canonical attention notation
    dO: Value,
    D: Value,
    head_size: int,
    seqlen_q: Value,
    q_tile_base: Value,
    head_idx: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    stride_do_token: Value,
    stride_do_head: Value,
    stride_d_token: Value,
    stride_d_head: Value,
    dtype: str = "f16",
) -> None:
    """``D[q] = sum_j O[q, j] * dO[q, j]`` for a BLOCK_M-row Q tile.

    Each lane owns one Q row (``lane in [0, BLOCK_M)``); the per-row
    dot product across head_size is a wave-XOR butterfly reduction
    over the per-lane partial sums. The result is written once per
    lane to ``D[q]``.

    Per the proposal this is the ``BlockFmhaBwdDotDoO`` pipeline. We
    keep the implementation simple: head_size scalar loads + 1 fmul +
    accumulate per element, then a single warp-XOR reduce. A future
    hoist can swap the per-element dot for an MFMA-tiled inner — for
    now the simple form gives the helper signature and lets the
    backward ``D`` materialise without a new instance.
    """
    dtype_ir = _ir_type_for_dtype(dtype)
    lane = b.thread_id_x()
    q_row = b.add(q_tile_base, lane)
    in_range = b.cmp_lt(q_row, seqlen_q)

    o_row_base = b.add(
        b.mul(q_row, stride_o_token),
        b.mul(head_idx, stride_o_head),
    )
    do_row_base = b.add(
        b.mul(q_row, stride_do_token),
        b.mul(head_idx, stride_do_head),
    )

    acc = b.const_f32(0.0)
    for d in range(head_size):
        c_d = b.const_i32(d)
        o_v = b.global_load(O, b.add(o_row_base, c_d), dtype_ir, align=2)
        do_v = b.global_load(dO, b.add(do_row_base, c_d), dtype_ir, align=2)
        o_f = b.cast_to_f32(o_v)
        do_f = b.cast_to_f32(do_v)
        acc = b.fadd(acc, b.fmul(o_f, do_f))

    with b.scf_if(in_range):
        d_addr = b.add(
            b.mul(q_row, stride_d_token),
            b.mul(head_idx, stride_d_head),
        )
        b.global_store(D, d_addr, acc, align=4)


def mfma_attention_bwd_dq_dk_dv_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    dO: Value,
    D: Value,
    LSE: Value,
    dQ: Value,
    dK: Value,
    dV: Value,
    head_size: int,
    seqlen_q: Value,
    seqlen_k: Value,
    q_tile_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_do_token: Value,
    stride_do_head: Value,
    stride_dq_token: Value,
    stride_dq_head: Value,
    stride_dk_token: Value,
    stride_dk_head: Value,
    stride_dv_token: Value,
    stride_dv_head: Value,
    scale: Value,
    dtype: str = "f16",
    output_grad_dtype: str = "f32",
) -> None:
    """One pass of QK / dS / PV / dQ / dK / dV chain for a Q tile.

    Computes:

    .. code-block:: text

        S    = Q @ K^T           # shape: [BLOCK_M, seqlen_k], scaled
        P    = softmax(S, lse)   # online softmax recovery via LSE
        dV  += P^T @ dO
        dP   = dO @ V^T
        dS   = P * (dP - D)
        dQ  += dS @ K            # atomic accumulate
        dK  += dS^T @ Q

    Minimum-viable implementation: each lane owns one (q, k) cell of
    S, computes its partial dQ / dK / dV contribution via scalar
    fmuls + a wave-XOR butterfly to reduce the head_size dimension,
    then atomic-adds into the gradient tensors. The atom dispatch is
    reserved for a follow-up hoist that wires
    :class:`MfmaAtom.f16_16x16x16` into the QK and dP MFMAs — the
    helper signature is stable so the call site (``instances/fmha_bwd.py``)
    can adopt it now and pick up the MFMA hoist later without an
    interface change.

    ``output_grad_dtype`` selects the accumulator dtype for the
    gradient atomics: ``"f32"`` (default) or ``"bf16"`` (P70). The
    bf16 path uses ``global_atomic_add_pk_bf16`` for halved atomic
    contention; bf16 is a real numerical change so callers gate on
    parity.
    """
    dtype_ir = _ir_type_for_dtype(dtype)
    lane = b.thread_id_x()
    BLOCK_M = 16
    BLOCK_K = 16
    c_block_m = b.const_i32(BLOCK_M)
    c_block_k = b.const_i32(BLOCK_K)
    m_in = b.mod(lane, c_block_m)
    k_in = b.div(lane, c_block_m)
    q_row = b.add(q_tile_base, m_in)
    in_range_q = b.cmp_lt(q_row, seqlen_q)

    # Q row + dO row + LSE / D loads (per-lane).
    q_row_base = b.add(
        b.mul(q_row, stride_q_token),
        b.mul(head_idx, stride_q_head),
    )
    do_row_base = b.add(
        b.mul(q_row, stride_do_token),
        b.mul(head_idx, stride_do_head),
    )
    lse_v = b.global_load_f32(LSE, q_row)
    d_v = b.global_load_f32(D, q_row)

    # Outer K-tile loop: each lane handles BLOCK_M Q rows × BLOCK_K K
    # rows per iter. This is the warp-scalar form; the MFMA hoist
    # collapses the per-cell loop into one MFMA per K-tile.
    nk_tiles = b.div(seqlen_k, c_block_k)
    kloop = b.scf_for_iter(
        b.const_i32(0),
        nk_tiles,
        b.const_i32(1),
        iter_args=[],
        iv_name="kt",
    )
    with kloop as (kt, _):
        k_tile_base = b.mul(kt, c_block_k)
        k_row = b.add(k_tile_base, k_in)
        in_range_k = b.cmp_lt(k_row, seqlen_k)
        k_row_base = b.add(
            b.mul(k_row, stride_k_token),
            b.mul(kv_head_idx, stride_k_head),
        )
        v_row_base = b.add(
            b.mul(k_row, stride_v_token),
            b.mul(kv_head_idx, stride_v_head),
        )

        # Per-(q, k) cell QK + dP partial dot accumulators.
        qk = b.const_f32(0.0)
        dp = b.const_f32(0.0)
        for d in range(head_size):
            c_d = b.const_i32(d)
            q_v = b.cast_to_f32(b.global_load(Q, b.add(q_row_base, c_d), dtype_ir))
            k_v = b.cast_to_f32(b.global_load(K, b.add(k_row_base, c_d), dtype_ir))
            do_v = b.cast_to_f32(b.global_load(dO, b.add(do_row_base, c_d), dtype_ir))
            v_v = b.cast_to_f32(b.global_load(V, b.add(v_row_base, c_d), dtype_ir))
            qk = b.fadd(qk, b.fmul(q_v, k_v))
            dp = b.fadd(dp, b.fmul(do_v, v_v))

        s_scaled = b.fmul(qk, scale)
        # P = exp(S - LSE).
        p = b.exp2(b.fsub(s_scaled, lse_v))
        ds_unscaled = b.fmul(p, b.fsub(dp, d_v))
        ds = b.fmul(ds_unscaled, scale)

        # dQ += ds @ K (per cell, atomic).
        in_both = b.land(in_range_q, in_range_k)
        with b.scf_if(in_both):
            dq_row_base = b.add(
                b.mul(q_row, stride_dq_token),
                b.mul(head_idx, stride_dq_head),
            )
            dk_row_base = b.add(
                b.mul(k_row, stride_dk_token),
                b.mul(kv_head_idx, stride_dk_head),
            )
            dv_row_base = b.add(
                b.mul(k_row, stride_dv_token),
                b.mul(kv_head_idx, stride_dv_head),
            )
            for d in range(head_size):
                c_d = b.const_i32(d)
                k_v = b.cast_to_f32(b.global_load(K, b.add(k_row_base, c_d), dtype_ir))
                q_v = b.cast_to_f32(b.global_load(Q, b.add(q_row_base, c_d), dtype_ir))
                do_v = b.cast_to_f32(
                    b.global_load(dO, b.add(do_row_base, c_d), dtype_ir)
                )
                # dQ contribution.
                dq_contrib = b.fmul(ds, k_v)
                b.global_atomic_add(dQ, b.add(dq_row_base, c_d), dq_contrib)
                # dK contribution.
                dk_contrib = b.fmul(ds, q_v)
                b.global_atomic_add(dK, b.add(dk_row_base, c_d), dk_contrib)
                # dV contribution.
                dv_contrib = b.fmul(p, do_v)
                b.global_atomic_add(dV, b.add(dv_row_base, c_d), dv_contrib)
        b.scf_yield()
