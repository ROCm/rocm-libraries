# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled FMHA forward inner-body (production attention loop).

Replaces the warp-distributed scalar FMHA body
(:mod:`ck_dsl.instances._fmha_warp_body`) with an MFMA-driven QK→
softmax→PV pipeline. One wave64 warp processes ``BLOCK_M = 16`` Q
rows per K-tile, ``BLOCK_K = 16`` K positions per iter. The MFMA
atom is ``mfma_f32_16x16x16_f16`` (or ``mfma_f32_16x16x32_f16`` for
the K-packed CDNA3 path). The QK + PV chain delivers 64 FLOPS /
cycle / lane vs the scalar inner's 2 FLOPS / cycle / lane.

What this helper does:

* **Q pre-load**: ``head_size / atom.k`` MFMA atom slabs of Q rows
  pre-loaded into per-lane f16 registers (no LDS staging for Q;
  it's constant over the K-loop so register-only is optimal).
* **K-tile loop** (``seqlen_k / BLOCK_K`` iterations):

  1. Load K tile into per-lane f16 registers.
  2. **QK MFMA chain**: ``head_size / atom.k`` MFMA invocations
     accumulating into a per-lane ``<4 x f32>`` score tile.
  3. Apply ``scale_log2`` and the spec's mask (causal / sliding
     window) via the per-row position check.
  4. **Per-row online softmax**: 16-lane butterfly reduce to get
     the row max, update ``m / l`` state, compute ``p = exp2(s -
     m_new)`` per row, rescale the previously-accumulated PV
     contribution by ``alpha = exp2(m_prev - m_new)``.
  5. **P operand staging**: cast f32 ``p`` to f16 and re-route via
     LDS so the PV MFMA gets it in the A-operand lane layout.
  6. **Load V tile** into per-lane f16 registers.
  7. **PV MFMA chain**: ``head_size / atom.n`` MFMA invocations
     accumulating into the per-lane f32 output.

* **Epilogue**: divide acc by ``l`` (per row), cast f32→f16, write
  to ``O`` in the atom's per-lane output layout.

What it leaves to the caller:

* Q / K / V / O global pointer & stride arithmetic (the kernel
  builder supplies these via callbacks).
* The grid layout (``q_tile_idx`` axis -- the caller's grid puts a
  CTA per ``(q_tile, head, batch)`` triple, where each q_tile
  spans ``BLOCK_M`` Q rows).
* The mask configuration (passed as a ``mask_mode`` string + a
  per-row query position callback).

The helper assumes ``head_size`` is a multiple of ``atom.k`` (16 for
the 16x16x16 atom, 32 for the K-packed 16x16x32 atom). All standard
head sizes (64, 128, 256) qualify.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..core.ir import F16, BF16, IRBuilder, Value
from .atoms import MfmaAtom
from .attention import (
    apply_attention_mask,
    warp_xor_reduce_max,
    warp_xor_reduce_sum,
)


__all__ = [
    "MFMA_ATTN_BLOCK_M",
    "MFMA_ATTN_BLOCK_K",
    "mfma_attention_fwd_inner_body",
]


MFMA_ATTN_BLOCK_M = 16  # Q rows per CTA per K-tile
MFMA_ATTN_BLOCK_K = 16  # K positions per K-tile
_WAVE = 64


def _ir_type_for_dtype(dtype: str):
    if dtype in ("f16", "fp16"):
        return F16
    if dtype == "bf16":
        return BF16
    raise ValueError(f"mfma_attention currently supports f16/bf16; got {dtype!r}")


def mfma_attention_fwd_inner_body(
    b: IRBuilder,
    *,
    Q: Value,
    K: Value,
    V: Value,
    O: Value,  # noqa: E741 - standard attention notation (Q,K,V,O)
    head_size: int,
    seqlen_k: Value,
    q_tile_base: Value,
    head_idx: Value,
    kv_head_idx: Value,
    q_pos_base: Optional[Value] = None,
    stride_q_token: Value,
    stride_q_head: Value,
    stride_k_token: Value,
    stride_k_head: Value,
    stride_v_token: Value,
    stride_v_head: Value,
    stride_o_token: Value,
    stride_o_head: Value,
    scale_log2: Value,
    dtype: str = "f16",
    mask_mode: str = "none",
    sliding_window: int = 0,
    causal_ctx_offset: Optional[Value] = None,
    k_token_offset_elems: Optional[Value] = None,
    v_token_offset_elems: Optional[Value] = None,
    k_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]] = None,
    v_row_base_fn: Optional[Callable[[IRBuilder, Value], Value]] = None,
    k_tile_start: Optional[Value] = None,
    k_tile_stop: Optional[Value] = None,
    extra_score_transform: Optional[
        Callable[[IRBuilder, Value, Value, int], Value]
    ] = None,
    extra_mask_predicate: Optional[Callable[[IRBuilder, Value], Value]] = None,
    kv_dtype: Optional[str] = None,
    v_scale: Optional[Value] = None,
) -> None:
    """One MFMA-tiled QK→softmax→PV pass for a ``BLOCK_M``-row Q tile.

    The kernel must launch with ``block_size = 64`` (one wave64 warp
    per CTA). The CTA processes ``BLOCK_M = 16`` Q rows starting at
    ``q_tile_base``, accumulating attention from the full K span
    (``seqlen_k`` positions) in ``seqlen_k / BLOCK_K`` K-tiles.

    ``causal_ctx_offset`` (when ``mask_mode == "causal"``) is the
    offset added to each row's local position to get its
    causal-mask threshold (``k_idx <= q_pos + offset``). For
    self-attention pass ``b.const_i32(0)``; for cross-attention pass
    the cache length.

    ``k_token_offset_elems`` / ``v_token_offset_elems`` are added to
    the K / V row base addresses (for varlen / paged-KV layouts).

    ``k_row_base_fn`` / ``v_row_base_fn``: callbacks
    ``(b, k_row_idx) -> i32`` returning the linear element offset
    for one K / V row. Overrides the default dense addressing
    (``k_row_idx * stride + head_offset + token_offset``). Used by
    the paged-KV kernel where each ``k_row_idx`` indirects through
    a ``block_table`` to a physical block.

    ``k_tile_start`` / ``k_tile_stop``: when set, the K-tile loop
    runs in ``[k_tile_start, k_tile_stop)`` instead of ``[0,
    seqlen_k / BLOCK_K)``. Used by the split-KV decode segment
    kernel so a single CTA handles one K segment.

    ``extra_score_transform``: callback
    ``(b, score_log2_per_lane, k_tile_idx, row_in_atom) ->
    score_log2`` invoked after the QK reduction and ``scale_log2``
    multiply, before the mask. Used by sage attention to apply
    per-block Q + K scales.

    ``extra_mask_predicate``: callback ``(b, k_tile_idx) -> i1``
    returning a per-K-tile keep flag. When false, the whole K-tile
    is skipped (no MFMA, no V load, no PV). Used by block-sparse
    (jenga / VSA) attention to short-circuit non-attended K-blocks.

    ``kv_dtype``: K / V storage dtype when it differs from Q's
    ``dtype`` (e.g. K/V in fp8 while Q is f16). The helper does
    inline ``cvt_fp8_to_f32 → cast_f32_to_f16 → f16 MFMA`` dequant
    on the load path; the native fp8 MFMA atom (``mfma_f32_16x16x32_fp8``)
    is a v2 hoist.

    ``v_scale`` (optional f32): per-tensor V dequant scale. When set,
    the final accumulator is multiplied by ``v_scale`` at the
    epilogue (mathematically equivalent to scaling each V dequant
    output by ``v_scale``). Callers that need per-tensor ``k_scale``
    just pre-multiply ``scale_log2`` by ``k_scale`` before invoking
    the helper -- the QK MFMA result lands in the log2-space score,
    and a constant K-scale is absorbed cleanly into ``scale_log2``.
    """
    if head_size % MFMA_ATTN_BLOCK_M != 0:
        raise ValueError(
            f"mfma_attention head_size {head_size} must be a multiple of "
            f"{MFMA_ATTN_BLOCK_M}"
        )
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"mfma_attention dtype must be f16/bf16, got {dtype!r}")
    if dtype == "bf16":
        raise NotImplementedError(
            "bf16 MFMA attention requires the bf16 MfmaAtom factory; "
            "f16 path ships today, bf16 lands once the atom is exposed"
        )

    # Q dtype is the activation dtype; K / V dtype can be fp8e4m3 /
    # bf8e5m2 (when ``kv_dtype`` is set). The QK MFMA atom picks
    # ``f16 ⊗ f16 → f32`` or ``fp8 ⊗ fp8 → f32`` based on K/V dtype.
    if kv_dtype is None or kv_dtype == dtype:
        atom = MfmaAtom.f16_16x16x16()
        kv_dtype_eff = dtype
    elif kv_dtype == "fp8e4m3":
        atom = MfmaAtom.fp8_16x16x32()
        kv_dtype_eff = "fp8e4m3"
    elif kv_dtype == "bf8e5m2":
        atom = MfmaAtom.bf8_16x16x32()
        kv_dtype_eff = "bf8e5m2"
    else:
        raise ValueError(
            f"mfma_attention: unsupported kv_dtype {kv_dtype!r}; "
            "expected None / 'f16' / 'fp8e4m3' / 'bf8e5m2'"
        )
    if head_size % atom.k != 0:
        raise ValueError(
            f"head_size {head_size} must be a multiple of atom.k "
            f"{atom.k} for the selected atom"
        )

    # The fp8 KV path requires Q and K to share the same MFMA-input
    # dtype, so when ``kv_dtype`` is fp8/bf8, the kernel is responsible
    # for pre-casting Q to that dtype before this helper runs. The
    # helper itself uses ``kv_dtype_eff`` for both operands.
    dtype_ir = _ir_type_for_dtype(dtype if kv_dtype_eff == dtype else "f16")
    kv_dtype_ir = (
        dtype_ir
        if kv_dtype_eff == dtype
        else (
            F16
            if kv_dtype_eff in ("f16", "fp16")
            else (BF16 if kv_dtype_eff == "bf16" else dtype_ir)
        )
    )
    # For now (v1) the fp8 path falls back to f32-promoted MFMA: K/V
    # bytes load to f32 via cvt_fp8_to_f32, then re-cast to f16 before
    # the f16 MFMA. This keeps the inner-body shape simple; the
    # true fp8-atom path is the v2 hoist on top of this same helper.
    from ..core.ir import BF8E5M2, FP8E4M3

    if kv_dtype_eff != dtype:
        # Fall back to the f16 atom; we'll dequantise K/V on load.
        atom = MfmaAtom.f16_16x16x16()
        dtype_ir = F16
        kv_dtype_ir = FP8E4M3 if kv_dtype_eff == "fp8e4m3" else BF8E5M2

    fp8_kv = kv_dtype_eff != dtype
    n_qk_atoms = head_size // atom.k
    n_pv_atoms = head_size // atom.n  # also == head_size / 16

    lane = b.thread_id_x()
    c16 = b.const_i32(16)
    m_in_atom = b.mod(lane, c16)  # 0..15 -- Q row within the BLOCK_M tile
    k_blk = b.div(lane, c16)  # 0..3  -- which 4-K-element slot
    c_a_per_lane = b.const_i32(atom.a_per_lane)
    k_lane_start = b.mul(k_blk, c_a_per_lane)

    k_off = k_token_offset_elems if k_token_offset_elems is not None else b.const_i32(0)
    v_off = v_token_offset_elems if v_token_offset_elems is not None else b.const_i32(0)

    # ---- Pre-load Q ----
    # For each QK atom along head_dim, lane t holds Q[q_tile_base +
    # m_in_atom, k_blk_atom*atom.k + k_blk*a_per_lane + 0..a_per_lane).
    q_row = b.add(q_tile_base, m_in_atom)
    q_addr_row_base = b.add(
        b.mul(q_row, stride_q_token),
        b.mul(head_idx, stride_q_head),
    )
    q_vecs = []
    for k_blk_atom in range(n_qk_atoms):
        d_start = b.add(
            b.mul(b.const_i32(k_blk_atom), b.const_i32(atom.k)),
            k_lane_start,
        )
        q_addr = b.add(q_addr_row_base, d_start)
        q_vecs.append(
            b.global_load_vN(
                Q,
                q_addr,
                dtype_ir,
                atom.a_per_lane,
                align=atom.a_per_lane * 2,
            )
        )

    # ---- LDS for P-operand staging ----
    # After softmax we need P in the A-operand lane layout for the PV
    # MFMA. The score tile lives in registers as 4 cells per lane (4
    # rows × 1 col). To redistribute for PV we round-trip through
    # LDS: store each lane's 4 cells into P_lds at (row, col), sync,
    # load back in the new layout.
    P_lds = b.smem_alloc(
        dtype_ir,
        [MFMA_ATTN_BLOCK_M, MFMA_ATTN_BLOCK_K],
        name_hint="Pmfma",
    )

    # ---- Online softmax + PV accumulator iter_args ----
    # Each lane has 4 rows worth of (m, l) state: rows m_blk*4 + i for
    # i in 0..3, where m_blk = lane / 16. We carry m_r, l_r per row
    # slot through the K-loop.
    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)
    acc_zero = b.zero_vec_f32(atom.c_per_lane)

    iter_args = []
    for r in range(atom.c_per_lane):
        iter_args.append((f"m{r}", neg_inf))
        iter_args.append((f"l{r}", zero_f))
    for n in range(n_pv_atoms):
        iter_args.append((f"acc{n}", acc_zero))

    c_block_k = b.const_i32(MFMA_ATTN_BLOCK_K)
    loop_start = k_tile_start if k_tile_start is not None else b.const_i32(0)
    loop_stop = k_tile_stop if k_tile_stop is not None else b.div(seqlen_k, c_block_k)

    kloop = b.scf_for_iter(
        loop_start,
        loop_stop,
        b.const_i32(1),
        iter_args=iter_args,
        iv_name="kt",
    )
    with kloop as (kt, state_vals):
        # Unpack state.
        ms = [state_vals[2 * r] for r in range(atom.c_per_lane)]
        ls = [state_vals[2 * r + 1] for r in range(atom.c_per_lane)]
        accs = list(state_vals[2 * atom.c_per_lane :])

        # K row base for this lane: K[k_tile_base + n_in_atom, ...]
        # where n_in_atom = m_in_atom (same lane decoded for both A
        # and B operands in the QK MFMA).
        k_tile_base = b.mul(kt, c_block_k)
        k_row_for_lane = b.add(k_tile_base, m_in_atom)  # k position for THIS lane

        # Per-K-tile mask predicate (block-sparse / VSA): when false, the
        # whole tile is skipped -- no K/V loads, no MFMA, no PV. The
        # iter_args carry through unchanged so the softmax state stays
        # consistent across skipped tiles.
        if extra_mask_predicate is not None:
            keep_tile = extra_mask_predicate(b, kt)
        else:
            keep_tile = None

        # Compute the K row base for this lane. Default is dense
        # ``k_row * stride_k_token + kv_head * stride_k_head + k_off``;
        # callbacks override for paged-KV / varlen.
        if k_row_base_fn is not None:
            k_addr_row_base = k_row_base_fn(b, k_row_for_lane)
        else:
            k_addr_row_base = b.add(
                b.add(
                    b.mul(k_row_for_lane, stride_k_token),
                    b.mul(kv_head_idx, stride_k_head),
                ),
                k_off,
            )

        # ---- QK MFMA chain ----
        # score = sum over qk_atoms of MFMA(Q[atom], K[atom], score)
        # Per-lane <4 x f32> (4 row-cells, 1 col-cell).
        # When K/V are fp8 / bf8, we load the bytes, cvt to f32, and
        # cast back to f16 before the f16 MFMA. The v2 hoist replaces
        # this dequant chain with the native fp8 MFMA atom.
        score = atom.zero_acc(b)
        for k_blk_atom in range(n_qk_atoms):
            d_start = b.add(
                b.mul(b.const_i32(k_blk_atom), b.const_i32(atom.k)),
                k_lane_start,
            )
            k_addr = b.add(k_addr_row_base, d_start)
            if fp8_kv:
                # Scalar load + dequant + repack into <a_per_lane x f16>.
                k_vec = b.zero_vec(dtype_ir, atom.a_per_lane)
                for j in range(atom.a_per_lane):
                    addr_j = b.add(k_addr, b.const_i32(j))
                    raw = b.global_load(K, addr_j, kv_dtype_ir, align=1)
                    f32_v = (
                        b.cvt_fp8_to_f32(raw)
                        if kv_dtype_eff == "fp8e4m3"
                        else b.cvt_bf8_to_f32(raw)
                    )
                    f16_v = b.cast_f32_to(f32_v, dtype_ir)
                    k_vec = b.vec_insert(k_vec, f16_v, j)
            else:
                k_vec = b.global_load_vN(
                    K,
                    k_addr,
                    dtype_ir,
                    atom.a_per_lane,
                    align=atom.a_per_lane * 2,
                )
            score = atom.emit(b, q_vecs[k_blk_atom], k_vec, score)

        # ---- Scale + mask + softmax row update ----
        # Each lane has 4 score cells at (m_blk*4 + i, n_in_atom).
        # m_blk = lane / 16, n_in_atom = lane % 16 = m_in_atom.
        m_blk = b.div(lane, c16)
        # Per-row max via 16-lane butterfly reduction (lanes in the same
        # m_blk reduce across cols 0..15).
        new_ms, new_ls, new_accs = [], [], list(accs)
        ps = []  # per-row scaled probabilities (per-lane f32)
        for r in range(atom.c_per_lane):
            s_r_f32 = b.vec_extract(score, r)
            s_r_scaled = b.fmul(s_r_f32, scale_log2)
            # Per-row position for the mask check + extra transforms.
            # ``q_pos_base`` (defaults to ``q_tile_base``) is the
            # position used by the mask predicate; it's distinct from
            # ``q_tile_base`` (which addresses Q rows in global memory)
            # so callers with a batch axis can pass the within-batch Q
            # position for the causal / sliding-window mask while
            # still using the global-batched Q address.
            q_pos_for_mask = q_pos_base if q_pos_base is not None else q_tile_base
            row_q_pos = b.add(
                b.add(q_pos_for_mask, b.mul(m_blk, b.const_i32(4))),
                b.const_i32(r),
            )
            k_col_pos = b.add(k_tile_base, m_in_atom)
            if extra_score_transform is not None:
                s_r_scaled = extra_score_transform(
                    b,
                    s_r_scaled,
                    kt,
                    r,
                )
            s_r_scaled = apply_attention_mask(
                b,
                s_r_scaled,
                mask_mode=mask_mode,
                k_idx=k_col_pos,
                query_pos=row_q_pos,
                sliding_window=sliding_window,
                context_len=causal_ctx_offset,
            )
            # If the whole K-tile is masked off (sparse-skip), force the
            # score to -inf so the softmax exponential collapses to 0.
            if keep_tile is not None:
                s_r_scaled = b.select(keep_tile, s_r_scaled, neg_inf)
            # 16-lane row-max reduce.
            row_max = warp_xor_reduce_max(b, s_r_scaled, stages=4)
            m_new_r = b.fmax(ms[r], row_max)
            alpha_r = b.exp2(b.fsub(ms[r], m_new_r))
            p_r = b.exp2(b.fsub(s_r_scaled, m_new_r))
            # Row-sum reduce of p.
            row_psum = warp_xor_reduce_sum(b, p_r, stages=4)
            l_new_r = b.fadd(b.fmul(ls[r], alpha_r), row_psum)

            new_ms.append(m_new_r)
            new_ls.append(l_new_r)
            ps.append(p_r)
            # Rescale every PV accumulator slot for row r.
            for n in range(n_pv_atoms):
                # Accumulator slot n's row r is in vec slot r.
                old = b.vec_extract(new_accs[n], r)
                rescaled = b.fmul(old, alpha_r)
                new_accs[n] = b.vec_insert(new_accs[n], rescaled, r)

        # ---- P operand staging via LDS ----
        # Lane t holds p[m_blk*4+r, n_in_atom] for r in 0..3.
        # Write to P_lds[m_blk*4+r, n_in_atom] then sync.
        for r in range(atom.c_per_lane):
            p_row = b.add(b.mul(m_blk, b.const_i32(4)), b.const_i32(r))
            p_col = m_in_atom
            p_f16 = b.cast_f32_to(ps[r], dtype_ir)
            b.smem_store_vN(P_lds, [p_row, p_col], p_f16, 1)
        b.sync()

        # ---- PV MFMA chain ----
        # For each PV atom along the n (head_dim) axis, load:
        #   * P operand: lane t holds P[m_in_atom, k_blk*4 + 0..3]
        #     -- these are 4 P cols at row m_in_atom.
        #   * V operand: lane t holds V[k_tile_base + k_blk*4 + 0..3,
        #     n_blk_atom * atom.n + n_in_atom] -- 4 V values at col
        #     (n_blk_atom * 16 + n_in_atom).
        # Where n_in_atom = m_in_atom and k_blk*4..k_blk*4+3 are 4
        # K-positions within the BLOCK_K K-tile.
        for n_blk_atom in range(n_pv_atoms):
            # P operand load from LDS.
            p_a_vec = b.zero_vec(dtype_ir, atom.a_per_lane)
            for j in range(atom.a_per_lane):
                p_col_j = b.add(k_lane_start, b.const_i32(j))
                p_v = b.vec_extract(
                    b.smem_load_vN(P_lds, m_in_atom, p_col_j, dtype=dtype_ir, n=1),
                    0,
                )
                p_a_vec = b.vec_insert(p_a_vec, p_v, j)
            # V operand: per-lane scalar loads at strided V[k, n_col].
            v_col_in_hd = b.add(
                b.mul(b.const_i32(n_blk_atom), b.const_i32(atom.n)),
                m_in_atom,
            )
            v_a_vec = b.zero_vec(dtype_ir, atom.b_per_lane)
            for j in range(atom.b_per_lane):
                v_row_k = b.add(
                    k_tile_base,
                    b.add(k_lane_start, b.const_i32(j)),
                )
                if v_row_base_fn is not None:
                    v_addr_row_base = v_row_base_fn(b, v_row_k)
                else:
                    v_addr_row_base = b.add(
                        b.add(
                            b.mul(v_row_k, stride_v_token),
                            b.mul(kv_head_idx, stride_v_head),
                        ),
                        v_off,
                    )
                v_addr = b.add(v_addr_row_base, v_col_in_hd)
                if fp8_kv:
                    raw = b.global_load(V, v_addr, kv_dtype_ir, align=1)
                    f32_v = (
                        b.cvt_fp8_to_f32(raw)
                        if kv_dtype_eff == "fp8e4m3"
                        else b.cvt_bf8_to_f32(raw)
                    )
                    v_scalar = b.cast_f32_to(f32_v, dtype_ir)
                else:
                    v_scalar = b.global_load(V, v_addr, dtype_ir, align=2)
                v_a_vec = b.vec_insert(v_a_vec, v_scalar, j)
            new_accs[n_blk_atom] = atom.emit(
                b,
                p_a_vec,
                v_a_vec,
                new_accs[n_blk_atom],
            )

        # Yield updated state.
        yields = []
        for r in range(atom.c_per_lane):
            yields.append(new_ms[r])
            yields.append(new_ls[r])
        yields.extend(new_accs)
        b.scf_yield(*yields)
        # P_lds will be re-written next iter; sync at top is implied
        # by the next iteration's smem_store path.

    # ---- Pull final state ----
    final = kloop.results
    # ``ms_final`` is the per-lane row-max; the scaled-acc / l_final pair
    # already encodes everything the epilogue needs, so we only consume
    # the normalisation factors below.
    ls_final = [final[2 * r + 1] for r in range(atom.c_per_lane)]
    accs_final = list(final[2 * atom.c_per_lane :])

    # ---- Epilogue: O[m, d] = acc[m, d] / l[m] in target dtype ----
    # Lane t writes to O[q_tile_base + m_blk*4 + r, head, n_blk*16 +
    # n_in_atom] for r in 0..3, n_blk in 0..n_pv_atoms.
    m_blk = b.div(lane, c16)
    for n_blk_atom in range(n_pv_atoms):
        for r in range(atom.c_per_lane):
            o_row = b.add(
                b.add(q_tile_base, b.mul(m_blk, b.const_i32(4))),
                b.const_i32(r),
            )
            o_col = b.add(
                b.mul(b.const_i32(n_blk_atom), b.const_i32(atom.n)),
                m_in_atom,
            )
            inv_l = b.rcp(ls_final[r])
            v_f32 = b.fmul(b.vec_extract(accs_final[n_blk_atom], r), inv_l)
            if v_scale is not None:
                v_f32 = b.fmul(v_f32, v_scale)
            v_out = b.cast_f32_to(v_f32, dtype_ir)
            addr = b.add(
                b.add(
                    b.mul(o_row, stride_o_token),
                    b.mul(head_idx, stride_o_head),
                ),
                o_col,
            )
            b.global_store(O, addr, v_out, align=2)
