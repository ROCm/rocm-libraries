# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Builders for the gated-delta-rule state-scan study.

Phase 2 (this file, so far) is the **GEMM1 probe**: deliberately the least
optimized possible form of

    v_new[BT, BV] = u[BT, BV] - w[BT, K] @ h[BV, K]^T

with padded (unswizzled) LDS and a single chunk, so that the *only* thing under
test is the MFMA operand mapping. That mapping is the study's highest risk: a
transposed operand load is a legal set of addresses and yields silently wrong
numbers rather than an error.

The three mappings, for a 16x16xK atom on wave64 (verified empirically against
a torch matmul before being used here):

===========  ===============================================================
A operand    lane holds ``A[m = lane % 16, k = (lane // 16) * 4 + e]``
B operand    lane holds ``B[k = (lane // 16) * 4 + e, n = lane % 16]``
C / D        lane holds ``D[m = (lane // 16) * 4 + e, n = lane % 16]``
===========  ===============================================================

Here ``B`` is ``h^T``, i.e. ``B[k, v] == h[v, k]``, so the B-operand read is a
run along K at fixed ``v`` — contiguous in the K-major LDS layout, same shape of
access as the A-operand read.

**The C-operand trap.** ``u`` is GEMM1's C operand, so it must be gathered in
the C-fragment layout: four *consecutive* BT rows at a fixed V column, which is
a strided read in a ``[T, V]``-major tensor. A row-contiguous load of the same
tile yields the same element count and stays in bounds — it just computes the
transpose. That is why ``u`` is read with scalar loads keyed off
``atom.lane_to_output`` rather than a vector load.
"""

from __future__ import annotations

from rocke.core.ir import BF16, F32, I16, I32, IRBuilder, PtrType
from rocke.helpers.atoms import mfma_atom

from .spec import GdnStateScanSpec

#: log2(e) — the kernel works in the log2 domain so the gate can use the raw
#: ``v_exp_f32`` (``exp2``) rather than a range-reduced ``exp``.
LOG2E = 1.4426950408889634

#: LDS row padding, in bf16 elements. Phase 2 uses padding rather than an XOR
#: swizzle: padding is the always-correct answer and keeps the operand mapping
#: the only variable under test. Swizzle is a later, numerics-neutral step.
LDS_PAD = 8

#: Global->LDS staging vector width, in bf16 elements (8 * 2 B = 128 bit).
STAGE_VEC = 8


def _stage_tile(b, *, src_ptr, smem, rows, cols, row_stride_src, src_row_base,
                block_threads, tid, pad=0, elem_off=None, clamp=None):
    """Cooperatively stage a ``[rows, cols]`` bf16 tile from global into LDS.

    Slot ``s = it * block_threads + tid`` walks the tile in units of
    ``STAGE_VEC`` contiguous columns, so ``cols // STAGE_VEC`` consecutive tids
    cover one row segment — coalesced on the global side, and contiguous on the
    LDS side (the pad sits at the end of a row, never inside a segment).

    ``elem_off`` is added to the flat source offset (used to select a head).
    ``clamp`` bounds the *source row*: rows at or past it read row 0 instead, so
    a tail chunk never reads out of range. The values are garbage but harmless —
    the tail-chunk row mask (N2) zeroes anything they feed.
    """
    vecs_per_row = cols // STAGE_VEC
    total_slots = rows * vecs_per_row
    assert total_slots % block_threads == 0, (
        f"tile [{rows},{cols}] = {total_slots} vec{STAGE_VEC} slots must tile "
        f"block_threads={block_threads}"
    )
    c_vpr = b.const_i32(vecs_per_row)
    c_vec = b.const_i32(STAGE_VEC)
    c_srcstride = b.const_i32(row_stride_src)

    for it in range(total_slots // block_threads):
        slot = b.add(b.const_i32(it * block_threads), tid)
        row = b.div(slot, c_vpr)
        col = b.mul(b.mod(slot, c_vpr), c_vec)
        src_row = b.add(row, src_row_base) if src_row_base is not None else row
        if clamp is not None:
            src_row = b.select(b.cmp_lt(src_row, clamp), src_row, b.const_i32(0))
        src_off = b.add(b.mul(src_row, c_srcstride), col)
        if elem_off is not None:
            src_off = b.add(src_off, elem_off)
        v = b.global_load_vN(src_ptr, src_off, BF16, STAGE_VEC)
        b.smem_store_vN(smem, [row, col], v, STAGE_VEC)


def _stage_k_transposed(b, *, src, smem, spec, tid, nthreads, t_base, i_hg,
                        T_val, Hg):
    """Stage ``k[BT, K]`` from global into ``sKT[K, BT]`` — a real transpose.

    gfx942 has no ``ds_read_*_tr_*`` (CDNA4 only), so a transposed operand must
    be built on the **store** side. This milestone keeps it simple and correct:
    a coalesced ``vec8`` read along K, then eight scalar LDS writes that scatter
    across eight ``sKT`` rows.

    That is the *slow* form. The FlyDSL kernel instead gives each thread four
    BT-consecutive rows at the same K columns so an in-register transpose turns
    each column's four values into one packed ``ds_write_b64``. Doing that here
    is a later optimization step (it is numerics-neutral); ``b.perm_b32`` is the
    rocKE vehicle for the in-register part.
    """
    BT, K = spec.BT, spec.K
    vecs_per_row = K // STAGE_VEC
    total = BT * vecs_per_row
    assert total % nthreads == 0
    c_vpr = b.const_i32(vecs_per_row)
    for it in range(total // nthreads):
        slot = b.add(b.const_i32(it * nthreads), tid)
        bt = b.div(slot, c_vpr)
        k0 = b.mul(b.mod(slot, c_vpr), b.const_i32(STAGE_VEC))
        t_abs = b.add(t_base, bt)
        t_safe = b.select(b.cmp_lt(t_abs, T_val), t_abs, b.const_i32(0))
        off = b.add(b.mul(b.add(b.mul(t_safe, b.const_i32(Hg)), i_hg),
                          b.const_i32(K)), k0)
        v = b.global_load_vN(src, off, BF16, STAGE_VEC)
        for j in range(STAGE_VEC):
            b.smem_store_vN(smem, [b.add(k0, b.const_i32(j)), bt],
                            b.vec_extract(v, j), 1)


def _drain_h(b, *, sH, dst, spec, tid, nthreads, i_t, i_h, v_base):
    """Drain the ``[BV, K]`` state snapshot from LDS to ``Hout[NT, H, V, K]``.

    All threads cooperate; each handles ``STAGE_VEC`` contiguous K values, which
    are contiguous in both LDS (K-major) and HBM (K innermost).
    """
    BV, K, V, H = spec.BV, spec.K, spec.V, spec.H
    vecs_per_row = K // STAGE_VEC
    total = BV * vecs_per_row
    assert total % nthreads == 0, (
        f"h drain ({total} vec slots) must tile block_threads={nthreads}")
    c_vpr = b.const_i32(vecs_per_row)
    for it in range(total // nthreads):
        slot = b.add(b.const_i32(it * nthreads), tid)
        row = b.div(slot, c_vpr)
        k0 = b.mul(b.mod(slot, c_vpr), b.const_i32(STAGE_VEC))
        v_abs = b.add(v_base, row)
        base = b.add(b.mul(b.add(b.mul(b.add(b.mul(i_t, b.const_i32(H)), i_h),
                                       b.const_i32(V)), v_abs),
                           b.const_i32(K)), k0)
        for j in range(STAGE_VEC):
            val = b.smem_load_vN(sH, row, b.add(k0, b.const_i32(j)),
                                 dtype=BF16, n=1)
            b.global_store(dst, b.add(base, b.const_i32(j)),
                           b.cast_to_f32(b.vec_extract(val, 0)), align=4)


def build_gemm1_probe(spec: GdnStateScanSpec) -> "object":
    """Build the Phase-2 GEMM1 probe kernel.

    Signature (all row-major, one chunk, one head)::

        W    bf16 [BT, K]
        Hst  bf16 [V,  K]     the state, in VK order
        U    bf16 [BT, V]
        OUT  f32  [BT, V]     v_new = u - w @ h^T

    Grid is ``(ceil(V / BV), 1, 1)``: each CTA owns a ``BV``-wide V stripe.
    """
    if spec.mfma_k * spec.k_steps_per_block != 64:
        raise ValueError("probe assumes k_steps_per_block covers a 64-wide K block")

    atom = mfma_atom("bf16", 16, 16, spec.mfma_k)
    BT, K, BV, V = spec.BT, spec.K, spec.BV, spec.V
    nthreads = spec.block_threads
    lds_cols = K + LDS_PAD

    b = IRBuilder(spec.kernel_name() + "_gemm1probe")
    b.kernel.attrs["max_workgroup_size"] = nthreads

    W = b.param("W", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Hst = b.param("Hst", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    U = b.param("U", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    OUT = b.param("OUT", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)

    sW = b.smem_alloc(BF16, [BT, lds_cols], name_hint="sW")
    sH = b.smem_alloc(BF16, [BV, lds_cols], name_hint="sH")

    tid = b.thread_id_x()
    i_v = b.block_id_x()
    v_base = b.mul(i_v, b.const_i32(BV))

    # ---- global -> LDS -------------------------------------------------
    _stage_tile(b, src_ptr=W, smem=sW, rows=BT, cols=K, row_stride_src=K,
                src_row_base=None, block_threads=nthreads, tid=tid, pad=LDS_PAD)
    _stage_tile(b, src_ptr=Hst, smem=sH, rows=BV, cols=K, row_stride_src=K,
                src_row_base=v_base, block_threads=nthreads, tid=tid, pad=LDS_PAD)
    b.sync()

    # ---- wave / lane decomposition -------------------------------------
    c16 = b.const_i32(16)
    c4 = b.const_i32(4)
    wid = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    wid_m = b.mod(wid, b.const_i32(spec.m_waves))
    wid_n = b.div(wid, b.const_i32(spec.m_waves))
    lane_n = b.mod(lane, c16)
    lmb = b.div(lane, c16)

    bt_row = b.add(b.mul(wid_m, c16), lane_n)       # A-operand row (m)
    k_lane = b.mul(lmb, c4)                          # this lane's K offset in a step

    cV = b.const_i32(V)

    for nr in range(spec.n_repeat_local):
        # this wave's global V tile index
        nr_g = b.add(b.mul(wid_n, b.const_i32(spec.n_repeat_local)), b.const_i32(nr))
        v_tile = b.mul(nr_g, c16)
        v_row = b.add(v_tile, lane_n)                # B-operand row: h[v, k]

        acc = atom.zero_acc(b)
        for kb in range(spec.num_k_blocks):
            for ks in range(spec.k_steps_per_block):
                k0 = b.add(b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane)
                af = b.smem_load_vN(sW, bt_row, k0, dtype=BF16, n=atom.a_per_lane)
                bf = b.smem_load_vN(sH, v_row, k0, dtype=BF16, n=atom.b_per_lane)
                acc = atom.emit(b, af, bf, acc)

        # ---- C fragment: v_new = u - acc --------------------------------
        # rows/cols come FROM the atom, not asserted alongside it.
        for e in range(atom.c_per_lane):
            r_in, c_in = atom.lane_to_output(b, lane, e)
            row = b.add(b.mul(wid_m, c16), r_in)          # BT row
            col = b.add(v_base, b.add(v_tile, c_in))      # global V column
            off = b.add(b.mul(row, cV), col)
            u_f32 = b.cast_to_f32(b.global_load_bf16(U, off))
            b.global_store(OUT, off, b.fsub(u_f32, b.vec_extract(acc, e)), align=4)

    b.ret()
    return b.kernel


# ---------------------------------------------------------------------------
# Phase 3 + 4 — the full K5 chunk recurrence
# ---------------------------------------------------------------------------
#
# Operand assignments, derived from the atom contract at the top of this file:
#
#   GEMM1  bv[BT, BV] = w[BT, K] @ h[BV, K]^T          contract over K
#     A <- w    from sW  : A[m=bt,   kc=k ]   row=bt,   4 consecutive k
#     B <- h    from sH  : B[kc=k,   n=v  ] == h[v, k], row=v, 4 consecutive k
#     D          -> bv[bt = wid_m*16 + lmb*4 + e, v = tile*16 + lane_n]
#
#   GEMM2  h[BV, K] += k^T[K, BT] @ v_new^T[BV, BT]    contract over BT
#     A <- k    from sKT : A[m=kdim, kc=bt]   row=kdim, 4 consecutive bt
#     B <- vnew from sVN : B[kc=bt,  n=v  ]   row=v,    4 consecutive bt
#     D          -> h[k = kb*64 + wid_m*16 + lmb*4 + e, v = tile*16 + lane_n]
#
# Note the axis swap: `wid_m` selects the **BT** tile in GEMM1 and the **K**
# tile in GEMM2. That is what makes each lane's four accumulator slots four
# *consecutive K values at one V* — exactly the VK storage order — so writing
# the state back out is a contiguous 4-wide store rather than a scatter.


def _to_bf16_fast(b, x):
    """Round-half-away-from-zero f32 -> bf16 (numerics item N1).

    ``(bitcast_u32(x) + 0x8000) >> 16``, truncated. CDNA3 has no
    ``v_cvt_pk_bf16_f32``, so the compiler's RNE path expands to several VALU
    ops per element; this is four. More importantly the bias is *symmetric* —
    plain truncation is one-sided, and over a serial scan that bias compounds.
    """
    bits = b.add(b.bitcast(x, I32), b.const_i32(0x8000))
    return b.bitcast(b.trunc(b.lshr(bits, b.const_i32(16)), I16), BF16)


def build_k5(spec: GdnStateScanSpec):
    """Full K5 chunk recurrence: snapshot, GEMM1, gate, GEMM2.

    Milestone restrictions (each asserted, each a later phase):
      * ``USE_GK`` only — per-channel gate, no v_new gating
      * non-varlen: one sequence per ``i_n``, ``T`` a runtime scalar
      * f32 SSM state

    Layouts (row-major, token-major)::

        Kt   bf16 [T, Hg, K]      Wt bf16 [T, H, K]      Ut bf16 [T, H, V]
        Gk   f32  [T, H,  K]      H0 f32  [N, H, V, K]
        Vnew f32  [T, H,  V]      Hout f32 [NT, H, V, K]  Ht f32 [N, H, V, K]

    Grid ``(ceil(V/BV), N*H, 1)``.
    """
    if spec.STATE_DTYPE_BF16:
        raise NotImplementedError("f32 state only in this milestone")

    atom = mfma_atom("bf16", 16, 16, spec.mfma_k)
    BT, K, BV, V, H, Hg = spec.BT, spec.K, spec.BV, spec.V, spec.H, spec.Hg
    NKB, NRL, MW = spec.num_k_blocks, spec.n_repeat_local, spec.m_waves
    nthreads = spec.block_threads
    wcols, tcols = K + LDS_PAD, BT + LDS_PAD

    b = IRBuilder(spec.kernel_name() + "_k5")
    b.kernel.attrs["max_workgroup_size"] = nthreads
    if spec.MFMA_VGPR_FORM:
        # Keep the accumulators in VGPRs: they are VALU-touched every chunk by
        # the gate multiply, and the AGPR form pays accvgpr copies for it.
        b.kernel.attrs["agpr_alloc"] = (0, 0)

    Kt = b.param("Kt", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Wt = b.param("Wt", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Ut = b.param("Ut", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    # One gate pointer, interpreted per spec: [T, H, K] on the per-channel
    # path, head-major [H, T] on the scalar path.
    Gate = b.param("Gate", PtrType(F32, "global"), noalias=True, readonly=True,
                   align=16)
    H0 = b.param("H0", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    Vn = b.param("Vnew", PtrType(F32, "global"), noalias=True, align=16)
    Ho = b.param("Hout", PtrType(F32, "global"), noalias=True, align=16)
    Ht = b.param("Ht", PtrType(F32, "global"), noalias=True, align=16)
    T_val = b.param("T_val", I32)
    NT_val = b.param("NT_val", I32)

    sW = b.smem_alloc(BF16, [BT, wcols], name_hint="sW")
    sH = b.smem_alloc(BF16, [BV, wcols], name_hint="sH")
    sKT = b.smem_alloc(BF16, [K, tcols], name_hint="sKT")
    sVN = b.smem_alloc(BF16, [BV, tcols], name_hint="sVN")

    tid = b.thread_id_x()
    i_v = b.block_id_x()
    nh = b.block_id_y()
    cH, cV, cK = b.const_i32(H), b.const_i32(V), b.const_i32(K)
    i_n = b.div(nh, cH)
    i_h = b.mod(nh, cH)
    i_hg = b.div(i_h, b.const_i32(spec.gqa_ratio))
    v_base = b.mul(i_v, b.const_i32(BV))

    c16, c4, c64 = b.const_i32(16), b.const_i32(4), b.const_i32(64)
    wid, lane = b.div(tid, c64), b.mod(tid, c64)
    wid_m, wid_n = b.mod(wid, b.const_i32(MW)), b.div(wid, b.const_i32(MW))
    lane_n, lmb = b.mod(lane, c16), b.div(lane, c16)
    k_lane = b.mul(lmb, c4)

    # this wave's V rows — shared by GEMM1's D, GEMM2's B, and the state layout
    v_tile = [b.mul(b.add(b.mul(wid_n, b.const_i32(NRL)), b.const_i32(nr)), c16)
              for nr in range(NRL)]
    v_row = [b.add(t, lane_n) for t in v_tile]

    # GEMM2 K tile owned by this wave: k = kb*64 + wid_m*16 + (lmb*4 + e)
    k_tile = [b.add(b.const_i32(kb * 64), b.mul(wid_m, c16)) for kb in range(NKB)]

    def state_off(kb, nr, seq):
        """Flat offset of this lane's 4 state values in an [N, H, V, K] tensor."""
        v_abs = b.add(v_base, v_row[nr])
        k0 = b.add(k_tile[kb], k_lane)
        return b.add(b.mul(b.add(b.mul(b.add(b.mul(seq, cH), i_h), cV), v_abs), cK), k0)

    # ---- initial state -> accumulators ---------------------------------
    inits = []
    for kb in range(NKB):
        for nr in range(NRL):
            v = (b.global_load_vN(H0, state_off(kb, nr, i_n), F32, 4)
                 if spec.USE_INITIAL_STATE else atom.zero_acc(b))
            inits.append((f"h_{kb}_{nr}", v))

    # ======================= chunk loop =================================
    for_op = b.scf_for_iter(b.const_i32(0), NT_val, b.const_i32(1), inits,
                            iv_name="i_t")
    with for_op as (i_t, hacc):
        t_base = b.mul(i_t, b.const_i32(BT))

        # -- phase A: accumulators -> sH (bf16); stage w -> sW ------------
        for kb in range(NKB):
            for nr in range(NRL):
                acc = hacc[kb * NRL + nr]
                for e in range(4):
                    kk = b.add(b.add(k_tile[kb], k_lane), b.const_i32(e))
                    b.smem_store_vN(sH, [v_row[nr], kk],
                                    _to_bf16_fast(b, b.vec_extract(acc, e)), 1)
        _stage_tile(b, src_ptr=Wt, smem=sW, rows=BT, cols=K, row_stride_src=H * K,
                    src_row_base=t_base, block_threads=nthreads, tid=tid,
                    elem_off=b.mul(i_h, cK), clamp=T_val)
        b.sync()

        if spec.STORE_H:
            _drain_h(b, sH=sH, dst=Ho, spec=spec, tid=tid, nthreads=nthreads,
                     i_t=i_t, i_h=i_h, v_base=v_base)

        # -- GEMM1: bv = w @ h^T ------------------------------------------
        bt_row = b.add(b.mul(wid_m, c16), lane_n)
        bv = []
        for nr in range(NRL):
            acc = atom.zero_acc(b)
            for kb in range(NKB):
                for ks in range(spec.k_steps_per_block):
                    k0 = b.add(b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane)
                    af = b.smem_load_vN(sW, bt_row, k0, dtype=BF16, n=atom.a_per_lane)
                    bf = b.smem_load_vN(sH, v_row[nr], k0, dtype=BF16,
                                        n=atom.b_per_lane)
                    acc = atom.emit(b, af, bf, acc)
            bv.append(acc)

        # -- last valid token of this chunk (shared by both gate paths) ----
        t_last = b.sub(b.smin(b.add(t_base, b.const_i32(BT)), T_val),
                       b.const_i32(1))

        # -- C-fragment row coordinates, derived ONCE ----------------------
        # Five things key off this mapping (the row mask, the v_new store
        # guard, the scalar gate, the u gather, the state write). Deriving it
        # once is what keeps them from silently disagreeing.
        frag_bt, frag_t, frag_ok = [], [], []
        for e in range(atom.c_per_lane):
            r_in, _ = atom.lane_to_output(b, lane, e)
            bt = b.add(b.mul(wid_m, c16), r_in)
            t_abs = b.add(t_base, bt)
            frag_bt.append(bt)
            frag_t.append(t_abs)
            frag_ok.append(b.cmp_lt(t_abs, T_val))

        # -- scalar-gate factors (USE_G only) ------------------------------
        # gate[e] = exp(g_last - g[t_e]) applied to v_new; h decays by
        # exp(g_last). g is head-major [H, T].
        if spec.USE_G:
            g_base = b.mul(i_h, T_val)
            g_last = b.global_load_f32(Gate, b.add(g_base, t_last))
            g_gate = []
            for e in range(atom.c_per_lane):
                t_safe = b.select(frag_ok[e], frag_t[e], b.const_i32(0))
                g_e = b.global_load_f32(Gate, b.add(g_base, t_safe))
                g_gate.append(b.exp2_fast(
                    b.fmul(b.fsub(g_last, g_e), b.const_f32(LOG2E))))
            h_decay = b.exp2_fast(b.fmul(g_last, b.const_f32(LOG2E)))

        # -- v_new = u - bv, with the tail-chunk row mask (N2) -------------
        # The mask is UNCONDITIONAL. On the scalar-gate path the gate happens to
        # zero out-of-range rows; on the per-channel path nothing else would.
        # Note v_new is reported UNGATED (matching the reference) while the LDS
        # copy that feeds GEMM2 is gated.
        vn = []
        for nr in range(NRL):
            per_e = []
            for e in range(atom.c_per_lane):
                _r, c_in = atom.lane_to_output(b, lane, e)
                ok, t_abs = frag_ok[e], frag_t[e]
                v_abs = b.add(v_base, b.add(v_tile[nr], c_in))
                t_safe = b.select(ok, t_abs, b.const_i32(0))
                off = b.add(b.mul(b.add(b.mul(t_safe, cH), i_h), cV), v_abs)
                u_f = b.cast_to_f32(b.global_load_bf16(Ut, off))
                val = b.select(ok, b.fsub(u_f, b.vec_extract(bv[nr], e)),
                               b.const_f32(0.0))
                gated = b.fmul(val, g_gate[e]) if spec.USE_G else val
                per_e.append((val, gated, off, ok))
                if spec.SAVE_NEW_VALUE:
                    with b.scf_if(ok):
                        b.global_store(Vn, off, val, align=4)
            vn.append(per_e)

        # -- v_new -> sVN [BV, BT]; k -> sKT [K, BT] ----------------------
        for nr in range(NRL):
            for e in range(atom.c_per_lane):
                _val, gated, _o, _ok = vn[nr][e]
                b.smem_store_vN(sVN, [v_row[nr], frag_bt[e]],
                                _to_bf16_fast(b, gated), 1)
        _stage_k_transposed(b, src=Kt, smem=sKT, spec=spec, tid=tid,
                            nthreads=nthreads, t_base=t_base, i_hg=i_hg,
                            T_val=T_val, Hg=Hg)
        b.sync()

        # -- state decay, then GEMM2 ---------------------------------------
        # USE_GK: h[v, k] *= exp(gk_last[k]) — per channel. Slot e is
        #         k = tile + lmb*4 + e, so the four factors are one f32x4 load.
        # USE_G : h *= exp(g_last)          — one scalar for the whole state.
        gk_row = b.mul(b.add(b.mul(t_last, cH), i_h), cK)
        out = []
        for kb in range(NKB):
            if spec.USE_GK:
                gk4 = b.global_load_vN(
                    Gate, b.add(gk_row, b.add(k_tile[kb], k_lane)), F32, 4)
            for nr in range(NRL):
                acc = hacc[kb * NRL + nr]
                dec = atom.zero_acc(b)
                for e in range(4):
                    f = (b.exp2_fast(b.fmul(b.vec_extract(gk4, e),
                                            b.const_f32(LOG2E)))
                         if spec.USE_GK else h_decay)
                    dec = b.vec_insert(dec, b.fmul(b.vec_extract(acc, e), f), e)
                k_row = b.add(k_tile[kb], lane_n)
                for bs in range(spec.bt_steps):
                    bt0 = b.add(b.const_i32(bs * spec.mfma_k), k_lane)
                    af = b.smem_load_vN(sKT, k_row, bt0, dtype=BF16,
                                        n=atom.a_per_lane)
                    bf = b.smem_load_vN(sVN, v_row[nr], bt0, dtype=BF16,
                                        n=atom.b_per_lane)
                    dec = atom.emit(b, af, bf, dec)
                out.append(dec)
        # No third barrier here. The two hazards that cross the loop back edge
        # are both already ordered:
        #   GEMM2(i) reads sKT/sVN  vs  writes to sKT/sVN in i+1 — a thread can
        #     only reach those writes after sync1(i+1), which every thread can
        #     only reach after finishing GEMM2(i).
        #   GEMM1(i) reads sW/sH    vs  writes to sW/sH in i+1 — separated by
        #     sync2(i), which is before the back edge.
        # Two barriers per chunk, not three.
        b.scf_yield(*out)

    # ---- final state ----------------------------------------------------
    if spec.STORE_FINAL_STATE:
        res = for_op.results
        for kb in range(NKB):
            for nr in range(NRL):
                b.global_store_vN(Ht, state_off(kb, nr, i_n), res[kb * NRL + nr], 4)
    b.ret()
    return b.kernel
