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

from rocke.core.ir import (BF16, F32, I16, I32, IRBuilder, PtrType,
                           VectorType)
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

#: XCD count on this part; drives the P7 chiplet remap.
NXCD = 8



class _Loader:
    """Global loads, either raw-pointer or through a bounds-checked descriptor.

    P6. With ``use_desc`` the resource carries ``num_records`` in *bytes*, so
    hardware returns 0 for any out-of-range access and the caller can drop its
    explicit row clamp. ``voffset`` is a byte offset; callers keep working in
    elements and the conversion happens here.
    """

    def __init__(self, b, ptr, *, n_elems=None, elem_bytes=2, use_desc=False):
        """``n_elems`` is a *thunk* returning a Value, so the size arithmetic is
        emitted only when a descriptor is actually built. With the flag off the
        emitted IR is identical to the raw-pointer version — no dead ops to
        perturb the already-verified default path."""
        self.b, self.ptr, self.eb = b, ptr, elem_bytes
        self.zero = b.const_i32(0) if use_desc else None
        self.rsrc = (b.buffer_rsrc(ptr, b.mul(n_elems(), b.const_i32(elem_bytes)))
                     if use_desc else None)

    @property
    def bounds_checked(self) -> bool:
        return self.rsrc is not None

    def _bytes(self, off):
        return self.b.mul(off, self.b.const_i32(self.eb))

    def vN(self, off, dtype, n):
        if self.rsrc is None:
            return self.b.global_load_vN(self.ptr, off, dtype, n)
        return self.b.buffer_load_vN(self.rsrc, self._bytes(off), self.zero, dtype, n)

    def scalar(self, off, dtype):
        if self.rsrc is None:
            return (self.b.global_load_bf16(self.ptr, off) if dtype is BF16
                    else self.b.global_load_f32(self.ptr, off))
        if dtype is BF16:
            return self.b.buffer_load_bf16(self.rsrc, self._bytes(off), self.zero)
        return self.b.buffer_load(self.rsrc, self._bytes(off), self.zero, dtype)


def _grp_col(b, row, col, ng):
    """Group-major + XOR swizzled column, matching the FlyDSL parent exactly.

    A logical ``[R, C]`` tile is stored ``[R][C/4][4]``: each row is ``ng = C/4``
    groups of 4 bf16 (8 B — one MFMA fragment), and the group index is XORed by
    a key derived from the row::

        mask = (row ^ (row >> 3)) & (ng - 1)
        col' = (col/4 ^ mask) * 4

    The ``>> 3`` fold matters: under the ``k`` store-transpose write pattern a
    plain ``row & (ng-1)`` takes too few distinct values across the 16 lanes of
    one fragment, which reintroduces the conflicts the swizzle exists to remove.

    ``col`` must be 4-aligned, and ``ng`` a power of two. Both hold here — every
    LDS access in this kernel is fragment-granular.

    **This is the only place the permutation is written.** Producer and consumer
    must agree and nothing checks that they do, so a second copy of this formula
    is a silent-wrong-answer waiting to happen.
    """
    grp = b.lshr(col, b.const_i32(2))
    mask = b.land(b.xor(row, b.lshr(row, b.const_i32(3))), b.const_i32(ng - 1))
    return b.shl(b.xor(grp, mask), b.const_i32(2))


def _swz(b, row, col, ng):
    """``_grp_col`` when swizzling, identity when not."""
    return col if ng is None else _grp_col(b, row, col, ng)


def _regroup(b, v, base, n=4):
    """Extract ``n`` consecutive lanes of a wider vector into their own vector."""
    out = b.vector_splat(b.vec_extract(v, base), n)
    for j in range(1, n):
        out = b.vec_insert(out, b.vec_extract(v, base + j), j)
    return out


def _stage_tile(b, *, src_ptr, smem, rows, cols, row_stride_src, src_row_base,
                block_threads, tid, ng=None, elem_off=None, clamp=None):
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
    regs = _stage_tile_load(b, src_ptr=src_ptr, rows=rows, cols=cols,
                            row_stride_src=row_stride_src,
                            src_row_base=src_row_base,
                            block_threads=block_threads, tid=tid,
                            elem_off=elem_off, clamp=clamp)
    _stage_tile_store(b, smem=smem, regs=regs, rows=rows, cols=cols,
                      block_threads=block_threads, tid=tid, ng=ng)


def _tile_slots(b, *, rows, cols, block_threads, tid):
    """Per-iteration ``(row, col)`` for the cooperative tile walk.

    Shared by the load and store halves so the two agree by construction — they
    now run in *different loop iterations* (P5), so a drifting decomposition
    would be a silent corruption rather than a compile error.
    """
    vecs_per_row = cols // STAGE_VEC
    total_slots = rows * vecs_per_row
    assert total_slots % block_threads == 0, (
        f"tile [{rows},{cols}] = {total_slots} vec{STAGE_VEC} slots must tile "
        f"block_threads={block_threads}")
    c_vpr = b.const_i32(vecs_per_row)
    out = []
    for it in range(total_slots // block_threads):
        slot = b.add(b.const_i32(it * block_threads), tid)
        out.append((b.div(slot, c_vpr),
                    b.mul(b.mod(slot, c_vpr), b.const_i32(STAGE_VEC))))
    return out


def _stage_tile_load(b, *, src_ptr, rows, cols, row_stride_src, src_row_base,
                     block_threads, tid, elem_off=None, clamp=None, ldr=None):
    """Issue only the global loads for a tile; return the raw vectors.

    **Raw loads only** — no conversion, no select on the *value*, no packing.
    Anything that consumes a load here would force a wait in the issuing
    iteration and defeat the prefetch (P5).
    """
    c_srcstride = b.const_i32(row_stride_src)
    regs = []
    for row, col in _tile_slots(b, rows=rows, cols=cols,
                                block_threads=block_threads, tid=tid):
        src_row = b.add(row, src_row_base) if src_row_base is not None else row
        # A bounds-checked descriptor makes the clamp redundant: an out-of-range
        # row reads a hardware zero instead of row 0, and the tail-chunk mask
        # (N2) zeroes whatever it feeds either way.
        if clamp is not None and not (ldr is not None and ldr.bounds_checked):
            src_row = b.select(b.cmp_lt(src_row, clamp), src_row, b.const_i32(0))
        off = b.add(b.mul(src_row, c_srcstride), col)
        if elem_off is not None:
            off = b.add(off, elem_off)
        regs.append(ldr.vN(off, BF16, STAGE_VEC) if ldr is not None
                    else b.global_load_vN(src_ptr, off, BF16, STAGE_VEC))
    return regs


def _stage_tile_store(b, *, smem, regs, rows, cols, block_threads, tid, ng=None):
    """Write previously-loaded tile registers into LDS."""
    slots = _tile_slots(b, rows=rows, cols=cols,
                        block_threads=block_threads, tid=tid)
    assert len(slots) == len(regs)
    for (row, col), v in zip(slots, regs):
        if ng is None:
            b.smem_store_vN(smem, [row, col], v, STAGE_VEC)
        else:
            # 8 contiguous elements span two 4-element groups, and the XOR can
            # place them anywhere in the row — so they become two stores.
            for g in range(STAGE_VEC // 4):
                c = b.add(col, b.const_i32(g * 4))
                b.smem_store_vN(smem, [row, _grp_col(b, row, c, ng)],
                                _regroup(b, v, g * 4), 4)


def _stage_k_transposed(b, *, src, smem, spec, tid, nthreads, t_base, i_hg,
                        T_val, Hg, ng=None, ldr=None):
    """Stage ``k[BT, K]`` from global into ``sKT[K, BT]`` — a real transpose.

    gfx942 has no ``ds_read_*_tr_*`` (CDNA4 only), so a transposed operand must
    be built on the **store** side.

    **P2, matching the FlyDSL parent's decomposition.** A thread takes four
    **BT-consecutive** rows at the same K columns. Per K column those four
    values then form one ``bt``-group — contiguous in ``sKT[k, bt]`` — so an
    in-register gather turns what would be four scattered ``ds_write_b16`` into
    a single packed ``ds_write_b64``.

    The row quad stays 4 regardless of block size (that is what makes the store
    packed), so a wider block costs *load* width rather than store width::

        256 thr -> vec8 (dwordx4) | 512 thr -> vec4 | 1024 thr -> vec2
    """
    BT, K = spec.BT, spec.K
    # Widest K run a thread can take while every (row-quad, col-group) slot
    # still tiles the block. Same formula as the parent.
    KVW = min(STAGE_VEC, max(2, (BT // 4) * K // nthreads))
    assert KVW & (KVW - 1) == 0, f"K vector width {KVW} must be a power of two"
    assert K % KVW == 0, f"K={K} must be divisible by the vector width {KVW}"
    col_groups = K // KVW
    row_quads = BT // 4
    slots = row_quads * col_groups
    assert slots % nthreads == 0, (
        f"k transpose slots ({slots}) must tile block_threads={nthreads}")

    c_cg = b.const_i32(col_groups)
    for it in range(slots // nthreads):
        slot = b.add(b.const_i32(it * nthreads), tid)
        bt0 = b.mul(b.div(slot, c_cg), b.const_i32(4))      # first of 4 BT rows
        k0 = b.mul(b.mod(slot, c_cg), b.const_i32(KVW))     # first of KVW k-cols
        rows = []
        for r in range(4):
            t_abs = b.add(b.add(t_base, bt0), b.const_i32(r))
            t_safe = (t_abs if (ldr is not None and ldr.bounds_checked)
                      else b.select(b.cmp_lt(t_abs, T_val), t_abs, b.const_i32(0)))
            off = b.add(b.mul(b.add(b.mul(t_safe, b.const_i32(Hg)), i_hg),
                              b.const_i32(K)), k0)
            rows.append(ldr.vN(off, BF16, KVW) if ldr is not None
                        else b.global_load_vN(src, off, BF16, KVW))
        for j in range(KVW):
            col = b.vector_splat(b.vec_extract(rows[0], j), 4)
            for r in range(1, 4):
                col = b.vec_insert(col, b.vec_extract(rows[r], j), r)
            row = b.add(k0, b.const_i32(j))
            b.smem_store_vN(smem, [row, _swz(b, row, bt0, ng)], col, 4)


def _drain_h(b, *, sH, dst, spec, tid, nthreads, i_t, i_h, v_base, ng=None):
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
        # P4: one wide LDS read, then f32 stores in 4-wide groups. The values
        # are contiguous in both LDS (K-major) and HBM (K innermost), so the
        # only reason this was ever scalar was expedience.
        for g in range(STAGE_VEC // 4):
            c = b.add(k0, b.const_i32(g * 4))
            val = b.smem_load_vN(sH, row, _swz(b, row, c, ng), dtype=BF16, n=4)
            quad = _pack_f32x4(
                b, [b.cast_to_f32(b.vec_extract(val, j)) for j in range(4)])
            b.global_store_vN(dst, b.add(base, b.const_i32(g * 4)), quad, 4)


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


def _to_bf16_fast(b, x, n: int = 1):
    """Round-half-away-from-zero f32 -> bf16 (numerics item N1).

    ``(bitcast_u32(x) + 0x8000) >> 16``, truncated. CDNA3 has no
    ``v_cvt_pk_bf16_f32``, so the compiler's RNE path expands to several VALU
    ops per element; this is four. More importantly the bias is *symmetric* —
    plain truncation is one-sided, and over a serial scan that bias compounds.

    With ``n > 1`` the whole conversion runs on an ``<n x f32>`` vector, so the
    cost is four VALU ops for the *group* rather than four per element. That is
    the form the FlyDSL parent uses, and it is what makes a packed
    ``ds_write_b64`` worth doing (P1).
    """
    if n == 1:
        bits = b.add(b.bitcast(x, I32), b.const_i32(0x8000))
        return b.bitcast(b.trunc(b.lshr(bits, b.const_i32(16)), I16), BF16)
    bits = b.bitcast(x, VectorType(I32, n))
    bits = b.add(bits, b.vector_splat(b.const_i32(0x8000), n))
    bits = b.lshr(bits, b.vector_splat(b.const_i32(16), n))
    return b.bitcast(b.trunc(bits, VectorType(I16, n)), VectorType(BF16, n))


def _pack_f32x4(b, vals):
    """Assemble four f32 scalars into an ``<4 x f32>``."""
    v = b.vector_splat(vals[0], 4)
    for i in range(1, 4):
        v = b.vec_insert(v, vals[i], i)
    return v


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
    # P3: swizzled buffers need no padding; the unswizzled fallback does.
    pad = 0 if spec.LDS_SWIZZLE else LDS_PAD
    wcols, tcols = K + pad, BT + pad
    ng_k = (K // 4) if spec.LDS_SWIZZLE else None    # sW, sH  (K-major rows)
    ng_t = (BT // 4) if spec.LDS_SWIZZLE else None   # sKT, sVN (BT-major rows)

    # P5 is gated: see PERF_PLAN.md for the unresolved ordering sensitivity.
    _PF_W = _PF_U = _PF_G = spec.PREFETCH

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
    N_val = b.param("N_val", I32)          # sequences; grid.y == N_val * H

    sW = b.smem_alloc(BF16, [BT, wcols], name_hint="sW")
    sH = b.smem_alloc(BF16, [BV, wcols], name_hint="sH")
    sKT = b.smem_alloc(BF16, [K, tcols], name_hint="sKT")
    sVN = b.smem_alloc(BF16, [BV, tcols], name_hint="sVN")

    tid = b.thread_id_x()
    cH, cV, cK = b.const_i32(H), b.const_i32(V), b.const_i32(K)

    if spec.XCD_REMAP:
        # P7. A head's GRID_V V-tiles all read the same V-independent w / k /
        # gate slices. Under the hardware default (flat block `xy` runs on XCD
        # `xy % NXCD`) those tiles scatter across XCDs and the slices land in up
        # to NXCD separate private L2s. Invert the round-robin so runs of GRID_V
        # consecutive logical ids share an XCD, then unflatten.
        #
        # Ported from the parent, tail guard included: ids past the last full
        # NXCD*GRID_V cycle pass through unchanged, since remapping them would
        # collide or run out of range. So any grid is at least as good as the
        # round-robin baseline.
        GV = spec.grid_v
        cGV, cNX = b.const_i32(GV), b.const_i32(NXCD)
        grid_total = b.mul(cGV, b.mul(N_val, cH))
        xy = b.add(b.block_id_x(), b.mul(cGV, b.block_id_y()))
        xcd = b.mod(xy, cNX)
        cycle = b.const_i32(NXCD * GV)
        last_full = b.mul(b.div(grid_total, cycle), cycle)
        local_id = b.div(xy, cNX)
        remapped = b.add(b.mul(b.div(local_id, cGV), cycle),
                         b.add(b.mul(xcd, cGV), b.mod(local_id, cGV)))
        logical = b.select(b.cmp_lt(xy, last_full), remapped, xy)
        i_v = b.mod(logical, cGV)
        nh = b.div(logical, cGV)
    else:
        i_v = b.block_id_x()
        nh = b.block_id_y()

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

    # P6: one bounds-checked resource per loaded tensor. Sizes are exact, so an
    # out-of-range access returns 0 rather than reading a neighbour.
    _bd = spec.BUFFER_DESC
    ld_w = _Loader(b, Wt, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, cH), cK))
    ld_k = _Loader(b, Kt, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, b.const_i32(Hg)), cK))
    ld_u = _Loader(b, Ut, elem_bytes=2, use_desc=_bd,
                   n_elems=lambda: b.mul(b.mul(T_val, cH), cV))
    ld_g = _Loader(b, Gate, elem_bytes=4, use_desc=_bd,
                   n_elems=(lambda: b.mul(b.mul(T_val, cH), cK)) if spec.USE_GK
                           else (lambda: b.mul(cH, T_val)))
    ld_h0 = _Loader(b, H0, elem_bytes=4, use_desc=_bd,
                    n_elems=lambda: b.mul(b.mul(b.mul(N_val, cH), cV), cK))

    # ---- P5: loop-carried prefetch -------------------------------------
    # Issue chunk i+1's HBM reads at the end of chunk i and carry the raw
    # values across the back edge, so their latency sits behind this chunk's
    # MFMA chain instead of in front of the next one.
    #
    # Two rules, both load-bearing:
    #   * RAW LOADS ONLY. No exp, no select-on-value, no packing at issue time
    #     — anything that consumes a load here forces a wait here.
    #   * issue/unpack are structural inverses. scf_yield ordering is unchecked
    #     and this list is long, so they walk the same sequence by construction.
    def _pf_issue(i_t_n):
        """Raw reads for chunk ``i_t_n``. Returns a flat list of Values."""
        t_b = b.mul(i_t_n, b.const_i32(BT))
        out = [] if not _PF_W else list(_stage_tile_load(
            b, src_ptr=Wt, rows=BT, cols=K, row_stride_src=H * K,
            src_row_base=t_b, block_threads=nthreads, tid=tid,
            elem_off=b.mul(i_h, cK), clamp=T_val, ldr=ld_w))
        _U_MARK = len(out)
        # u, gathered in the MMA's C-fragment layout
        for nr in (range(NRL) if _PF_U else []):
            for e in range(atom.c_per_lane):
                r_in, c_in = atom.lane_to_output(b, lane, e)
                t_abs = b.add(t_b, b.add(b.mul(wid_m, c16), r_in))
                t_safe = b.select(b.cmp_lt(t_abs, T_val), t_abs, b.const_i32(0))
                v_abs = b.add(v_base, b.add(v_tile[nr], c_in))
                out.append(ld_u.scalar(
                    b.add(b.mul(b.add(b.mul(t_safe, cH), i_h), cV), v_abs), BF16))
        _G_MARK = len(out)
        # gate
        t_last = b.sub(b.smin(b.add(t_b, b.const_i32(BT)), T_val), b.const_i32(1))
        if not _PF_G:
            pass
        elif spec.USE_GK:
            row = b.mul(b.add(b.mul(t_last, cH), i_h), cK)
            for kb in range(NKB):
                out.append(ld_g.vN(b.add(row, b.add(k_tile[kb], k_lane)), F32, 4))
        elif True:
            g_base = b.mul(i_h, T_val)
            out.append(ld_g.scalar(b.add(g_base, t_last), F32))
            for e in range(atom.c_per_lane):
                r_in, _c = atom.lane_to_output(b, lane, e)
                t_abs = b.add(t_b, b.add(b.mul(wid_m, c16), r_in))
                t_safe = b.select(b.cmp_lt(t_abs, T_val), t_abs, b.const_i32(0))
                out.append(ld_g.scalar(b.add(g_base, t_safe), F32))
        # swap: [w][u][gate] -> [w][gate][u]
        u_part = out[_U_MARK:_G_MARK]
        g_part = out[_G_MARK:]
        return out[:_U_MARK] + g_part + u_part

    def _pf_unpack(vals):
        """Structural inverse of :func:`_pf_issue`."""
        it = iter(vals)
        n_w = len(_tile_slots(b, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid))
        w_regs = [next(it) for _ in range(n_w)] if _PF_W else None
        if not _PF_G:
            gate = None
        elif spec.USE_GK:
            gate = [next(it) for _ in range(NKB)]
        else:
            gate = [next(it) for _ in range(1 + atom.c_per_lane)]
        u_vals = ([[next(it) for _ in range(atom.c_per_lane)] for _ in range(NRL)]
                  if _PF_U else None)
        assert next(it, None) is None, "prefetch unpack did not consume every value"
        return w_regs, u_vals, gate

    _pf0 = _pf_issue(b.const_i32(0))

    # ---- initial state -> accumulators ---------------------------------
    inits = []
    for kb in range(NKB):
        for nr in range(NRL):
            v = (ld_h0.vN(state_off(kb, nr, i_n), F32, 4)
                 if spec.USE_INITIAL_STATE else atom.zero_acc(b))
            inits.append((f"h_{kb}_{nr}", v))

    n_acc = len(inits)
    inits = inits + [(f"pf_{i}", v) for i, v in enumerate(_pf0)]

    # ======================= chunk loop =================================
    for_op = b.scf_for_iter(b.const_i32(0), NT_val, b.const_i32(1), inits,
                            iv_name="i_t")
    with for_op as (i_t, carried):
        hacc, pf = carried[:n_acc], carried[n_acc:]
        w_regs, u_pf, gate_pf = _pf_unpack(pf)
        t_base = b.mul(i_t, b.const_i32(BT))

        # -- phase A: accumulators -> sH (bf16); stage w -> sW ------------
        for kb in range(NKB):
            for nr in range(NRL):
                # P1: the four accumulator slots are four consecutive k, and sH
                # is K-major, so this is one packed ds_write, not four scalars.
                k0 = b.add(k_tile[kb], k_lane)
                b.smem_store_vN(sH, [v_row[nr], _swz(b, v_row[nr], k0, ng_k)],
                                _to_bf16_fast(b, hacc[kb * NRL + nr], 4), 4)
        if _PF_W:
            _stage_tile_store(b, smem=sW, regs=w_regs, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid, ng=ng_k)
        else:
            _stage_tile_store(b, smem=sW, rows=BT, cols=K,
                              block_threads=nthreads, tid=tid, ng=ng_k,
                              regs=_stage_tile_load(
                                  b, src_ptr=Wt, rows=BT, cols=K,
                                  row_stride_src=H * K, src_row_base=t_base,
                                  block_threads=nthreads, tid=tid,
                                  elem_off=b.mul(i_h, cK), clamp=T_val, ldr=ld_w))
        b.sync()

        if spec.STORE_H:
            _drain_h(b, sH=sH, dst=Ho, spec=spec, tid=tid, nthreads=nthreads,
                     i_t=i_t, i_h=i_h, v_base=v_base, ng=ng_k)

        # -- GEMM1: bv = w @ h^T ------------------------------------------
        bt_row = b.add(b.mul(wid_m, c16), lane_n)
        bv = []
        for nr in range(NRL):
            acc = atom.zero_acc(b)
            for kb in range(NKB):
                for ks in range(spec.k_steps_per_block):
                    k0 = b.add(b.const_i32(kb * 64 + ks * spec.mfma_k), k_lane)
                    af = b.smem_load_vN(sW, bt_row, _swz(b, bt_row, k0, ng_k),
                                        dtype=BF16, n=atom.a_per_lane)
                    bf = b.smem_load_vN(sH, v_row[nr], _swz(b, v_row[nr], k0, ng_k),
                                        dtype=BF16, n=atom.b_per_lane)
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
            gb = b.mul(i_h, T_val)
            g_last = gate_pf[0] if _PF_G else ld_g.scalar(b.add(gb, t_last), F32)
            g_gate = []
            for e in range(atom.c_per_lane):
                ge = (gate_pf[1+e] if _PF_G else ld_g.scalar(
                      b.add(gb, b.select(frag_ok[e], frag_t[e], b.const_i32(0))), F32))
                g_gate.append(b.exp2_fast(b.fmul(b.fsub(g_last, ge), b.const_f32(LOG2E))))
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
                u_f = b.cast_to_f32(u_pf[nr][e] if _PF_U else ld_u.scalar(off, BF16))
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
            # P1: frag_bt[e] = wid_m*16 + lmb*4 + e -> four consecutive bt, and
            # sVN is BT-major, so the gated values pack into one ds_write.
            packed = _pack_f32x4(b, [vn[nr][e][1] for e in range(4)])
            b.smem_store_vN(sVN, [v_row[nr], _swz(b, v_row[nr], frag_bt[0], ng_t)],
                            _to_bf16_fast(b, packed, 4), 4)
        _stage_k_transposed(b, src=Kt, smem=sKT, spec=spec, tid=tid,
                            nthreads=nthreads, t_base=t_base, i_hg=i_hg,
                            T_val=T_val, Hg=Hg, ng=ng_t, ldr=ld_k)
        b.sync()

        # -- state decay, then GEMM2 ---------------------------------------
        # USE_GK: h[v, k] *= exp(gk_last[k]) — per channel. Slot e is
        #         k = tile + lmb*4 + e, so the four factors are one f32x4 load.
        # USE_G : h *= exp(g_last)          — one scalar for the whole state.
        gk_row = b.mul(b.add(b.mul(t_last, cH), i_h), cK)
        out = []
        for kb in range(NKB):
            if spec.USE_GK:
                gk4 = (gate_pf[kb] if _PF_G else ld_g.vN(
                    b.add(gk_row, b.add(k_tile[kb], k_lane)), F32, 4))
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
                    af = b.smem_load_vN(sKT, k_row, _swz(b, k_row, bt0, ng_t),
                                        dtype=BF16, n=atom.a_per_lane)
                    bf = b.smem_load_vN(sVN, v_row[nr], _swz(b, v_row[nr], bt0, ng_t),
                                        dtype=BF16, n=atom.b_per_lane)
                    dec = atom.emit(b, af, bf, dec)
                out.append(dec)
        # P5: issue chunk i+1's reads here, after GEMM2 has been emitted, so the
        # MFMA chain sits between the load and its first use next iteration.
        nxt = _pf_issue(b.add(i_t, b.const_i32(1)))
        # No third barrier here. The two hazards that cross the loop back edge
        # are both already ordered:
        #   GEMM2(i) reads sKT/sVN  vs  writes to sKT/sVN in i+1 — a thread can
        #     only reach those writes after sync1(i+1), which every thread can
        #     only reach after finishing GEMM2(i).
        #   GEMM1(i) reads sW/sH    vs  writes to sW/sH in i+1 — separated by
        #     sync2(i), which is before the back edge.
        # Two barriers per chunk, not three.
        b.scf_yield(*out, *nxt)

    # ---- final state ----------------------------------------------------
    if spec.STORE_FINAL_STATE:
        res = for_op.results
        for kb in range(NKB):
            for nr in range(NRL):
                b.global_store_vN(Ht, state_off(kb, nr, i_n), res[kb * NRL + nr], 4)
    b.ret()
    return b.kernel
