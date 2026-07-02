# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hand-authored grouped bf16 GEMM built directly on the raw ``IRBuilder``
(NO ``gemm_universal`` emitter) and lowered through the HIP-C++ -> hipcc
backend (``compile_kernel_via_hipcc``).

Instead of declaring a ``TileSpec`` and letting the generic emitter + comgr
schedule the kernel, we author the warp tiling, LDS staging, double-buffered
software pipeline, and the ``s_setprio`` MFMA-priority hints by hand, and rely
on the hipcc toolchain to preserve that schedule. The kernel borrows the
*correct* MFMA lane-mapping helpers from ``helpers.mfma_gemm_inner``
(``mfma_atom_for_dtype`` / ``decode_mfma_lanes`` / ``atom.lane_to_output``) so
the operand-frag and epilogue layouts are exact.

ABI matches ``GroupedGemmSingleLaunchRunner``: ``(A, B, C, M, N, K,
stride_a, stride_b, stride_c)``; grid ``(ceil(N/TN), ceil(M/TM), E)`` with
``block_id_z`` selecting the expert. RCR layout (A ``[M,K]``, B ``[N,K]``,
C ``[M,N]`` row-major, per expert, packed contiguously over E).

Levers (env-gated, default = the measured sweet spot for M/E=8192, N=1024,
K=512, E=64 on gfx950 / MI355X):

* ``DTL=1``   async direct-to-LDS DMA (``async_buffer_load_lds_addr``,
              DRAM->LDS, double-buffered, vmcnt-overlapped). Best path.
* ``DB=1``    register-prefetch double buffer (used when ``DTL=0``).
* ``PRIO=1``  ``s_setprio(1/0)`` around the MFMA cluster (survives hipcc;
              the comgr/LLVM-IR path could not keep this hint).
* ``TM/TN/TK/WM/WN`` tile + warp-grid geometry.

* ``SWZ=1``  st_16x32 LDS XOR swizzle (bank-conflict-free ds_reads); default on.
* ``CHIP=1`` chiplet/super-tile grid remap (L2 locality across XCDs via a
             chunked super-tile transform + WGM grouping); default on. ``XCDS``/
             ``WGM`` tune the XCD count / super-tile group height.
* ``EPIFUSE=1`` interleave each accumulator's C store with its final MFMA so
             the store-issue + address VALU overlap the last K-tile's MFMAs;
             default on.
* ``PF=1``   register-prefetch-ahead (read next K-chunk frags before current
             MFMAs). Neutral here -- the compiler already overlaps ds_read->mma.
* ``PIN=1``  pin wave-uniform tile bases/expert offsets into SGPRs (~neutral).

Measured progression (this harness, default shape): single-buffer 446 ->
double-buffer 552 -> +s_setprio 648 -> prefetch-before-barrier 676 ->
async-DTL 729 -> +st_16x32 swizzle 750 -> +chiplet grid swizzle ~790 ->
+epilogue-store interleave ~807 TFLOPS median / ~815 peak (vs the generic
emitter's 750). Producer/consumer wave specialization does not pay off on
gfx950 (unified VGPR file -- no per-wave register reallocation, and occupancy
is LDS-bound at 1 block/CU). See CASE_STUDY.md for the full analysis.

Run:
    PYTHONPATH=Python python3 \
        Python/rocke/examples/gfx950/grouped_gemm/grouped_gemm_hip.py
"""
from __future__ import annotations

import math
import os

from rocke.core.ir import BF16, I32, I64, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes, mfma_atom_for_dtype
from rocke.helpers.spec import SignatureBuilder


def build_custom_grouped(
    M,
    N,
    K,
    E,
    *,
    TM=256,
    TN=256,
    TK=64,
    WM=2,
    WN=4,
    dtl=True,
    db=True,
    prio=True,
    swz=False,
    prefetch=False,
    pin=False,
    chiplet=False,
    chiplet_wgm=8,
    chiplet_xcds=8,
    asm_reads=False,
    tpb=1,
    name="grouped_gemm",
):
    """Build the hand-scheduled grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. ``M`` here is the per-expert
    row count (``M_total // E``).
    """
    # MFMA32: the 32x32x8 bf16 atom (a/b_per_lane=4, c_per_lane=16). Does 4x the
    # output per MFMA vs 16x16x32 at the same MAC/op, so it loads HALF the operand
    # bf16 per MAC -> halves LDS-read pressure (rocBLAS's NN trick, LdsUtil ~18% vs
    # our ~45%). And b_per_lane=4 == one ds_read_b64_tr_b16 -> the NN transpose-read
    # is a SINGLE op (vs 2x for 16x16x32).
    mfma32 = os.environ.get("MFMA32", "0") == "1"
    if mfma32:
        from rocke.helpers.atoms import MfmaAtom

        atom = MfmaAtom.bf16_32x32x8()
    else:
        atom = mfma_atom_for_dtype("bf16", 16, 16)  # 16x16x32 bf16
    AM, AN, AK = atom.m, atom.n, atom.k
    apl, bpl = atom.a_per_lane, atom.b_per_lane  # 16x16x32: 8,8 ; 32x32x8: 4,4
    BS = WM * WN * 64
    WTM, WTN = TM // WM, TN // WN  # per-warp output (rows, cols)
    mm, nn = WTM // AM, WTN // AN  # MFMA tiles per warp
    kchunks = TK // AK
    n_ktiles = K // TK
    assert TK % AK == 0 and TM % (WM * AM) == 0 and TN % (WN * AN) == 0
    assert (TM * TK) % BS == 0 and (TN * TK) % BS == 0

    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    b.param("M", I32)
    b.param("N", I32)
    b.param("K", I32)  # ABI
    sa = b.param("stride_a", I32)
    sb = b.param("stride_b", I32)
    sc = b.param("stride_c", I32)

    tid = b.thread_id_x()
    warp = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    warp_row = b.div(warp, b.const_i32(WN))
    warp_col = b.mod(warp, b.const_i32(WN))

    def U(v):
        return b.to_sgpr_u32(v) if pin else v

    n_pid_m = (M + TM - 1) // TM
    n_pid_n = (N + TN - 1) // TN

    def set_tile(bid_n, bid_m, expert):
        # Returns a per-tile coords holder; the K-loop / epilogue read from it so
        # the pipelined path can store tile j-1 while computing tile j.
        tb = {}
        if chiplet:
            from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

            wgid_flat = b.add(b.mul(bid_m, b.const_i32(n_pid_n)), bid_n)
            swz = chiplet_aware_super_tile_dynamic(
                b,
                wgid_flat,
                num_pid_m=b.const_i32(n_pid_m),
                num_pid_n=b.const_i32(n_pid_n),
                wgm=chiplet_wgm,
                num_xcds=chiplet_xcds,
                chunk_size=64,
            )
            tb["m_base"] = U(b.mul(swz.row, b.const_i32(TM)))
            tb["n_base"] = U(b.mul(swz.col, b.const_i32(TN)))
        else:
            tb["m_base"] = U(b.mul(bid_m, b.const_i32(TM)))
            tb["n_base"] = U(b.mul(bid_n, b.const_i32(TN)))
        tb["a_eoff"] = U(b.mul(expert, sa))
        tb["b_eoff"] = U(b.mul(expert, sb))
        tb["c_eoff"] = U(b.mul(expert, sc))
        return tb

    nbuf = 2 if (db or dtl) else 1
    ld = decode_mfma_lanes(b, atom, lane)
    cshuf = os.environ.get("CSHUF", "0") == "1"
    # RRR weight layout (B = [E,K,N], N-contiguous, like the Triton/CK references):
    # stage B into LDS as a plain [TK,TN] tile and read the MFMA b-operand via the
    # CK transpose LDS read (ds_read_b64_tr_b16), which delivers the SAME b-operand
    # layout mma_ABt already consumes -> the MFMA is unchanged. Requires DTL.
    b_rrr = os.environ.get("BRRR", "0") == "1"
    if b_rrr:
        assert dtl and not cshuf, "BRRR requires DTL and is incompatible with CSHUF"
    if cshuf:
        # One flat LDS buffer that the K-loop slices into As0/As1/Bs0/Bs1 (row
        # bands) and the epilogue reuses as the [TM][TN] C-stage (the DTL buffers
        # are dead after the K-loop). Requires TM==TN so the stacked band size
        # (2*nbuf*TM rows x TK) exactly equals the C-tile (TM*TN elems).
        assert TM == TN and TK > 0 and (TM * TN) == (2 * nbuf * TM) * TK
        AB = b.smem_alloc(BF16, [2 * nbuf * TM, TK], name_hint="AB")
        As = [(AB, i * TM) for i in range(nbuf)]  # (buffer, row_base)
        Bs = [(AB, (nbuf + i) * TM) for i in range(nbuf)]
    else:
        As = [
            (b.smem_alloc(BF16, [TM, TK], name_hint=f"As{i}"), 0) for i in range(nbuf)
        ]
        # RRR: B staged as [TK,TN] (K rows, N cols) for the transpose-read; else
        # RCR [TN,TK] (N rows, K cols) for the direct ds_read.
        b_shape = [TK, TN] if b_rrr else [TN, TK]
        Bs = [(b.smem_alloc(BF16, b_shape, name_hint=f"Bs{i}"), 0) for i in range(nbuf)]
    accs = [atom.zero_acc(b) for _ in range(mm * nn)]

    def _row(buf, r):  # absolute row index within buf's smem handle
        return b.add(r, b.const_i32(buf[1])) if buf[1] else r

    def _addr(buf):  # i64 LDS base address of this buf's region
        a = b.smem_addr_of(buf[0])
        return (
            b.smem_ptr_add(a, b.zext(b.const_i32(buf[1] * TK * 2), I64))
            if buf[1]
            else a
        )

    def issue_loads(gptr, eoff, row_base, rows, kt):
        """Global vector loads (in flight); return [(r, col, vec8)] to LDS-store."""
        p = (rows * TK) // BS
        out = []
        k_base = b.const_i32(kt * TK)
        for c in range(p // 8):
            e = b.add(b.mul(tid, b.const_i32(p)), b.const_i32(c * 8))
            r = b.div(e, b.const_i32(TK))
            col = b.mod(e, b.const_i32(TK))
            g = b.add(
                eoff,
                b.add(b.mul(b.add(row_base, r), b.const_i32(K)), b.add(k_base, col)),
            )
            out.append((r, col, b.global_load_vN(gptr, g, BF16, 8, align=16)))
        return out

    def store_regs(buf, regs):
        for r, col, v in regs:
            b.smem_store_vN(buf[0], [_row(buf, r), col], v, 8)

    # Per-(mi)/(nj) LDS row bases (lane-relative), reused every kk.
    a_rows = [
        b.add(
            b.add(b.mul(warp_row, b.const_i32(WTM)), b.const_i32(mi * AM)), ld.m_in_atom
        )
        for mi in range(mm)
    ]
    b_rows = [
        b.add(
            b.add(b.mul(warp_col, b.const_i32(WTN)), b.const_i32(nj * AN)), ld.n_in_atom
        )
        for nj in range(nn)
    ]
    k_lane = b.mul(ld.k_blk, b.const_i32(apl))

    def swz_col(col, row):
        # st_16x32 XOR swizzle: toggle the 16-half (32-byte) column group on
        # rows with bit 3 set -> col ^ (((row>>3)&1)<<4). Applied identically to
        # the DTL store column and the ds_read column so physical addresses
        # agree (bank-conflict-free reads, zero LDS overhead).
        if not swz:
            return col
        return b.xor(
            col,
            b.shl(b.mod(b.lshr(row, b.const_i32(3)), b.const_i32(2)), b.const_i32(4)),
        )

    # Inline-asm ds_read path (gated): single base LDS address + a compile-time
    # immediate byte offset per fragment ("ds_read_b128 base offset:N"),
    # bypassing the compiler's per-read address recompute. Result read as a
    # 16-byte i32x4 (forces an aligned 4-VGPR tuple, matching ds_read_b128's
    # write), then bitcast to the bf16 operand fragment.
    from rocke.core.ir import VectorType

    a_frag_ty = VectorType(BF16, apl)
    b_frag_ty = VectorType(BF16, bpl)
    raw_ty = VectorType(I32, apl // 2)
    a_base_row = b.add(b.mul(warp_row, b.const_i32(WTM)), ld.m_in_atom)
    b_base_row = b.add(b.mul(warp_col, b.const_i32(WTN)), ld.n_in_atom)

    # RRR transpose-read: CK TransposeLDSLayout lane formulas (proven in the
    # attention PV path) read a plain [TK,TN] LDS tile and deliver the MFMA
    # b-operand (B[n,k] per lane) without the strided/transposed scatter the
    # references' RRR layout would otherwise need. AK=32 -> two ds_read_tr16_b64
    # (read 0/1) concatenated into the 8-wide bf16 b-fragment.
    if b_rrr:
        from rocke.helpers.layouts import TransposeLdsReader

        tr_reader = TransposeLdsReader(K=AK, M=AN).bind(b, lane)
        # per-warp N base for atom nj added at read time; tr_reader.col is the
        # lane's column component within the 16-wide N tile.
        b_warp_n = b.mul(warp_col, b.const_i32(WTN))
        # Hoist the per-nj column addresses (constant across all K-chunks) so the
        # transpose-read addressing VALU runs once, not every chunk.
        b_n_cols = [
            b.add(b.add(b_warp_n, b.const_i32(nj * AN)), tr_reader.col)
            for nj in range(nn)
        ]

    def ds_read_imm(buf, base_row, col_swz, fi, frag_ty):
        base_off = b.mul(
            b.add(b.mul(base_row, b.const_i32(TK)), col_swz), b.const_i32(2)
        )
        addr = b.add(b.trunc(_addr(buf), I32), base_off)
        imm = fi * AM * TK * 2  # bytes (row stride * dtype)
        ds_w = apl * 16  # fragment bits: 8 bf16 -> b128, 4 bf16 -> b64
        raw = b.inline_asm(
            f"ds_read_b{ds_w} $0, $1 offset:{imm}",
            "=v,v",
            [addr],
            result_type=raw_ty,
            sideeffect=True,
        )
        return b.vec_bitcast(raw, frag_ty)

    # ds_read_b128_tr_b16 would halve the B LDS-read ops, but llvm.amdgcn.ds.read.
    # tr16.b128 is not in this ROCm's LLVM (only the b64 variant links) -> default off.
    tr128 = os.environ.get("TR128", "0") == "1"  # 1 wide transpose-read vs 2x b64

    def _read_b_rrr(bi, kk):
        # B-operand from the plain [TK,TN] LDS tile via CK transpose-reads.
        r0 = tr_reader.row(b, k_offset=kk * AK, read=0)
        out = []
        if tr128:
            # One ds_read_b128_tr_b16 delivers all 8 bf16 (the full 16x16x32
            # b-operand) per N-tile -> half the LDS-read ops of the 2x b64 form.
            for nj in range(nn):
                out.append(b.ds_read_tr16_b128(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16))
            return out
        r1 = tr_reader.row(b, k_offset=kk * AK, read=1)
        for nj in range(nn):
            f0 = b.ds_read_tr16_b64(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16)
            f1 = b.ds_read_tr16_b64(Bs[bi][0], r1, b_n_cols[nj], dtype=BF16)
            out.append(b.vec_concat(f0, f1))
        return out

    def read_chunk(ai, bi, kk):
        col = b.add(b.const_i32(kk * AK), k_lane)
        if asm_reads:
            ca = swz_col(col, a_base_row)
            a = [ds_read_imm(As[ai], a_base_row, ca, mi, a_frag_ty) for mi in range(mm)]
        else:
            a = [
                b.smem_load_vN(
                    As[ai][0],
                    _row(As[ai], a_rows[mi]),
                    swz_col(col, a_rows[mi]),
                    dtype=BF16,
                    n=apl,
                )
                for mi in range(mm)
            ]
        if b_rrr:
            return a, _read_b_rrr(bi, kk)
        if asm_reads:
            cb = swz_col(col, b_base_row)
            bb = [
                ds_read_imm(Bs[bi], b_base_row, cb, nj, b_frag_ty) for nj in range(nn)
            ]
            return a, bb
        bb = [
            b.smem_load_vN(
                Bs[bi][0],
                _row(Bs[bi], b_rows[nj]),
                swz_col(col, b_rows[nj]),
                dtype=BF16,
                n=bpl,
            )
            for nj in range(nn)
        ]
        return a, bb

    permmasched = os.environ.get("PERMMASCHED", "0") == "1"
    # CK HotLoopScheduler: emit explicit sched_group_barrier MFMA:DS-read groups so
    # the backend spreads the next chunk's (transpose) LDS reads evenly across the
    # current chunk's MFMAs, hiding the ds_read latency behind the matrix ops.
    cksched = os.environ.get("CKSCHED", "0") == "1"
    SGB_MFMA, SGB_DSREAD = 0x008, 0x100
    # Deep operand pipeline (full-drain current chunk, then issue next chunk's
    # ds_reads so they overlap the MFMAs). Default ON with asm reads: small but
    # consistent win (~+4 TF) over the compiler's schedule.
    deeppipe = os.environ.get("DEEPPIPE", "1") == "1"
    # BURSTPRIO: per-burst priority ping-pong. Drop s_setprio(0) for the
    # lgkmcnt-drain + ds_read section of each chunk and raise s_setprio(1) only
    # around the MFMA burst. With 2 waves/SIMD this yields the issue slots to the
    # sibling wave's MFMAs while this wave stalls on its read drain (and vice
    # versa) -- the producer/consumer overlap that hides ds_read+barrier latency.
    burstprio = os.environ.get("BURSTPRIO", "0") == "1"

    def _mma_block(a_fr, b_fr, acc, n_ds=0, store_after=None):
        nmma = mm * nn
        ds_done = 0
        for mi in range(mm):
            for nj in range(nn):
                idx = mi * nn + nj
                if permmasched:
                    # Per-mma scheduling: raise priority around EACH MFMA
                    # and drop a sched_barrier fence after it so the backend can't
                    # hoist the next chunk's ds_reads across the matrix op.
                    b.s_setprio(1)
                    acc[idx] = atom.emit(b, a_fr[mi], b_fr[nj], acc[idx])
                    b.s_setprio(0)
                    b.sched_barrier(0)
                else:
                    acc[idx] = atom.emit(b, a_fr[mi], b_fr[nj], acc[idx])
                if store_after is not None:
                    # EPIFUSE: this is acc[idx]'s final MFMA -> store it now so its
                    # stores overlap the remaining MFMAs of this last K-tile.
                    store_after(acc, idx)
                if cksched and n_ds:
                    # CK ratio: 1 MFMA, then a proportional slice of the in-flight
                    # ds_reads, so the N transpose-reads spread across the M MFMAs.
                    b.sched_group_barrier(SGB_MFMA, 1, 0)
                    pos = idx + 1
                    want = (pos * n_ds) // nmma
                    if want > ds_done:
                        b.sched_group_barrier(SGB_DSREAD, want - ds_done, 0)
                        ds_done = want

    def compute(ai, bi, acc=None, store_on_last=None):
        if acc is None:
            acc = accs
        if prio and not permmasched and not burstprio:
            b.s_setprio(1)
        if deeppipe and asm_reads:  # noqa: store_on_last handled in last chunk
            # Software-pipelined operand reads. Each iteration: (1) full-drain the
            # CURRENT chunk's ds_reads with lgkmcnt(0) -- safe, since at this point
            # only the current chunk's reads are outstanding (the next chunk hasn't
            # been issued yet), so no MFMA ever consumes a fragment whose LDS read
            # hasn't landed; (2) issue the NEXT chunk's ds_reads, which then overlap
            # the current chunk's MFMAs; (3) MFMAs. This overlaps reads under the
            # matrix ops without the out-of-order-completion hazard of a
            # partial lgkmcnt (ds_reads can retire out of order, so a partial count
            # is unsafe -> intermittent NaN).
            # ds_reads issued per next-chunk prefetch: mm A-frag reads + the B reads
            # (RRR transpose: nn x1 if tr128 else nn x2; RCR: nn).
            n_ds_b = (nn if tr128 else nn * 2) if b_rrr else nn
            n_ds = mm + n_ds_b
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                if burstprio:
                    b.s_setprio(0)  # yield to sibling wave during the read drain
                b.s_waitcnt(lgkmcnt=0)  # drain current operands (next not yet issued)
                na = nb = None
                if kk + 1 < kchunks:
                    na, nb = read_chunk(ai, bi, kk + 1)  # issue next -> overlaps MFMAs
                if burstprio:
                    b.s_setprio(1)  # raise for this chunk's MFMA burst
                if kk + 1 < kchunks:
                    _mma_block(a_fr, b_fr, acc, n_ds=n_ds)
                else:
                    _mma_block(a_fr, b_fr, acc, store_after=store_on_last)
                if kk + 1 < kchunks:
                    a_fr, b_fr = na, nb
            if burstprio:
                b.s_setprio(0)  # drop priority before the K-loop barrier
        else:
            # Register-prefetch-ahead: pull the next K-chunk's operand frags into
            # registers before issuing the current chunk's MFMAs (PF), so the
            # ds_read latency overlaps the matrix ops.
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                nxt = None
                if kk + 1 < kchunks and prefetch:
                    nxt = read_chunk(ai, bi, kk + 1)  # prefetch BEFORE current MFMAs
                if asm_reads:
                    # Opaque inline-asm ds_read: compiler won't auto-insert the
                    # lgkmcnt wait, so force a full drain before the consuming MFMA.
                    b.s_waitcnt(lgkmcnt=0)
                last_chunk = kk == kchunks - 1
                _mma_block(
                    a_fr, b_fr, acc, store_after=store_on_last if last_chunk else None
                )
                if kk + 1 < kchunks:
                    a_fr, b_fr = nxt if nxt is not None else read_chunk(ai, bi, kk + 1)
        if prio and not permmasched and not burstprio:
            b.s_setprio(0)

    if dtl:
        HALVES, DWORDS = 8, 4  # 16 bytes/lane
        cpr = TK // HALVES
        passes_a = (TM * TK // HALVES) // BS
        passes_b = (TN * TK // HALVES) // BS
        assert TK % HALVES == 0
        warp_id = b.div(tid, b.const_i32(64))
        wave_off = b.zext(b.mul(warp_id, b.const_i32(64 * 16)), I64)
        zsoff = b.const_i32(0)
        big = b.const_i32(0x7FFF0000)
        a_rsrc = b.buffer_rsrc(A, big)
        b_rsrc = b.buffer_rsrc(Bp, big)
        c_cpr = b.const_i32(cpr)
        c_hv = b.const_i32(HALVES)

        def dtl_load(rsrc, eoff, buf, row_base, kt, passes, coh):
            base = b.smem_ptr_add(_addr(buf), wave_off)
            for p in range(passes):
                lds = (
                    base
                    if p == 0
                    else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
                )
                ci = b.add(tid, b.const_i32(p * BS))
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                off = b.add(
                    eoff,
                    b.add(
                        b.mul(b.add(row_base, row), b.const_i32(K)),
                        b.add(b.const_i32(kt * TK), swz_col(col, row)),
                    ),
                )
                b.async_buffer_load_lds_addr(
                    rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
                )

        c_cpr_n = b.const_i32(TN // HALVES)  # N-fast groups per row (RRR B load)

        def dtl_load_b_rrr(buf, eoff, n_base, kt, coh):
            # Stage B[E,K,N] (N-contiguous) into a plain [TK,TN] LDS tile: row=k,
            # col=n, 8 contiguous N per lane. No swizzle (tr16 reads plain layout).
            base = b.smem_ptr_add(_addr(buf), wave_off)
            for p in range(passes_b):
                lds = (
                    base
                    if p == 0
                    else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
                )
                ci = b.add(tid, b.const_i32(p * BS))
                row_k = b.div(ci, c_cpr_n)
                col_n = b.mul(b.mod(ci, c_cpr_n), c_hv)
                off = b.add(
                    eoff,
                    b.add(
                        b.mul(b.add(b.const_i32(kt * TK), row_k), b.const_i32(N)),
                        b.add(n_base, col_n),
                    ),
                )
                b.async_buffer_load_lds_addr(
                    b_rsrc,
                    lds,
                    b.mul(off, b.const_i32(2)),
                    zsoff,
                    DWORDS,
                    coherency=coh,
                )

        def load_tile_dtl(buf, kt, tb):
            dtl_load(a_rsrc, tb["a_eoff"], As[buf], tb["m_base"], kt, passes_a, 0)
            if b_rrr:
                dtl_load_b_rrr(Bs[buf], tb["b_eoff"], tb["n_base"], kt, 0)
            else:
                dtl_load(b_rsrc, tb["b_eoff"], Bs[buf], tb["n_base"], kt, passes_b, 0)

        import os as _os_ls

        light_sync = _os_ls.environ.get("LIGHTSYNC", "0") == "1"

    def kloop(acc, tb, final_storer=None):
        if dtl:
            load_tile_dtl(0, 0, tb)  # prologue
            for kt in range(n_ktiles):
                cur = kt % 2
                sol = final_storer if kt == n_ktiles - 1 else None
                # The cooperative DTL load distributes the A/B tile rows across ALL
                # warps, so each warp's compute reads LDS rows written by *other*
                # warps -> the workgroup barrier is REQUIRED (dropping it fails
                # verification). DTL writes are vmcnt (buffer_load_lds), not
                # lgkmcnt, and the compute ds_reads are already drained by the
                # MFMAs, so lgkmcnt(0) in a full sync() is redundant: the minimal
                # correct barrier is vmcnt(0) + s_barrier (LIGHTSYNC).
                if light_sync:
                    b.s_waitcnt(vmcnt=0)
                    b.s_barrier_bare()
                else:
                    b.sync()  # drain cur tile's DTL writes + barrier
                if kt + 1 < n_ktiles:
                    load_tile_dtl(1 - cur, kt + 1, tb)  # next-tile loads overlap
                compute(cur, cur, acc, store_on_last=sol)
        elif db:
            ar = issue_loads(A, tb["a_eoff"], tb["m_base"], TM, 0)
            br = issue_loads(Bp, tb["b_eoff"], tb["n_base"], TN, 0)
            for kt in range(n_ktiles):
                cur = kt % 2
                store_regs(As[cur], ar)
                store_regs(Bs[cur], br)
                nxt = None
                if kt + 1 < n_ktiles:
                    nxt = (
                        issue_loads(A, tb["a_eoff"], tb["m_base"], TM, kt + 1),
                        issue_loads(Bp, tb["b_eoff"], tb["n_base"], TN, kt + 1),
                    )
                b.sync()
                compute(cur, cur, acc)  # double-buffer => no trailing sync needed
                if nxt is not None:
                    ar, br = nxt
        else:
            for kt in range(n_ktiles):
                store_regs(As[0], issue_loads(A, tb["a_eoff"], tb["m_base"], TM, kt))
                store_regs(Bs[0], issue_loads(Bp, tb["b_eoff"], tb["n_base"], TN, kt))
                b.sync()
                compute(0, 0, acc)
                b.sync()

    # Epilogue. The C-store *address* depends only on (lane, mi, nj, i) -- NOT on
    # the accumulator value -- and decomposes as:
    #   addr = [c_eoff + (m_base+warp_m_off)*N + (n_base+warp_n_off)]   (per-lane base L)
    #          + (row_in[i]*N + col_in)                                 (per-(lane,i) only!)
    #          + (mi*AM*N + nj*AN)                                      (compile-time const)
    # Since (row_in[i]*N + col_in) is independent of (mi,nj), the expensive row*N
    # multiply runs c_per_lane times (per i), not mm*nn*c_per_lane times -- and
    # each store reduces to base + a compile-time constant. This collapses the
    # serial-tail address VALU (~128 muls + 64-bit addr math) that hides behind
    # no MFMA in this one-tile-per-block kernel.
    cN = b.const_i32(N)
    warp_m_off = b.mul(warp_row, b.const_i32(WTM))
    warp_n_off = b.mul(warp_col, b.const_i32(WTN))
    nostore = os.environ.get("NOSTORE", "0") == "1"  # diagnostic: skip C stores
    # diagnostic: keep all stores (so accs are consumed -> MFMAs survive DCE) but
    # collapse their addresses to c_per_lane per-lane slots -> write-combined, ~0
    # HBM store traffic. (full - storesink) isolates the real store-BW cost with
    # compute intact, unlike NOSTORE which DCEs the whole MFMA chain.
    storesink = os.environ.get("STORESINK", "0") == "1"

    def store_c(addr, val):
        b.global_store(C, addr, val, align=2)

    # EPIFUSE: instead of a serial store tail, issue each accumulator's stores right
    # after its FINAL MFMA (last K-tile), so the store-issue + address VALU overlap
    # the remaining MFMAs of that K-tile instead of running exposed after the loop.
    epifuse = os.environ.get("EPIFUSE", "1") == "1"  # +~10-20 TF, default on

    def make_storer(tb):
        """Precompute the per-i lane base addresses for tile `tb`; return a
        store_one(acc, idx) that writes acc[idx]'s c_per_lane elements."""
        L = b.add(
            tb["c_eoff"],
            b.add(
                b.mul(b.add(tb["m_base"], warp_m_off), cN),
                b.add(tb["n_base"], warp_n_off),
            ),
        )
        Li = []
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            Li.append(b.add(L, b.add(b.mul(row_in, cN), col_in)))

        def store_one(acc, idx):
            mi, nj = idx // nn, idx % nn
            mn_const = mi * AM * N + nj * AN  # compile-time
            for i in range(atom.c_per_lane):
                addr = b.add(Li[i], b.const_i32(mn_const))
                store_c(addr, b.cast_f32_to(b.vec_extract(acc[idx], i), BF16))

        return store_one

    # EPIFUSE applies only to the DTL scalar-store path (the dtl K-loop is the only
    # branch that threads the per-acc store callback; non-dtl/cshuf/sink fall back
    # to the separate epilogue() to stay correct).
    epifuse_active = epifuse and dtl and not cshuf and not nostore and not storesink

    def epilogue(acc, tb):
        if nostore:
            return
        if cshuf:
            # Store-coalescing epilogue: re-use the now-free flat LDS buffer AB as
            # the [TM][TN] C-stage. (1) scatter each lane's accumulators into AB at
            # their C-tile-local (row,col) [LDS ds_write, cheaper than global
            # scatter]; (2) cooperatively read AB in contiguous 8-wide runs and
            # issue WIDE coalesced global stores (b128).
            cTN = b.const_i32(TN)
            cTK = b.const_i32(TK)
            b.sync()  # last compute's AB reads done before reuse as C-stage
            idx = 0
            for mi in range(mm):
                for nj in range(nn):
                    for i in range(atom.c_per_lane):
                        ro, ci = atom.lane_to_output(b, lane, i)
                        lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), ro)
                        lc = b.add(b.add(warp_n_off, b.const_i32(nj * AN)), ci)
                        flat = b.add(b.mul(lr, cTN), lc)
                        val = b.cast_f32_to(b.vec_extract(acc[idx], i), BF16)
                        b.smem_store_vN(
                            AB, [b.div(flat, cTK), b.mod(flat, cTK)], val, 1
                        )
                    idx += 1
            b.sync()
            nblk = (TM * TN) // (BS * 8)
            for blk in range(nblk):
                flat = b.add(b.const_i32(blk * BS * 8), b.mul(tid, b.const_i32(8)))
                vec = b.smem_load_vN(
                    AB, b.div(flat, cTK), b.mod(flat, cTK), dtype=BF16, n=8
                )
                r = b.div(flat, cTN)
                c = b.mod(flat, cTN)
                addr = b.add(
                    tb["c_eoff"],
                    b.add(b.mul(b.add(tb["m_base"], r), cN), b.add(tb["n_base"], c)),
                )
                b.global_store_vN(C, addr, vec, n=8, align=16)
            return

        L = b.add(
            tb["c_eoff"],
            b.add(
                b.mul(b.add(tb["m_base"], warp_m_off), cN),
                b.add(tb["n_base"], warp_n_off),
            ),
        )
        # per-i base = L + row_in[i]*N + col_in  (lane-dependent, mi/nj-independent)
        Li = []
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            Li.append(b.add(L, b.add(b.mul(row_in, cN), col_in)))
        if storesink:
            # Reduce ALL accumulators into c_per_lane sums (data-dependent on every
            # MFMA -> none get DCE'd), then store only those c_per_lane values per
            # lane. Compute stays fully live; store traffic ~ c_per_lane/(mm*nn*
            # c_per_lane) of full. (full - storesink) = real store-BW cost.
            sums = [None] * atom.c_per_lane
            for idx in range(mm * nn):
                for i in range(atom.c_per_lane):
                    e = b.vec_extract(acc[idx], i)
                    sums[i] = e if sums[i] is None else b.fadd(sums[i], e)
            for i in range(atom.c_per_lane):
                store_c(Li[i], b.cast_f32_to(sums[i], BF16))
            return

        store_one = make_storer(tb)
        for idx in range(mm * nn):
            store_one(acc, idx)

    def zero(acc):
        for i in range(len(acc)):
            acc[i] = atom.zero_acc(b)

    def do_tile(tb):
        # Each output tile is independent -> fresh accumulators per tile. compute()
        # rewrites acc[idx] in place across the K-loop, so re-zero here.
        zero(accs)
        if epifuse_active:
            # Fuse stores into the last K-tile's MFMAs (no separate store tail).
            kloop(accs, tb, final_storer=make_storer(tb))
        else:
            kloop(accs, tb)
            epilogue(accs, tb)

    # Persistent grid: when tpb>1, each block walks a grid-strided set of output
    # tiles. PIPE adds a second accumulator set so tile j's epilogue stores (which
    # read acc_prev) overlap tile (j+1)'s K-loop MFMAs (which write acc_cur) --
    # without a WAR hazard forcing the K-loop's s_waitcnt to drain the stores.
    pipe = os.environ.get("PIPE", "0") == "1"
    accs2 = [atom.zero_acc(b) for _ in range(mm * nn)] if (pipe and tpb > 1) else None
    n_mt = (M + TM - 1) // TM
    n_nt = (N + TN - 1) // TN
    tiles_per_expert = n_mt * n_nt
    total_tiles = E * tiles_per_expert

    def tile_coords(t):
        c_tpe = b.const_i32(tiles_per_expert)
        c_nnt = b.const_i32(n_nt)
        expert = b.div(t, c_tpe)
        rem = b.mod(t, c_tpe)
        return b.mod(rem, c_nnt), b.div(rem, c_nnt), expert  # (nt, mt, expert)

    vgprprobe = os.environ.get("VGPRPROBE", "0") == "1"
    if vgprprobe:
        # Feasibility probe for the bf16-packed store/compute overlap: keep tile
        # T's accumulators, cast to bf16 (~64 VGPR), LIVE across tile T+1's full
        # K-loop (128 f32 accs). The resulting .vgpr_count tells us whether the
        # overlap fits <=256 (2 waves/SIMD, no spill). Not meant to run correctly.
        tb = set_tile(b.block_id_x(), b.block_id_y(), b.block_id_z())
        zero(accs)
        kloop(accs, tb)  # tile T
        # PACKED carry: each f32 accumulator vector -> one bf16x{c_per_lane} vector
        # (c_per_lane=4 bf16 = 8 B = 2 VGPR). mm*nn=32 -> ~64 VGPR (vs 128 unpacked).
        carry = [b.vec_cast_f32_to(accs[idx], BF16) for idx in range(mm * nn)]
        zero(accs)
        kloop(accs, tb)  # tile T+1 compute, `carry` still live
        epilogue(accs, tb)
        cbase = b.add(tb["c_eoff"], b.mul(lane, b.const_i32(atom.c_per_lane)))
        for idx, v in enumerate(carry):
            addr = b.add(cbase, b.const_i32(idx * BS * atom.c_per_lane))
            b.global_store_vN(C, addr, v, n=atom.c_per_lane, align=8)  # packed sink
        b.ret()
        return b.kernel, BS, TM, TN

    if tpb <= 1:
        tb = set_tile(b.block_id_x(), b.block_id_y(), b.block_id_z())
        do_tile(tb)
    else:
        # Grid-strided decode assumes every block walks exactly tpb valid tiles.
        assert (
            total_tiles % tpb == 0
        ), f"total_tiles={total_tiles} not divisible by tpb={tpb}"
        n_blocks = total_tiles // tpb
        bf = b.block_id_x()
        c_nblk = b.const_i32(n_blocks)
        tids = [b.add(bf, b.mul(b.const_i32(j), c_nblk)) for j in range(tpb)]
        tbs = [set_tile(*tile_coords(t)) for t in tids]
        if accs2 is None:
            for tb in tbs:
                do_tile(tb)
        else:
            # Software-pipelined across tiles with ping-pong accumulators:
            #   compute(t0); for t: { store(t-1) || compute(t) }; store(last)
            bufs = [accs, accs2]
            zero(bufs[0])
            kloop(bufs[0], tbs[0])
            for j in range(1, tpb):
                cur = j % 2
                epilogue(bufs[1 - cur], tbs[j - 1])  # stores for tile j-1
                zero(bufs[cur])
                kloop(bufs[cur], tbs[j])  # compute tile j into the other set
            epilogue(bufs[(tpb - 1) % 2], tbs[tpb - 1])
    b.ret()
    return b.kernel, BS, TM, TN


def _main() -> int:
    import torch  # local import: only the benchmark/verify path needs torch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from rocke.instances.gfx950.grouped_gemm import GroupedGemmSpec, build_grouped_gemm

    M_total, N, K, E = 524288, 1024, 512, 64
    m = M_total // E
    flops = 2 * M_total * N * K

    # Build spec from env (defaults = production opts)
    spec = GroupedGemmSpec(
        M=m,
        N=N,
        K=K,
        E=E,
        TM=int(os.environ.get("TM", 256)),
        TN=int(os.environ.get("TN", 256)),
        TK=int(os.environ.get("TK", 64)),
        WM=int(os.environ.get("WM", 2)),
        WN=int(os.environ.get("WN", 4)),
        dtl=os.environ.get("DTL", "1") == "1",
        db=os.environ.get("DB", "1") == "1",
        prio=os.environ.get("PRIO", "1") == "1",
        swz=os.environ.get("SWZ", "1") == "1",
        chiplet=os.environ.get("CHIP", "1") == "1",
        chiplet_xcds=int(os.environ.get("XCDS", "8")),
        chiplet_wgm=int(os.environ.get("WGM", "8")),
        asm_reads=os.environ.get("ASM", "1") == "1",
        deeppipe=os.environ.get("DEEPPIPE", "1") == "1",
        epifuse=os.environ.get("EPIFUSE", "1") == "1",
        b_rrr=os.environ.get("BRRR", "0") == "1",
        tpb=int(os.environ.get("TPB", "1")),
    )

    kernel, BS, tm, tn = build_grouped_gemm(spec)
    print(
        f"[ggemm] tile={spec.TM}x{spec.TN}x{spec.TK} warps={spec.WM}x{spec.WN} BS={BS} "
        f"dtl={spec.dtl} prio={spec.prio} swz={spec.swz} asm={spec.asm_reads} "
        f"deeppipe={spec.deeppipe} epifuse={spec.epifuse} "
        f"chiplet={spec.chiplet} wgm={spec.chiplet_wgm} xcds={spec.chiplet_xcds} "
        f"brrr={spec.b_rrr} tpb={spec.tpb}"
    )
    art = compile_kernel_via_hipcc(kernel, arch="gfx950")
    print(f"[ggemm] hipcc built {kernel.name}: {len(art.hsaco)} B")

    from rocke.instances.gfx950.grouped_gemm import grouped_gemm_signature

    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=grouped_gemm_signature(),
        cache_key=(kernel.name,),
    )
    dt = torch.bfloat16
    A = torch.randn(E, m, K, dtype=dt, device="cuda")
    C = torch.empty(E, m, N, dtype=dt, device="cuda")
    if spec.b_rrr:
        # RRR: weights as [E,K,N] (N-contiguous), like the Triton/CK references.
        B = torch.randn(E, K, N, dtype=dt, device="cuda") * 0.05
        ref = torch.bmm(A.float(), B.float())  # A[m,K] @ B[K,N]
    else:
        B = torch.randn(E, N, K, dtype=dt, device="cuda") * 0.05
        ref = torch.bmm(A.float(), B.float().transpose(-1, -2))
    if spec.tpb > 1:
        total_tiles = E * math.ceil(N / tn) * math.ceil(m / tm)
        grid = (math.ceil(total_tiles / spec.tpb), 1, 1)
    else:
        grid = (math.ceil(N / tn), math.ceil(m / tm), E)
    blk = (BS, 1, 1)

    def call():
        launcher(
            {
                "A": A.data_ptr(),
                "B": B.data_ptr(),
                "C": C.data_ptr(),
                "M": m,
                "N": N,
                "K": K,
                "stride_a": m * K,
                "stride_b": N * K,
                "stride_c": m * N,
            },
            config=LaunchConfig(stream=0, grid=grid, block=blk),
        )

    C.zero_()
    call()
    torch.cuda.synchronize()
    err = (C.float() - ref).abs().max().item()
    print(f"[ggemm] grid={grid} max_abs={err:.4f}  {'PASS' if err < 0.1 else 'FAIL'}")
    if (
        err >= 0.1
        and os.environ.get("NOSTORE", "0") != "1"
        and os.environ.get("STORESINK", "0") != "1"
    ):
        return 1
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            call()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 50)
    ts.sort()
    print(
        f"[ggemm] med={flops / (ts[2] * 1e9):.1f} TF  peak={flops / (ts[0] * 1e9):.1f} TF"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
