"""Ragged grouped GEMM kernel builder for gfx950 (MI355X).

Pure ragged grouped bf16 GEMM with variable M per expert. No gather/scatter,
no top-k, no routing weights. Each expert owns a contiguous slice of A and C;
a per-tile schedule (expert_ids / m_offsets / m_valid) addresses the rows.

    A = [M_total, K]     (row-major, expert-sorted, contiguous per expert)
    B = [E, N, K] (NT)   or [E, K, N] (NN, b_rrr)
    C = [M_total + 1, N] (last row is the sink for padded tile rows)

Each block computes one (m_tile, n_tile): it reads A[m_base : m_base+TM] and
B[expert] directly and writes C[m_base : m_base+TM] directly. Tile rows past the
expert's valid count (m_valid) redirect to the sink row instead of a branch.

CDNA4 optimizations: DTL cooperative loads, epilogue fusion, inline-asm ds_reads,
chiplet-aware tiling, int64 C addressing for large shapes.

Author: yraparti <yraparti@amd.com>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import BF16, I32, I64, IRBuilder, PtrType, VectorType
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes, mfma_atom_for_dtype
from rocke.helpers.spec import SignatureBuilder


@dataclass(frozen=True)
class RaggedGemmSpec:
    """Ragged grouped GEMM specification (gfx950/CDNA4).

    Geometry (problem + tile sizes):
        N, K, E: problem dimensions (output width, reduction, num experts)
        TM, TN, TK: tile sizes (M/N/K tile per block)
        WM, WN: warp layout (num warps in M/N dimensions)

    Production optimizations (defaults = full opts):
        hoist: Hoist loop-invariant A/B-load offsets
        epifuse: Fuse stores into final K-tile MFMAs (default epilogue)
        swz: st_16x32 XOR swizzle (bank-conflict-free LDS reads)
        asm_reads: Inline-asm ds_read with manual lgkmcnt
        deeppipe: Deep operand prefetch pipeline
        prio: s_setprio(1) around MFMA blocks
        burstprio: Toggle s_setprio per k-chunk
        chiplet: Chiplet-aware super-tile remap (L2 locality)
        pin: Hoist wave-uniform indices to SGPRs
        b_rrr: NN weight layout B=[E,K,N] (vs NT B=[E,N,K])

    Experimental levers (measured NOT wins, default off):
        cshuf: C-shuffle wide-store epilogue (contiguous C rows -> coalesced)
        cksched: CK-HotLoop sched_group_barrier interleave
        permmasched: Per-MFMA s_setprio + sched_barrier
        tr128: ds_read_tr16_b128 (vs 2x b64, needs ROCm support)

    Hardening (optional compile-time constants):
        nmt_const: Bake num_m_tiles as immediate (div-by-const in chiplet remap)
    """

    # Problem dimensions
    N: int
    K: int
    E: int

    # Tile geometry
    TM: int = 256
    TN: int = 256
    TK: int = 64
    WM: int = 2
    WN: int = 4

    # Production optimizations (defaults = full opts)
    hoist: bool = True  # Offset hoisting
    epifuse: bool = True  # Epilogue fusion (default epilogue)
    swz: bool = True  # LDS bank-conflict swizzle
    asm_reads: bool = True  # Inline-asm ds_read
    deeppipe: bool = True  # Deep prefetch pipeline
    prio: bool = True  # s_setprio around MFMAs
    burstprio: bool = False  # Per-chunk prio toggle
    chiplet: bool = True  # Super-tile remap
    pin: bool = True  # SGPR index hoisting
    b_rrr: bool = False  # NN layout (vs NT default)

    # Chiplet remap params (active when chiplet=True)
    chiplet_wgm: int = 8  # Workgroups per super-tile
    chiplet_xcds: int = 8  # Number of XCDs

    # Experimental levers (measured NOT wins)
    cshuf: bool = False  # C-shuffle wide stores
    cksched: bool = False  # sched_group_barrier interleave
    permmasched: bool = False  # Per-MFMA barriers
    tr128: bool = False  # b128 transpose read (ROCm support)

    # Hardening (optional compile-time constants)
    nmt_const: Optional[int] = None

    # Kernel name
    name: str = "ragged_gemm"


def build_ragged_gemm(spec: RaggedGemmSpec):
    """Build the pure ragged grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. N/K/E are compile-time; the number
    of m-tiles is a runtime grid dimension (depends on the token distribution),
    so it never appears in the kernel body -- each block computes exactly one
    (m_tile, n_tile).
    """
    N, K, E = spec.N, spec.K, spec.E
    TM, TN, TK = spec.TM, spec.TN, spec.TK
    WM, WN = spec.WM, spec.WN
    nmt_const = spec.nmt_const

    atom = mfma_atom_for_dtype("bf16", 16, 16)  # 16x16x32 bf16
    AM, AN, AK = atom.m, atom.n, atom.k
    apl, bpl = atom.a_per_lane, atom.b_per_lane  # 8, 8
    BS = WM * WN * 64
    WTM, WTN = TM // WM, TN // WN
    mm, nn = WTM // AM, WTN // AN
    kchunks = TK // AK
    n_ktiles = K // TK
    assert TK % AK == 0 and TM % (WM * AM) == 0 and TN % (WN * AN) == 0
    assert (TM * TK) % BS == 0 and (TN * TK) % BS == 0

    # Unpack levers from spec
    swz = spec.swz
    asm_reads = spec.asm_reads
    deeppipe = spec.deeppipe
    burstprio = spec.burstprio
    prio = spec.prio
    chiplet = spec.chiplet
    pin = spec.pin
    chiplet_wgm = spec.chiplet_wgm
    chiplet_xcds = spec.chiplet_xcds
    n_pid_n = (N + TN - 1) // TN
    b_rrr = spec.b_rrr
    hoist = spec.hoist
    epifuse = spec.epifuse
    nbuf = 2
    cshuf = spec.cshuf
    if cshuf and (b_rrr or TM != TN or TN != 2 * nbuf * TK):
        cshuf = False
    cksched = spec.cksched
    permmasched = spec.permmasched
    SGB_MFMA, SGB_DSREAD = 0x008, 0x100
    tr128 = spec.tr128
    if cshuf:
        epifuse = False

    b = IRBuilder(spec.name)
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    EIDS = b.param(
        "expert_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    MOFF = b.param(
        "m_offsets", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    MVAL = b.param(
        "m_valid", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    M = b.param("M", I32)  # total valid rows; also the sink row index for padding
    NMT = b.param("num_m_tiles", I32)  # runtime grid.y extent (for chiplet remap)
    # HARDEN: for a fixed ("hardened") schedule, bake num_m_tiles as a compile-time
    # constant. This turns the chiplet super-tile remap's div/mod by num_m_tiles
    # from a runtime integer-reciprocal sequence (run every tile) into a
    # divide-by-constant. The param stays in the ABI.
    NMT_v = b.const_i32(nmt_const) if nmt_const is not None else NMT

    tid = b.thread_id_x()
    warp = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    warp_row = b.div(warp, b.const_i32(WN))
    warp_col = b.mod(warp, b.const_i32(WN))
    ld = decode_mfma_lanes(b, atom, lane)

    def U(v):
        return b.to_sgpr_u32(v) if pin else v

    cK = b.const_i32(K)
    cN64 = b.const_i64(N)  # wide N stride: the C row offset orow*N can exceed 2^31
    # (M_total*N > 2.1e9 at large row counts / N), so the row term must be computed
    # in i64 to avoid signed-int32 address overflow.
    c_npn = b.const_i32(n_pid_n)

    def c_rowbase(orow, col_i32):
        # i64 C base address = orow*N + col, with orow*N widened to 64-bit and the
        # (small, < N) column offset zext'd in. Callers add compile-time per-n-tile
        # constants (nj*AN) to this.
        return b.add(b.mul(b.zext(orow, I64), cN64), b.zext(col_i32, I64))

    # Per-tile coords, assigned by set_tile(); the DTL-load / epilogue closures
    # read these as free variables.
    m_base = n_base = m_valid = expert = b_eoff = None
    # Per-pass A-load / B-load offsets, precomputed once per tile (HOIST lever);
    # read as free variables by dtl_load_a() / dtl_load_b*().
    a_load_base = None
    b_load_base = None

    def set_tile(wgid):
        # Decode a flat workgroup id -> (m_tile, n_tile). The chiplet/super-tile
        # remap keeps adjacent m-tiles on the same XCD for L2 reuse (dynamic
        # num_pid_m = num_m_tiles).
        nonlocal m_base, n_base, m_valid, expert, b_eoff
        if chiplet:
            from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

            sw = chiplet_aware_super_tile_dynamic(
                b,
                wgid,
                num_pid_m=NMT_v,
                num_pid_n=c_npn,
                wgm=chiplet_wgm,
                num_xcds=chiplet_xcds,
                chunk_size=64,
            )
            mt, nt = sw.row, sw.col
        else:
            mt, nt = b.div(wgid, c_npn), b.mod(wgid, c_npn)
        m_base = U(b.global_load(MOFF, mt, I32))  # absolute A/C row start of tile
        m_valid = U(b.global_load(MVAL, mt, I32))  # valid rows in this tile (<= TM)
        n_base = U(b.mul(nt, b.const_i32(TN)))
        expert = U(b.global_load(EIDS, mt, I32))
        b_eoff = U(b.mul(expert, b.const_i32(N * K)))

    # A/B LDS buffers are carried as (buffer, row_base) tuples so CSHUF can slice
    # them out of one flat pool (and reuse that pool as the C-stage). Non-CSHUF
    # keeps distinct allocations with row_base=0.
    if cshuf:
        # One flat [2*nbuf*TM, TK] pool: As0..As{nbuf-1} then Bs0..Bs{nbuf-1} as
        # row bands; the epilogue reuses the whole pool as the [TM,TN] C-stage
        # (TM*TN == 2*nbuf*TM*TK by the CSHUF guard above).
        AB = b.smem_alloc(BF16, [2 * nbuf * TM, TK], name_hint="AB")
        As = [(AB, i * TM) for i in range(nbuf)]
        Bs = [(AB, (nbuf + i) * TM) for i in range(nbuf)]
    else:
        AB = None
        As = [
            (b.smem_alloc(BF16, [TM, TK], name_hint=f"As{i}"), 0) for i in range(nbuf)
        ]
        # RRR stages B as [TK,TN] (K rows, N cols) for the transpose-read; RCR uses
        # [TN,TK] (N rows, K cols) for the direct ds_read.
        b_shape = [TK, TN] if b_rrr else [TN, TK]
        Bs = [(b.smem_alloc(BF16, b_shape, name_hint=f"Bs{i}"), 0) for i in range(nbuf)]

    def _addr(buf):  # i64 LDS base address of this buf's band
        a = b.smem_addr_of(buf[0])
        return (
            b.smem_ptr_add(a, b.zext(b.const_i32(buf[1] * TK * 2), I64))
            if buf[1]
            else a
        )

    def _row(buf, r):  # absolute row within buf's smem handle (adds the band base)
        return b.add(r, b.const_i32(buf[1])) if buf[1] else r

    accs = [atom.zero_acc(b) for _ in range(mm * nn)]

    warp_m_off = b.mul(warp_row, b.const_i32(WTM))
    warp_n_off = b.mul(warp_col, b.const_i32(WTN))

    def swz_col(col, row):
        # st_16x32 XOR swizzle (bank-conflict-free ds_reads): col ^ (((row>>3)&1)<<4).
        if not swz:
            return col
        return b.xor(
            col,
            b.shl(b.mod(b.lshr(row, b.const_i32(3)), b.const_i32(2)), b.const_i32(4)),
        )

    # ---- operand-read setup (RCR layout, mirrors the dense kernel) ----
    a_frag_ty = VectorType(BF16, apl)
    b_frag_ty = VectorType(BF16, bpl)
    raw_ty = VectorType(I32, apl // 2)
    a_base_row = b.add(b.mul(warp_row, b.const_i32(WTM)), ld.m_in_atom)
    b_base_row = b.add(b.mul(warp_col, b.const_i32(WTN)), ld.n_in_atom)
    k_lane = b.mul(ld.k_blk, b.const_i32(apl))
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

    # RRR transpose-read setup: read a plain [TK,TN] LDS tile and deliver the
    # MFMA b-operand (B[n,k] per lane) via CK's TransposeLdsReader. AK=32 -> two
    # ds_read_tr16_b64 (read 0/1) concatenated into the 8-wide b-fragment.
    if b_rrr:
        from rocke.helpers.layouts import TransposeLdsReader

        tr_reader = TransposeLdsReader(K=AK, M=AN).bind(b, lane)
        b_warp_n = b.mul(warp_col, b.const_i32(WTN))
        b_n_cols = [
            b.add(b.add(b_warp_n, b.const_i32(nj * AN)), tr_reader.col)
            for nj in range(nn)
        ]
        # HOIST: the transpose-read LDS row indices depend only on the k-chunk
        # position within the TK tile (kk), not on the K-tile -- the LDS layout is
        # identical every K-tile. Precompute the (read0, read1) rows for each kk
        # once instead of recomputing them in every _read_b_rrr call.
        tr_rows = [
            (
                tr_reader.row(b, k_offset=kk * AK, read=0),
                tr_reader.row(b, k_offset=kk * AK, read=1),
            )
            for kk in range(kchunks)
        ]

    def _read_b_rrr(bi, kk):
        r0, r1 = tr_rows[kk]
        out = []
        if tr128:
            # One b128 transpose-read delivers the full 8-wide B fragment per nj.
            for nj in range(nn):
                out.append(b.ds_read_tr16_b128(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16))
            return out
        for nj in range(nn):
            f0 = b.ds_read_tr16_b64(Bs[bi][0], r0, b_n_cols[nj], dtype=BF16)
            f1 = b.ds_read_tr16_b64(Bs[bi][0], r1, b_n_cols[nj], dtype=BF16)
            out.append(b.vec_concat(f0, f1))
        return out

    def ds_read_imm(buf, base_row, col_swz, fi, frag_ty):
        base_off = b.mul(
            b.add(b.mul(base_row, b.const_i32(TK)), col_swz), b.const_i32(2)
        )
        addr = b.add(b.trunc(_addr(buf), I32), base_off)
        imm = fi * AM * TK * 2
        ds_w = apl * 16
        raw = b.inline_asm(
            f"ds_read_b{ds_w} $0, $1 offset:{imm}",
            "=v,v",
            [addr],
            result_type=raw_ty,
            sideeffect=True,
        )
        return b.vec_bitcast(raw, frag_ty)

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

    def _mma_block(a_fr, b_fr, acc, store_row_cb=None, n_ds=0):
        # EPIFUSE: after finishing an m-row's nn MFMAs on the final K-chunk, store
        # that row immediately so its stores overlap the later m-rows' MFMAs.
        # CKSCHED: interleave 1 MFMA + a proportional slice of the next chunk's
        # in-flight ds_reads so the backend spreads reads across the matrix ops.
        nmma = mm * nn
        ds_done = 0
        for mi in range(mm):
            for nj in range(nn):
                idx = mi * nn + nj
                if permmasched:
                    b.s_setprio(1)
                    acc[idx] = atom.emit(b, a_fr[mi], b_fr[nj], acc[idx])
                    b.s_setprio(0)
                    b.sched_barrier(0)
                else:
                    acc[idx] = atom.emit(b, a_fr[mi], b_fr[nj], acc[idx])
                if cksched and n_ds:
                    b.sched_group_barrier(SGB_MFMA, 1, 0)
                    want = ((idx + 1) * n_ds) // nmma
                    if want > ds_done:
                        b.sched_group_barrier(SGB_DSREAD, want - ds_done, 0)
                        ds_done = want
            if store_row_cb is not None:
                store_row_cb(acc, mi)

    # ds_reads issued per next-chunk prefetch: mm A-frags + the B reads (RRR
    # transpose: nn x2 b64; RCR: nn b128). Used to size the CKSCHED interleave.
    n_ds_prefetch = mm + (nn * 2 if b_rrr else nn)

    def compute(ai, bi, acc, store_on_last=None):
        if prio and not burstprio:
            b.s_setprio(1)
        if deeppipe and asm_reads:
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                if burstprio:
                    b.s_setprio(0)
                b.s_waitcnt(lgkmcnt=0)
                na = nb = None
                has_next = kk + 1 < kchunks
                if has_next:
                    na, nb = read_chunk(ai, bi, kk + 1)
                if burstprio:
                    b.s_setprio(1)
                sr = store_on_last if kk == kchunks - 1 else None
                _mma_block(
                    a_fr,
                    b_fr,
                    acc,
                    store_row_cb=sr,
                    n_ds=n_ds_prefetch if has_next else 0,
                )
                if has_next:
                    a_fr, b_fr = na, nb
            if burstprio:
                b.s_setprio(0)
        else:
            a_fr, b_fr = read_chunk(ai, bi, 0)
            for kk in range(kchunks):
                if asm_reads:
                    b.s_waitcnt(lgkmcnt=0)
                sr = store_on_last if kk == kchunks - 1 else None
                _mma_block(a_fr, b_fr, acc, store_row_cb=sr)
                if kk + 1 < kchunks:
                    a_fr, b_fr = read_chunk(ai, bi, kk + 1)
        if prio and not burstprio:
            b.s_setprio(0)

    # ---- DTL cooperative direct-to-LDS loads ----
    HALVES, DWORDS = 8, 4  # 16 bytes/lane
    cpr = TK // HALVES
    passes_a = (TM * TK // HALVES) // BS
    passes_b = (TN * TK // HALVES) // BS
    assert TK % HALVES == 0
    wave_off = b.zext(b.mul(warp, b.const_i32(64 * 16)), I64)
    zsoff = b.const_i32(0)
    big = b.const_i32(0x7FFF0000)
    a_rsrc = b.buffer_rsrc(A, big)
    b_rsrc = b.buffer_rsrc(Bp, big)
    c_cpr = b.const_i32(cpr)
    c_hv = b.const_i32(HALVES)
    c_cpr_n = b.const_i32(TN // HALVES)  # N-fast groups per row (RRR B load)

    def precompute_b():
        # HOIST: the B DTL-load source offset per pass is loop-invariant across
        # K-tiles except for a compile-time +kt*TK (RCR) / +kt*TK*N (RRR). Compute
        # the invariant part once per tile (div/mod + row*stride + expert/n base +
        # swizzle) and hold it; the K-loop then adds a compile-time constant.
        nonlocal b_load_base
        vals = []
        for p in range(passes_b):
            ci = b.add(tid, b.const_i32(p * BS))
            if b_rrr:
                row_k = b.div(ci, c_cpr_n)
                col_n = b.mul(b.mod(ci, c_cpr_n), c_hv)
                vals.append(
                    b.add(
                        b_eoff,
                        b.add(b.mul(row_k, b.const_i32(N)), b.add(n_base, col_n)),
                    )
                )
            else:
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                vals.append(
                    b.add(
                        b_eoff,
                        b.add(
                            b.mul(b.add(n_base, row), cK),
                            swz_col(col, row),
                        ),
                    )
                )
        b_load_base = vals

    def dtl_load_b_rrr(buf, kt, coh):
        # B=[E,K,N] (N-contiguous): stage into a plain [TK,TN] LDS tile (row=k,
        # col=n, no swizzle -- tr16 reads the plain layout).
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_b):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            if hoist:
                off = b.add(b_load_base[p], b.const_i32(kt * TK * N))
            else:
                ci = b.add(tid, b.const_i32(p * BS))
                row_k = b.div(ci, c_cpr_n)
                col_n = b.mul(b.mod(ci, c_cpr_n), c_hv)
                off = b.add(
                    b_eoff,
                    b.add(
                        b.mul(b.add(b.const_i32(kt * TK), row_k), b.const_i32(N)),
                        b.add(n_base, col_n),
                    ),
                )
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def dtl_load_b(buf, kt, coh):
        # B = [E,N,K]: stage rows n_base..+TN, K-contiguous, into [TN,TK] LDS.
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_b):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            if hoist:
                off = b.add(b_load_base[p], b.const_i32(kt * TK))
            else:
                ci = b.add(tid, b.const_i32(p * BS))
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                off = b.add(
                    b_eoff,
                    b.add(
                        b.mul(b.add(n_base, row), cK),
                        b.add(b.const_i32(kt * TK), swz_col(col, row)),
                    ),
                )
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def precompute_a():
        # HOIST: the A-load source offset for each pass is loop-invariant across
        # K-tiles except for the +kt*TK term. Compute the invariant part once per
        # tile ((m_base + row)*K + swizzle) and hold it in a register; the K-loop
        # then only adds the compile-time kt*TK.
        nonlocal a_load_base
        vals = []
        for p in range(passes_a):
            ci = b.add(tid, b.const_i32(p * BS))
            row = b.div(ci, c_cpr)
            col = b.mul(b.mod(ci, c_cpr), c_hv)
            grow = b.add(m_base, row)  # absolute A row (contiguous, no gather)
            vals.append(b.add(b.mul(grow, cK), swz_col(col, row)))
        a_load_base = vals

    def dtl_load_a(buf, kt, coh):
        # A = [M_total(+TM pad),K]: this tile owns the contiguous rows
        # m_base..m_base+TM. The last tile may address rows past M_total; the
        # caller pads A by TM rows so those reads stay mapped. Their (garbage)
        # results are never stored -- the m_valid guard sinks them to row M.
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_a):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            if hoist:
                off = b.add(a_load_base[p], b.const_i32(kt * TK))
            else:
                ci = b.add(tid, b.const_i32(p * BS))
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                grow = b.add(m_base, row)
                off = b.add(
                    b.mul(grow, cK),
                    b.add(b.const_i32(kt * TK), swz_col(col, row)),
                )
            b.async_buffer_load_lds_addr(
                a_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def load_tile_b(buf, kt):
        if b_rrr:
            dtl_load_b_rrr(Bs[buf], kt, 0)
        else:
            dtl_load_b(Bs[buf], kt, 0)

    def load_tile(buf, kt):
        dtl_load_a(As[buf], kt, 0)
        load_tile_b(buf, kt)

    def kloop(acc, final_storer=None):
        if hoist:
            precompute_a()  # hoist the A-load per-pass base offsets
            precompute_b()  # hoist the B-load per-pass base offsets
        load_tile(0, 0)  # prologue
        for kt in range(n_ktiles):
            cur = kt % 2
            b.sync()  # drain cur tile's DTL writes + workgroup barrier
            if kt + 1 < n_ktiles:
                load_tile(1 - cur, kt + 1)  # next-tile loads overlap MFMAs
            # EPIFUSE: on the last K-tile, fuse the store into the MFMAs.
            sol = final_storer if kt == n_ktiles - 1 else None
            compute(cur, cur, acc, store_on_last=sol)

    def store_row(acc, mi):
        # Store one m-row (all nn n-tiles x c_per_lane elements) directly to
        # C[m_base + local_row]. Rows past the tile's valid count redirect to the
        # sink row (M) instead of branching. The row-base depends on (mi, i) only,
        # not nj -> hoist over nj.
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
            orow = b.select(b.cmp_lt(lr, m_valid), b.add(m_base, lr), M)  # sink=M
            # i64 row base: orow*N can overflow i32 (see cN64). The small column
            # term (n_base + warp_n_off + col_in) stays < N, folded in via c_rowbase.
            col_off = b.add(n_base, b.add(warp_n_off, col_in))
            row_col = c_rowbase(orow, col_off)
            for nj in range(nn):
                idx = mi * nn + nj
                addr = b.add(row_col, b.zext(b.const_i32(nj * AN), I64))
                val = b.vec_extract(acc[idx], i)
                b.global_store(C, addr, b.cast_f32_to(val, BF16), align=2)

    def epilogue(acc):
        # Non-fused fallback (EPIFUSE off): store every m-row after the K-loop.
        for mi in range(mm):
            store_row(acc, mi)

    def store_c_shuffle(acc):
        # CSHUF: reuse the A/B pool (dead after the K-loop) as a [TM,TN] C-stage.
        # (1) scatter each lane's accumulators into the stage at their tile-local
        # (row,col) via ds_write -- the C-lane transpose; (2) cooperatively read the
        # stage in contiguous 8-wide runs and issue WIDE (b128) per-row global
        # stores, redirecting each padded row to the sink row (M). The N dimension
        # is contiguous in both the stage and the destination C row, so the store
        # widens 16x vs the per-element store.
        cTN = b.const_i32(TN)
        cTK = b.const_i32(TK)
        b.sync()  # last compute's operand ds_reads on AB done before reuse
        for mi in range(mm):
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
                for nj in range(nn):
                    idx = mi * nn + nj
                    lc = b.add(b.add(warp_n_off, b.const_i32(nj * AN)), col_in)
                    flat = b.add(b.mul(lr, cTN), lc)
                    val = b.cast_f32_to(b.vec_extract(acc[idx], i), BF16)
                    b.smem_store_vN(AB, [b.div(flat, cTK), b.mod(flat, cTK)], val, 1)
        b.sync()
        nblk = (TM * TN) // (BS * 8)
        for blk in range(nblk):
            flat = b.add(b.const_i32(blk * BS * 8), b.mul(tid, b.const_i32(8)))
            vec = b.smem_load_vN(
                AB, b.div(flat, cTK), b.mod(flat, cTK), dtype=BF16, n=8
            )
            r = b.div(flat, cTN)
            c = b.mod(flat, cTN)
            orow = b.select(b.cmp_lt(r, m_valid), b.add(m_base, r), M)  # sink row = M
            addr = c_rowbase(orow, b.add(n_base, c))  # i64 (orow*N can overflow i32)
            b.global_store_vN(C, addr, vec, n=8, align=16)

    def do_tile_body():
        for i in range(len(accs)):
            accs[i] = atom.zero_acc(b)
        if cshuf:
            kloop(accs)
            store_c_shuffle(accs)  # coalesced wide-store epilogue via LDS C-stage
        elif epifuse:
            kloop(accs, final_storer=store_row)  # store fused into last K-tile
        else:
            kloop(accs)
            epilogue(accs)

    # One (m_tile, n_tile) per block: grid = (ceil(N/TN), num_m_tiles, 1).
    set_tile(b.add(b.mul(b.block_id_y(), c_npn), b.block_id_x()))
    do_tile_body()
    b.ret()
    return b.kernel, BS, TM, TN


def ragged_gemm_signature():
    return (
        SignatureBuilder()
        .ptr("A", "bf16")
        .ptr("B", "bf16")
        .ptr("C", "bf16")
        .ptr("expert_ids", "i32")
        .ptr("m_offsets", "i32")
        .ptr("m_valid", "i32")
        .scalar("M", "i32")
        .scalar("num_m_tiles", "i32")
        .build()
    )
