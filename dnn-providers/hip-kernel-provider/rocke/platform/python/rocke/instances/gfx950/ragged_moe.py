"""Ragged MoE grouped GEMM kernel builder for gfx950 (MI355X).

Fused gather (A via token indices) + grouped bf16 GEMM + weighted scatter (C).
CDNA4 optimizations: DTL cooperative loads, prologue overlap (PLOG), epilogue
fusion (EPIFUSE), inline-asm ds_reads, chiplet-aware tiling, int64 addressing.

Author: yraparti <yraparti@amd.com>
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from rocke.core.ir import BF16, F32, I32, I64, IRBuilder, PtrType, VectorType
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes, mfma_atom_for_dtype
from rocke.helpers.spec import SignatureBuilder


@dataclass(frozen=True)
class RaggedMoeSpec:
    """Ragged MoE grouped GEMM specification (gfx950/CDNA4).

    Geometry (problem + tile sizes):
        N, K, E, TOPK: problem dimensions (output width, reduction, experts, top-k)
        TM, TN, TK: tile sizes (M/N/K tile per block)
        WM, WN: warp layout (num warps in M/N dimensions)

    Optimization levers (all bool, defaults = production config):
        plog: Overlap prologue (B tile-0 load before token barrier)
        hoist: Hoist loop-invariant A-gather + B-load offsets
        rwlds: Stage routing weights in LDS (required by epifuse)
        epifuse: Fuse scatter into final K-tile MFMAs (default epilogue)
        swz: st_16x32 XOR swizzle (bank-conflict-free LDS reads)
        asm_reads: Inline-asm ds_read with manual lgkmcnt
        deeppipe: Deep operand prefetch pipeline
        prio: s_setprio(1) around MFMA blocks
        burstprio: Toggle s_setprio per k-chunk
        chiplet: Chiplet-aware super-tile remap (L2 locality)
        pin: Hoist wave-uniform indices to SGPRs
        b_rrr: NN weight layout B=[E,K,N] (vs NT B=[E,N,K])

    Experimental levers (measured NOT wins, default off):
        cshuf: C-shuffle wide-store epilogue (neutral vs epifuse)
        cksched: CK-HotLoop sched_group_barrier interleave
        permmasched: Per-MFMA s_setprio + sched_barrier
        tr128: ds_read_tr16_b128 (vs 2× b64, needs ROCm support)
        opsw: Operand swap (packed stores, slower)
        combine: Fused atomic top-k combine (3.7× slower)
        storesink: Diagnostic (collapse stores, isolate compute cost)

    Hardening (optional compile-time constants):
        nexp_const: Bake num_expanded as immediate (folds padding checks)
        nmt_const: Bake num_m_tiles as immediate (div-by-const in chiplet remap)
    """

    # Problem dimensions
    N: int
    K: int
    E: int
    TOPK: int = 2

    # Tile geometry
    TM: int = 256
    TN: int = 256
    TK: int = 64
    WM: int = 2
    WN: int = 4

    # Production optimizations (defaults = full opts)
    plog: bool = True  # Prologue overlap (+5-15 TF)
    hoist: bool = True  # Offset hoisting (embedded in plog win)
    rwlds: bool = True  # Routing weights in LDS (required by epifuse)
    epifuse: bool = True  # Epilogue fusion (default epilogue)
    swz: bool = True  # LDS bank-conflict swizzle
    asm_reads: bool = True  # Inline-asm ds_read (neutral at K=512)
    deeppipe: bool = True  # Deep prefetch pipeline
    prio: bool = True  # s_setprio around MFMAs
    burstprio: bool = False  # Per-chunk prio toggle
    chiplet: bool = True  # Super-tile remap (+~40 TF)
    pin: bool = True  # SGPR index hoisting
    b_rrr: bool = False  # NN layout (vs NT default)

    # Chiplet remap params (active when chiplet=True)
    chiplet_wgm: int = 8  # Workgroups per super-tile
    chiplet_xcds: int = 8  # Number of XCDs

    # Experimental levers (measured NOT wins)
    cshuf: bool = False  # C-shuffle wide stores (neutral)
    cksched: bool = False  # sched_group_barrier interleave
    permmasched: bool = False  # Per-MFMA barriers
    tr128: bool = False  # b128 transpose read (ROCm support)
    opsw: bool = False  # Operand swap (slower)
    combine: bool = False  # Atomic combine (3.7× slower)
    storesink: bool = False  # Diagnostic (collapse stores)

    # Hardening (optional compile-time constants)
    nexp_const: Optional[int] = None
    nmt_const: Optional[int] = None

    # Kernel name
    name: str = "ragged_moe"


def build_ragged_moe(spec: RaggedMoeSpec):
    """Build the ragged-MoE grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. N/K/E/TOPK are compile-time; the
    number of m-tiles is a runtime grid dimension (depends on the token
    distribution), so it never appears in the kernel body -- each block computes
    exactly one (m_tile, n_tile).
    """
    N, K, E, TOPK = spec.N, spec.K, spec.E, spec.TOPK
    TM, TN, TK = spec.TM, spec.TN, spec.TK
    WM, WN = spec.WM, spec.WN
    nexp_const, nmt_const = spec.nexp_const, spec.nmt_const

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
    rwlds = spec.rwlds
    epifuse = spec.epifuse
    nbuf = 2
    cshuf = spec.cshuf
    if cshuf and (b_rrr or TM != TN or TN != 2 * nbuf * TK):
        cshuf = False
    cksched = spec.cksched
    permmasched = spec.permmasched
    SGB_MFMA, SGB_DSREAD = 0x008, 0x100
    tr128 = spec.tr128
    opsw = spec.opsw
    combine = spec.combine
    if combine:
        opsw = True  # combine packs 2 contiguous N per atomic -> needs OPSW layout
    if opsw and not b_rrr:
        opsw = False
        combine = False  # combine depends on the OPSW layout
    if opsw:
        cshuf = False  # OPSW has its own packed-store epilogue
    if cshuf:
        epifuse = False
    # epifuse, cshuf, and opsw all read the routing weight from LDS in the
    # store path (rw_lds), so they require the RWL buffer to be allocated.
    if epifuse or cshuf or opsw:
        rwlds = True
    plog = spec.plog
    storesink = spec.storesink

    b = IRBuilder(spec.name)
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(BF16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(BF16, "global"), noalias=True, writeonly=True, align=16)
    STOK = b.param(
        "sorted_token_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    EIDS = b.param(
        "expert_ids", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    RW = b.param(
        "routing_weights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    NEXP = b.param("num_expanded", I32)
    NMT = b.param("num_m_tiles", I32)  # runtime grid.y extent (for chiplet remap)
    # HARDEN: for a fixed ("hardened") shape/routing, bake num_expanded and
    # num_m_tiles as compile-time constants. This folds the padding-sentinel
    # comparisons (tok < num_expanded) to immediates and, crucially, turns the
    # chiplet super-tile remap's div/mod by num_m_tiles from a runtime integer
    # reciprocal sequence (v_cvt_f32_u32 / v_rcp_iflag_f32 / s_mul_hi chain, run
    # every tile) into a divide-by-constant. The params stay in the ABI.
    NEXP_v = b.const_i32(nexp_const) if nexp_const is not None else NEXP
    NMT_v = b.const_i32(nmt_const) if nmt_const is not None else NMT

    tid = b.thread_id_x()
    warp = b.div(tid, b.const_i32(64))
    lane = b.mod(tid, b.const_i32(64))
    warp_row = b.div(warp, b.const_i32(WN))
    warp_col = b.mod(warp, b.const_i32(WN))
    ld = decode_mfma_lanes(b, atom, lane)

    def U(v):
        return b.to_sgpr_u32(v) if pin else v

    cN = b.const_i32(N)
    cN64 = b.const_i64(N)  # wide N stride: the C-scatter row offset orow*N can
    # exceed 2^31 (num_expanded*N > 2.1e9 at large token counts / N), so the row
    # term must be computed in i64 to avoid signed-int32 address overflow.
    cTOPK = b.const_i32(TOPK)
    c_npn = b.const_i32(n_pid_n)

    def c_rowbase(orow, col_i32):
        # i64 C-scatter base address = orow*N + col, with orow*N widened to 64-bit
        # and the (small, < N) column offset zext'd in. Callers add compile-time
        # per-n-tile constants (nj*AN) to this.
        return b.add(b.mul(b.zext(orow, I64), cN64), b.zext(col_i32, I64))

    # Per-tile coords, assigned by set_tile(); the DTL-load / epilogue closures
    # read these as free variables.
    m_base = n_base = expert = b_eoff = None
    # Per-pass A-gather / B-load offsets, precomputed once per tile (HOIST lever);
    # read as free variables by dtl_load_a_gather() / dtl_load_b*().
    a_gather_base = None
    b_load_base = None

    def set_tile(wgid):
        # Decode a flat workgroup id -> (m_tile, n_tile). The chiplet/super-tile
        # remap keeps adjacent m-tiles on the same XCD for L2 reuse (dynamic
        # num_pid_m = num_m_tiles).
        nonlocal m_base, n_base, expert, b_eoff
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
        m_base = U(b.mul(mt, b.const_i32(TM)))  # offset into sorted_token_ids
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

    def _row(buf, r):  # absolute row within buf's smem handle (adds the band base)
        return b.add(r, b.const_i32(buf[1])) if buf[1] else r

    def _addr(buf):  # i64 LDS base address of this buf's band
        a = b.smem_addr_of(buf[0])
        return (
            b.smem_ptr_add(a, b.zext(b.const_i32(buf[1] * TK * 2), I64))
            if buf[1]
            else a
        )

    # Per-tile token indices staged once into LDS (they are constant across all
    # K-tiles). This replaces the redundant per-element global sorted_token_ids
    # loads in the A-gather and the scatter epilogue with cheap LDS reads.
    STOKL = b.smem_alloc(I32, [TM, 1], name_hint="stok")
    # RWLDS: per-row routing weights staged into LDS (one f32 per tile row) so the
    # scatter reads them from LDS instead of a scattered per-output-row global load.
    RWL = b.smem_alloc(F32, [TM, 1], name_hint="rw") if rwlds else None
    c_zero = b.const_i32(0)
    accs = [atom.zero_acc(b) for _ in range(mm * nn)]

    warp_m_off = b.mul(warp_row, b.const_i32(WTM))
    warp_n_off = b.mul(warp_col, b.const_i32(WTN))

    def stage_tokens():
        # Cooperatively load sorted_token_ids[m_base : m_base+TM] into LDS.
        # Threads past TM clamp to the last slot (redundant same-value write).
        # RWLDS: co-stage each row's routing weight (RW[token], token clamped for
        # padded rows) so the scatter reads it from LDS.
        for p in range((TM + BS - 1) // BS):
            idx = b.add(tid, b.const_i32(p * BS))
            sidx = b.select(b.cmp_lt(idx, b.const_i32(TM)), idx, b.const_i32(TM - 1))
            val = b.global_load(STOK, b.add(m_base, sidx), I32)
            b.smem_store_vN(STOKL, [sidx, c_zero], val, 1)
            if rwlds:
                wv = b.global_load(
                    RW, b.select(b.cmp_lt(val, NEXP_v), val, b.const_i32(0)), F32
                )
                b.smem_store_vN(RWL, [sidx, c_zero], wv, 1)

    def stok_lds(row):
        return b.vec_extract(b.smem_load_vN(STOKL, row, c_zero, dtype=I32, n=1), 0)

    def rw_lds(row):
        return b.vec_extract(b.smem_load_vN(RWL, row, c_zero, dtype=F32, n=1), 0)

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
        # EPIFUSE: after finishing an m-row's nn MFMAs on the final K-chunk,
        # scatter that row immediately so its stores overlap the later m-rows'
        # MFMAs (store_row_cb reads its operands from LDS, computed transiently).
        # CKSCHED: interleave 1 MFMA + a proportional slice of the next chunk's
        # in-flight ds_reads so the backend spreads reads across the matrix ops.
        nmma = mm * nn
        ds_done = 0
        for mi in range(mm):
            for nj in range(nn):
                idx = mi * nn + nj
                # OPSW: swap operands -> acc[idx] holds C[m-tile mi, n-tile nj]
                # with N as the per-lane (c_per_lane) fast dimension.
                op_a, op_b = (b_fr[nj], a_fr[mi]) if opsw else (a_fr[mi], b_fr[nj])
                if permmasched:
                    b.s_setprio(1)
                    acc[idx] = atom.emit(b, op_a, op_b, acc[idx])
                    b.s_setprio(0)
                    b.sched_barrier(0)
                else:
                    acc[idx] = atom.emit(b, op_a, op_b, acc[idx])
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
                            b.mul(b.add(n_base, row), b.const_i32(K)),
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
                        b.mul(b.add(n_base, row), b.const_i32(K)),
                        b.add(b.const_i32(kt * TK), swz_col(col, row)),
                    ),
                )
            b.async_buffer_load_lds_addr(
                b_rsrc, lds, b.mul(off, b.const_i32(2)), zsoff, DWORDS, coherency=coh
            )

    def precompute_a_gather():
        # HOIST: the A-gather source offset for each pass is loop-invariant across
        # K-tiles except for the +kt*TK term. Compute the invariant part once per
        # tile (token LDS read + token//TOPK + row*K + swizzle) and hold it in a
        # register; the K-loop then only adds the compile-time kt*TK.
        nonlocal a_gather_base
        vals = []
        for p in range(passes_a):
            ci = b.add(tid, b.const_i32(p * BS))
            row = b.div(ci, c_cpr)
            col = b.mul(b.mod(ci, c_cpr), c_hv)
            tok = stok_lds(row)
            a_row = b.select(b.cmp_lt(tok, NEXP_v), b.div(tok, cTOPK), b.const_i32(0))
            vals.append(b.add(b.mul(a_row, b.const_i32(K)), swz_col(col, row)))
        a_gather_base = vals

    def dtl_load_a_gather(buf, kt, coh):
        # A = [num_tokens,K]: gather each tile row's token via sorted_token_ids,
        # activation row = token // TOPK. Padded rows (tok >= NEXP) clamp to row 0
        # (their output is discarded to the sink row in the epilogue).
        base = b.smem_ptr_add(_addr(buf), wave_off)
        for p in range(passes_a):
            lds = (
                base
                if p == 0
                else b.smem_ptr_add(base, b.zext(b.const_i32(p * BS * 16), I64))
            )
            if hoist:
                off = b.add(a_gather_base[p], b.const_i32(kt * TK))
            else:
                ci = b.add(tid, b.const_i32(p * BS))
                row = b.div(ci, c_cpr)
                col = b.mul(b.mod(ci, c_cpr), c_hv)
                tok = stok_lds(row)  # LDS-staged token index (no global reload)
                a_row = b.select(
                    b.cmp_lt(tok, NEXP_v), b.div(tok, cTOPK), b.const_i32(0)
                )
                off = b.add(
                    b.mul(a_row, b.const_i32(K)),
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
        dtl_load_a_gather(As[buf], kt, 0)
        load_tile_b(buf, kt)

    def kloop(acc, final_storer=None):
        if plog:
            # Overlap the prologue: B's tile-0 load + offset precompute are
            # token-independent, so issue them BEFORE the token-staging barrier.
            # The B global->LDS DMA then overlaps the token global loads + the
            # staging sync. Only the A-gather (reads LDS-staged tokens) waits.
            if hoist:
                precompute_b()
            load_tile_b(0, 0)  # B tile-0 DMA in flight during token staging
            stage_tokens()  # token indices (+ weights) -> LDS (overlaps B DMA)
            b.sync()  # drains token loads AND the B tile-0 DMA
            if hoist:
                precompute_a_gather()  # tokens in LDS -> hoist A-gather offsets
            dtl_load_a_gather(As[0], 0, 0)  # A tile-0 DMA (post-token)
        else:
            stage_tokens()  # stage per-tile token indices (+ weights) into LDS once
            b.sync()
            if hoist:
                precompute_a_gather()  # tokens are in LDS now -> hoist gather offsets
                precompute_b()  # hoist the B-load per-pass base offsets
            load_tile(0, 0)  # prologue
        for kt in range(n_ktiles):
            cur = kt % 2
            b.sync()  # drain cur tile's DTL writes + workgroup barrier
            if kt + 1 < n_ktiles:
                load_tile(1 - cur, kt + 1)  # next-tile loads overlap MFMAs
            # EPIFUSE: on the last K-tile, fuse the scatter into the MFMAs.
            sol = final_storer if kt == n_ktiles - 1 else None
            compute(cur, cur, acc, store_on_last=sol)

    def store_row(acc, mi):
        if opsw:
            # OPSW layout: for this lane the c_per_lane accumulator values are
            # CONSECUTIVE N (row = (lane//16)*4 + i) for a SINGLE token (col =
            # lane%16). Pack them (scaled by the row's weight) into one aligned
            # b64 store to C[token, n0..n0+c_per_lane-1].
            r0, c0 = atom.lane_to_output(b, lane, 0)  # r0 = n-in-atom base, c0 = m
            lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), c0)  # token row (m)
            otok = stok_lds(lr)
            valid = b.cmp_lt(otok, NEXP_v)
            w = rw_lds(lr)
            if combine:
                # COMBINE: atomically accumulate into C[token = otok//TOPK, n],
                # halving the output rows. Predicate out padding so its sentinel
                # rows don't contend on a single sink.
                token = b.div(otok, cTOPK)
                col_off = b.add(n_base, b.add(warp_n_off, r0))
                row_base = c_rowbase(token, col_off)  # i64 (token*N can overflow i32)
                with b.scf_if(valid):
                    for nj in range(nn):
                        idx = mi * nn + nj
                        addr = b.add(row_base, b.zext(b.const_i32(nj * AN), I64))
                        for p in range(atom.c_per_lane // 2):
                            pk = b.vec_pack(
                                [
                                    b.cast_f32_to(
                                        b.fmul(b.vec_extract(acc[idx], 2 * p + q), w),
                                        BF16,
                                    )
                                    for q in range(2)
                                ],
                                BF16,
                            )
                            b.global_atomic_add_pk_bf16(
                                C, b.add(addr, b.zext(b.const_i32(2 * p), I64)), pk
                            )
                return
            orow = b.select(valid, otok, NEXP_v)  # padding -> sink row
            col_off = b.add(n_base, b.add(warp_n_off, r0))
            row_base = c_rowbase(orow, col_off)  # i64 (orow*N can overflow i32)
            for nj in range(nn):
                idx = mi * nn + nj
                addr = b.add(row_base, b.zext(b.const_i32(nj * AN), I64))
                comps = [
                    b.cast_f32_to(b.fmul(b.vec_extract(acc[idx], i), w), BF16)
                    for i in range(atom.c_per_lane)
                ]
                b.global_store_vN(
                    C, addr, b.vec_pack(comps, BF16), n=atom.c_per_lane, align=8
                )
            return
        # Scatter one m-row (all nn n-tiles x c_per_lane elements) to
        # C[sorted_token_ids[row]] (expanded index), scaled by the routing weight.
        # The token/weight/row-base depend on (mi, i) only, not nj -> hoist over nj.
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
            otok = stok_lds(lr)  # LDS-staged token index (no global reload)
            valid = b.cmp_lt(otok, NEXP_v)
            orow = b.select(valid, otok, NEXP_v)  # padding -> sink row (NEXP)
            if rwlds:
                w = rw_lds(lr)  # LDS-staged routing weight (no global reload)
            else:
                w = b.global_load(RW, b.select(valid, otok, b.const_i32(0)), F32)
            # i64 row base: orow*N can overflow i32 (see cN64). The small column
            # term (n_base + warp_n_off + col_in) stays < N, folded in via c_rowbase.
            col_off = b.add(n_base, b.add(warp_n_off, col_in))
            row_col = c_rowbase(orow, col_off)
            for nj in range(nn):
                idx = mi * nn + nj
                addr = b.add(row_col, b.zext(b.const_i32(nj * AN), I64))
                val = b.fmul(b.vec_extract(acc[idx], i), w)
                b.global_store(C, addr, b.cast_f32_to(val, BF16), align=2)

    def epilogue(acc):
        # Non-fused fallback (EPIFUSE off): scatter every m-row after the K-loop.
        for mi in range(mm):
            store_row(acc, mi)

    def store_c_shuffle(acc):
        # CSHUF: reuse the A/B pool (dead after the K-loop) as a [TM,TN] C-stage.
        # (1) scatter each lane's weighted accumulators into the stage at their
        # tile-local (row,col) via ds_write -- the C-lane transpose; (2)
        # cooperatively read the stage in contiguous 8-wide runs and issue WIDE
        # (b128) per-row global stores, redirecting each row to its token (padding
        # -> sink row NEXP). The N dimension is contiguous in both the stage and
        # the destination C row, so the store widens 16x vs the per-element scatter.
        cTN = b.const_i32(TN)
        cTK = b.const_i32(TK)
        b.sync()  # last compute's operand ds_reads on AB done before reuse
        for mi in range(mm):
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                lr = b.add(b.add(warp_m_off, b.const_i32(mi * AM)), row_in)
                w = rw_lds(lr)  # per-row routing weight applied on the way into LDS
                for nj in range(nn):
                    idx = mi * nn + nj
                    lc = b.add(b.add(warp_n_off, b.const_i32(nj * AN)), col_in)
                    flat = b.add(b.mul(lr, cTN), lc)
                    val = b.cast_f32_to(b.fmul(b.vec_extract(acc[idx], i), w), BF16)
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
            otok = stok_lds(r)
            orow = b.select(b.cmp_lt(otok, NEXP_v), otok, NEXP_v)  # padding -> sink row
            addr = c_rowbase(orow, b.add(n_base, c))  # i64 (orow*N can overflow i32)
            b.global_store_vN(C, addr, vec, n=8, align=16)

    def store_sink(acc):
        # DIAGNOSTIC (STORESINK=1): reduce every accumulator into c_per_lane sums
        # (data-dependent on all MFMAs, so none are DCE'd) and store only those few
        # values. Compute+load stay fully live; store traffic ~ 0. (full-run TF minus
        # STORESINK TF) isolates the real C-store-tail cost.
        sums = [None] * atom.c_per_lane
        for idx in range(mm * nn):
            for i in range(atom.c_per_lane):
                e = b.vec_extract(acc[idx], i)
                sums[i] = e if sums[i] is None else b.fadd(sums[i], e)
        base = b.mul(lane, b.const_i32(atom.c_per_lane))
        for i in range(atom.c_per_lane):
            b.global_store(
                C, b.add(base, b.const_i32(i)), b.cast_f32_to(sums[i], BF16), align=2
            )

    def do_tile_body():
        for i in range(len(accs)):
            accs[i] = atom.zero_acc(b)
        if storesink:
            kloop(accs)
            store_sink(accs)
        elif cshuf:
            kloop(accs)
            store_c_shuffle(accs)  # coalesced wide-store epilogue via LDS C-stage
        elif epifuse:
            kloop(accs, final_storer=store_row)  # scatter fused into last K-tile
        else:
            kloop(accs)
            epilogue(accs)

    # One (m_tile, n_tile) per block: grid = (ceil(N/TN), num_m_tiles, 1).
    set_tile(b.add(b.mul(b.block_id_y(), c_npn), b.block_id_x()))
    do_tile_body()
    b.ret()
    return b.kernel, BS, TM, TN


def ragged_moe_signature():
    return (
        SignatureBuilder()
        .ptr("A", "bf16")
        .ptr("B", "bf16")
        .ptr("C", "bf16")
        .ptr("sorted_token_ids", "i32")
        .ptr("expert_ids", "i32")
        .ptr("routing_weights", "f32")
        .scalar("num_expanded", "i32")
        .scalar("num_m_tiles", "i32")
        .build()
    )
