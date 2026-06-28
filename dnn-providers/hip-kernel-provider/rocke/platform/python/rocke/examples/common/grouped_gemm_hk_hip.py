# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hand-authored grouped bf16 GEMM built directly on the raw ``IRBuilder``
(NO ``gemm_universal`` emitter) and lowered through the HIP-C++ -> hipcc
backend (``compile_kernel_via_hipcc``).

This is the HipKittens-style experiment: instead of declaring a ``TileSpec``
and letting the generic emitter + comgr schedule the kernel, we author the
warp tiling, LDS staging, double-buffered software pipeline, and the
``s_setprio`` MFMA-priority hints by hand, and rely on the hipcc toolchain
to preserve that schedule (the same toolchain HipKittens uses). The kernel
only borrows the *correct* MFMA lane-mapping helpers from
``helpers.mfma_gemm_inner`` (``mfma_atom_for_dtype`` / ``decode_mfma_lanes`` /
``atom.lane_to_output``) so the operand-frag and epilogue layouts are exact.

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
* ``CHIP=1`` chiplet/super-tile grid remap (L2 locality across XCDs, HK's
             chiplet_transform_chunked + WGM grouping); default on. ``XCDS``/
             ``WGM`` tune the XCD count / super-tile group height.
* ``PF=1``   register-prefetch-ahead (read next K-chunk frags before current
             MFMAs). Neutral here -- the compiler already overlaps ds_read->mma.
* ``PIN=1``  pin wave-uniform tile bases/expert offsets into SGPRs (~neutral).

Measured progression (this harness, default shape): single-buffer 446 ->
double-buffer 552 -> +s_setprio 648 -> prefetch-before-barrier 676 ->
async-DTL 729 -> +st_16x32 swizzle 750 -> +chiplet grid swizzle ~790 TFLOPS
(vs the generic emitter's 750 and HipKittens' ~849). See the kernel-authoring
notes for why HipKittens' per-MFMA interleave + barrier time-slicing regress
when transplanted in isolation (co-designed with its swizzled LDS + exact
wait counts), and why producer/consumer wave specialization does not pay off
on gfx950 (no Hopper-style per-wave register reallocation).

Run:
    PYTHONPATH=Python python3 -m rocke.examples.common.grouped_gemm_hk_hip
    DTL=1 TM=256 TN=256 TK=64 WM=2 WN=4 PYTHONPATH=Python \
        python3 -m rocke.examples.common.grouped_gemm_hk_hip
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
    name="grouped_gemm_hk",
):
    """Build the hand-scheduled grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. ``M`` here is the per-expert
    row count (``M_total // E``).
    """
    atom = mfma_atom_for_dtype("bf16", 16, 16)  # 16x16x32 bf16
    AM, AN, AK = atom.m, atom.n, atom.k
    apl, bpl = atom.a_per_lane, atom.b_per_lane  # 8, 8
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
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    expert = b.block_id_z()

    # Register pinning (HK-style): force the wave-uniform tile bases / expert
    # offsets into SGPRs (readfirstlane + an "+s" asm pin) so they aren't
    # re-materialised into VGPRs at every consumer across the unrolled K-loop.
    def U(v):
        return b.to_sgpr_u32(v) if pin else v

    if chiplet:
        # HK-style L2-locality grid remap: flatten (by, bx) -> linear wgid, run
        # it through the chiplet XCD transform + WGM super-tile grouping so
        # consecutive workgroups share B/A tiles in the same XCD's L2 slice.
        from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

        n_pid_m = (M + TM - 1) // TM
        n_pid_n = (N + TN - 1) // TN
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
        m_base = U(b.mul(swz.row, b.const_i32(TM)))
        n_base = U(b.mul(swz.col, b.const_i32(TN)))
    else:
        m_base = U(b.mul(bid_m, b.const_i32(TM)))
        n_base = U(b.mul(bid_n, b.const_i32(TN)))

    # Fold the per-expert offset into the element index (global_ptr_add lowers
    # to a const pointer in the HIP backend, which breaks the C store).
    a_eoff = U(b.mul(expert, sa))
    b_eoff = U(b.mul(expert, sb))
    c_eoff = U(b.mul(expert, sc))

    nbuf = 2 if (db or dtl) else 1
    ld = decode_mfma_lanes(b, atom, lane)
    As = [b.smem_alloc(BF16, [TM, TK], name_hint=f"As{i}") for i in range(nbuf)]
    Bs = [b.smem_alloc(BF16, [TN, TK], name_hint=f"Bs{i}") for i in range(nbuf)]
    accs = [atom.zero_acc(b) for _ in range(mm * nn)]

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

    def store_regs(smem, regs):
        for r, col, v in regs:
            b.smem_store_vN(smem, [r, col], v, 8)

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

    def read_chunk(ai, bi, kk):
        col = b.add(b.const_i32(kk * AK), k_lane)
        a = [
            b.smem_load_vN(
                As[ai], a_rows[mi], swz_col(col, a_rows[mi]), dtype=BF16, n=apl
            )
            for mi in range(mm)
        ]
        bb = [
            b.smem_load_vN(
                Bs[bi], b_rows[nj], swz_col(col, b_rows[nj]), dtype=BF16, n=bpl
            )
            for nj in range(nn)
        ]
        return a, bb

    def compute(ai, bi):
        if prio:
            b.s_setprio(1)
        # Register-prefetch-ahead: pull the next K-chunk's operand frags into
        # registers before issuing the current chunk's MFMAs, so the ds_read
        # latency overlaps the matrix ops (the register-level pipeline atop the
        # LDS double buffer).
        a_fr, b_fr = read_chunk(ai, bi, 0)
        for kk in range(kchunks):
            nxt = None
            if kk + 1 < kchunks and prefetch:
                nxt = read_chunk(ai, bi, kk + 1)  # prefetch BEFORE current MFMAs
            for mi in range(mm):
                for nj in range(nn):
                    idx = mi * nn + nj
                    accs[idx] = atom.emit(b, a_fr[mi], b_fr[nj], accs[idx])
            if kk + 1 < kchunks:
                a_fr, b_fr = nxt if nxt is not None else read_chunk(ai, bi, kk + 1)
        if prio:
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

        def dtl_load(rsrc, eoff, smem, row_base, kt, passes, coh):
            base = b.smem_ptr_add(b.smem_addr_of(smem), wave_off)
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

        def load_tile_dtl(buf, kt):
            dtl_load(a_rsrc, a_eoff, As[buf], m_base, kt, passes_a, 0)
            dtl_load(b_rsrc, b_eoff, Bs[buf], n_base, kt, passes_b, 0)

        load_tile_dtl(0, 0)  # prologue
        for kt in range(n_ktiles):
            cur = kt % 2
            b.sync()  # drain cur tile's DTL writes + barrier
            if kt + 1 < n_ktiles:
                load_tile_dtl(1 - cur, kt + 1)  # next-tile async loads overlap compute
            compute(cur, cur)
    elif db:
        ar = issue_loads(A, a_eoff, m_base, TM, 0)
        br = issue_loads(Bp, b_eoff, n_base, TN, 0)
        for kt in range(n_ktiles):
            cur = kt % 2
            store_regs(As[cur], ar)
            store_regs(Bs[cur], br)
            nxt = None
            if kt + 1 < n_ktiles:
                nxt = (
                    issue_loads(A, a_eoff, m_base, TM, kt + 1),
                    issue_loads(Bp, b_eoff, n_base, TN, kt + 1),
                )
            b.sync()
            compute(cur, cur)  # double-buffer makes a trailing sync unnecessary
            if nxt is not None:
                ar, br = nxt
    else:
        for kt in range(n_ktiles):
            store_regs(As[0], issue_loads(A, a_eoff, m_base, TM, kt))
            store_regs(Bs[0], issue_loads(Bp, b_eoff, n_base, TN, kt))
            b.sync()
            compute(0, 0)
            b.sync()

    cN = b.const_i32(N)
    idx = 0
    for mi in range(mm):
        for nj in range(nn):
            mtb = b.add(
                m_base, b.add(b.mul(warp_row, b.const_i32(WTM)), b.const_i32(mi * AM))
            )
            ntb = b.add(
                n_base, b.add(b.mul(warp_col, b.const_i32(WTN)), b.const_i32(nj * AN))
            )
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(mtb, row_in)
                col = b.add(ntb, col_in)
                addr = b.add(c_eoff, b.add(b.mul(row, cN), col))
                val = b.cast_f32_to(b.vec_extract(accs[idx], i), BF16)
                b.global_store(C, addr, val, align=2)
            idx += 1
    b.ret()
    return b.kernel, BS, TM, TN


def grouped_gemm_hk_signature():
    return (
        SignatureBuilder()
        .ptr("A", "bf16")
        .ptr("B", "bf16")
        .ptr("C", "bf16")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
        .build()
    )


def _main() -> int:
    import torch  # local import: only the benchmark/verify path needs torch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    M_total, N, K, E = 524288, 1024, 512, 64
    m = M_total // E
    flops = 2 * M_total * N * K
    TM = int(os.environ.get("TM", 256))
    TN = int(os.environ.get("TN", 256))
    TK = int(os.environ.get("TK", 64))
    WM = int(os.environ.get("WM", 2))
    WN = int(os.environ.get("WN", 4))
    dtl = os.environ.get("DTL", "1") == "1"
    db = os.environ.get("DB", "1") == "1"
    prio = os.environ.get("PRIO", "1") == "1"
    swz = os.environ.get("SWZ", "1") == "1"  # st_16x32 swizzle: +54 TF, default on
    prefetch = os.environ.get("PF", "0") == "1"
    pin = os.environ.get("PIN", "0") == "1"
    chiplet = os.environ.get("CHIP", "1") == "1"  # L2-locality grid remap: +40 TF
    chiplet_xcds = int(os.environ.get("XCDS", "8"))
    chiplet_wgm = int(os.environ.get("WGM", "8"))

    kernel, BS, tm, tn = build_custom_grouped(
        m,
        N,
        K,
        E,
        TM=TM,
        TN=TN,
        TK=TK,
        WM=WM,
        WN=WN,
        dtl=dtl,
        db=db,
        prio=prio,
        swz=swz,
        prefetch=prefetch,
        pin=pin,
        chiplet=chiplet,
        chiplet_wgm=chiplet_wgm,
        chiplet_xcds=chiplet_xcds,
    )
    print(
        f"[hk-hip] tile={TM}x{TN}x{TK} warps={WM}x{WN} BS={BS} "
        f"dtl={dtl} prio={prio} swz={swz} pf={prefetch} pin={pin} "
        f"chiplet={chiplet} wgm={chiplet_wgm} xcds={chiplet_xcds}"
    )
    art = compile_kernel_via_hipcc(kernel, arch="gfx950")
    print(f"[hk-hip] hipcc built {kernel.name}: {len(art.hsaco)} B")

    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=kernel.name,
        signature=grouped_gemm_hk_signature(),
        cache_key=(kernel.name,),
    )
    dt = torch.bfloat16
    A = torch.randn(E, m, K, dtype=dt, device="cuda")
    B = torch.randn(E, N, K, dtype=dt, device="cuda") * 0.05
    C = torch.empty(E, m, N, dtype=dt, device="cuda")
    ref = torch.bmm(A.float(), B.float().transpose(-1, -2))
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
    print(f"[hk-hip] grid={grid} max_abs={err:.4f}  {'PASS' if err < 0.1 else 'FAIL'}")
    if err >= 0.1:
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
        f"[hk-hip] med={flops / (ts[2] * 1e9):.1f} TF  peak={flops / (ts[0] * 1e9):.1f} TF"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
