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

Levers are command-line flags (defaults = the measured sweet spot for
M/E=8192, N=1024, K=512, E=64 on gfx950 / MI355X); use ``--no-<flag>`` to
disable a boolean lever:

* ``--dtl``   async direct-to-LDS DMA (``async_buffer_load_lds_addr``,
              DRAM->LDS, double-buffered, vmcnt-overlapped). Best path.
* ``--db``    register-prefetch double buffer (used when ``--no-dtl``).
* ``--prio``  ``s_setprio(1/0)`` around the MFMA cluster (survives hipcc;
              the comgr/LLVM-IR path could not keep this hint).
* ``--tm/--tn/--tk/--wm/--wn`` tile + warp-grid geometry.
* ``--swz``   st_16x32 LDS XOR swizzle (bank-conflict-free ds_reads); default on.
* ``--chip``  chiplet/super-tile grid remap (L2 locality across XCDs via a
              chunked super-tile transform + WGM grouping); default on.
              ``--xcds``/``--wgm`` tune the XCD count / super-tile group height.
* ``--epifuse`` interleave each accumulator's C store with its final MFMA so
              the store-issue + address VALU overlap the last K-tile's MFMAs;
              default on.
* ``--brrr``  NN weight layout (B=[E,K,N]) via the transpose-read path.

Measured progression (this harness, default shape): single-buffer 446 ->
double-buffer 552 -> +s_setprio 648 -> prefetch-before-barrier 676 ->
async-DTL 729 -> +st_16x32 swizzle 750 -> +chiplet grid swizzle ~790 ->
+epilogue-store interleave ~807 TFLOPS median / ~815 peak (vs the generic
emitter's 750). Producer/consumer wave specialization does not pay off on
gfx950 (unified VGPR file -- no per-wave register reallocation, and occupancy
is LDS-bound at 1 block/CU). See CASE_STUDY.md for the full analysis.

Run:
    PYTHONPATH=python python3 \
        python/rocke/examples/gfx950/grouped_gemm/grouped_gemm_hip.py [--no-swz ...]
"""
from __future__ import annotations

import argparse
import math

from rocke.helpers.compile import compile_kernel_via_hipcc

# The kernel builder now lives in the instance module. Re-exported here so
# scripts that historically did ``from grouped_gemm_hip import
# build_custom_grouped`` keep working.
from rocke.instances.gfx950.grouped_gemm import build_custom_grouped  # noqa: F401


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="gfx950 hand-scheduled grouped bf16 GEMM harness"
    )
    # Problem shape.
    p.add_argument("--m-total", type=int, default=524288)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--e", type=int, default=64)
    # Tile / warp geometry + persistent-grid / chiplet knobs.
    p.add_argument("--tm", type=int, default=256)
    p.add_argument("--tn", type=int, default=256)
    p.add_argument("--tk", type=int, default=64)
    p.add_argument("--wm", type=int, default=2)
    p.add_argument("--wn", type=int, default=4)
    p.add_argument("--xcds", type=int, default=8)
    p.add_argument("--wgm", type=int, default=8)
    p.add_argument("--tpb", type=int, default=1)
    # Boolean levers (defaults = measured production sweet spot). Use --no-<flag>
    # to disable, e.g. --no-swz.
    B = argparse.BooleanOptionalAction
    p.add_argument("--dtl", action=B, default=True, help="async direct-to-LDS DMA")
    p.add_argument("--db", action=B, default=True, help="register double-buffer")
    p.add_argument("--prio", action=B, default=True, help="s_setprio around MFMAs")
    p.add_argument("--swz", action=B, default=True, help="st_16x32 LDS XOR swizzle")
    p.add_argument("--chip", action=B, default=True, help="chiplet super-tile remap")
    p.add_argument("--asm", action=B, default=True, help="inline-asm ds_read")
    p.add_argument("--deeppipe", action=B, default=True, help="deep operand pipeline")
    p.add_argument("--epifuse", action=B, default=True, help="epilogue store fusion")
    p.add_argument(
        "--brrr", action=B, default=False, help="NN weight layout (B=[E,K,N])"
    )
    return p.parse_args(argv)


def _main(argv=None) -> int:
    import torch  # local import: only the benchmark/verify path needs torch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig
    from rocke.instances.gfx950.grouped_gemm import GroupedGemmSpec, build_grouped_gemm

    args = _parse_args(argv)
    M_total, N, K, E = args.m_total, args.n, args.k, args.e
    m = M_total // E
    flops = 2 * M_total * N * K

    # Build the spec from command-line args (defaults = production opts).
    spec = GroupedGemmSpec(
        M=m,
        N=N,
        K=K,
        E=E,
        TM=args.tm,
        TN=args.tn,
        TK=args.tk,
        WM=args.wm,
        WN=args.wn,
        dtl=args.dtl,
        db=args.db,
        prio=args.prio,
        swz=args.swz,
        chiplet=args.chip,
        chiplet_xcds=args.xcds,
        chiplet_wgm=args.wgm,
        asm_reads=args.asm,
        deeppipe=args.deeppipe,
        epifuse=args.epifuse,
        b_rrr=args.brrr,
        tpb=args.tpb,
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
        f"[ggemm] med={flops / (ts[2] * 1e9):.1f} TF  peak={flops / (ts[0] * 1e9):.1f} TF"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
