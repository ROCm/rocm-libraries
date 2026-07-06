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

from rocke.helpers.compile import compile_kernel_via_hipcc

# The kernel builder now lives in the instance module. Re-exported here so
# scripts that historically did ``from grouped_gemm_hip import
# build_custom_grouped`` keep working.
from rocke.instances.gfx950.grouped_gemm import build_custom_grouped  # noqa: F401


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
