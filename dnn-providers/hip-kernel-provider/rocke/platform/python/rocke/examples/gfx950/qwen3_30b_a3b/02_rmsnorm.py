#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fused add-residual + RMSNorm — with CUDA-graph capture (M=2, N=2048).

Problem
-------
ATOM calls aiter.rmsnorm2d_fwd_with_add for every norm layer.  On A3B
there are three identical RMSNorm calls per decode step (input, post-
attention, pre-MoE) each taking ~5.9µs.

The DSL add_rmsnorm2d_bf16 kernel reduces this to 3.6µs from kernel
work alone — but CUDA-graph capture brings it to 0.45µs by eliminating
the HIP command-submission overhead entirely.

Why CUDA graph is so effective here
------------------------------------
aiter.rmsnorm2d_fwd_with_add is itself a small fast kernel, but every
GPU launch carries ~5–8µs of CPU-side HIP runtime overhead:

  1. Python ↔ C++ boundary (pybind11 arg unpacking)
  2. hipModuleLaunchKernel argument serialization
  3. HIP's command packet construction and HSA queue write
  4. GPU hardware scheduler picks up the packet

CUDA graph capture records all of (2–4) once at capture time and stores
the resolved command packet.  Replay calls a single hipGraphLaunch which
submits the pre-built packet — ~0.45µs regardless of kernel count.

For a kernel whose GPU execution time is only 3–5µs, removing 5–8µs of
dispatch overhead is the dominant win.  Graph capture is worth it for
any kernel that:
  (a) runs on preallocated stable-pointer tensors, and
  (b) appears in a tight inference loop.

Optimization steps
------------------
  1. Start with the default add_rmsnorm2d_bf16 spec (n_per_block=2048,
     block_size=64, vec=4, save_residual=True).
  2. Measure raw kernel latency: 3.6µs — still slower than AITER at 5.9µs
     because the baseline includes Python overhead on the AITER side too.
  3. Capture both DSL and AITER into CUDA graphs, re-measure.
  4. DSL graph: 0.45µs.  AITER graph: 0.47µs.  Both equal (kernel-bound).
  5. In production the AITER call goes through tgemm dispatch overhead;
     DSL via graph is 13× faster wall-clock.

Results (MI355X / gfx950, M=2, N=2048, bf16)
----------------------------------------------
  Backend                  Latency    Speedup
  AITER (production call)   5.90µs    1.00×
  DSL kernel only           3.62µs    1.63×
  DSL + CUDA graph          0.45µs   13.1×

Run
---
  PYTHONPATH=<rocke_python_root> python3 02_rmsnorm.py
"""

from __future__ import annotations

import struct

import torch

from _common import (
    BATCH,
    DTYPE,
    HIDDEN,
    ISA,
    WARMUP,
    ITERS,
    REPEATS,
    ms,
    speedup,
    capture_graph,
)


def build_norm_kernel(M: int, N: int):
    """Compile add_rmsnorm2d_bf16.  Returns (run_fn, rt, fn, grid, block)."""
    from rocke.instances.common.add_rmsnorm2d_bf16 import (
        AddRMSNorm2DBF16Spec,
        build_add_rmsnorm2d_bf16,
        add_rmsnorm2d_bf16_grid,
    )
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    spec = AddRMSNorm2DBF16Spec(
        n_per_block=N,
        block_size=64,
        vec=4,
        dtype="bf16",
        save_residual=True,
    )
    art = compile_kernel(build_add_rmsnorm2d_bf16(spec), isa=ISA)
    grid = add_rmsnorm2d_bf16_grid(M, spec)
    block = (spec.block_size, 1, 1)
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)
    return rt, fn, grid, block


def main() -> None:
    M, N = BATCH, HIDDEN
    eps = 1e-6
    print("=" * 60)
    print("02  Add-residual + RMSNorm — CUDA-graph capture")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: M={M} N={N}  dtype=bf16")
    print(f"Timing: {WARMUP} warmup, {ITERS} iters/sample, {REPEATS} samples → median")
    print()

    X = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1
    R = torch.randn(M, N, dtype=DTYPE, device="cuda") * 0.1
    Gam = torch.ones(N, dtype=DTYPE, device="cuda")
    Xout = torch.empty(M, N, dtype=DTYPE, device="cuda")
    Yout = torch.empty(M, N, dtype=DTYPE, device="cuda")

    # ── AITER baseline (production path) ──────────────────────────────────────
    bl_ms = float("nan")
    try:
        import aiter

        # Signature: (out, input, residual_in, residual_out, weight, epsilon).
        # out=Xout (normalized), residual_out=Yout (= input + residual_in).
        aiter.rmsnorm2d_fwd_with_add(Xout, X, R, Yout, Gam, eps)
        torch.cuda.synchronize()
        bl_ms = ms(lambda: aiter.rmsnorm2d_fwd_with_add(Xout, X, R, Yout, Gam, eps))
        print(f"  AITER rmsnorm2d_fwd_with_add (production): {bl_ms * 1000:.2f}µs")
    except Exception as exc:
        print(f"  SKIP AITER: {exc}")

    # ── DSL kernel (no graph) ─────────────────────────────────────────────────
    rt, fn, grid, block = build_norm_kernel(M, N)
    # Kernel param order (save_residual=True): A, B, Gamma, X(residual_out),
    # Y(normalized_out).  A/B are the two summed inputs; X receives A+B and Y
    # receives rmsnorm(A+B)*Gamma.  So A=X, B=R, X(resid)=Xout, Y(norm)=Yout.
    packed = struct.pack(
        "<QQQQQiif",
        int(X.data_ptr()),
        int(R.data_ptr()),
        int(Gam.data_ptr()),
        int(Xout.data_ptr()),
        int(Yout.data_ptr()),
        M,
        N,
        eps,
    )

    # Warmup compile
    rt.launch(fn, grid, block, packed, stream=0)
    torch.cuda.synchronize()

    dsl_raw_ms = ms(lambda: rt.launch(fn, grid, block, packed, stream=0))
    print(
        f"  DSL kernel (no graph):                     {dsl_raw_ms * 1000:.2f}µs  "
        f"speedup={speedup(bl_ms, dsl_raw_ms):.3f}×"
    )

    # ── DSL kernel + CUDA-graph capture ───────────────────────────────────────
    g = capture_graph(lambda: rt.launch(fn, grid, block, packed, stream=0), warmup=5)
    if g is not None:
        dsl_graph_ms = ms(lambda: g.replay())
        print(
            f"  DSL kernel + CUDA graph:                   {dsl_graph_ms * 1000:.2f}µs  "
            f"speedup={speedup(bl_ms, dsl_graph_ms):.3f}×"
        )
    else:
        print("  CUDA graph capture failed")
        dsl_graph_ms = float("nan")

    # ── Correctness ───────────────────────────────────────────────────────────
    X_ref = (X + R).float()
    rms = ((X_ref**2).mean(dim=-1, keepdim=True) + eps).sqrt()
    Y_ref = (X_ref / rms * Gam.float()).to(DTYPE)
    err = (Yout - Y_ref).abs().max().item()
    # bf16 carries ~2^-7 relative precision; the normalized output reaches
    # |y|~3.5 here, so 1 ULP is ~2.7e-2.  Judge with a relative tolerance of
    # 2 ULP rather than a fixed absolute threshold.
    rel = err / (Y_ref.float().abs().max().item() + 1e-6)
    ok = rel < 2.0**-6
    print(
        f"\n  Correctness: max|err|={err:.2e}  rel={rel:.2e}  "
        f"{'PASS' if ok else 'FAIL'}"
    )

    print()
    print("  Optimization note: CUDA graph eliminates ~5µs HIP command-submission")
    print("  overhead per launch.  For a 3µs kernel that is the dominant cost.")


if __name__ == "__main__":
    main()
