# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf check for the ``preshuffle_b=True`` BatchedGemm path.

The kernel under test is :func:`build_batched_gemm` with
``trait.preshuffle_b=True``. The kernel assumes B has been pre-shuffled
on the host into ``(batch, k_tiles, n_tiles, block_n, block_k)``
contiguous layout. The expected per-K-tile B-load is one wide
``buffer_load_dwordx<N>`` per wavefront: this is what closes the gap to
CK Tile's preshuffled-B path (`§12.1.H` of the runbook).

This script:
  1. Builds two BatchedGemm kernels (preshuffle_b=False / True) for
     identical (B, M, N, K) and TileSpec.
  2. Generates a random A and B, computes a numpy reference.
  3. Launches both kernels (B preshuffled on the host for the second).
  4. Asserts numeric parity to the reference, then times both with HIP
     events.

Run with::

    cd <repo>/dnn-providers/hip-kernel-provider/rocke/platform/python
    python rocke/examples/gfx950/moe/test_preshuffle_b.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


import numpy as np  # noqa: E402

from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from rocke.instances.common.batched_gemm import (  # noqa: E402
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec  # noqa: E402
from rocke.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from rocke.runtime.hip_module import Runtime, get_device_arch  # noqa: E402
from rocke.runtime.host_buffers import as_u8_buffer  # noqa: E402
from rocke.runtime.launcher import (  # noqa: E402
    DeviceMem,
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


def host_preshuffle_b(B: np.ndarray, block_n: int, block_k: int) -> np.ndarray:
    """``(E, N, K)`` row-major B  ->  ``(E, k_tiles, n_tiles, block_n, block_k)``
    contiguous, matching the layout expected by ``preshuffle_b=True``."""
    E, N, K = B.shape
    assert N % block_n == 0, f"N={N} must be a multiple of block_n={block_n}"
    assert K % block_k == 0, f"K={K} must be a multiple of block_k={block_k}"
    n_tiles, k_tiles = N // block_n, K // block_k
    # Canonical (E, n_tiles, block_n, k_tiles, block_k)  ->
    # preshuffled (E, k_tiles, n_tiles, block_n, block_k).
    return np.ascontiguousarray(
        B.reshape(E, n_tiles, block_n, k_tiles, block_k).transpose(0, 3, 1, 2, 4)
    )


def build_launcher(spec: BatchedGemmSpec) -> KernelLauncher:
    k = build_batched_gemm(spec)
    hsaco, _ = build_hsaco_from_llvm_ir(lower_kernel_to_llvm(k))
    return KernelLauncher(
        hsaco=hsaco, kernel_name=k.name, signature=batched_gemm_signature(spec)
    )


def run(
    Bsz: int = 4,
    M: int = 256,
    N: int = 256,
    K: int = 256,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pipeline: str = "compv4",
    scheduler: str = "intrawave",
    epilogue: str = "default",
    iters: int = 80,
    warmup: int = 20,
) -> int:
    arch = get_device_arch(0)
    if arch is None:
        print("skipping: no HIP device visible")
        return 3
    # The 32x32x16 f16 MFMA atom this example builds (warp_tile=32x32x16) is
    # CDNA 4 / gfx950-only -- gfx942 (CDNA 3) tops out at 32x32x8 f16 and cannot
    # select it, and build_launcher targets gfx950 unconditionally, so any other
    # device would fail at code-object load. Skip with rc=3 (distinct from a
    # rc=0 pass) rather than launch a wrong-arch code object.
    if arch != "gfx950":
        print(f"skipping: device {arch} is not gfx950 (CDNA 4)")
        return 3
    rng = np.random.default_rng(0)
    # Pick warp_m / warp_n so warp_m * warp_tile_m == tile_m and
    # warp_n * warp_tile_n == tile_n with warp_tile=32x32. The TileSpec warp_*
    # fields are AMD wavefront tiling; this mirrors how the FusedMoe
    # orchestrator chooses the wavefront split from the tile dims.
    warp_m = max(1, tile_m // 32)
    warp_n = max(1, tile_n // 32)
    tile = TileSpec(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )
    spec_base = BatchedGemmSpec(
        name="preb_bgemm",
        tile=tile,
        trait=TraitSpec(
            pipeline=pipeline,
            scheduler=scheduler,
            epilogue=epilogue,
            preshuffle_b=False,
        ),
    )
    spec_pre = BatchedGemmSpec(
        name="preb_bgemm",
        tile=tile,
        trait=TraitSpec(
            pipeline=pipeline,
            scheduler=scheduler,
            epilogue=epilogue,
            preshuffle_b=True,
        ),
    )
    print(f"[shape] B={Bsz} M={M} N={N} K={K}, tile=({tile_m},{tile_n},{tile_k})")
    print(f"[trait] {pipeline}/{scheduler}/{epilogue}")

    A = rng.standard_normal((Bsz, M, K)).astype(np.float16)
    Bm = rng.standard_normal((Bsz, N, K)).astype(np.float16)
    Bm_pre = host_preshuffle_b(Bm, tile_n, tile_k)
    C_base = np.zeros((Bsz, M, N), dtype=np.float16)
    C_pre = np.zeros((Bsz, M, N), dtype=np.float16)

    launcher_base = build_launcher(spec_base)
    launcher_pre = build_launcher(spec_pre)
    grid = batched_gemm_grid(Bsz, M, N, spec_base)
    cfg = LaunchConfig(grid=grid, block=(spec_base.block_size, 1, 1))

    # Torch-free device I/O: DeviceMem RAII buffers, numpy host arrays uploaded
    # as raw bytes; the DeviceMem objects go straight into the launcher values
    # (pack_args reads their .ptr()). base and preshuffle share the A buffer.
    rt = Runtime()
    a_dev = DeviceMem(A.nbytes)
    bm_dev = DeviceMem(Bm.nbytes)
    bmpre_dev = DeviceMem(Bm_pre.nbytes)
    cbase_dev = DeviceMem(C_base.nbytes)
    cpre_dev = DeviceMem(C_pre.nbytes)
    rt.memcpy_h2d(a_dev.ptr(), as_u8_buffer(A), A.nbytes)
    rt.memcpy_h2d(bm_dev.ptr(), as_u8_buffer(Bm), Bm.nbytes)
    rt.memcpy_h2d(bmpre_dev.ptr(), as_u8_buffer(Bm_pre), Bm_pre.nbytes)
    rt.memset(cbase_dev.ptr(), 0, C_base.nbytes)
    rt.memset(cpre_dev.ptr(), 0, C_pre.nbytes)

    base_vals = {
        "A": a_dev,
        "B": bm_dev,
        "C": cbase_dev,
        "M": M,
        "N": N,
        "K": K,
        "stride_a": M * K,
        "stride_b": N * K,
        "stride_c": M * N,
    }
    pre_vals = dict(base_vals, B=bmpre_dev, C=cpre_dev)
    launcher_base(base_vals, config=cfg)
    launcher_pre(pre_vals, config=cfg)
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(C_base), cbase_dev.ptr(), C_base.nbytes)
    rt.memcpy_d2h(as_u8_buffer(C_pre), cpre_dev.ptr(), C_pre.nbytes)

    # numpy fp32 matmul reference (accumulate in fp32, cast back to fp16),
    # matching the kernel's fp32-accumulate / fp16-output contract.
    ref = (A.astype(np.float32) @ Bm.astype(np.float32).transpose(0, 2, 1)).astype(
        np.float16
    )

    def _max_abs_diff(actual: np.ndarray) -> float:
        return float(np.abs(actual.astype(np.float32) - ref.astype(np.float32)).max())

    base_err = _max_abs_diff(C_base)
    pre_err = _max_abs_diff(C_pre)
    # f16 GEMM accumulation noise grows roughly as sqrt(K); the numpy ref
    # accumulates in fp32 then casts back. Use a K-scaled tolerance (matches
    # the heuristic in `examples/moe/fused_moe_e2e_perf.py`).
    tol = max(7e-2, 7e-2 * (K / 256) ** 0.5)
    cross_err = float(
        np.abs(C_base.astype(np.float32) - C_pre.astype(np.float32)).max()
    )
    print(f"[parity] baseline max|err|={base_err:.4f}  (tol={tol:.3f})")
    print(f"[parity] preshuf  max|err|={pre_err:.4f}  (tol={tol:.3f})")
    print(f"[parity] base vs preshuf max|delta|={cross_err:.4f}  (need == 0.0)")
    base_ok = base_err <= tol
    pre_ok = pre_err <= tol
    cross_ok = cross_err <= 1e-3
    if not (base_ok and pre_ok and cross_ok):
        print(
            "[FAIL] numeric mismatch — kernel did not produce parity to numpy reference"
        )
        print("ref[0,:2,:2]    =", ref[0, :2, :2].astype(np.float32))
        print("base[0,:2,:2]   =", C_base[0, :2, :2].astype(np.float32))
        print("preshuf[0,:2,:2]=", C_pre[0, :2, :2].astype(np.float32))
        return 1

    base_us = (
        time_launches(
            lambda: launcher_base(base_vals, config=cfg), warmup=warmup, iters=iters
        )
        * 1e-3
    )
    pre_us = (
        time_launches(
            lambda: launcher_pre(pre_vals, config=cfg), warmup=warmup, iters=iters
        )
        * 1e-3
    )
    speedup = base_us / pre_us if pre_us > 0 else float("nan")
    print(f"[time] baseline    = {base_us * 1e6:8.2f} us")
    print(f"[time] preshuf     = {pre_us * 1e6:8.2f} us")
    print(f"[time] speedup     = {speedup:.3f}x")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--tile_m", type=int, default=128)
    ap.add_argument("--tile_n", type=int, default=128)
    ap.add_argument("--tile_k", type=int, default=64)
    ap.add_argument("--pipeline", default="compv4")
    ap.add_argument("--scheduler", default="intrawave")
    ap.add_argument("--epilogue", default="default")
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()
    return run(
        args.B,
        args.M,
        args.N,
        args.K,
        args.tile_m,
        args.tile_n,
        args.tile_k,
        args.pipeline,
        args.scheduler,
        args.epilogue,
        args.iters,
    )


if __name__ == "__main__":
    sys.exit(main())
