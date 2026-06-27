# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf check for the ``preshuffle_b=True`` BatchedGemm path.

The kernel under test is :func:`build_batched_gemm` with
``trait.preshuffle_b=True``. The kernel assumes B has been pre-shuffled
on the host into ``(batch, k_tiles, n_tiles, block_n, block_k)``
contiguous layout. The expected per-K-tile B-load is one wide
``buffer_load_dwordx<N>`` per warp: this is what closes the gap to
CK Tile's preshuffled-B path (`§12.1.H` of the runbook).

This script:
  1. Builds two BatchedGemm kernels (preshuffle_b=False / True) for
     identical (B, M, N, K) and TileSpec.
  2. Generates a random A and B, computes a torch reference.
  3. Launches both kernels (B preshuffled on the host for the second).
  4. Asserts numeric parity to the reference, then times both with HIP
     events.

Run with::

    cd <repo>/dnn-providers/hip-kernel-provider/rocKE/Python
    python rocke/examples/moe/test_preshuffle_b.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if "rocke" not in sys.modules:
    HERE = Path(__file__).resolve()
    pkg_root = HERE.parents[4]
    sys.path.insert(0, str(pkg_root))

import torch  # noqa: E402

from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402
from rocke.instances.common.batched_gemm import (  # noqa: E402
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
)
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec  # noqa: E402
from rocke.runtime.comgr import build_hsaco_from_llvm_ir  # noqa: E402
from rocke.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
    synchronize_and_release,
    time_launches,
)


def host_preshuffle_b(B: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
    """``(E, N, K)`` row-major B  ->  ``(E, k_tiles, n_tiles, block_n, block_k)``
    contiguous, matching the layout expected by ``preshuffle_b=True``."""
    E, N, K = B.shape
    assert N % block_n == 0, f"N={N} must be a multiple of block_n={block_n}"
    assert K % block_k == 0, f"K={K} must be a multiple of block_k={block_k}"
    n_tiles, k_tiles = N // block_n, K // block_k
    # Canonical (E, n_tiles, block_n, k_tiles, block_k)  ->
    # preshuffled (E, k_tiles, n_tiles, block_n, block_k).
    return (
        B.view(E, n_tiles, block_n, k_tiles, block_k)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
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
    if not torch.cuda.is_available():
        print("skipping: no CUDA device")
        return 0
    torch.manual_seed(0)
    # Pick warp_m / warp_n so warp_m * warp_tile_m == tile_m and
    # warp_n * warp_tile_n == tile_n with warp_tile=32x32. This
    # mirrors how the FusedMoe orchestrator chooses warps from the
    # tile dims.
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

    A = torch.randn(Bsz, M, K, dtype=torch.float16, device="cuda")
    Bm = torch.randn(Bsz, N, K, dtype=torch.float16, device="cuda")
    Bm_pre = host_preshuffle_b(Bm, tile_n, tile_k)
    C_base = torch.zeros(Bsz, M, N, dtype=torch.float16, device="cuda")
    C_pre = torch.zeros(Bsz, M, N, dtype=torch.float16, device="cuda")

    launcher_base = build_launcher(spec_base)
    launcher_pre = build_launcher(spec_pre)
    grid = batched_gemm_grid(Bsz, M, N, spec_base)
    cfg = LaunchConfig(grid=grid, block=(spec_base.block_size, 1, 1))

    base_vals = {
        "A": A,
        "B": Bm,
        "C": C_base,
        "M": M,
        "N": N,
        "K": K,
        "stride_a": M * K,
        "stride_b": N * K,
        "stride_c": M * N,
    }
    pre_vals = dict(base_vals, B=Bm_pre, C=C_pre)
    launcher_base(base_vals, config=cfg)
    launcher_pre(pre_vals, config=cfg)
    synchronize_and_release()

    ref = (A.float() @ Bm.float().transpose(-1, -2)).half()

    def _max_abs_diff(actual: torch.Tensor) -> float:
        return float((actual - ref).abs().max().cpu())

    base_err = _max_abs_diff(C_base)
    pre_err = _max_abs_diff(C_pre)
    # f16 GEMM accumulation noise grows roughly as sqrt(K); the
    # torch ref accumulates in fp32 then casts back. Use a K-scaled
    # tolerance (matches the heuristic in `examples/moe/fused_moe_e2e_perf.py`).
    tol = max(7e-2, 7e-2 * (K / 256) ** 0.5)
    cross_err = float((C_base - C_pre).abs().max().cpu())
    print(f"[parity] baseline max|err|={base_err:.4f}  (tol={tol:.3f})")
    print(f"[parity] preshuf  max|err|={pre_err:.4f}  (tol={tol:.3f})")
    print(f"[parity] base vs preshuf max|delta|={cross_err:.4f}  (need == 0.0)")
    base_ok = base_err <= tol
    pre_ok = pre_err <= tol
    cross_ok = cross_err <= 1e-3
    if not (base_ok and pre_ok and cross_ok):
        print(
            "[FAIL] numeric mismatch — kernel did not produce parity to torch reference"
        )
        print("ref[0,:2,:2]    =", ref[0, :2, :2].float().cpu().numpy())
        print("base[0,:2,:2]   =", C_base[0, :2, :2].float().cpu().numpy())
        print("preshuf[0,:2,:2]=", C_pre[0, :2, :2].float().cpu().numpy())
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
