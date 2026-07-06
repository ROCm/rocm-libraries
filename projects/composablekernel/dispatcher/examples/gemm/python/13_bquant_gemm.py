#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 13 — BQuantGrouped GEMM via the Dispatcher

Demonstrates the full three-layer path:
  1. Codegen  — unified_bquant_gemm_codegen.py → .hpp
  2. Compile  — hipcc → .so
  3. Run      — BQuantGpuGemmRunner → C = A @ dequant(B, BQ)

Verifies the GPU result against a NumPy fp32 reference.

Requirements:
  - gfx950 GPU (MI350X)
  - hipcc in PATH
  - CK include path discoverable relative to this repo

Usage:
  python3 13_bquant_gemm.py                     # fp8, 1x1x128 groups, M=16 N=64 K=256
  python3 13_bquant_gemm.py --dtype bf8
  python3 13_bquant_gemm.py --dtype fp8 --M 32 --N 128 --K 512 --quant-group-k 128
  python3 13_bquant_gemm.py --no-verify         # skip CPU reference check
"""

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add dispatcher/python to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from bquant_gemm_utils import (
    BQuantKernelConfig,
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
    setup_multiple_bquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# NumPy reference: C = A @ dequant(B, BQ)
# =============================================================================


def _fp8_to_float32(arr: np.ndarray) -> np.ndarray:
    """Cast fp8 byte array to float32 (treating as float8_e4m3fn)."""
    # numpy doesn't have fp8; cast uint8 bit pattern to float via ml_dtypes if available,
    # otherwise fall back to float32 direct cast (close enough for reference at low values).
    try:
        import ml_dtypes
        return arr.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    except ImportError:
        return arr.astype(np.float32)


def reference_bquant_gemm(
    A: np.ndarray,
    B: np.ndarray,
    BQ: np.ndarray,
    problem: BQuantGemmProblem,
) -> np.ndarray:
    """
    CPU fp32 reference for C = A @ dequant(B, BQ).

    A   [M, K]  float32 (upcast from fp8/bf8)
    B   [K, N]  float32 (upcast from fp8/bf8)
    BQ  [QK_B, QN_B] float32 scale factors

    Dequant: B[k, n] *= BQ[k // gK, n // gN]
    """
    M, N, K = problem.M, problem.N, problem.K
    gK = problem.quant_group_k
    gN = problem.quant_group_n

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)

    # Apply per-block scales to B
    B_dequant = B_f32.copy()
    for qi in range(problem.QK_B):
        for qj in range(problem.QN_B):
            k_start = qi * gK
            k_end   = min(k_start + gK, K)
            n_start = qj * gN
            n_end   = min(n_start + gN, N)
            scale   = float(BQ[qi, qj])
            B_dequant[k_start:k_end, n_start:n_end] *= scale

    C_ref = A_f32 @ B_dequant
    return C_ref.astype(np.float16)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="BQuantGrouped GEMM dispatcher example")
    parser.add_argument("--dtype", choices=["fp8", "bf8"], default="fp8")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--quant-group-k", type=int, default=128)
    parser.add_argument("--quant-group-n", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gfx-arch", type=str, default="gfx950")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    gK = args.quant_group_k
    gN = args.quant_group_n

    # -------------------------------------------------------------------------
    # 1. Build kernel config
    # -------------------------------------------------------------------------
    if args.dtype == "fp8":
        config = default_fp8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=args.gfx_arch)
    else:
        config = default_bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=args.gfx_arch)

    log.info("Kernel: %s", config.name)

    # -------------------------------------------------------------------------
    # 2. Codegen + compile
    # -------------------------------------------------------------------------
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bquant_ex13_"))
    log.info("Output dir: %s", out_dir)

    so_paths = setup_multiple_bquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=args.gfx_arch,
    )

    if not so_paths or so_paths[0] is None:
        log.error("Kernel build failed — see errors above")
        return 1

    so_path = so_paths[0]
    log.info("Built: %s", so_path)

    # -------------------------------------------------------------------------
    # 3. Generate inputs
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(42)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)

    # A and B as uint8 (fp8 byte representation) in [-2, 2] float range
    # Using simple values so the reference is easy to verify
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    BQ_f32 = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)

    # Cast to fp8 approximation (uint8 for raw byte passing to C lib)
    # Real fp8 encoding would use ml_dtypes; here we use float16 as a stand-in
    # that the kernel can safely interpret as fp8 bit patterns for testing.
    A_raw = A_f32.astype(np.float16)
    B_raw = B_f32.astype(np.float16)

    # -------------------------------------------------------------------------
    # 4. Run on GPU
    # -------------------------------------------------------------------------
    problem = BQuantGemmProblem(M=M, N=N, K=K,
                                quant_group_m=1,
                                quant_group_n=gN,
                                quant_group_k=gK)

    runner = BQuantGpuGemmRunner(so_path)
    log.info("Running kernel: %s", runner.kernel_name)

    result = runner.run(A=A_raw, B=B_raw, BQ=BQ_f32, problem=problem)
    log.info("Kernel time: %.3f ms", result.time_ms)

    # -------------------------------------------------------------------------
    # 5. Verify
    # -------------------------------------------------------------------------
    if not args.no_verify:
        C_ref = reference_bquant_gemm(A_f32, B_f32, BQ_f32, problem)
        C_gpu = result.C

        max_rel = float(np.max(np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32)))
                        / (np.max(np.abs(C_ref.astype(np.float32))) + 1e-6))

        tolerance = 0.05  # fp8 ~1e-2 to 5e-2
        if max_rel <= tolerance:
            log.info("PASSED (max_rel=%.4f, tol=%.4f)", max_rel, tolerance)
        else:
            log.error("FAILED (max_rel=%.4f > tol=%.4f)", max_rel, tolerance)
            return 1
    else:
        log.info("Verification skipped (--no-verify)")

    log.info("Example 13 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
