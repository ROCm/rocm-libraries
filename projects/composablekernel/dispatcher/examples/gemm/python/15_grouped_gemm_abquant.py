#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 15 — GroupedGemm ABQuant via the Dispatcher

Demonstrates the full three-layer path:
  1. Codegen  — unified_grouped_gemm_abquant_codegen.py -> .hpp
  2. Compile  — hipcc -> .so
  3. Run      — ABQuantGpuGemmRunner -> C = dequant(A, AQ) @ dequant(B, BQ)

ABQuant: both A-side and B-side quantization active simultaneously.
  C[M,N] = dequant(A[M,K], AQ[ceil(M/aM), ceil(K/aK)]) @ dequant(B[K,N], BQ[ceil(K/bK), ceil(N/bN)])

Verifies the GPU result against a NumPy fp32 reference.

Requirements:
  - gfx950 GPU (MI350X)
  - hipcc in PATH
  - CK include path discoverable relative to this repo

Usage:
  python3 15_grouped_gemm_abquant.py                     # fp8, 128x128x128, no preshuffle
  python3 15_grouped_gemm_abquant.py --dtype bf8
  python3 15_grouped_gemm_abquant.py --pipeline eightwaves --M 192 --N 256 --K 128
  python3 15_grouped_gemm_abquant.py --no-verify
"""

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from grouped_gemm_abquant_utils import (
    ABQuantKernelConfig,
    ABQuantGemmProblem,
    ABQuantGpuGemmRunner,
    setup_multiple_abquant_dispatchers,
    default_fp8_compv3_config,
    default_bf8_compv3_config,
    default_fp8_eightwaves_config,
    default_bf8_eightwaves_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# fp8 encode/decode helpers
# =============================================================================


def _float32_to_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.astype(ml_t).view(np.uint8)
    except ImportError:
        clamped = np.clip(arr, -2.0, 2.0)
        return (clamped * 64).astype(np.int8).view(np.uint8)


def _fp8_to_float32(arr: np.ndarray, dtype: str) -> np.ndarray:
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.view(ml_t).astype(np.float32)
    except ImportError:
        return arr.view(np.int8).astype(np.float32) / 64.0


# =============================================================================
# CPU reference: C = dequant(A, AQ) @ dequant(B, BQ)
# =============================================================================


def reference_abquant_gemm(
    A: np.ndarray,
    B: np.ndarray,
    AQ: np.ndarray,
    BQ: np.ndarray,
    problem: ABQuantGemmProblem,
) -> np.ndarray:
    """
    CPU fp32 reference for C = dequant(A, AQ) @ dequant(B, BQ).

    A   [M, K]  float32 — decoded from fp8 bytes
    B   [K, N]  float32 — decoded from fp8 bytes
    AQ  [QM_A, QK_A]  float32 per-block scales for A
    BQ  [QK_B, QN_B]  float32 per-block scales for B
    """
    M, N, K = problem.M, problem.N, problem.K
    aM = problem.aquant_group_m
    aK = problem.aquant_group_k
    bK = problem.bquant_group_k
    bN = problem.bquant_group_n

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)

    # Dequantize A: A[m, k] *= AQ[m // aM, k // aK]
    A_dequant = A_f32.copy()
    for qi in range(problem.QM_A):
        for qk in range(problem.QK_A):
            m_start = qi * aM
            m_end   = min(m_start + aM, M)
            k_start = qk * aK
            k_end   = min(k_start + aK, K)
            A_dequant[m_start:m_end, k_start:k_end] *= float(AQ[qi, qk])

    # Dequantize B: B[k, n] *= BQ[k // bK, n // bN]
    B_dequant = B_f32.copy()
    for qk in range(problem.QK_B):
        for qn in range(problem.QN_B):
            k_start = qk * bK
            k_end   = min(k_start + bK, K)
            n_start = qn * bN
            n_end   = min(n_start + bN, N)
            B_dequant[k_start:k_end, n_start:n_end] *= float(BQ[qk, qn])

    C_ref = A_dequant @ B_dequant
    return C_ref.astype(np.float16)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="GroupedGemm ABQuant dispatcher example")
    parser.add_argument("--dtype", choices=["fp8", "bf8"], default="fp8")
    parser.add_argument("--pipeline", choices=["compv3", "eightwaves", "preshuffleb"],
                        default="compv3")
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--quant-group-k", type=int, default=128)
    parser.add_argument("--bquant-group-n", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gfx-arch", type=str, default="gfx950")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    gK = args.quant_group_k
    bN = args.bquant_group_n

    # -------------------------------------------------------------------------
    # 1. Build kernel config
    # -------------------------------------------------------------------------
    if args.pipeline == "compv3":
        if args.dtype == "fp8":
            config = default_fp8_compv3_config(quant_group_k=gK, bquant_group_n=bN,
                                               gfx_arch=args.gfx_arch)
        else:
            config = default_bf8_compv3_config(quant_group_k=gK, bquant_group_n=bN,
                                               gfx_arch=args.gfx_arch)
    elif args.pipeline == "eightwaves":
        if args.dtype == "fp8":
            config = default_fp8_eightwaves_config(quant_group_k=gK, bquant_group_n=bN,
                                                   gfx_arch=args.gfx_arch)
        else:
            config = default_bf8_eightwaves_config(quant_group_k=gK, bquant_group_n=bN,
                                                   gfx_arch=args.gfx_arch)
    else:
        from grouped_gemm_abquant_utils import default_fp8_preshuffleb_config, default_bf8_preshuffleb_config
        if args.dtype == "fp8":
            config = default_fp8_preshuffleb_config(quant_group_k=gK, bquant_group_n=bN,
                                                    gfx_arch=args.gfx_arch)
        else:
            config = default_bf8_preshuffleb_config(quant_group_k=gK, bquant_group_n=bN,
                                                    gfx_arch=args.gfx_arch)

    log.info("Kernel: %s", config.name)

    # -------------------------------------------------------------------------
    # 2. Codegen + compile
    # -------------------------------------------------------------------------
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="abquant_ex15_"))
    log.info("Output dir: %s", out_dir)

    so_paths = setup_multiple_abquant_dispatchers(
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
    QK_A = math.ceil(K / gK)
    QM_A = 1  # aquant_group_m=1 by default → QM_A == M
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / bN)

    # Recompute using actual problem to be safe
    problem = ABQuantGemmProblem(
        M=M, N=N, K=K,
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=gK,
        bquant_group_m=1, bquant_group_n=bN, bquant_group_k=gK,
    )

    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ_f32 = rng.uniform(0.5, 2.0, (problem.QM_A, problem.QK_A)).astype(np.float32)
    BQ_f32 = rng.uniform(0.5, 2.0, (problem.QK_B, problem.QN_B)).astype(np.float32)

    A_raw = _float32_to_fp8(A_f32, args.dtype)
    B_raw = _float32_to_fp8(B_f32, args.dtype)

    A_dec = _fp8_to_float32(A_raw, args.dtype)
    B_dec = _fp8_to_float32(B_raw, args.dtype)

    # -------------------------------------------------------------------------
    # 4. Run on GPU
    # -------------------------------------------------------------------------
    runner = ABQuantGpuGemmRunner(so_path)
    log.info("Running kernel: %s", runner.kernel_name)

    result = runner.run(A=A_raw, B=B_raw, AQ=AQ_f32, BQ=BQ_f32, problem=problem)
    log.info("Kernel time: %.3f ms", result.time_ms)

    # -------------------------------------------------------------------------
    # 5. Verify
    # -------------------------------------------------------------------------
    if not args.no_verify:
        C_ref = reference_abquant_gemm(A_dec, B_dec, AQ_f32, BQ_f32, problem)
        C_gpu = result.C

        max_abs_err = float(np.max(np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32))))
        max_ref = float(np.max(np.abs(C_ref.astype(np.float32)))) + 1e-6
        max_rel = max_abs_err / max_ref

        tolerance = 0.05
        if max_rel <= tolerance:
            log.info("PASSED (max_rel=%.4f, tol=%.4f)", max_rel, tolerance)
        else:
            log.error("FAILED (max_rel=%.4f > tol=%.4f)", max_rel, tolerance)
            return 1
    else:
        log.info("Verification skipped (--no-verify)")

    log.info("Example 15 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
