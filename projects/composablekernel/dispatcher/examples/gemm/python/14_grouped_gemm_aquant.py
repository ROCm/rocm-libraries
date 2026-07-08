#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 14 — GroupedGemm AQuant via the Dispatcher

Demonstrates the full three-layer path:
  1. Codegen  — unified_grouped_gemm_aquant_codegen.py -> .hpp
  2. Compile  — hipcc -> .so
  3. Run      — AQuantGpuGemmRunner -> C = dequant(A, AQ) @ B

AQuant: A-side activation quantization.
  C[M,N] = dequant(A[M,K], AQ[ceil(M/gM), ceil(K/gK)]) @ B[K,N]
  AQ is the per-group scale for A; B is unquantized.

Verifies the GPU result against a NumPy fp32 reference.

Requirements:
  - gfx950 GPU (MI350X)
  - hipcc in PATH
  - CK include path discoverable relative to this repo

Usage:
  python3 14_grouped_gemm_aquant.py                     # fp8, 1x1x128 groups, M=16 N=64 K=256
  python3 14_grouped_gemm_aquant.py --dtype bf8
  python3 14_grouped_gemm_aquant.py --dtype fp8 --M 32 --N 128 --K 512 --quant-group-k 128
  python3 14_grouped_gemm_aquant.py --no-verify         # skip CPU reference check
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

from grouped_gemm_aquant_utils import (
    AQuantKernelConfig,
    AQuantGemmProblem,
    AQuantGpuGemmRunner,
    setup_multiple_aquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# fp8 encode/decode helpers
# =============================================================================


def _float32_to_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Encode float32 values as fp8 bytes (uint8 view of the fp8 bit pattern).

    dtype: "fp8" -> float8_e4m3fn, "bf8" -> float8_e5m2.
    Falls back to clamping when ml_dtypes is not installed.
    """
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.astype(ml_t).view(np.uint8)
    except ImportError:
        clamped = np.clip(arr, -2.0, 2.0)
        return (clamped * 64).astype(np.int8).view(np.uint8)


def _fp8_to_float32(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Decode fp8 bytes (uint8 view) back to float32."""
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.view(ml_t).astype(np.float32)
    except ImportError:
        return arr.view(np.int8).astype(np.float32) / 64.0


# =============================================================================
# CPU reference: C = dequant(A, AQ) @ B
# =============================================================================


def reference_aquant_gemm(
    A: np.ndarray,
    B: np.ndarray,
    AQ: np.ndarray,
    problem: AQuantGemmProblem,
) -> np.ndarray:
    """
    CPU fp32 reference for C = dequant(A, AQ) @ B.

    A   [M, K]  float32 — decoded from fp8 bytes
    B   [K, N]  float32 — decoded from fp8 bytes
    AQ  [QM_A, QK_A] float32 per-block scales for A

    Dequant: A[m, k] *= AQ[m // gM, k // gK]
    """
    M, N, K = problem.M, problem.N, problem.K
    gM = problem.quant_group_m
    gK = problem.quant_group_k

    A_f32 = A.astype(np.float32)
    B_f32 = B.astype(np.float32)

    # Apply per-block scales to A
    A_dequant = A_f32.copy()
    for qi in range(problem.QM_A):
        for qk in range(problem.QK_A):
            m_start = qi * gM
            m_end   = min(m_start + gM, M)
            k_start = qk * gK
            k_end   = min(k_start + gK, K)
            scale   = float(AQ[qi, qk])
            A_dequant[m_start:m_end, k_start:k_end] *= scale

    C_ref = A_dequant @ B_f32
    return C_ref.astype(np.float16)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="GroupedGemm AQuant dispatcher example")
    parser.add_argument("--dtype", choices=["fp8", "bf8"], default="fp8")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--quant-group-k", type=int, default=128)
    parser.add_argument("--quant-group-m", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gfx-arch", type=str, default="gfx950")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    gK = args.quant_group_k
    gM = args.quant_group_m

    # -------------------------------------------------------------------------
    # 1. Build kernel config
    # -------------------------------------------------------------------------
    if args.dtype == "fp8":
        config = default_fp8_config(quant_group_k=gK, quant_group_m=gM, gfx_arch=args.gfx_arch)
    else:
        config = default_bf8_config(quant_group_k=gK, quant_group_m=gM, gfx_arch=args.gfx_arch)

    log.info("Kernel: %s", config.name)

    # -------------------------------------------------------------------------
    # 2. Codegen + compile
    # -------------------------------------------------------------------------
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="aquant_ex14_"))
    log.info("Output dir: %s", out_dir)

    so_paths = setup_multiple_aquant_dispatchers(
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
    QM_A = math.ceil(M / gM)

    # Generate float32 values in fp8 representable range, then encode as fp8 bytes.
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    # AQ scale: per-group scale for A, shape [QM_A, QK_A], dtype float32
    AQ_f32 = rng.uniform(0.5, 2.0, (QM_A, QK_A)).astype(np.float32)

    # Encode A and B as fp8 bytes (uint8 view, 1 byte/element)
    A_raw = _float32_to_fp8(A_f32, args.dtype)
    B_raw = _float32_to_fp8(B_f32, args.dtype)

    # Decode back so the CPU reference uses the same rounded values the kernel sees
    A_dec = _fp8_to_float32(A_raw, args.dtype)
    B_dec = _fp8_to_float32(B_raw, args.dtype)

    # -------------------------------------------------------------------------
    # 4. Run on GPU
    # -------------------------------------------------------------------------
    problem = AQuantGemmProblem(
        M=M, N=N, K=K,
        quant_group_m=gM,
        quant_group_n=1,
        quant_group_k=gK,
    )

    runner = AQuantGpuGemmRunner(so_path)
    log.info("Running kernel: %s", runner.kernel_name)

    result = runner.run(A=A_raw, B=B_raw, AQ=AQ_f32, problem=problem)
    log.info("Kernel time: %.3f ms", result.time_ms)

    # -------------------------------------------------------------------------
    # 5. Verify
    # -------------------------------------------------------------------------
    if not args.no_verify:
        C_ref = reference_aquant_gemm(A_dec, B_dec, AQ_f32, problem)
        C_gpu = result.C

        max_abs_err = float(np.max(np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32))))
        max_ref = float(np.max(np.abs(C_ref.astype(np.float32)))) + 1e-6
        max_rel = max_abs_err / max_ref

        tolerance = 0.05  # fp8 ~1e-2 to 5e-2
        if max_rel <= tolerance:
            log.info("PASSED (max_rel=%.4f, tol=%.4f)", max_rel, tolerance)
        else:
            log.error("FAILED (max_rel=%.4f > tol=%.4f)", max_rel, tolerance)
            return 1
    else:
        log.info("Verification skipped (--no-verify)")

    log.info("Example 14 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
