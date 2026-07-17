#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for BQuant GEMM dispatcher — C4 and H3 coverage.

Requires a gfx950 GPU (MI350X / MI355X) and hipcc in PATH.

Tests:
  C4 — fp8, bf8: GPU output is non-zero and within 5% max-relative-error
       vs. a fp32 CPU reference.
  H3 — mx_bf16bf16, mx_bf16bf8, mx_bf16fp4: same non-zero / rel-error checks,
       plus verify QuantType::BQuantGrouped + e8m0 pipeline compiles and runs.
  M2 — timing: time_ms is non-zero when timing is requested.

Run:
  python3 test_bquant_gpu_correctness.py
  python3 test_bquant_gpu_correctness.py -v          # verbose hipcc output
  python3 test_bquant_gpu_correctness.py --gfx gfx950
"""

import argparse
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_bquant_utils import (
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
    setup_multiple_bquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _encode_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Encode float32 → fp8 bytes (uint8 view). Uses ml_dtypes when available."""
    try:
        import ml_dtypes
        # fp8 = OCP e4m3fn (bias=7); bf8 = OCP e5m2.
        # CK kernels on gfx950 are compiled with -DCK_TILE_USE_OCP_FP8 so they
        # use the same OCP format — bit patterns are compatible.
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.astype(ml_t).view(np.uint8)
    except ImportError:
        return (np.clip(arr, -2.0, 2.0) * 64).astype(np.int8).view(np.uint8)


def _decode_fp8(arr: np.ndarray, dtype: str) -> np.ndarray:
    try:
        import ml_dtypes
        ml_t = ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
        return arr.view(ml_t).astype(np.float32)
    except ImportError:
        return arr.view(np.int8).astype(np.float32) / 64.0


def _encode_bf8(arr: np.ndarray) -> np.ndarray:
    return _encode_fp8(arr, "bf8")

def _decode_bf8(arr: np.ndarray) -> np.ndarray:
    return _decode_fp8(arr, "bf8")


def _encode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Encode float32 scale values → e8m0 uint8 (MX block scale format).

    e8m0 stores a power-of-two exponent: byte b represents 2^(b - 127).
    Scales must be positive; zero maps to 0 (subnormal/zero in e8m0).
    """
    arr = np.asarray(arr, dtype=np.float32)
    # Clamp to representable range: 2^-127 … 2^127
    arr = np.clip(arr, 0.0, np.float32(2.0 ** 127))
    nonzero = arr > 0.0
    out = np.zeros(arr.shape, dtype=np.uint8)
    # biased exponent = floor(log2(s)) + 127, clamped to [0, 254]
    exp = np.floor(np.log2(arr[nonzero])).astype(np.int32) + 127
    out[nonzero] = np.clip(exp, 0, 254).astype(np.uint8)
    return out


def _decode_e8m0(arr: np.ndarray) -> np.ndarray:
    """Decode e8m0 uint8 → float32 scale values (2^(b - 127))."""
    arr = np.asarray(arr, dtype=np.uint8)
    return np.exp2(arr.astype(np.float32) - 127.0)


def _bf16_raw_to_f32(arr: np.ndarray) -> np.ndarray:
    """Reinterpret a uint16 array of bf16 bit patterns as float32."""
    u16 = arr.flatten().astype(np.uint16)
    words = np.zeros(len(u16) * 2, dtype=np.uint16)
    words[1::2] = u16  # bf16 occupies upper 2 bytes of float32 (little-endian)
    return words.view(np.float32).reshape(arr.shape)


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

def _reference_gemm(A_f32: np.ndarray, B_f32: np.ndarray,
                    BQ: np.ndarray, problem: BQuantGemmProblem,
                    c_dtype=np.float16) -> np.ndarray:
    """C = A @ dequant(B, BQ) in fp32, cast to c_dtype."""
    gK, gN = problem.quant_group_k, problem.quant_group_n
    B_dq = B_f32.copy()
    for qi in range(problem.QK_B):
        for qj in range(problem.QN_B):
            k0, k1 = qi * gK, min((qi + 1) * gK, problem.K)
            n0, n1 = qj * gN, min((qj + 1) * gN, problem.N)
            B_dq[k0:k1, n0:n1] *= float(BQ[qi, qj])
    return (A_f32.astype(np.float32) @ B_dq.astype(np.float32)).astype(c_dtype)


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    num = np.abs(C_gpu.astype(np.float32) - C_ref.astype(np.float32))
    den = np.abs(C_ref.astype(np.float32)) + 1e-6
    return float(np.max(num / den))


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"


def _run_one(label: str, config, M: int, N: int, K: int,
             A_raw: np.ndarray, A_f32: np.ndarray,
             B_raw: np.ndarray, B_f32: np.ndarray,
             BQ: np.ndarray,
             out_dir: Path,
             c_dtype=np.float16,
             c_decode_fn=None,
             BQ_ref: np.ndarray = None,
             gfx_arch: str = "gfx950") -> tuple[str, str]:
    """
    Build, run, and verify one kernel.

    BQ       -- raw buffer passed to the GPU kernel (may be e8m0 uint8 for MX variants)
    BQ_ref   -- float32 scales used for the CPU reference; defaults to BQ if not given
                (caller should supply the decoded float32 version when BQ is e8m0)
    c_decode_fn -- optional fn(raw_C_array) -> float32 array, used when c_dtype is uint16
                   (bf16 raw output) to convert before comparison

    Returns (PASS|FAIL, detail_message).
    """
    if BQ_ref is None:
        BQ_ref = BQ.astype(np.float32)

    problem = BQuantGemmProblem(
        M=M, N=N, K=K,
        quant_group_m=config.quant_group_m,
        quant_group_n=config.quant_group_n,
        quant_group_k=config.quant_group_k,
    )

    so_paths = setup_multiple_bquant_dispatchers(
        configs=[config],
        output_dir=out_dir,
        gfx_arch=gfx_arch,
    )
    if not so_paths or so_paths[0] is None:
        return FAIL, f"{label}: kernel build failed"

    runner = BQuantGpuGemmRunner(so_paths[0])

    # Run once (timing is collected internally by the runner)
    result = runner.run(A=A_raw, B=B_raw, BQ=BQ, problem=problem, c_dtype=c_dtype)
    C_gpu_raw = result.C
    # Convert raw output to float32 for validation (needed for bf16 output via uint16 buffer)
    C_gpu = c_decode_fn(C_gpu_raw) if c_decode_fn is not None else C_gpu_raw.astype(np.float32)

    # Non-zero check (C4 / H3 smoke)
    if np.all(C_gpu == 0):
        return FAIL, f"{label}: GPU output is all-zero"
    if not np.all(np.isfinite(C_gpu.astype(np.float32))):
        return FAIL, f"{label}: GPU output contains NaN/Inf"

    # Correctness check vs. CPU reference (always in float32)
    C_ref = _reference_gemm(A_f32, B_f32, BQ_ref, problem, np.float32)
    mre = _max_rel_err(C_gpu, C_ref)
    if mre > TOLERANCE:
        return FAIL, (f"{label}: max_rel_err={mre:.4f} > tol={TOLERANCE:.4f} "
                      f"(shape M={M} N={N} K={K})")

    # Timing check (M2 sanity)
    result_timed = runner.run(A=A_raw, B=B_raw, BQ=BQ, problem=problem,
                              c_dtype=c_dtype)
    if result_timed.time_ms <= 0.0:
        return FAIL, f"{label}: time_ms={result_timed.time_ms:.4f} is not positive"

    return PASS, (f"{label}: max_rel_err={mre:.4f}, "
                  f"time_ms={result_timed.time_ms:.3f}")


# ---------------------------------------------------------------------------
# Individual test cases
# ---------------------------------------------------------------------------

def _make_fp8_inputs(M, N, K, gK, gN, dtype="fp8", seed=42):
    rng = np.random.default_rng(seed)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    BQ    = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)
    A_raw = _encode_fp8(A_f32, dtype)
    B_raw = _encode_fp8(B_f32, dtype)
    A_dec = _decode_fp8(A_raw, dtype)
    B_dec = _decode_fp8(B_raw, dtype)
    return A_raw, A_dec, B_raw, B_dec, BQ


def _to_bf16_raw(x: np.ndarray) -> np.ndarray:
    """Encode float32 array → uint16 array of bfloat16 bit patterns."""
    packed = np.frombuffer(x.astype(np.float32).tobytes(), dtype=np.uint16)
    # Little-endian: bf16 occupies the upper 2 bytes of each float32,
    # which are at odd indices (1, 3, 5, ...) in the uint16 view.
    return packed[1::2].reshape(x.shape)


def _make_bf16_inputs(M, N, K, gK, gN, seed=42):
    rng = np.random.default_rng(seed)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    # BQ for MX: scales stored as e8m0 (uint8, one byte per group).
    # Generate float32 scales in [0.5, 1.5], encode to e8m0, then decode back
    # to float32 for the CPU reference (so reference uses the same quantised values).
    BQ_f32 = rng.uniform(0.5, 1.5, (QK_B, QN_B)).astype(np.float32)
    BQ_e8m0 = _encode_e8m0(BQ_f32)          # uint8, matches kernel's e8m0_t
    BQ_f32_dec = _decode_e8m0(BQ_e8m0)       # float32 after e8m0 round-trip
    # A and B as uint16 bfloat16; decode back for reference
    A_raw = _to_bf16_raw(A)
    B_raw = _to_bf16_raw(B)
    A_dec = _bf16_raw_to_f32(A_raw)
    B_dec = _bf16_raw_to_f32(B_raw)
    return A_raw, A_dec, B_raw, B_dec, BQ_e8m0, BQ_f32_dec


def test_c4_fp8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=512 = 2*TileK(256): CompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, N, K, gK, gN = 16, 64, 512, 128, 1
    cfg = default_fp8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "fp8")
    return _run_one("C4/fp8", cfg, M, N, K,
                    A_raw, A_dec, B_raw, B_dec, BQ,
                    out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)


def test_c4_bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=512 = 2*TileK(256): CompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, N, K, gK, gN = 16, 64, 512, 128, 1
    cfg = default_bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "bf8")
    return _run_one("C4/bf8", cfg, M, N, K,
                    A_raw, A_dec, B_raw, B_dec, BQ,
                    out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)


def test_h3_mx_bf16bf16(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, N, K, gK, gN = 128, 128, 256, 32, 1
    cfg = default_mx_bf16bf16_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    # C is bf16_t (2 bytes); use uint16 buffer + decode fn so size matches exactly.
    A_raw, A_dec, B_raw, B_dec, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
    return _run_one("H3/mx_bf16bf16", cfg, M, N, K,
                    A_raw, A_dec, B_raw, B_dec, BQ_e8m0,
                    out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                    BQ_ref=BQ_f32, gfx_arch=gfx_arch)


def test_h3_mx_bf16bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, N, K, gK, gN = 128, 128, 256, 128, 1
    cfg = default_mx_bf16bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
    # B is bf8: encode K*N float32 values as bf8 bytes
    B_f32 = np.random.default_rng(43).uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    B_raw_bf8 = _encode_fp8(B_f32, "bf8")
    B_dec_bf8 = _decode_fp8(B_raw_bf8, "bf8")
    return _run_one("H3/mx_bf16bf8", cfg, M, N, K,
                    A_raw, A_dec, B_raw_bf8, B_dec_bf8, BQ_e8m0,
                    out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                    BQ_ref=BQ_f32, gfx_arch=gfx_arch)


def test_h3_mx_bf16fp4(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # pk_fp4: 2 fp4 values per byte → B buffer is K*N/2 bytes.
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch.
    M, N, K, gK, gN = 128, 128, 256, 32, 1
    cfg = default_mx_bf16fp4_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
    rng = np.random.default_rng(44)
    # pk_fp4: K*N values packed 2-per-byte
    B_raw = rng.integers(0, 256, size=(K * N // 2,), dtype=np.uint8)
    # Reference: approximate fp4 values as small floats in [-1,1]
    B_f32_approx = (B_raw.astype(np.float32) / 128.0 - 1.0).reshape(K // 2, N)
    B_f32_approx = np.repeat(B_f32_approx, 2, axis=0)  # rough unpack approximation
    return _run_one("H3/mx_bf16fp4", cfg, M, N, K,
                    A_raw, A_dec, B_raw, B_f32_approx, BQ_e8m0,
                    out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                    BQ_ref=BQ_f32, gfx_arch=gfx_arch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("C4/fp8",          test_c4_fp8),
    ("C4/bf8",          test_c4_bf8),
    ("H3/mx_bf16bf16",  test_h3_mx_bf16bf16),
    ("H3/mx_bf16bf8",   test_h3_mx_bf16bf8),
    ("H3/mx_bf16fp4",   test_h3_mx_bf16fp4),
]


def main():
    parser = argparse.ArgumentParser(description="BQuant GPU correctness tests (C4 + H3)")
    parser.add_argument("--gfx", default="gfx950", help="GPU arch (default: gfx950)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bquant_gpu_test_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, args.gfx)
        except Exception as exc:
            status, detail = FAIL, f"{name}: exception: {exc}"
        results.append((name, status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    passed = sum(1 for _, s, _ in results if s == PASS)
    for name, status, detail in results:
        print(f"  [{status:4s}] {detail}")
    print(f"\n{passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
