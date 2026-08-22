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

# Non-grouped gemm_bquant bridge (38_block_scale_gemm), the subject of PR #9982.
from gemm_bquant_utils import (
    BQuantGemmProblem,
    BQuantGpuGemmRunner,
    setup_multiple_bquant_dispatchers,
    _detect_gpu_arch,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)

log = logging.getLogger(__name__)

TOLERANCE = 0.05  # 5% max relative error — fp8/bf8 precision floor


# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

def _uses_ocp_fp8(gfx_arch: str) -> bool:
    """True when ck_tile::fp8_t is OCP (not FNUZ) for ``gfx_arch``.

    Mirrors the C++ arch defines in gemm_bquant_utils._compile_bquant_kernel:
    gfx950 / gfx12* build with -DCK_TILE_USE_OCP_FP8 (OCP e4m3/e5m2), everything
    else (gfx942 / gfx90a) uses FNUZ e4m3fnuz / e5m2fnuz.  Hardcoding OCP made
    gfx942 read NaN / mismatched fp8 values.
    """
    return ("gfx950" in gfx_arch) or ("gfx12" in gfx_arch)


def _ml_fp8_dtype(dtype: str, gfx_arch: str):
    import ml_dtypes
    if _uses_ocp_fp8(gfx_arch):
        return ml_dtypes.float8_e4m3fn if dtype == "fp8" else ml_dtypes.float8_e5m2
    return ml_dtypes.float8_e4m3fnuz if dtype == "fp8" else ml_dtypes.float8_e5m2fnuz


def _encode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str = "gfx950") -> np.ndarray:
    """Encode float32 → fp8 bytes (uint8 view), ARCH-AWARE (FNUZ on gfx942)."""
    try:
        ml_t = _ml_fp8_dtype(dtype, gfx_arch)
        return arr.astype(ml_t).view(np.uint8)
    except ImportError:
        return (np.clip(arr, -2.0, 2.0) * 64).astype(np.int8).view(np.uint8)


def _decode_fp8(arr: np.ndarray, dtype: str, gfx_arch: str = "gfx950") -> np.ndarray:
    try:
        ml_t = _ml_fp8_dtype(dtype, gfx_arch)
        return arr.view(ml_t).astype(np.float32)
    except ImportError:
        return arr.view(np.int8).astype(np.float32) / 64.0


def _encode_bf8(arr: np.ndarray, gfx_arch: str = "gfx950") -> np.ndarray:
    return _encode_fp8(arr, "bf8", gfx_arch)

def _decode_bf8(arr: np.ndarray, gfx_arch: str = "gfx950") -> np.ndarray:
    return _decode_fp8(arr, "bf8", gfx_arch)


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


# OCP FP4 E2M1 lookup table (from pk_fp4.hpp e2m1_to_fp32_table).
# Index i (0-15) gives the float32 value for the 4-bit code i.
_FP4_E2M1_LUT: np.ndarray = np.array([
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def _decode_fp4(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """Unpack K*N OCP FP4 E2M1 values from K*N/2 packed bytes.

    pk_fp4_t packing (from pk_fp4.hpp _pack/_unpack):
      byte = (element1 << 4) | (element0 & 0xF)
    Low nibble = element at flat index 2i; high nibble = element at flat index 2i+1.
    The flat layout is row-major [K, N], so each byte contains two consecutive N-elements.
    """
    flat = packed.flatten()
    lo = (flat & 0x0F).astype(np.uint8)
    hi = ((flat >> 4) & 0x0F).astype(np.uint8)
    out = np.empty(K * N, dtype=np.float32)
    out[0::2] = _FP4_E2M1_LUT[lo]
    out[1::2] = _FP4_E2M1_LUT[hi]
    return out.reshape(K, N)


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
    C_gpu_f = C_gpu.astype(np.float32)
    C_ref_f = C_ref.astype(np.float32)
    num = np.abs(C_gpu_f - C_ref_f)
    # Use 1% of the global max magnitude as the denominator floor to avoid
    # inflating the relative error when individual elements are near zero
    # (a common occurrence with random inputs that partially cancel in GEMM).
    ref_max = float(np.abs(C_ref_f).max())
    den = np.abs(C_ref_f) + max(ref_max * 1e-2, 1e-6)
    return float(np.max(num / den))


# ---------------------------------------------------------------------------
# Core test helper
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


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
    nan_mask = ~np.isfinite(C_gpu.astype(np.float32))
    if np.any(nan_mask):
        nan_frac = nan_mask.mean()
        sample = C_gpu.astype(np.float32).flat[:8].tolist()
        log.debug("%s: NaN/Inf fraction=%.3f, first 8 elements=%s", label, nan_frac, sample)
        return FAIL, f"{label}: GPU output contains NaN/Inf (frac={nan_frac:.3f})"

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

def _make_fp8_inputs(M, N, K, gK, gN, dtype="fp8", seed=42, gfx_arch="gfx950"):
    rng = np.random.default_rng(seed)
    QK_B = math.ceil(K / gK)
    QN_B = math.ceil(N / gN)
    A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f32 = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    BQ    = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)
    A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
    B_raw = _encode_fp8(B_f32, dtype, gfx_arch)
    A_dec = _decode_fp8(A_raw, dtype, gfx_arch)
    B_dec = _decode_fp8(B_raw, dtype, gfx_arch)
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


# N-tile sweep: N=128 is a single N-tile; N=256/512 span 2 and 4 TileN blocks and
# exercise the round-6 per-N-tile PermuteN de-permute (the round-5 global riffle
# scrambled columns at N>=256).  gN must divide N; TileN is 64 (decode) / 128 (MX).
_N_SWEEP = (128, 256, 512)


def test_c4_fp8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=768 = 3*TileK(256): use num_loop=3 (TailNumber::Odd) for better coverage.
    # num_loop=2 works but exercises only the no-hot-loop/Even tail path; 3 gives
    # the no-hot-loop/Odd tail path and exercises the BQ scale prefetch more robustly.
    M, K, gK, gN = 16, 768, 128, 1
    cfg = default_fp8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "fp8",
                                                          gfx_arch=gfx_arch)
        status, detail = _run_one(f"C4/fp8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ,
                                  out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"C4/fp8: PASS for N in {_N_SWEEP}"


def test_c4_bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=768 = 3*TileK(256): use num_loop=3 for the same reason as test_c4_fp8.
    M, K, gK, gN = 16, 768, 128, 1
    cfg = default_bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, B_raw, B_dec, BQ = _make_fp8_inputs(M, N, K, gK, gN, "bf8",
                                                          gfx_arch=gfx_arch)
        status, detail = _run_one(f"C4/bf8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ,
                                  out_dir, c_dtype=np.float16, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"C4/bf8: PASS for N in {_N_SWEEP}"


def test_h3_mx_bf16bf16(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch
    M, K, gK, gN = 128, 256, 32, 1
    cfg = default_mx_bf16bf16_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        # C is bf16_t (2 bytes); use uint16 buffer + decode fn so size matches exactly.
        A_raw, A_dec, B_raw, B_dec, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        status, detail = _run_one(f"H3/mx_bf16bf16/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_dec, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16bf16: PASS for N in {_N_SWEEP}"


def test_h3_mx_bf16bf8(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # K=384 = 3*TileK(128): use num_loop=3 (TailNumber::Odd) for broader pipeline coverage.
    M, K, gK, gN = 128, 384, 128, 1
    cfg = default_mx_bf16bf8_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        # B is bf8: encode K*N float32 values as bf8 bytes
        B_f32 = np.random.default_rng(43).uniform(-1.0, 1.0, (K, N)).astype(np.float32)
        B_raw_bf8 = _encode_fp8(B_f32, "bf8", gfx_arch)
        B_dec_bf8 = _decode_fp8(B_raw_bf8, "bf8", gfx_arch)
        status, detail = _run_one(f"H3/mx_bf16bf8/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw_bf8, B_dec_bf8, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16bf8: PASS for N in {_N_SWEEP}"


def test_h3_mx_bf16fp4(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    # pk_fp4: 2 fp4 values per byte → B buffer is K*N/2 bytes.
    # K=256 = 2*TileK(128): MicroscaleCompV3 needs num_loop>=2 to avoid OOB second prefetch.
    M, K, gK, gN = 128, 256, 32, 1
    cfg = default_mx_bf16fp4_config(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
    for N in _N_SWEEP:
        A_raw, A_dec, _, _, BQ_e8m0, BQ_f32 = _make_bf16_inputs(M, N, K, gK, gN)
        rng = np.random.default_rng(44)
        # rcr kernel reads pk_fp4 B COLUMN-MAJOR: consecutive K-elements share a byte
        # (low nibble = k even, high nibble = k odd), per reference_mx_gemm_bquant's (k&1)
        # branch. Generate logical (K,N) fp4 codes, pack column-major, LUT-decode reference.
        codes = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
        B_f32_approx = _FP4_E2M1_LUT[codes]                  # reference values (K,N)
        _flat = codes.flatten(order='F').astype(np.uint8)    # col-major: idx = n*K + k
        _lo = _flat[0::2] & 0x0F                             # k even -> low nibble
        _hi = _flat[1::2] & 0x0F                             # k odd  -> high nibble
        B_raw = (_lo | (_hi << 4)).astype(np.uint8)
        status, detail = _run_one(f"H3/mx_bf16fp4/N{N}", cfg, M, N, K,
                                  A_raw, A_dec, B_raw, B_f32_approx, BQ_e8m0,
                                  out_dir, c_dtype=np.uint16, c_decode_fn=_bf16_raw_to_f32,
                                  BQ_ref=BQ_f32, gfx_arch=gfx_arch)
        if status != PASS:
            return status, detail
    return PASS, f"H3/mx_bf16fp4: PASS for N in {_N_SWEEP}"


def test_c_i4(out_dir: Path, gfx_arch: str) -> tuple[str, str]:
    """Round-6: fp8i4 / bf8i4 must be exact once BQ is encoded to the kernel's
    QDataType (fp8/bf8, 1 byte).  The round-5 float32 BQ produced NaN.  Swept over
    N to also exercise the per-N-tile de-permute.  B is pk_int4 (2 per byte)."""
    M, K, gK, gN = 16, 256, 128, 1
    results = []
    for variant, ctor, dtype in (("fp8i4", default_fp8i4_config, "fp8"),
                                 ("bf8i4", default_bf8i4_config, "bf8")):
        cfg = ctor(quant_group_k=gK, quant_group_n=gN, gfx_arch=gfx_arch)
        for N in _N_SWEEP:
            rng = np.random.default_rng(45)
            A_f32 = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
            A_raw = _encode_fp8(A_f32, dtype, gfx_arch)
            A_dec = _decode_fp8(A_raw, dtype, gfx_arch)
            # B is pk_int4: 2 K-elements per byte, packed COLUMN-MAJOR (rcr).
            #
            # Ground-truth nibble convention (Old-TE reference_gemm_quant +
            # pk_int4.hpp with CK_TILE_USE_PK4_LAYOUT_SHUFFLE=1):
            #   * value(code) = code - 8   (code in 0..15 -> value in [-8, +7]),
            #     NOT the two's-complement (codes<8?codes:codes-16) round-5 used.
            #   * pk_int4_t_to_fp32x2_t returns {x_h, x_l} (SHUFFLE swap), and
            #     load_b picks (k&1) ? .hi : .lo, so for the byte covering
            #     (k even, k odd) of a fixed n:
            #         k even -> HIGH nibble;  k odd -> LOW nibble.
            #   The device conversion consumes the same bytes after the
            #   UNCONDITIONAL permute_vectors_i4x4_b in the ctypes lib, so the
            #   host byte layout must match this reference exactly.
            codes = rng.integers(0, 16, size=(K, N), dtype=np.uint8)  # 0..15
            B_dec = codes.astype(np.float32) - 8.0                    # value = code-8
            _flat = codes.flatten(order='F').astype(np.uint8)  # col-major idx=n*K+k
            _even = _flat[0::2] & 0x0F   # k even -> HIGH nibble
            _odd = _flat[1::2] & 0x0F    # k odd  -> LOW nibble
            B_raw = (_odd | (_even << 4)).astype(np.uint8)
            QK_B = math.ceil(K / gK)
            QN_B = math.ceil(N / gN)
            # BQ supplied as float32 -> the runner encodes it to fp8/bf8 (QDataType,
            # arch-aware).  The kernel therefore sees the fp8/bf8-rounded scale, so
            # the CPU reference must use the same round-tripped value to stay fair.
            BQ = rng.uniform(0.5, 2.0, (QK_B, QN_B)).astype(np.float32)
            BQ_ref = _decode_fp8(_encode_fp8(BQ, dtype, gfx_arch), dtype, gfx_arch)
            BQ_ref = BQ_ref.reshape(QK_B, QN_B).astype(np.float32)
            status, detail = _run_one(f"C/{variant}/N{N}", cfg, M, N, K,
                                      A_raw, A_dec, B_raw, B_dec, BQ,
                                      out_dir, c_dtype=np.float16,
                                      BQ_ref=BQ_ref, gfx_arch=gfx_arch)
            if status != PASS:
                return status, detail
        results.append(variant)
    return PASS, f"C/i4: PASS ({', '.join(results)}) for N in {_N_SWEEP}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("C4/fp8",          test_c4_fp8),
    ("C4/bf8",          test_c4_bf8),
    ("C/i4",            test_c_i4),
    ("H3/mx_bf16bf16",  test_h3_mx_bf16bf16),
    ("H3/mx_bf16bf8",   test_h3_mx_bf16bf8),
    ("H3/mx_bf16fp4",   test_h3_mx_bf16fp4),
]


def _gpu_and_hipcc_available() -> bool:
    """True only if both hipcc and a ROCm GPU are present.

    Lets the standalone test SKIP cleanly (exit 0) on CPU-only CI runners
    instead of failing when the kernel build/run cannot proceed.
    """
    import shutil
    import subprocess as _sp
    if shutil.which("hipcc") is None:
        return False
    try:
        out = _sp.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return False
    return "gfx" in out


def main():
    parser = argparse.ArgumentParser(description="BQuant GPU correctness tests (C4 + H3)")
    parser.add_argument(
        "--gfx",
        default=None,
        help="GPU arch (default: auto-detect the running device via rocm_agent_enumerator)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if not _gpu_and_hipcc_available():
        print("SKIP: no GPU/hipcc detected; skipping bquant GPU correctness")
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gfx_arch = args.gfx or _detect_gpu_arch()
    log.info("Target GPU arch: %s", gfx_arch)
    out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="bquant_gpu_test_"))
    log.info("Kernel output dir: %s", out_dir)

    results = []
    for name, fn in TESTS:
        log.info("--- Running %s ---", name)
        try:
            status, detail = fn(out_dir, gfx_arch)
        except Exception as exc:
            # MX (H3) variants need gfx950 e8m0 microscaling hardware; on any
            # other arch (e.g. gfx1250/MI400) that is an expected skip, not a
            # failure -- the fp8/bf8/i4 block-scale paths still run there.
            if "requires gfx950" in str(exc):
                status, detail = SKIP, f"{name}: skipped on {gfx_arch} (MX is gfx950-only)"
            else:
                status, detail = FAIL, f"{name}: exception: {exc}"
        results.append((name, status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    passed = sum(1 for _, s, _ in results if s == PASS)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    for name, status, detail in results:
        print(f"  [{status:4s}] {detail}")
    print(f"\n{passed}/{len(results)} passed, {skipped} skipped, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
