#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
On-device GPU correctness test for the RowColQuant GEMM dispatcher bridge
(PR #9979).

Mirrors test_bquant_gpu_correctness.py: build the op's block-scale dispatcher
.so, run it on the GPU via RowColQuantGpuGemmRunner, and compare against a
NumPy fp32 reference. RowColQuant applies a per-row scale to A and a per-col
scale to B (equivalently, a per-column scale of the output):

    C[m, n] = AQ[m] * BQ[n] * sum_k A[m, k] * B[k, n]

The reference rounds A/B through the SAME fp8/bf8 quantization the kernel sees,
so the tolerance only absorbs GEMM accumulation error.

Block-scale traps handled explicitly:
  1. fp8 flavour: FNUZ on gfx942, OCP on gfx950/gfx12 (wrong flavour NaNs the
     reference). Uses the utils' own encode_fp8_bytes / quantize_dequantize_fp8,
     which select the flavour from the arch.
  2. warp_tile_k=32 on gfx942 (128 silently all-zeros): default_fp8_config /
     default_bf8_config derive it from the arch; the test asserts it matches.

Run:
    python3 -m pytest test_rowcolquant_gpu_correctness.py -v
    python3 test_rowcolquant_gpu_correctness.py          # standalone
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()


def _find_python_dir() -> Path:
    for c in (_HERE.parent.parent / "python", _HERE.parent / "python"):
        if (c / "gemm_rowcolquant_utils.py").is_file():
            return c
    for parent in _HERE.parents:
        cand = parent / "dispatcher" / "python"
        if (cand / "gemm_rowcolquant_utils.py").is_file():
            return cand
    raise RuntimeError("could not locate dispatcher/python/gemm_rowcolquant_utils.py")


sys.path.insert(0, str(_find_python_dir()))

import gemm_rowcolquant_utils as u  # noqa: E402
from gemm_rowcolquant_utils import (  # noqa: E402
    RowColQuantGemmProblem,
    RowColQuantGpuGemmRunner,
    setup_multiple_rowcolquant_dispatchers,
    default_fp8_config,
    default_bf8_config,
    _detect_gpu_arch,
    _warp_tile_k_for,
)


def _have_ml_dtypes() -> bool:
    return u.fp8_encoding_available()


def _have_gpu() -> bool:
    # Require BOTH a ROCm GPU (rocminfo) and hipcc: this test builds a kernel .so
    # at runtime via hipcc, so on GPU-but-no-hipcc nodes it must skip cleanly
    # rather than run and then fail at build time.
    if shutil.which("hipcc") is None:
        return False
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30)
        return "gfx" in out.stdout
    except Exception:
        return False


# =============================================================================
# NumPy reference: C[m,n] = AQ[m] * BQ[n] * sum_k A[m,k] B[k,n]
# =============================================================================


def reference_rowcolquant_gemm(A_dec, B_dec, AQ, BQ) -> np.ndarray:
    acc = A_dec.astype(np.float32) @ B_dec.astype(np.float32)   # [M, N]
    acc = acc * AQ.astype(np.float32)[:, None]                  # per-row scale
    acc = acc * BQ.astype(np.float32)[None, :]                  # per-col scale
    return acc.astype(np.float32)


# =============================================================================
# build + run + verify
# =============================================================================


def _run_case(dtype: str, M: int, N: int, K: int, out_dir: Path):
    arch = _detect_gpu_arch()
    make_cfg = default_fp8_config if dtype == "fp8" else default_bf8_config
    config = make_cfg(gfx_arch=arch)

    expected_wtk = _warp_tile_k_for(dtype, arch)
    assert config.warp_tile_k == expected_wtk, (
        f"warp_tile_k arch trap: got {config.warp_tile_k}, expected {expected_wtk} for {arch}"
    )

    so_paths = setup_multiple_rowcolquant_dispatchers(
        configs=[config], output_dir=out_dir, gfx_arch=arch
    )
    assert so_paths and so_paths[0] is not None, "rowcolquant kernel build failed"
    so_path = so_paths[0]

    rng = np.random.default_rng(1234)
    A_f = rng.uniform(-2.0, 2.0, (M, K)).astype(np.float32)
    B_f = rng.uniform(-2.0, 2.0, (K, N)).astype(np.float32)
    AQ = rng.uniform(0.5, 1.5, (M,)).astype(np.float32)
    BQ = rng.uniform(0.5, 1.5, (N,)).astype(np.float32)

    # Encode real fp8/bf8 bytes (arch-correct flavour), round the reference the same way.
    A_raw = u.encode_fp8_bytes(A_f, dtype, arch)
    B_raw = u.encode_fp8_bytes(B_f, dtype, arch)
    A_dec = u.quantize_dequantize_fp8(A_f, dtype, arch)
    B_dec = u.quantize_dequantize_fp8(B_f, dtype, arch)

    problem = RowColQuantGemmProblem(M=M, N=N, K=K)
    runner = RowColQuantGpuGemmRunner(so_path)
    result = runner.run(A_raw, B_raw, AQ, BQ, problem)

    C_gpu = np.asarray(result.C, dtype=np.float32)
    assert np.max(np.abs(C_gpu)) > 1e-3, "GPU output all-zeros (warp_tile_k arch trap?)"

    C_ref = reference_rowcolquant_gemm(A_dec, B_dec, AQ, BQ)
    max_rel = float(np.max(np.abs(C_gpu - C_ref)) / (np.max(np.abs(C_ref)) + 1e-6))

    tol = 0.05  # fp8/bf8 block-scale ~1e-2 .. 5e-2
    assert max_rel <= tol, f"max_rel={max_rel:.4f} > tol={tol} ({dtype} {M}x{N}x{K})"
    return max_rel, result.time_ms


_SKIP_NO_GPU = pytest.mark.skipif(not _have_gpu(), reason="no ROCm GPU detected")
_SKIP_NO_MLD = pytest.mark.skipif(not _have_ml_dtypes(), reason="ml_dtypes not installed")


@_SKIP_NO_GPU
@_SKIP_NO_MLD
@pytest.mark.parametrize("dtype", ["fp8", "bf8"])
def test_rowcolquant_gpu_matches_reference(dtype, tmp_path):
    max_rel, _ = _run_case(dtype, M=256, N=256, K=512, out_dir=tmp_path)
    assert max_rel <= 0.05


@_SKIP_NO_GPU
@_SKIP_NO_MLD
def test_rowcolquant_gpu_not_all_zeros(tmp_path):
    _run_case("fp8", M=256, N=256, K=512, out_dir=tmp_path)


if __name__ == "__main__":
    if not _have_gpu():
        print("SKIP: no GPU"); raise SystemExit(0)
    if not _have_ml_dtypes():
        print("SKIP: ml_dtypes not installed"); raise SystemExit(0)
    d = Path(tempfile.mkdtemp(prefix="rowcolquant_gputest_"))
    ok = True
    for dt in ("fp8", "bf8"):
        try:
            mr, t = _run_case(dt, 256, 256, 512, d)
            print(f"PASS rowcolquant {dt}: max_rel={mr:.4f} time_ms={t:.3f}")
        except Exception as e:
            ok = False
            print(f"FAIL rowcolquant {dt}: {e}")
    raise SystemExit(0 if ok else 1)
