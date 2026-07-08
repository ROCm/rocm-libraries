#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Standalone reproducer / regression test for rocm-libraries #8645.

On GPUs whose grid Y/Z dimension is limited to 65536 (e.g. gfx1201 / RDNA4), a
gemm with a large free (``N``) dimension launched a Tensile solution whose
``grid.y = ceil(N / MacroTileN)`` exceeded that limit, so the trailing
workgroups were never dispatched and those output columns were left
uninitialized (silently wrong / NaN results, non-deterministic across runs).

``rocblas_internal_gemm`` now chunks ``N`` around the Tensile call
(see ``library/src/blas3/Tensile/gemm_templates.cpp``), matching the chunking
already done on the source-GEMM and ``_64`` paths.

The equivalent gtest case is ``gemm_large_n_grid_y_limit`` in
``gemm_gtest.yaml`` (``M=3, N=2097153, K=15``, ``single``/``double``); this
Python version is a dependency-light reproducer that maps the same tall/skinny
problem through PyTorch (row-major ``A @ B`` becomes a column-major gemm whose
free dimension is the large one, i.e. ``N`` just over ``65536 * 32 == 2^21``).

It is skipped unless a ROCm PyTorch build and a GPU are available.
"""

import pytest

torch = pytest.importorskip("torch")

# N just past the fp64 grid.y boundary (65536 * 32 == 2^21 == 2097152).
N = 2097153
K = 15
M = 3


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_gemm_large_n_grid_y_limit(dtype):
    if not torch.cuda.is_available():
        pytest.skip("no GPU / ROCm device available")

    device = "cuda"
    # Row-major A @ B: rocBLAS computes it as a column-major gemm whose free
    # (N) dimension is the large leading dimension of A -> exercises grid.y.
    a = torch.randn(N, K, dtype=dtype, device=device)
    b = torch.randn(K, M, dtype=dtype, device=device)

    got = a @ b
    ref = (a.cpu() @ b.cpu()).to(device)

    assert torch.isfinite(got).all(), "gemm produced non-finite (NaN/Inf) output"

    tol = 1e-10 if dtype == torch.float64 else 1e-3
    max_err = (got - ref).abs().max().item()
    assert max_err < tol, f"max abs error {max_err} exceeds tol {tol} (dtype={dtype})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
