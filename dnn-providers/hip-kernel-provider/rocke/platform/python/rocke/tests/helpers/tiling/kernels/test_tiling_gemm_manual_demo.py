# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end test for the manual GEMM demo.

The demo writes every tensor in the GEMM axis order -- A (M, K), B (N, K), C (M, N) -- with the
memory layout only in the strides (A and B K-contiguous, C M-contiguous). This proves the result is
bit-exact against a numpy golden on integer inputs, including shapes where M, N and K all differ, so
an operand read with its axes swapped cannot pass by symmetry.

The test skips without the rocke substrate. Without a gfx90a GPU, including on a host whose HIP
runtime loads but sees no device, it skips, or fails under ``ROCKE_REQUIRE_GFX90A=1`` (the
``gfx90a_gpu`` fixture). torch-free.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "rocke.helpers.compile", reason="rocke substrate required (run on the gfx90a host)"
)

from rocke.helpers.tiling.kernels.tiling_gemm_manual_demo import (
    run_and_verify_manual,
)


@pytest.mark.parametrize("shape", [(256, 256, 256), (64, 32, 48), (32, 96, 64)])
@pytest.mark.usefixtures("gfx90a_gpu")
def test_manual_gemm_bit_exact_on_gfx90a(shape: tuple[int, int, int]) -> None:
    report = run_and_verify_manual(*shape, arch="gfx90a")
    assert report["bit_exact"], report
    assert report["max_abs_diff"] == 0.0, report
