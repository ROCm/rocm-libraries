# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regressions for the multi-wave AQuant memory-pipeline tail on gfx1250."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from gemm_aquant_utils import (
    AQuantGemmProblem,
    AQuantGpuGemmRunner,
    AQuantKernelConfig,
    setup_multiple_aquant_dispatchers,
)


# Each first shape/seed reproduced a wrong answer in the default-config sweep.
# The additional three/four-tile shapes exercise the same build's hot-loop path.
CASES = [
    ("ccr", "cshuffle", (128, 64, 128), (1, 4), 64,
     (384, 128, 256), 1609182115),
    ("ccr", "default", (128, 128, 128), (2, 4), 128,
     (128, 256, 256), 1855132628),
    ("rrr", "cshuffle", (192, 64, 256), (4, 2), 128,
     (576, 192, 512), 2349328619),
    ("rrr", "default", (64, 128, 256), (1, 4), 128,
     (256, 128, 512), 1060679394),
]


@pytest.mark.parametrize(
    "layout,epilogue,tile,warps,warp_k,shape,seed", CASES,
    ids=["ccr-cshuffle-4wave", "ccr-default-8wave",
         "rrr-cshuffle-8wave", "rrr-default-4wave"],
)
def test_aquant_mem_tail(
    layout, epilogue, tile, warps, warp_k, shape, seed,
    gpu_arch, skip_without_ml_dtypes, tmp_path,
):
    if gpu_arch != "gfx1250":
        pytest.skip("regression configurations use gfx1250 WMMA tiles")
    import ml_dtypes

    config = AQuantKernelConfig(
        variant_key="fp8", gfx_arch=gpu_arch, layout=layout,
        pipeline="mem", scheduler="intrawave", epilogue=epilogue,
        tile_m=tile[0], tile_n=tile[1], tile_k=tile[2],
        warp_m=warps[0], warp_n=warps[1], warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_k,
        quant_group_m=1, quant_group_n=1, quant_group_k=128,
        preshuffle_aquant=False, double_smem_buffer=False, k_block_per_cu=1,
    )
    libraries = setup_multiple_aquant_dispatchers(
        configs=[config], output_dir=tmp_path, gfx_arch=gpu_arch,
    )
    assert libraries and libraries[0] is not None, "regression kernel failed to build"
    runner = AQuantGpuGemmRunner(libraries[0], layout=layout)

    for index, k_tiles in enumerate((2, 3, 4)):
        m, n = shape[:2]
        k = k_tiles * tile[2]
        rng = np.random.default_rng(seed + index)
        a = rng.normal(0, 1, (m, k)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        b = rng.normal(0, 1, (k, n)).astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        aq = rng.uniform(0.5, 1.5, (m, k // 128)).astype(np.float32)
        reference = (a.astype(np.float32) * aq[:, np.arange(k) // 128]) @ b.astype(np.float32)
        problem = AQuantGemmProblem(M=m, N=n, K=k, quant_group_k=128)
        result = runner.run(a.view(np.uint8), aq, b.view(np.uint8), problem)
        output = result.C.astype(np.float32)
        assert np.isfinite(output).all(), f"nonfinite output with {k_tiles} K tiles"
        error = np.max(np.abs(output - reference)) / (np.max(np.abs(reference)) + 1e-6)
        assert error <= 0.05, f"{k_tiles} K tiles: normalized error {error}"
