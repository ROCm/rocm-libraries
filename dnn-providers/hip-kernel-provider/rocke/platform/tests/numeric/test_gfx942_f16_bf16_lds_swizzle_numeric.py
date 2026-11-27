# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 f16/bf16 universal GEMM ``lds_swizzle`` regression, run on gfx942 hardware."""
import numpy as np
import pytest
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    is_valid_spec,
)
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16
from rocke.tests.arch_marks import arch_specific
from rocke.tests.engines import available_engines
from rocke.tests.gpu import current_device_arch_in
from rocke.tests.universal_gemm import launch_universal_gemm, lower_universal_gemm


def _swizzle_spec(tile, warp_tile, pipeline, dtype):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="swizzle",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, *warp_tile),
        trait=TraitSpec(pipeline=pipeline, epilogue="default", lds_swizzle=True),
        data=DataSpec(dtype, dtype, dtype, "fp32"),
    )


# The LDS store granule must cover max(a_per_lane, load_vec); when it tracked
# only a_per_lane a load_vec-wide store landed at a different swizzle column
# than the per-element ds_reads. These are the tiles that regressed.
@arch_specific(
    ("gfx942",),
    reason="regression for the gfx942 lds_swizzle store granule, run on gfx942 hardware",
)
@pytest.mark.parametrize(
    "tile, warp_tile, pipeline, dtype",
    [
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv4", "f16"),
        ((64, 64, 64, 2, 2), (16, 16, 16), "compv4", "f16"),
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv3", "f16"),
        ((64, 64, 32, 2, 2), (32, 32, 8), "compv4", "f16"),
        ((32, 32, 8, 1, 1), (32, 32, 8), "mem", "f16"),
        ((64, 64, 32, 2, 2), (16, 16, 16), "compv4", "bf16"),
        ((64, 64, 64, 2, 2), (16, 16, 16), "compv4", "bf16"),
        ((64, 64, 32, 2, 2), (32, 32, 8), "compv4", "bf16"),
        ((32, 32, 8, 1, 1), (32, 32, 8), "mem", "bf16"),
    ],
)
def test_gfx942_f16_bf16_lds_swizzle_numeric(tile, warp_tile, pipeline, dtype):
    arch = current_device_arch_in(("gfx942",))
    spec = _swizzle_spec(tile, warp_tile, pipeline, dtype)
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    m, n, k = 2 * tile_m, 3 * tile_n, 4 * tile_k
    rng = np.random.default_rng(tile_m * 100 + tile_k)
    a_int = rng.integers(-1, 2, (m, k))
    bt_int = rng.integers(-1, 2, (n, k))
    # A, B entries are -1/0/1, exact in f16/bf16, so the product is exact.
    expected = a_int.astype(np.float64) @ bt_int.astype(np.float64).T
    if dtype == "bf16":
        a = f32_to_bf16(a_int.astype(np.float32))
        bt = f32_to_bf16(bt_int.astype(np.float32))
        c = np.full((m, n), 0x7FC0, dtype=np.uint16)
    else:
        a = a_int.astype(np.float16)
        bt = bt_int.astype(np.float16)
        c = np.full((m, n), np.nan, dtype=np.float16)
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(spec, engine, arch=arch)
        result = launch_universal_gemm(llvm_ir, spec, a, bt, arch=arch, c=c)
        got = bf16_to_f32(result).astype(np.float64) if dtype == "bf16" else result
        np.testing.assert_array_equal(got.astype(np.float64), expected)
