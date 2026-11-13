# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 WMMA universal GEMM: the compv4 double buffer and ``lds_swizzle``, on hardware.

Both knobs change only where a tile sits in LDS. compv4 alternates the K tiles
between two LDS halves, so the WMMA fragment reads must follow the parity the
loop filled; ``lds_swizzle`` XORs the LDS column on the store, so the reads must
apply the same XOR. Either read mistake still emits, runs, and returns
plausible numbers, so the check is an exact integer GEMM. Every shape runs
several workgroups per axis and at least four K tiles, so both LDS halves are
read more than once.
"""
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


def _spec(tile, pipeline, swizzle, dtype):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="gfx1201_wmma",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, 16, 16, 16),
        trait=TraitSpec(pipeline=pipeline, epilogue="default", lds_swizzle=swizzle),
        data=DataSpec(dtype, dtype, dtype, "fp32"),
        wave_size=32,
    )


@arch_specific(
    ("gfx1201",),
    reason="compv4 and lds_swizzle on the WMMA path are admitted on gfx1201 only",
)
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "tile, pipeline, swizzle",
    [
        ((128, 128, 32, 2, 2), "compv4", False),
        ((128, 128, 32, 2, 2), "compv4", True),
        ((128, 64, 32, 2, 1), "compv4", False),
        ((128, 64, 32, 2, 1), "compv4", True),
        ((128, 128, 16, 4, 1), "compv4", True),
        ((128, 128, 32, 2, 2), "mem", True),
        ((64, 32, 16, 2, 1), "mem", True),
    ],
)
def test_gfx1201_wmma_compv4_and_swizzle_numeric(tile, pipeline, swizzle, dtype):
    arch = current_device_arch_in(("gfx1201",))
    spec = _spec(tile, pipeline, swizzle, dtype)
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    # 2x3 workgroups and 5 K tiles: an odd count ends the double buffer on the
    # half it started on, so a read pinned to either half fails.
    m, n, k = 2 * tile_m, 3 * tile_n, 5 * tile_k
    rng = np.random.default_rng(tile_m * 1000 + tile_n * 10 + tile_k)
    a_int = rng.integers(-2, 3, (m, k))
    bt_int = rng.integers(-2, 3, (n, k))
    # A, B entries are in [-2, 2], exact in f16/bf16, and |C| <= 4K stays exact.
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
        np.testing.assert_array_equal(got.astype(np.float64), expected, err_msg=engine)
