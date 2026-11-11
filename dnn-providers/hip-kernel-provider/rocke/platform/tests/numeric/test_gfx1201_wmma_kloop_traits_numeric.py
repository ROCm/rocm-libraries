# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 WMMA universal GEMM: the K-loop placement, raster and stagger traits, on hardware.

``lds_store_after_mma`` moves the next tile's LDS stores after the WMMAs,
``group_m`` remaps which output tile each workgroup computes, and ``k_stagger``
changes the order each workgroup walks K. A wrong store placement reads a half
before it is written; a wrong raster writes some tiles twice and others never;
a wrong stagger skips or repeats K tiles. All three still run and return
plausible numbers, so the check is bit-exact against NumPy: small-integer
inputs make the fp32 accumulation exact, and the reference is rounded to C's
dtype the way the kernel's store rounds (round-to-nearest-even). C is
pre-filled with NaN. The shapes cover an M-tile count that leaves a partial
GROUP_M group, K-tile counts the stagger spans do not divide (so the start
offsets wrap), power-of-two K, and odd K-tile counts (both LDS halves read).
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

_TRAITS = {
    "store_after_mma": dict(lds_store_after_mma=True),
    "group_m": dict(group_m=4),
    "k_stagger": dict(k_stagger=8, k_stagger_step=4),
    "k_stagger_wrap": dict(k_stagger=3, k_stagger_step=2),
    "all": dict(lds_store_after_mma=True, group_m=4, k_stagger=8, k_stagger_step=4),
}

# (M-tiles, N-tiles, K-tiles): 5 M-tiles leave a partial group of 4; 7 and 9
# K-tiles are odd and not multiples of the stagger spans (8*4 and 3*2 tiles);
# 32 K-tiles of 32 make K a power of two.
_GRIDS = [(5, 3, 7), (2, 3, 9), (4, 2, 32)]


def _spec(tile, dtype, **traits):
    tile_m, tile_n, tile_k, warp_m, warp_n = tile
    return UniversalGemmSpec(
        name="gfx1201_kloop_traits",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, warp_n, 1, 16, 16, 16),
        trait=TraitSpec(
            pipeline="compv4", epilogue="default", lds_swizzle=True, **traits
        ),
        data=DataSpec(dtype, dtype, dtype, "fp32"),
        wave_size=32,
    )


@arch_specific(
    ("gfx1201",),
    reason="the WMMA K-loop traits are admitted on gfx1201 only",
)
@pytest.mark.parametrize("grid", _GRIDS, ids=lambda g: "x".join(map(str, g)))
@pytest.mark.parametrize("traits", sorted(_TRAITS))
@pytest.mark.parametrize(
    "tile, dtype",
    [((128, 128, 32, 2, 2), "f16"), ((128, 64, 32, 2, 1), "bf16")],
    ids=["128x128x32-f16", "128x64x32-bf16"],
)
def test_gfx1201_wmma_kloop_traits_numeric(tile, dtype, traits, grid):
    _check_numeric(tile, dtype, _TRAITS[traits], grid)


@arch_specific(
    ("gfx1201",),
    reason="the one-workgroup LDS double buffer is admitted on gfx1201 only",
)
@pytest.mark.parametrize("grid", _GRIDS, ids=lambda g: "x".join(map(str, g)))
@pytest.mark.parametrize("dtype", ("f16", "bf16"))
@pytest.mark.parametrize(
    "tile",
    ((128, 128, 64, 2, 4), (256, 128, 32, 4, 2), (128, 256, 32, 2, 4)),
    ids=("128x128x64-w2x4", "256x128x32-w4x2", "128x256x32-w2x4"),
)
def test_gfx1201_wmma_db1wg_numeric(tile, dtype, grid):
    _check_numeric(tile, dtype, dict(lds_db_allow_1wg=True, **_TRAITS["all"]), grid)


@arch_specific(
    ("gfx1201",),
    reason="prepared global reads are admitted on gfx1201 FP16 compv4 only",
)
@pytest.mark.parametrize(
    "grid", _GRIDS + [(2, 2, 1)], ids=lambda g: "x".join(map(str, g))
)
@pytest.mark.parametrize("addressing", ("auto", "global64"))
@pytest.mark.parametrize(
    "tile, one_wg",
    (
        ((128, 128, 32, 2, 2), False),
        ((64, 64, 64, 2, 2), False),
        ((128, 128, 64, 2, 4), True),
    ),
    ids=("128x128x32-w2x2", "64x64x64-w2x2", "128x128x64-w2x4"),
)
def test_gfx1201_wmma_prepared_reads_numeric(tile, one_wg, addressing, grid):
    # (2, 2, 1) is one K tile: the double-buffered loop runs zero iterations.
    traits = dict(
        global_addressing=addressing, lds_db_allow_1wg=one_wg, **_TRAITS["all"]
    )
    _check_numeric(tile, "fp16", traits, grid)


def _check_numeric(tile, dtype, traits, grid):
    arch = current_device_arch_in(("gfx1201",))
    spec = _spec(tile, dtype, **traits)
    assert is_valid_spec(spec, arch=arch) == (True, "ok")
    tile_m, tile_n, tile_k = tile[:3]
    m, n, k = grid[0] * tile_m, grid[1] * tile_n, grid[2] * tile_k
    rng = np.random.default_rng(m + 7 * n + 31 * k)
    a_int = rng.integers(-2, 3, (m, k))
    bt_int = rng.integers(-2, 3, (n, k))
    # Entries in [-2, 2] are exact in f16/bf16 and |C| <= 4K stays exact in the
    # fp32 accumulator, so the only rounding is the one cast to C's dtype:
    # round the exact product the same way and compare bit for bit.
    exact = (a_int.astype(np.float64) @ bt_int.astype(np.float64).T).astype(np.float32)
    if dtype == "bf16":
        a = f32_to_bf16(a_int.astype(np.float32))
        bt = f32_to_bf16(bt_int.astype(np.float32))
        c = np.full((m, n), 0x7FC0, dtype=np.uint16)
        expected = bf16_to_f32(f32_to_bf16(exact)).astype(np.float64)
    else:
        a = a_int.astype(np.float16)
        bt = bt_int.astype(np.float16)
        c = np.full((m, n), np.nan, dtype=np.float16)
        expected = exact.astype(np.float16).astype(np.float64)
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(spec, engine, arch=arch)
        result = launch_universal_gemm(llvm_ir, spec, a, bt, arch=arch, c=c.copy())
        got = bf16_to_f32(result).astype(np.float64) if dtype == "bf16" else result
        np.testing.assert_array_equal(got.astype(np.float64), expected, err_msg=engine)
