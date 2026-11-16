# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1201 ``direct_a`` GEMM on hardware, bit-exact against NumPy.

The pipeline's failure modes all return plausible numbers. A B tile stored
before every wave has finished reading the previous one corrupts some K tiles;
a wrong raster or stagger writes or sums the wrong tiles; a partial tile that
reads past a row instead of past the buffer adds a neighbour's data; an
unmasked store writes into the next row of C. Small-integer inputs make the
fp32 accumulation exact, so the only rounding is the cast to C's type (none
for an FP32 C). C is pre-filled with NaN, and guard rows after C catch a store
outside it.

Shapes: M and N that are not whole tiles (in either or both), one K tile (the
loops never run), odd and even K-tile counts (the one-tile remainder loop),
a K-tile count the stagger span does not divide, and a single partial tile
smaller than the tile itself.
"""
import numpy as np
import pytest
from rocke.instances.common.gemm_direct_a import direct_a_launch_supported
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

# (tile_m, tile_n, tile_k, warp_m): with and without B read-ahead, four and two
# waves, and a single K atom per tile.
_TILES = [
    (128, 64, 64, 4),
    (128, 96, 64, 4),
    (64, 64, 64, 4),
    (64, 32, 32, 2),
    (64, 32, 16, 2),
]
_TRAITS = {
    "plain": dict(),
    "swizzle": dict(lds_swizzle=True),
    "palette": dict(lds_swizzle=True, group_m=4, k_stagger=8, k_stagger_step=4),
    "wrap": dict(lds_swizzle=True, group_m=8, k_stagger=3, k_stagger_step=2),
}
# (M, N, K-tiles)
_SHAPES = [
    (300, 200, 1),
    (128, 192, 2),
    (1000, 1000, 7),
    (257, 97, 5),
    (40, 24, 3),
    (640, 385, 16),
]
_GUARD = 64


def _check(tile, traits, shape, ab, c):
    arch = current_device_arch_in(("gfx1201",))
    tile_m, tile_n, tile_k, warp_m = tile
    spec = UniversalGemmSpec(
        name="gfx1201_direct_a",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, 1, 1, 16, 16, 16),
        trait=TraitSpec(
            pipeline="direct_a", scheduler="intrawave", epilogue="default", **traits
        ),
        data=DataSpec(ab, ab, c),
        wave_size=32,
    )
    assert is_valid_spec(spec, arch) == (True, "ok")
    m, n, k_tiles = shape
    k = k_tiles * tile_k
    assert direct_a_launch_supported(spec, m, n, k) == (True, "ok")
    rng = np.random.default_rng(np.random.SeedSequence([1201, m, n, k, tile_n]))
    a_int = rng.integers(-2, 3, (m, k)).astype(np.float32)
    bt_int = rng.integers(-2, 3, (n, k)).astype(np.float32)
    exact = (a_int.astype(np.float64) @ bt_int.astype(np.float64).T).astype(np.float32)
    if ab == "bf16":
        a, bt = f32_to_bf16(a_int), f32_to_bf16(bt_int)
    else:
        a, bt = a_int.astype(np.float16), bt_int.astype(np.float16)
    # C is the first m rows of a taller NaN buffer: the kernel's bound is
    # m * n, so the guard rows must come back untouched.
    if c == "fp32":
        expected = exact
        c_init = np.full((m + _GUARD, n), np.nan, dtype=np.float32)
        decode = lambda raw: raw
    elif c == "bf16":
        expected = bf16_to_f32(f32_to_bf16(exact))
        c_init = f32_to_bf16(np.full((m + _GUARD, n), np.nan, dtype=np.float32))
        decode = bf16_to_f32
    else:
        expected = exact.astype(np.float16).astype(np.float32)
        c_init = np.full((m + _GUARD, n), np.nan, dtype=np.float16)
        decode = lambda raw: raw.astype(np.float32)
    for engine in available_engines():
        llvm_ir = lower_universal_gemm(spec, engine, arch=arch)
        got = decode(launch_universal_gemm(llvm_ir, spec, a, bt, arch=arch, c=c_init))
        np.testing.assert_array_equal(got[:m], expected, err_msg=engine)
        assert np.isnan(got[m:]).all(), engine


@arch_specific(("gfx1201",), reason="direct_a is a gfx1201 pipeline")
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("traits", sorted(_TRAITS))
@pytest.mark.parametrize("tile", _TILES, ids=lambda t: "x".join(map(str, t)))
def test_gfx1201_direct_a_matches_exact_reference(tile, traits, shape):
    _check(tile, _TRAITS[traits], shape, "fp16", "fp16")


@arch_specific(("gfx1201",), reason="direct_a is a gfx1201 pipeline")
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(
    "ab, c", [("bf16", "bf16"), ("fp16", "fp32"), ("bf16", "fp32")], ids=lambda v: v
)
@pytest.mark.parametrize("tile", _TILES[:3], ids=lambda t: "x".join(map(str, t)))
def test_gfx1201_direct_a_bf16_and_fp32_output(tile, ab, c, shape):
    _check(tile, _TRAITS["palette"], shape, ab, c)
