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
a K-tile count the stagger span does not divide, a single partial tile
smaller than the tile itself, and K that ends in a partial tile, including K
shorter than one tile. Row strides are tested with NaN in the padding of A and
B, so a read that strays into it shows in C.
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

# (tile_m, tile_n, tile_k, warp_m): with and without B read-ahead, four, two
# and one waves, the K-128 tiles, and a single K atom per tile.
_TILES = [
    (128, 64, 64, 4),
    (128, 96, 64, 4),
    (64, 64, 64, 4),
    (32, 32, 128, 2),
    (16, 16, 128, 1),
    (64, 32, 32, 2),
    (64, 32, 16, 2),
]
_TRAITS = {
    "plain": dict(),
    "swizzle": dict(lds_swizzle=True),
    "palette": dict(lds_swizzle=True, group_m=4, k_stagger=8, k_stagger_step=1),
    "wrap": dict(lds_swizzle=True, group_m=8, k_stagger=3, k_stagger_step=2),
}
# (M, N, whole K tiles, K atoms after them)
_SHAPES = [
    (300, 200, 1, 0),
    (128, 192, 2, 0),
    (1000, 1000, 7, 0),
    (257, 97, 5, 0),
    (40, 24, 3, 0),
    (640, 385, 16, 0),
    (300, 200, 0, 1),
    (257, 97, 1, 1),
    (640, 385, 8, 3),
    (1000, 1000, 5, 2),
]
# Padding elements after the K elements of each A and B row.
_STRIDES = [(32, 32), (16, 0), (0, 48)]
_GUARD = 64


def _padded(values, pad):
    """``values`` with ``pad`` NaN elements after each row."""
    if not pad:
        return values
    rows, k = values.shape
    out = np.full((rows, k + pad), np.nan, dtype=np.float32)
    out[:, :k] = values
    return out


def _check(tile, traits, shape, ab, c, strides=None):
    arch = current_device_arch_in(("gfx1201",))
    tile_m, tile_n, tile_k, warp_m = tile
    spec = UniversalGemmSpec(
        name="gfx1201_direct_a",
        tile=TileSpec(tile_m, tile_n, tile_k, warp_m, 1, 1, 16, 16, 16),
        trait=TraitSpec(
            pipeline="direct_a",
            scheduler="intrawave",
            epilogue="default",
            row_strides=strides is not None,
            **traits,
        ),
        data=DataSpec(ab, ab, c),
        wave_size=32,
    )
    assert is_valid_spec(spec, arch) == (True, "ok")
    m, n, k_tiles, k_atoms = shape
    k = k_tiles * tile_k + k_atoms * 16
    pad_a, pad_b = strides or (0, 0)
    assert direct_a_launch_supported(spec, m, n, k, k + pad_a, k + pad_b) == (
        True,
        "ok",
    )
    rng = np.random.default_rng(np.random.SeedSequence([1201, m, n, k, tile_n]))
    a_int = rng.integers(-2, 3, (m, k)).astype(np.float32)
    bt_int = rng.integers(-2, 3, (n, k)).astype(np.float32)
    exact = (a_int.astype(np.float64) @ bt_int.astype(np.float64).T).astype(np.float32)
    a_rows, bt_rows = _padded(a_int, pad_a), _padded(bt_int, pad_b)
    if ab == "bf16":
        a, bt = f32_to_bf16(a_rows), f32_to_bf16(bt_rows)
    else:
        a, bt = a_rows.astype(np.float16), bt_rows.astype(np.float16)
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
        got = decode(
            launch_universal_gemm(llvm_ir, spec, a, bt, arch=arch, c=c_init, k=k)
        )
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


@arch_specific(("gfx1201",), reason="direct_a is a gfx1201 pipeline")
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("strides", _STRIDES, ids=lambda s: "pad%dx%d" % s)
@pytest.mark.parametrize(
    "tile", _TILES[:5:2] + _TILES[1:2], ids=lambda t: "x".join(map(str, t))
)
def test_gfx1201_direct_a_row_strides(tile, strides, shape):
    _check(tile, _TRAITS["palette"], shape, "fp16", "fp16", strides)


@arch_specific(("gfx1201",), reason="direct_a is a gfx1201 pipeline")
@pytest.mark.parametrize("ab, c", [("bf16", "bf16"), ("fp16", "fp32")], ids=lambda v: v)
def test_gfx1201_direct_a_row_strides_bf16_and_fp32_output(ab, c):
    _check(_TILES[0], _TRAITS["palette"], (1000, 1000, 5, 2), ab, c, (32, 48))
