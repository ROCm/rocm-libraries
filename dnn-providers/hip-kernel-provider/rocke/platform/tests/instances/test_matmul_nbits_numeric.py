# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MatMulNBits computes ``C = A @ (W * scales)^T`` on the GPU, for every family.

``build_matmul_nbits`` dispatches on ``spec.family``: ``large_n`` and
``skinny_n`` share the WMMA tiled body (``optimized=True`` selects the body
that applies the group scale to the accumulator), ``decode_gemv`` is the scalar
one-thread-per-column body. The weights ``W`` are signed int4 ``[N, K]``, packed
two per byte with the low nibble holding the even ``k``
(``pack_i4_weights_for_matmul_nbits``), and each group of 32 consecutive ``k``
of a row has its own scale.

The reference is written here from the logical weights (it does not use the
module's dequant helpers). Weight rows 0 and 1 hold all 16 int4 values in both
slots, so a nibble order, sign-extension or group-index error changes the
result. A is small integers and every scale a power of two, so every product
and partial sum is representable in fp32 and IEEE fp32 summation in any order
is exact. gfx1151's WMMA accumulation still rounded (outputs whose exact value
is 0 came back as small negative fp16 subnormals there; gfx1201 matched bit for
bit), so each output may differ from the exact value by one fp16 spacing plus
``2**-20`` of the sum of its terms' magnitudes, far less than the change a
wrong nibble, sign or group scale makes to the outputs of weight rows 0 and 1.
C starts as NaN, so an unwritten output fails. The WMMA bodies need M to be a
multiple of ``tile_m`` (in-bounds A loads); the decode body guards M and N, so
it runs a ragged N.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytest
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.gemm_universal import TileSpec
from rocke.instances.common.matmul_nbits import (
    MatMulNBitsSpec,
    build_matmul_nbits,
    is_valid_spec,
    matmul_nbits_grid,
    matmul_nbits_signature,
    pack_i4_weights_for_matmul_nbits,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

# The arches this test has run green on. MatMulNBits targets gfx1151 and
# gfx1201 only; enable each once the test has run there.
_RUN_ARCHES = ("gfx1151", "gfx1201")


_WMMA_TILE = {"warp_k": 1, "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 16}


@dataclass(frozen=True)
class _Case:
    family: str
    tile: TileSpec
    M: int
    N: int
    K: int
    scale_dtype: str = "fp16"
    optimized: bool = False


_CASES = {
    "large_n": _Case(
        "large_n",
        TileSpec(tile_m=64, tile_n=128, tile_k=16, warp_m=2, warp_n=2, **_WMMA_TILE),
        M=128,
        N=256,
        K=128,
    ),
    "large_n_f32_scales": _Case(
        "large_n",
        TileSpec(tile_m=64, tile_n=128, tile_k=16, warp_m=2, warp_n=2, **_WMMA_TILE),
        M=64,
        N=128,
        K=96,
        scale_dtype="fp32",
    ),
    # Scale on the accumulator: tile_k equals the group size.
    "large_n_optimized": _Case(
        "large_n",
        TileSpec(tile_m=64, tile_n=128, tile_k=32, warp_m=2, warp_n=2, **_WMMA_TILE),
        M=128,
        N=256,
        K=128,
        optimized=True,
    ),
    "skinny_n": _Case(
        "skinny_n",
        TileSpec(tile_m=64, tile_n=32, tile_k=16, warp_m=2, warp_n=1, **_WMMA_TILE),
        M=64,
        N=32,
        K=128,
    ),
    # One thread per output column; N is not a multiple of the 256-thread block,
    # and M > 1 runs the body's outer row loop.
    "decode_gemv": _Case(
        "decode_gemv",
        TileSpec(tile_m=1, tile_n=256, tile_k=16, warp_m=1, warp_n=8, **_WMMA_TILE),
        M=3,
        N=300,
        K=128,
    ),
}


@pytest.fixture(scope="module")
def gpu_arch():
    return current_device_arch_in(_RUN_ARCHES, subject="the MatMulNBits numeric test")


def _spec(case: _Case) -> MatMulNBitsSpec:
    return MatMulNBitsSpec(
        name="matmul_nbits_numeric",
        N=case.N,
        K=case.K,
        tile=case.tile,
        scale_dtype=case.scale_dtype,
        family=case.family,
        optimized=case.optimized,
    )


def test_the_optimized_body_is_refused_off_gfx1201():
    # Its LDS-staged A and LDS epilogue follow gfx1201's WMMA layout; on
    # gfx1151 it computed the wrong tile.
    spec = _spec(_CASES["large_n_optimized"])
    reason = "optimized=True is built for gfx1201 only; got gfx1151"
    assert is_valid_spec(spec, "gfx1151") == (False, reason)
    with pytest.raises(ValueError, match=re.escape(reason)):
        build_matmul_nbits(spec, arch="gfx1151")


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matmul_nbits_matches_the_exact_product_on_gpu(gpu_arch, name):
    case = _CASES[name]
    if case.optimized and gpu_arch != "gfx1201":
        pytest.skip(f"the optimized body is refused on {gpu_arch}")
    spec = _spec(case)
    rng = np.random.default_rng(0x4B)
    a = rng.integers(-2, 3, (case.M, case.K)).astype(np.float16)
    weights = rng.integers(-8, 8, (case.N, case.K))
    # Every int4 value in both nibble slots: row 0 puts the even values at even
    # k (low nibble), row 1 shifted by one puts them at odd k (high nibble).
    weights[0, :16] = np.arange(-8, 8)
    weights[1, :16] = np.roll(np.arange(-8, 8), 1)
    groups = case.K // spec.group_size
    scale_type = np.float16 if case.scale_dtype == "fp16" else np.float32
    scales = (2.0 ** rng.integers(-2, 2, (case.N, groups))).astype(scale_type)

    buffers = {
        "A": a,
        "B": pack_i4_weights_for_matmul_nbits(weights, spec),
        "Scales": scales,
        "C": np.full((case.M, case.N), np.nan, dtype=np.float16),
    }
    launch_with_buffers(
        compile_kernel(build_matmul_nbits(spec, arch=gpu_arch), arch=gpu_arch),
        matmul_nbits_signature(spec),
        buffers,
        grid=matmul_nbits_grid(case.M, spec),
        block=(spec.block_size, 1, 1),
        scalars={"M": case.M},
    )

    scaled = weights * np.repeat(scales.astype(np.float64), spec.group_size, axis=1)
    exact = a.astype(np.float64) @ scaled.T
    magnitude = np.abs(a.astype(np.float64)) @ np.abs(scaled).T
    got = buffers["C"].astype(np.float64)
    spacing = np.spacing(np.abs(exact).astype(np.float16)).astype(np.float64)
    allowed = spacing + 2.0**-20 * magnitude
    wrong = np.argwhere(~(np.abs(got - exact) <= allowed))
    assert wrong.size == 0, (
        f"{len(wrong)} of {got.size} outputs wrong, first at (m, n) = "
        f"{tuple(int(i) for i in wrong[0])}: got {got[tuple(wrong[0])]}, "
        f"want {exact[tuple(wrong[0])]} within {allowed[tuple(wrong[0])]:.3g}"
    )
