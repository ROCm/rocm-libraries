# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""FlatMM computes the batched product ``C[b] = A[b] @ B[b]^T`` on the GPU.

``build_flatmm`` builds the batched universal GEMM (RCR: A is ``[M, K]``
row-major, B is ``[N, K]`` row-major, C is ``[M, N]`` row-major) under the
FlatMM kernel name; its launch is ``flatmm_grid`` / ``flatmm_signature`` with
per-batch element strides ``stride_a`` / ``stride_b`` / ``stride_c``. The
parity emitters prove the two engines emit the same IR; this file proves the
kernel's numbers.

The inputs are small integers in ``[-3, 3]``, so every product and every
partial sum is an integer that fp16 holds exactly (at most ``9 * K = 1152``),
whatever the MFMA accumulation order: the output must equal the numpy product
exactly. The batch strides leave a gap after each batch, and C starts as NaN
everywhere, so an element written outside its ``[M, N]`` block or one left
unwritten fails the check. The padded configuration runs M, N and K that are
not multiples of the tile.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.flatmm import (
    FlatMMSpec,
    TraitSpec,
    build_flatmm,
    flatmm_config16,
    flatmm_config32,
    flatmm_grid,
    flatmm_signature,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers
from rocke.tests.lanes import skip_needing_arch

pytestmark = pytest.mark.gpu

# The arches this test has run green on; add one once it has.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

# The canonical FlatMM tiles use the K-packed f16 atoms that only gfx950 has
# (flatmm_config32 / flatmm_config16); on the other arches the same 128x128x64
# tile takes the narrow 32x32x8 atom (build_flatmm's docstring).
_TILES = {
    "config32": lambda arch: (
        flatmm_config32("f16")
        if arch == "gfx950"
        else replace(flatmm_config32("f16"), warp_tile_k=8)
    ),
    "config16": lambda arch: flatmm_config16("f16") if arch == "gfx950" else None,
}

# (batch, M, N, K, padded): full tiles unpadded; partial M, N and K tiles with
# pad_m / pad_n / pad_k.
_SHAPES = {
    "full_tiles": (3, 256, 256, 128, False),
    "partial_tiles": (2, 200, 136, 96, True),
}

# Elements left between consecutive batches of each operand.
_BATCH_GAP = 64


@pytest.fixture(scope="module")
def gpu_arch():
    return current_device_arch_in(_RUN_ARCHES, subject="the FlatMM numeric test")


@pytest.mark.parametrize("shape", sorted(_SHAPES))
@pytest.mark.parametrize("tile_name", sorted(_TILES))
def test_flatmm_matches_the_exact_batched_product_on_gpu(gpu_arch, tile_name, shape):
    tile = _TILES[tile_name](gpu_arch)
    if tile is None:
        skip_needing_arch(
            ("gfx950",),
            reason=f"{tile_name} uses an atom {gpu_arch} does not have (gfx950 only)",
        )
    batch, m, n, k, padded = _SHAPES[shape]
    spec = FlatMMSpec(
        tile=tile,
        trait=TraitSpec(epilogue="cshuffle", pad_m=padded, pad_n=padded, pad_k=padded),
    )
    stride_a, stride_b, stride_c = (
        rows * cols + _BATCH_GAP for rows, cols in ((m, k), (n, k), (m, n))
    )
    rng = np.random.default_rng(18)
    a_values = rng.integers(-3, 4, (batch, m, k)).astype(np.float16)
    b_values = rng.integers(-3, 4, (batch, n, k)).astype(np.float16)

    def batched(values, stride, fill):
        flat = np.full(batch * stride, fill, dtype=np.float16)
        rows, cols = values.shape[1:]
        for index in range(batch):
            start = index * stride
            flat[start : start + rows * cols] = values[index].ravel()
        return flat

    buffers = {
        "A": batched(a_values, stride_a, 0),
        "B": batched(b_values, stride_b, 0),
        "C": np.full(batch * stride_c, np.nan, dtype=np.float16),
    }
    launch_with_buffers(
        compile_kernel(build_flatmm(spec, arch=gpu_arch), arch=gpu_arch),
        flatmm_signature(spec),
        buffers,
        grid=flatmm_grid(spec, batch, m, n),
        block=(spec.block_size, 1, 1),
        scalars={
            "M": m,
            "N": n,
            "K": k,
            "stride_a": stride_a,
            "stride_b": stride_b,
            "stride_c": stride_c,
        },
    )

    want = np.einsum(
        "bmk,bnk->bmn", a_values.astype(np.float64), b_values.astype(np.float64)
    ).astype(np.float16)
    got = buffers["C"]
    for index in range(batch):
        start = index * stride_c
        block = got[start : start + m * n].reshape(m, n)
        wrong = np.argwhere(block.view(np.uint16) != want[index].view(np.uint16))
        assert wrong.size == 0, (
            f"batch {index}: {len(wrong)} of {m * n} outputs wrong, first at "
            f"(m, n) = {tuple(int(i) for i in wrong[0])}: got {block[tuple(wrong[0])]}, "
            f"want {want[index][tuple(wrong[0])]}"
        )
        gap = got[start + m * n : start + stride_c]
        assert np.isnan(gap).all(), f"batch {index}: wrote past its [M, N] block"
