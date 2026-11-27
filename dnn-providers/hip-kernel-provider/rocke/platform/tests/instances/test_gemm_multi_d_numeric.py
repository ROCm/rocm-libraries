# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The multi-D and multi-A/B/D GEMM instances compute their fused epilogue on the GPU.

``build_gemm_multi_d`` computes ``E = f(A @ B.T, D0, D1, ...)`` for row-major
``A (M, K)``, ``B (N, K)`` (RCR) and row-major ``(M, N)`` side inputs, where
``f`` applies each D's op (``add`` or ``mul``) in the order the spec lists
them: ``((A @ B.T) op0 D0) op1 D1 ...``. ``build_gemm_multi_abd`` (one A, one
B today) delegates to it, or to the plain universal GEMM when it has no D.

Each case runs the kernel on small-integer inputs whose products, sums and
every intermediate fit exactly in the I/O dtype, so the result is exact
whatever precision the epilogue combines in, and compares it bit for bit with
a NumPy reference written here from that definition. The cases cover each D
load kind (``vector``, ``tiled``, ``stock``) on exact tiles, each op in either
order, one to three Ds, both dtypes, and padded kernels on M, N and K that are
not multiples of the tile.

The output starts as NaN, so an element the
kernel does not write fails the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytest
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.gemm_multi_abd import (
    GemmMultiAbdSpec,
    build_gemm_multi_abd,
    gemm_multi_abd_grid,
    gemm_multi_abd_signature,
)
from rocke.instances.common.gemm_multi_d import (
    DataSpec,
    GemmMultiDSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_gemm_multi_d,
    gemm_multi_d_grid,
    gemm_multi_d_signature,
    is_valid_spec,
)
from rocke.numeric import storage
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

# The arches this test has run green on; add an arch once it has.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

# Input bounds that keep every value exact: fp16 holds every integer up to 2048
# and bf16 up to 256. With |A|, |B| <= the bound, K <= 64 and |D| <= 4, every
# case stays inside that range (checked per case). An unpadded kernel needs M,
# N and K to be multiples of its tile, so those cases use exact tiles.
_AB_BOUND = {"fp16": 2, "bf16": 1}
_D_BOUND = 4


def _base(dtype: str, tile: TileSpec, *, padded: bool) -> UniversalGemmSpec:
    return UniversalGemmSpec(
        name="multi_d_numeric",
        tile=tile,
        trait=TraitSpec(epilogue="cshuffle", pad_m=padded, pad_n=padded, pad_k=padded),
        data=DataSpec(dtype_a=dtype, dtype_b=dtype, dtype_c=dtype),
    )


_TILE_64 = TileSpec(64, 64, 32, 2, 2, warp_tile_m=16, warp_tile_n=16, warp_tile_k=16)
_TILE_64_ATOM32 = TileSpec(
    64, 64, 32, 2, 2, warp_tile_m=32, warp_tile_n=32, warp_tile_k=8
)


@dataclass(frozen=True)
class _Case:
    d_operands: tuple[tuple[str, str], ...]
    dtype: str
    d_load_kind: str
    tile: TileSpec
    m: int
    n: int
    k: int
    padded: bool

    def multi_d_spec(self) -> GemmMultiDSpec:
        return GemmMultiDSpec(
            base=_base(self.dtype, self.tile, padded=self.padded),
            d_operands=self.d_operands,
            d_dtype=self.dtype,
            d_load_kind=self.d_load_kind,
        )

    def multi_abd_spec(self) -> GemmMultiAbdSpec:
        return GemmMultiAbdSpec(
            base=_base(self.dtype, self.tile, padded=self.padded),
            a_operands=(("A", self.dtype),),
            b_operands=(("B", self.dtype),),
            d_operands=self.d_operands,
            d_dtype=self.dtype,
            d_load_kind=self.d_load_kind,
        )


_MULTI_D_CASES = {
    "vector_add_mul_fp16": _Case(
        (("D0", "add"), ("D1", "mul")), "fp16", "vector", _TILE_64, 128, 192, 64, False
    ),
    "tiled_mul_add_add_fp16": _Case(
        (("D0", "mul"), ("D1", "add"), ("D2", "add")),
        "fp16",
        "tiled",
        _TILE_64_ATOM32,
        64,
        128,
        32,
        False,
    ),
    "stock_add_fp16": _Case(
        (("D0", "add"),), "fp16", "stock", _TILE_64, 128, 128, 64, False
    ),
    "vector_add_mul_bf16": _Case(
        (("D0", "add"), ("D1", "mul")),
        "bf16",
        "vector",
        _TILE_64_ATOM32,
        64,
        64,
        32,
        False,
    ),
    "padded_add_mul_fp16_partial_tiles": _Case(
        (("D0", "add"), ("D1", "mul")), "fp16", "vector", _TILE_64, 100, 72, 40, True
    ),
}

_MULTI_ABD_CASES = {
    "no_d_fp16_padded_partial_tiles": _Case(
        (), "fp16", "vector", _TILE_64, 77, 65, 48, True
    ),
    "tiled_mul_add_bf16": _Case(
        (("D0", "mul"), ("D1", "add")), "bf16", "tiled", _TILE_64, 64, 128, 64, False
    ),
}


def _run(case: _Case, kernel, signature, grid, block_size: int) -> None:
    arch = current_device_arch_in(_RUN_ARCHES, subject="multi-D GEMM")
    rng = np.random.default_rng(19)
    ab = _AB_BOUND[case.dtype]
    a = rng.integers(-ab, ab + 1, (case.m, case.k))
    b = rng.integers(-ab, ab + 1, (case.n, case.k))
    ds = {
        name: rng.integers(-_D_BOUND, _D_BOUND + 1, (case.m, case.n))
        for name, _ in case.d_operands
    }
    # The exactness the bit-for-bit check relies on (see _AB_BOUND): every
    # intermediate of the epilogue, not only the result, stays exact.
    limit = 2048 if case.dtype == "fp16" else 256
    want = (a @ b.T).astype(np.float64)
    assert np.abs(want).max() <= limit, "A @ B.T too large to stay exact"
    for name, op in case.d_operands:
        want = want + ds[name] if op == "add" else want * ds[name]
        assert np.abs(want).max() <= limit, f"{op} {name} too large to stay exact"

    c = storage.encode(np.full((case.m, case.n), np.nan), case.dtype)
    buffers = {
        "A": storage.encode(a, case.dtype),
        "B": storage.encode(b, case.dtype),
        "C": c,
        **{name: storage.encode(value, case.dtype) for name, value in ds.items()},
    }
    launch_with_buffers(
        compile_kernel(kernel(arch), arch=arch),
        signature,
        buffers,
        grid=grid,
        block=(block_size, 1, 1),
        scalars={"M": case.m, "N": case.n, "K": case.k},
    )
    got = storage.decode(buffers["C"], case.dtype).astype(np.float64)
    wrong = np.argwhere(got != want)
    assert not wrong.size, (
        f"{len(wrong)} of {got.size} outputs wrong, first at {tuple(wrong[0])}: "
        f"got {got[tuple(wrong[0])]}, want {want[tuple(wrong[0])]}"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(_MULTI_D_CASES))
def test_multi_d_gemm_applies_each_d_in_order_on_gpu(name):
    case = _MULTI_D_CASES[name]
    spec = case.multi_d_spec()
    _run(
        case,
        lambda arch: build_gemm_multi_d(spec, arch=arch),
        gemm_multi_d_signature(spec),
        gemm_multi_d_grid(spec, case.m, case.n),
        spec.base.block_size,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(_MULTI_ABD_CASES))
def test_multi_abd_gemm_with_one_a_and_one_b_on_gpu(name):
    case = _MULTI_ABD_CASES[name]
    spec = case.multi_abd_spec()
    _run(
        case,
        lambda arch: build_gemm_multi_abd(spec, arch=arch),
        gemm_multi_abd_signature(spec),
        gemm_multi_abd_grid(spec, case.m, case.n),
        spec.base.block_size,
    )


@pytest.mark.parametrize(("c_dtype", "d_dtype"), [("fp16", "bf16"), ("bf16", "fp16")])
def test_a_d_dtype_other_than_the_c_dtype_is_refused(c_dtype, d_dtype):
    # The epilogue combines each D with the C tile in C's element type.
    spec = GemmMultiDSpec(
        base=_base(c_dtype, _TILE_64, padded=False),
        d_operands=(("D0", "add"),),
        d_dtype=d_dtype,
    )
    reason = (
        f"d_dtype {d_dtype!r} must be the C dtype {c_dtype!r}: "
        "each D combines with C in C's type"
    )
    assert is_valid_spec(spec, "gfx942") == (False, reason)
    with pytest.raises(ValueError, match=re.escape(reason)):
        build_gemm_multi_d(spec, arch="gfx942")
