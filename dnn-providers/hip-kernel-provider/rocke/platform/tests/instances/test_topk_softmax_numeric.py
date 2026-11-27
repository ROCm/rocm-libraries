# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The top-K + softmax instance picks the right entries and weights on the GPU.

For each row of ``X`` (``M x N``), ``build_topk_softmax`` writes the indices of
the ``K`` largest entries, largest first, to ``Idx`` and the softmax over those
``K`` values to ``Y``. Equal values are taken in index order: the smaller index
wins a tie (the documented argmax tie-break, which keeps the ``K`` picks
deterministic for inputs with duplicates).

The reference is numpy on the values the kernel reads: a stable sort by
descending value gives the picks, and the softmax runs in float64. Indices must
match exactly; weights within the rounding of the output dtype. One test runs
rows of distinct values (exact in every input dtype, so rounding makes no
ties); another runs rows made of ties; a third runs the values whose order is
easiest to get wrong: signed zeros (equal, so a tie), the input dtype's
smallest subnormals, NaN (never picked), the largest finite values, and ties
at the last indices of a long row. The configurations cover a row length
that is not a multiple of the block (masked tail lanes), one- and multi-wave
blocks, and each input and output dtype.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.core.dtypes import normalize_dtype
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.topk_softmax import (
    TopkSoftmaxSpec,
    build_topk_softmax,
    is_valid_spec,
    topk_softmax_grid,
    topk_softmax_signature,
)
from rocke.numeric import storage
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

pytestmark = pytest.mark.gpu

# The arches this test has run green on; add one once it has.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

_ROWS = 37


def _encode(values: np.ndarray, dtype: str) -> np.ndarray:
    """Host storage of ``values`` in the spec dtype (``f32``, ``f16``, ``bf16``)."""
    return storage.encode(values, normalize_dtype(dtype))


def _decode(stored: np.ndarray, dtype: str) -> np.ndarray:
    return storage.decode(stored, normalize_dtype(dtype)).astype(np.float64)


def _reference(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(indices, softmax weights) of each row's ``k`` largest entries; a tie
    goes to the smaller index."""
    order = np.argsort(-x, axis=1, kind="stable")[:, :k]
    picked = np.take_along_axis(x, order, axis=1)
    weights = np.exp(picked - picked[:, :1])
    return order.astype(np.int32), weights / weights.sum(axis=1, keepdims=True)


def _distinct_rows(spec: TopkSoftmaxSpec, rng: np.random.Generator) -> np.ndarray:
    """Rows of distinct values: a shuffle of ``N`` multiples of 1/8 centred on
    0. ``N <= 256``, so each value has at most 8 significant bits and is exact
    in bf16, f16 and f32."""
    n = spec.n_per_row
    assert n <= 256, "more values than bf16 holds exactly in this pattern"
    return np.stack([(rng.permutation(n) - n // 2) / 8.0 for _ in range(_ROWS)])


def _tie_rows(spec: TopkSoftmaxSpec) -> np.ndarray:
    """Rows made of ties: every entry equal; a four-value pattern repeated
    along the row; and every third entry equal to the row's maximum."""
    n = spec.n_per_row
    rows = np.zeros((3, n))
    rows[0] = 1.5
    rows[1] = np.resize([2.0, -1.0, 2.0, 0.5], n)
    rows[2] = -np.arange(n) / 8.0
    rows[2, ::3] = 7.25
    return rows


# Per input dtype: the smallest positive subnormal, the largest finite value,
# and the finite value just below it.
_SMALLEST_SUBNORMAL = {"f32": 2.0**-149, "f16": 2.0**-24, "bf16": 2.0**-133}
_LARGEST_FINITE = {
    "f32": float(np.finfo(np.float32).max),
    "f16": 65504.0,
    "bf16": (2.0 - 2.0**-7) * 2.0**127,
}
_BELOW_LARGEST_FINITE = {
    "f32": float(np.nextafter(np.float32(_LARGEST_FINITE["f32"]), np.float32(0))),
    "f16": 65472.0,
    "bf16": (2.0 - 2.0**-6) * 2.0**127,
}


def _edge_rows(spec: TopkSoftmaxSpec) -> np.ndarray:
    """Rows of the values whose order is easiest to get wrong.

    Row 0: signed zeros, which tie, at indices on both sides of a block
    boundary and at the last index; the dtype's smallest subnormals on either
    side of zero; NaN, which is never picked. Row 1: the largest finite value
    twice (a tie) and the value just below it twice. Row 2: a tie of the
    largest value at the last indices of the row. The other entries are equal
    and smaller, so a ``k`` past the special entries picks them in index
    order."""
    n = spec.n_per_row
    assert n >= 9, "the edge rows need nine distinct positions"
    tiny = _SMALLEST_SUBNORMAL[spec.dtype]
    rows = np.full((3, n), -2.0)
    rows[0, [0, 5]] = np.nan
    rows[0, 1] = -tiny
    rows[0, [3, n - 1]] = -0.0
    rows[0, [7, n // 2]] = 0.0
    rows[0, n - 2] = tiny
    rows[1, [2, n // 3]] = _LARGEST_FINITE[spec.dtype]
    rows[1, [0, n - 1]] = _BELOW_LARGEST_FINITE[spec.dtype]
    rows[2, [n - 5, n - 3, n - 2, n - 1]] = 3.0
    return rows


_SPECS = [
    TopkSoftmaxSpec(n_per_row=64, k=4, dtype="f32", out_dtype="f32"),
    # A row longer than the block, not a multiple of it: masked tail lanes.
    TopkSoftmaxSpec(n_per_row=37, k=8, dtype="f16", out_dtype="f16", block_size=32),
    TopkSoftmaxSpec(n_per_row=200, k=6, dtype="bf16", out_dtype="f32"),
    # Two and four waves per block: the LDS reduction path.
    TopkSoftmaxSpec(
        n_per_row=256,
        k=32,
        dtype="f32",
        out_dtype="bf16",
        block_size=128,
    ),
    TopkSoftmaxSpec(n_per_row=128, k=1, dtype="f16", out_dtype="f32", block_size=256),
]


# A 4096-entry row (16 block widths): indices up to 4095 in a 256-lane block.
_LONG_ROW_SPEC = TopkSoftmaxSpec(
    n_per_row=4096, k=8, dtype="f32", out_dtype="f32", block_size=256
)


def _spec_id(spec: TopkSoftmaxSpec) -> str:
    return spec.kernel_name()


# Output dtype -> absolute tolerance on a softmax weight in [0, 1]: half an ulp
# at 1 of the output dtype, plus f32 exp error.
_WEIGHT_ATOL = {"f32": 1e-6, "f16": 2.0**-11 + 1e-6, "bf16": 2.0**-8 + 1e-6}


def _run_and_check(spec: TopkSoftmaxSpec, x: np.ndarray) -> None:
    """Run ``spec`` on the rows of ``x`` (values exact in the input dtype) and
    compare its picks and weights with the reference."""
    arch = current_device_arch_in(_RUN_ARCHES, subject="topk_softmax")
    ok, why = is_valid_spec(spec, arch)
    assert ok, why
    rows = x.shape[0]
    buffers = {
        "X": _encode(x, spec.dtype),
        "Y": _encode(np.full((rows, spec.k), np.nan), spec.out_dtype),
        "Idx": np.full((rows, spec.k), -1, dtype=np.int32),
    }
    stored = _decode(buffers["X"], spec.dtype)
    assert np.array_equal(stored, x, equal_nan=True), "inputs not exact"
    assert np.array_equal(np.signbit(stored), np.signbit(x)), "signs not kept"
    launch_with_buffers(
        compile_kernel(build_topk_softmax(spec, arch), arch=arch),
        topk_softmax_signature(spec),
        buffers,
        grid=topk_softmax_grid(rows, spec),
        block=(spec.block_size, 1, 1),
        scalars={"M": rows, "N": spec.n_per_row},
    )
    want_idx, want_weights = _reference(x, spec.k)
    wrong_rows = np.flatnonzero((buffers["Idx"] != want_idx).any(axis=1))
    assert wrong_rows.size == 0, (
        f"{wrong_rows.size} rows picked the wrong entries; row {wrong_rows[0]}: "
        f"got {buffers['Idx'][wrong_rows[0]]}, want {want_idx[wrong_rows[0]]}"
    )
    got_weights = _decode(buffers["Y"], spec.out_dtype)
    np.testing.assert_allclose(
        got_weights, want_weights, rtol=0, atol=_WEIGHT_ATOL[spec.out_dtype]
    )


@pytest.mark.parametrize("spec", _SPECS, ids=_spec_id)
def test_topk_softmax_picks_the_largest_distinct_values_on_gpu(spec):
    rng = np.random.default_rng(spec.n_per_row * 31 + spec.k)
    _run_and_check(spec, _distinct_rows(spec, rng))


@pytest.mark.parametrize("spec", _SPECS, ids=_spec_id)
def test_topk_softmax_breaks_ties_toward_the_smaller_index_on_gpu(spec):
    _run_and_check(spec, _tie_rows(spec))


@pytest.mark.parametrize("spec", [*_SPECS, _LONG_ROW_SPEC], ids=_spec_id)
def test_topk_softmax_ranks_signed_zeros_subnormals_nan_and_extremes_on_gpu(spec):
    _run_and_check(spec, _edge_rows(spec))
