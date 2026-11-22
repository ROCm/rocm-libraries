# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The 2D pooling instance computes max / avg / sum pooling on the GPU.

``build_pooling2d`` reduces, for each NHWC output ``(n, ho, wo, c)``, the input
cells ``hi = ho*sH - pH + y*dH``, ``wi = wo*sW - pW + x*dW`` of its ``Y x X``
window that lie inside the image: pad cells never win a max, add nothing to a
sum and are left out of an average's count. Each case runs the kernel on random
NHWC input and compares it with a float64 NumPy pooling written here from that
definition. The fp16 cases cover stride, padding, dilation, windows that are
mostly padding, every ``vec`` width and a grid whose last block is only partly
used. The max cases use only negative inputs, so a pad cell read as 0 would win
and fail the check. bf16 has no output store yet, so it is refused up front
(checked on the host).

The output starts as all-ones bytes (NaN in fp16 and bf16), so an output the
kernel does not write fails too.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from rocke.core.dtypes import normalize_dtype
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.pooling import (
    Pooling2DSpec,
    PoolingProblem,
    build_pooling2d,
    is_valid_spec,
    pooling2d_grid,
    pooling2d_signature,
)
from rocke.numeric import storage
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

# The arches this test has run green on; add an arch once it has.
_RUN_ARCHES = ("gfx90a", "gfx950")

# One f32-accumulated reduction then one rounding to the I/O dtype: within one
# unit in the last place of the result (relative 2**-10 for fp16, 2**-7 for
# bf16), plus a floor for results near zero.
_RELATIVE_TOLERANCE = {"f16": 2.0**-10, "bf16": 2.0**-7}
_ABSOLUTE_TOLERANCE = 1e-6

# id -> (op, dtype, vec, problem)
_CASES = {
    "max_f16_v1_stride2_pad1": (
        "max",
        "f16",
        1,
        PoolingProblem(N=2, H=9, W=11, C=6, Y=3, X=3, sH=2, sW=2, pH=1, pW=1),
    ),
    "avg_f16_v4_dilation2_pad": (
        "avg",
        "f16",
        4,
        PoolingProblem(N=1, H=10, W=7, C=12, Y=3, X=2, sH=1, sW=2, pH=1, pW=1, dH=2),
    ),
    "sum_f16_v8_stride2": (
        "sum",
        "f16",
        8,
        PoolingProblem(N=1, H=5, W=5, C=16, Y=2, X=2, sH=2, sW=2),
    ),
    "max_f16_v2_pad2": (
        "max",
        "f16",
        2,
        PoolingProblem(N=3, H=6, W=6, C=4, Y=5, X=5, pH=2, pW=2),
    ),
    # 3x3 windows with pad 2 at stride 1: the corner windows hold one image
    # cell and eight pad cells, so the count must be 1 there.
    "avg_f16_v1_mostly_pad_corners": (
        "avg",
        "f16",
        1,
        PoolingProblem(N=1, H=4, W=5, C=3, Y=3, X=3, pH=2, pW=2),
    ),
}


def _reference(x: np.ndarray, problem: PoolingProblem, op: str) -> np.ndarray:
    """float64 NHWC pooling over the in-image cells of each window."""
    p = problem
    reduce = {"max": np.max, "sum": np.sum, "avg": np.mean}[op]
    out = np.empty((p.N, p.Ho, p.Wo, p.C))
    for ho in range(p.Ho):
        rows = [ho * p.sH - p.pH + y * p.dH for y in range(p.Y)]
        for wo in range(p.Wo):
            cols = [wo * p.sW - p.pW + col * p.dW for col in range(p.X)]
            cells = [
                x[:, hi, wi, :]
                for hi in rows
                if 0 <= hi < p.H
                for wi in cols
                if 0 <= wi < p.W
            ]
            out[:, ho, wo, :] = reduce(np.stack(cells), axis=0)
    return out


# is_valid_spec admits bf16, but build_pooling2d cannot emit a bf16 output store
# (store_tile: "buffer ... store not yet wired for dtype bf16"). Strict, so the
# cases fail loudly once the store is wired and must then be un-marked.
@pytest.mark.parametrize("case", sorted(_CASES))
def test_pooling_matches_the_window_reduction_on_gpu(case):
    op, dtype, vec, problem = _CASES[case]
    arch = current_device_arch_in(_RUN_ARCHES, subject="pooling2d")
    spec = Pooling2DSpec(problem=problem, dtype=dtype, op=op, vec=vec, block_size=64)
    # The last block is partly used, so out-of-range threads must not store.
    assert problem.total_out // vec % spec.block_size, "the grid has no partial block"

    values = np.random.default_rng(3).standard_normal(
        (problem.N, problem.H, problem.W, problem.C)
    )
    if op == "max":
        values = -np.abs(values) - 0.25
    x = storage.encode(values, normalize_dtype(dtype))
    y = np.empty((problem.N, problem.Ho, problem.Wo, problem.C), dtype=x.dtype)
    y.view(np.uint8)[...] = 0xFF
    launch_with_buffers(
        compile_kernel(build_pooling2d(spec, arch=arch), arch=arch),
        pooling2d_signature(spec),
        {"X": x, "Y": y},
        grid=pooling2d_grid(spec),
        block=(spec.block_size, 1, 1),
        scalars={"X_bytes": x.nbytes, "Y_bytes": y.nbytes},
    )

    got = storage.decode(y, normalize_dtype(dtype)).astype(np.float64)
    want = _reference(
        storage.decode(x, normalize_dtype(dtype)).astype(np.float64), problem, op
    )
    assert np.isfinite(got).all(), f"{case}: unwritten or non-finite output"
    if op == "max":
        np.testing.assert_array_equal(got, want, err_msg=case)
    else:
        bound = _RELATIVE_TOLERANCE[dtype] * np.abs(want) + _ABSOLUTE_TOLERANCE
        worst = np.abs(got - want) - bound
        assert (
            worst <= 0
        ).all(), (
            f"{case}: {(worst > 0).sum()} outputs off, worst excess {worst.max():.3g}"
        )


def test_bf16_is_refused_before_the_build():
    # store_tile has a buffer store for f16 only, so is_valid_spec refuses bf16
    # rather than letting the builder fail on the store.
    problem = PoolingProblem(N=1, H=4, W=4, C=4, Y=2, X=2, sH=2, sW=2)
    spec = Pooling2DSpec(problem=problem, dtype="bf16", op="max", vec=1, block_size=64)
    reason = "bf16 output is not supported (the store is f16 only); use f16"
    assert is_valid_spec(spec, "gfx942") == (False, reason)
    with pytest.raises(
        ValueError, match=re.escape(f"invalid pooling2d spec: {reason}")
    ):
        build_pooling2d(spec, arch="gfx942")
