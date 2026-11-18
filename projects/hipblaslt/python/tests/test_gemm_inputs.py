# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gemm() refuses what it can't do, before it needs a GPU.

The prototype's gemm() looked types up in a table that also held int8, int32 and the fp8 types,
so it accepted those arrays and ran them with the wrong compute type.
"""

import ml_dtypes
import numpy as np
import pytest

import hipblaslt


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int32, ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2fnuz],
    ids=lambda d: np.dtype(d).name,
)
def test_unsupported_types_are_refused(dtype):
    a = np.zeros((4, 4), dtype)
    with pytest.raises(ValueError, match=f"{np.dtype(dtype).name} isn't supported"):
        hipblaslt.gemm(a, a)


def test_mixed_types_are_refused():
    with pytest.raises(ValueError, match="must match"):
        hipblaslt.gemm(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float16))


@pytest.mark.parametrize(
    "shapes", [((2, 3), (2, 3)), ((2, 3, 1), (3, 2)), ((3,), (3,))]
)
def test_shapes_must_multiply(shapes):
    with pytest.raises(ValueError, match="can't be multiplied"):
        hipblaslt.gemm(np.zeros(shapes[0], np.float32), np.zeros(shapes[1], np.float32))
