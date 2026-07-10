# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core

# Validation fires before any HIP call — no GPU needed for these tests.


def test_dtype_mismatch_raises():
    a = np.arange(4, dtype=np.float64)  # f64 host
    with pytest.raises(ValueError):
        hipblaslt.from_numpy(a, c.DataType.R_32F)  # asked for f32


def test_non_contiguous_raises():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]  # non-contiguous
    with pytest.raises(ValueError):
        hipblaslt.from_numpy(a, c.DataType.R_32F)
