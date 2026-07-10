# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import ml_dtypes
import pytest
import hipblaslt

c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


def test_bf16_in_map():
    assert hipblaslt._DTYPE_TO_NP[c.DataType.R_16BF] is ml_dtypes.bfloat16


@requires_gpu
def test_fp8_e4m3_roundtrip():
    a = np.arange(8).astype(ml_dtypes.float8_e4m3fn)
    da = hipblaslt.from_numpy(a, c.DataType.R_8F_E4M3)
    back = da.to_numpy()
    np.testing.assert_array_equal(back.astype(np.float32), a.astype(np.float32))
