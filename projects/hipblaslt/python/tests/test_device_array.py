# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu


@requires_gpu
def test_roundtrip_f32():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    assert da.ptr != 0
    assert da.nbytes == a.nbytes
    back = da.to_numpy()
    np.testing.assert_array_equal(back, a)


@requires_gpu
def test_copy_from_host_reuse():
    a = np.zeros((2, 2), dtype=np.float32)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    b = np.ones((2, 2), dtype=np.float32)
    da.copy_from_host(b)
    np.testing.assert_array_equal(da.to_numpy(), b)
