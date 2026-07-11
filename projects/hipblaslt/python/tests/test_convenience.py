# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu


@requires_gpu
def test_gemm_f32():
    a = np.random.rand(48, 32).astype(np.float32)
    b = np.random.rand(32, 16).astype(np.float32)
    out = hipblaslt.gemm(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-3)
