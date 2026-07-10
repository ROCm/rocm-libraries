# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_all_algos_agree():
    m = n = k = 32
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    ref = A @ B
    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        dA = c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), c.DataType.R_32F)
        dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), c.DataType.R_32F)
        dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 32)
        assert results
        for r in results:
            ws = c.DeviceArray.from_numpy(np.zeros(max(1, r.workspace_size), np.uint8), c.DataType.R_8I)
            c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, r.algo, ws)
            out = dD.to_numpy().reshape(n, m).T
            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3,
                                       err_msg=f"algo #{r.algo.index} disagrees")
