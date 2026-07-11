# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu


@requires_gpu
def test_matmul_f32_matches_numpy():
    m = n = k = 64
    # hipBLASLt uses column-major (Fortran) layout by default.
    # For a matrix X with shape (rows, cols) in C/numpy row-major order,
    # the column-major representation is X.T stored contiguously — so:
    #   MatrixLayout(dtype, rows, cols, leading_dim=rows)  (column-major ld = rows)
    # Given A (m x k) and B (k x n), D = A @ B (m x n).
    # We store each as its transpose (.T.copy()) so that hipBLASLt sees a
    # column-major matrix with the correct shape.  The result D is retrieved
    # from the GPU in column-major form and transposed back to row-major.
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    ref = A @ B

    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)

        # Store matrices in column-major order: A.T is shape (k, m) stored
        # contiguously in row-major, which is the same as A in column-major.
        dA = c.DeviceArray.from_numpy(np.ascontiguousarray(A.T), c.DataType.R_32F)
        dB = c.DeviceArray.from_numpy(np.ascontiguousarray(B.T), c.DataType.R_32F)
        dC = c.DeviceArray.from_numpy(
            np.ascontiguousarray(np.zeros((n, m), np.float32)), c.DataType.R_32F
        )
        dD = c.DeviceArray.from_numpy(
            np.ascontiguousarray(np.zeros((n, m), np.float32)), c.DataType.R_32F
        )

        # Column-major layouts: MatrixLayout(dtype, rows, cols, ld) where ld = rows
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)

        pref = c.Preference()
        pref.set_max_workspace(32 * 1024 * 1024)
        res = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        assert len(res) > 0, "heuristic returned no results"

        ws = c.DeviceArray.from_numpy(
            np.zeros(max(1, res[0].workspace_size), np.uint8), c.DataType.R_8I
        )
        c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, res[0].algo, ws)

        # Retrieve result: GPU produced D in column-major (m x n), stored as
        # (n, m) in host memory (same bytes).  Transpose back to (m, n) row-major.
        out = dD.to_numpy().reshape(n, m).T
        np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
