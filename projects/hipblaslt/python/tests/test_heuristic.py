# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu


@requires_gpu
def test_heuristic_returns_algos():
    m = n = k = 64
    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        la = c.MatrixLayout(c.DataType.R_32F, m, k, m)
        lb = c.MatrixLayout(c.DataType.R_32F, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference()
        pref.set_max_workspace(32 * 1024 * 1024)
        results = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        assert len(results) > 0
        assert results[0].workspace_size >= 0
        assert isinstance(results[0].algo.index, int)
