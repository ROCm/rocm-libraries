# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
ml_dtypes = pytest.importorskip("ml_dtypes")


def _fp8_gemm(a_dtype, mld_type):
    m = n = k = 64
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    a8 = A.astype(mld_type); b8 = B.astype(mld_type)
    ref = a8.astype(np.float32) @ b8.astype(np.float32)
    with c.Handle() as h:
        desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
        dA = c.DeviceArray.from_numpy(np.ascontiguousarray(a8.T), a_dtype)
        dB = c.DeviceArray.from_numpy(np.ascontiguousarray(b8.T), a_dtype)
        dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
        la = c.MatrixLayout(a_dtype, m, k, m)
        lb = c.MatrixLayout(a_dtype, k, n, k)
        lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
        pref = c.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
        res = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
        if not res:
            raise c.HipblasLtError("HIPBLAS_STATUS_NOT_SUPPORTED (no algo)")
        ws = c.DeviceArray.from_numpy(np.zeros(max(1, res[0].workspace_size), np.uint8), c.DataType.R_8I)
        c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, res[0].algo, ws)
        return dD.to_numpy().reshape(n, m).T, ref


@requires_gpu
def test_fnuz_fp8_gemm():
    # RUNS on gfx942 / MI300 (FNUZ is the MI300 fp8 format).
    fnuz = getattr(c.DataType, "R_8F_E4M3_FNUZ", None)
    mld = getattr(ml_dtypes, "float8_e4m3fnuz", None)
    if fnuz is None or mld is None:
        pytest.skip("FNUZ fp8 unavailable in this build/ml_dtypes")
    out, ref = _fp8_gemm(fnuz, mld)
    np.testing.assert_allclose(out, ref, rtol=0.1, atol=0.1)


@pytest.mark.mi350
@requires_gpu
def test_ocp_fp8_gemm():
    # DEFERRED to MI350 (gfx950): OCP E4M3 GEMM. On MI300 this raises
    # NOT_SUPPORTED and skips. See handoff doc.
    try:
        out, ref = _fp8_gemm(c.DataType.R_8F_E4M3, ml_dtypes.float8_e4m3fn)
        np.testing.assert_allclose(out, ref, rtol=0.1, atol=0.1)
    except c.HipblasLtError as e:
        if "NOT_SUPPORTED" in str(e):
            pytest.skip(f"OCP fp8 GEMM unsupported on this arch: {e}")
        raise
