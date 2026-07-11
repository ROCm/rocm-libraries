# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
from hipblaslt import mx
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


def test_build_block_scales_shapes():
    ref = np.random.rand(64, 64).astype(np.float32)
    scales, scaled = mx.build_block_scales(ref, block=32)
    assert scaled.shape == ref.shape
    # one scale per 32-element block along the innermost dim
    assert scales.shape == (64, 64 // 32)


def test_apply_inverts_build():
    ref = np.random.rand(32, 32).astype(np.float32)
    scales, scaled = mx.build_block_scales(ref, block=32)
    recon = mx.apply_block_scales(scaled, scales, block=32)
    # reconstruction is within fp8 block-scaling error of the original
    np.testing.assert_allclose(recon, ref, rtol=0.1, atol=0.1)


def test_swizzle_roundtrip():
    scales = np.arange(32 * 8, dtype=np.uint8).reshape(32, 8)
    sw = mx.swizzle_scales(scales, tile=(32, 8, 4))
    back = mx.unswizzle_scales(sw, tile=(32, 8, 4), shape=scales.shape)
    np.testing.assert_array_equal(back, scales)


@pytest.mark.mi350
@pytest.mark.gpu
@requires_gpu
def test_mx_gemm_matches_reference():
    """DEFERRED to MI350 (gfx950). Full MX GEMM: build canonical UE8M0 scales,
    set A_SCALE_MODE/B_SCALE_MODE=VEC32_UE8M0 + A/B_SCALE_POINTER, run matmul,
    compare against the apply_block_scales numpy reference. On gfx942 this raises
    NOT_SUPPORTED and skips. See docs/superpowers/handoff/2026-07-10-mi350-verification.md.
    """
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    m = n = k = 128
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    a_scales, a_scaled = mx.build_block_scales(A, block=32)   # canonical (row, col) UE8M0
    b_scales, b_scaled = mx.build_block_scales(B, block=32)
    ref = mx.apply_block_scales(a_scaled, a_scales) @ mx.apply_block_scales(b_scaled, b_scales)
    try:
        # Element tensors: fp8 e4m3 (OCP) of the block-scaled values.
        a_fp8 = a_scaled.astype(ml_dtypes.float8_e4m3fn)
        b_fp8 = b_scaled.astype(ml_dtypes.float8_e4m3fn)
        with c.Handle() as h:
            desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
            dA = c.DeviceArray.from_numpy(np.ascontiguousarray(a_fp8.T), c.DataType.R_8F_E4M3)
            dB = c.DeviceArray.from_numpy(np.ascontiguousarray(b_fp8.T), c.DataType.R_8F_E4M3)
            dsa = c.DeviceArray.from_numpy(np.ascontiguousarray(a_scales.T), c.DataType.R_8I)  # UE8M0 bytes
            dsb = c.DeviceArray.from_numpy(np.ascontiguousarray(b_scales.T), c.DataType.R_8I)
            dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
            dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
            la = c.MatrixLayout(c.DataType.R_8F_E4M3, m, k, m)
            lb = c.MatrixLayout(c.DataType.R_8F_E4M3, k, n, k)
            lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
            ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
            desc.set_attribute_int(c.MatmulDescAttr.A_SCALE_MODE, int(c.ScaleMode.VEC32_UE8M0))
            desc.set_attribute_int(c.MatmulDescAttr.B_SCALE_MODE, int(c.ScaleMode.VEC32_UE8M0))
            desc.set_attribute_ptr(c.MatmulDescAttr.A_SCALE_POINTER, dsa.ptr)
            desc.set_attribute_ptr(c.MatmulDescAttr.B_SCALE_POINTER, dsb.ptr)
            pref = c.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
            res = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
            ws = c.DeviceArray.from_numpy(np.zeros(max(1, res[0].workspace_size), np.uint8), c.DataType.R_8I)
            c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, res[0].algo, ws)
            out = dD.to_numpy().reshape(n, m).T
            # VERIFY-ON-MI350: tolerance and the exact scale-tensor layout/transpose
            # for VEC32_UE8M0 are unverified without gfx950; adjust rtol and the
            # dsa/dsb layout until this matches, then tighten. Cross-ref
            # DataInitialization.cpp for the canonical scale layout the kernel expects.
            np.testing.assert_allclose(out, ref, rtol=0.15, atol=0.15)
    except c.HipblasLtError as e:
        if "NOT_SUPPORTED" in str(e):
            pytest.skip(f"MX unsupported on this arch: {e}")
        raise


@pytest.mark.mi350
@requires_gpu
def test_mx_gemm_preswizzle_mode1001():
    """DEFERRED to MI350 (gfx950). Same as test_mx_gemm_matches_reference but with
    A_SCALE_MODE=BLK32_UE8M0_32_8_EXT (1001) and PRE-SWIZZLED scale tensors
    (mx.swizzle_scales). Reference still uses canonical scales. Skips on gfx942.
    """
    if not hasattr(c.ScaleMode, "BLK32_UE8M0_32_8_EXT"):
        pytest.skip("BLK32_UE8M0_32_8_EXT not available in this SDK version")
    pytest.importorskip("ml_dtypes")
    import ml_dtypes
    m = n = k = 128
    A = np.random.rand(m, k).astype(np.float32)
    B = np.random.rand(k, n).astype(np.float32)
    a_scales, a_scaled = mx.build_block_scales(A, block=32)
    b_scales, b_scaled = mx.build_block_scales(B, block=32)
    ref = mx.apply_block_scales(a_scaled, a_scales) @ mx.apply_block_scales(b_scaled, b_scales)
    try:
        a_fp8 = a_scaled.astype(ml_dtypes.float8_e4m3fn)
        b_fp8 = b_scaled.astype(ml_dtypes.float8_e4m3fn)
        # VERIFY-ON-MI350: pre-swizzle the canonical scales for mode 1001. The
        # swizzle permutation (mx.swizzle_scales) is only roundtrip-verified on
        # gfx942; correctness of the forward layout is confirmed here on MI350.
        a_sw = mx.swizzle_scales(a_scales, tile=(32, 8, 4))
        b_sw = mx.swizzle_scales(b_scales, tile=(32, 8, 4))
        with c.Handle() as h:
            desc = c.MatmulDesc(c.ComputeType.COMPUTE_32F, c.DataType.R_32F)
            dA = c.DeviceArray.from_numpy(np.ascontiguousarray(a_fp8.T), c.DataType.R_8F_E4M3)
            dB = c.DeviceArray.from_numpy(np.ascontiguousarray(b_fp8.T), c.DataType.R_8F_E4M3)
            dsa = c.DeviceArray.from_numpy(np.ascontiguousarray(a_sw), c.DataType.R_8I)
            dsb = c.DeviceArray.from_numpy(np.ascontiguousarray(b_sw), c.DataType.R_8I)
            dC = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
            dD = c.DeviceArray.from_numpy(np.zeros((n, m), np.float32), c.DataType.R_32F)
            la = c.MatrixLayout(c.DataType.R_8F_E4M3, m, k, m)
            lb = c.MatrixLayout(c.DataType.R_8F_E4M3, k, n, k)
            lc = c.MatrixLayout(c.DataType.R_32F, m, n, m)
            ld = c.MatrixLayout(c.DataType.R_32F, m, n, m)
            desc.set_attribute_int(c.MatmulDescAttr.A_SCALE_MODE, int(c.ScaleMode.BLK32_UE8M0_32_8_EXT))
            desc.set_attribute_int(c.MatmulDescAttr.B_SCALE_MODE, int(c.ScaleMode.BLK32_UE8M0_32_8_EXT))
            desc.set_attribute_ptr(c.MatmulDescAttr.A_SCALE_POINTER, dsa.ptr)
            desc.set_attribute_ptr(c.MatmulDescAttr.B_SCALE_POINTER, dsb.ptr)
            pref = c.Preference(); pref.set_max_workspace(64 * 1024 * 1024)
            res = c.heuristic(h, desc, la, lb, lc, ld, pref, 16)
            ws = c.DeviceArray.from_numpy(np.zeros(max(1, res[0].workspace_size), np.uint8), c.DataType.R_8I)
            c.matmul(h, desc, 1.0, dA, la, dB, lb, 0.0, dC, lc, dD, ld, res[0].algo, ws)
            out = dD.to_numpy().reshape(n, m).T
            np.testing.assert_allclose(out, ref, rtol=0.15, atol=0.15)
    except c.HipblasLtError as e:
        if "NOT_SUPPORTED" in str(e):
            pytest.skip(f"mode-1001 pre-swizzle unsupported on this arch: {e}")
        raise
