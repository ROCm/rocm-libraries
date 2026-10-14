# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Lane-layout and numerics check for the 4x4x4 MFMA atoms (f16 and bf16).

One wave issues a single ``mfma_f32_4x4x4_{f16,bf16}``. The test pins the
16-batch lane map the direct-conv 4c kernel relies on:

* lane ``l`` belongs to batch ``blk = l // 4`` and carries ``i = l % 4``;
* operand ``a`` of lane ``l`` is row ``i`` of ``A_blk`` (K = 0..3);
* operand ``b`` of lane ``l`` is column ``i`` of ``B_blk`` (K = 0..3);
* result slot ``m`` of lane ``l`` is ``D_blk[m][i]``.

The IR / lowering checks run everywhere; the on-device numeric check runs only
where a gfx942 / gfx950 GPU is visible (numpy only, torch is not required).
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest
from rocke.core.ir import BF16, F16, F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm

_INTRIN = {
    "f16": "@llvm.amdgcn.mfma.f32.4x4x4f16(",
    "bf16": "@llvm.amdgcn.mfma.f32.4x4x4bf16.1k(",
}


def _build(dtype: str):
    elem = BF16 if dtype == "bf16" else F16
    b = IRBuilder(f"mfma_4x4x4_{dtype}_probe")
    b.kernel.attrs["max_workgroup_size"] = 64
    A = b.param("A", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    lane = b.thread_id_x()
    idx = b.mul(lane, b.const_i32(4))
    a = b.global_load_vN(A, idx, elem, 4)
    bv = b.global_load_vN(B, idx, elem, 4)
    zero = b.zero_vec_f32(4)
    if dtype == "bf16":
        acc = b.mfma_f32_4x4x4_bf16(a, bv, zero)
    else:
        acc = b.mfma_f32_4x4x4_f16(a, bv, zero)
    b.global_store_vN(D, idx, acc, 4)
    b.ret()
    return b.kernel


def _to_bits(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        # round-to-nearest-even f32 -> bf16 bit pattern
        u = x.astype(np.float32).view(np.uint32)
        u = u + 0x7FFF + ((u >> 16) & 1)
        return (u >> 16).astype(np.uint16)
    return x.astype(np.float16).view(np.uint16)


def _from_bits(u: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "bf16":
        return (u.astype(np.uint32) << 16).view(np.float32)
    return u.view(np.float16).astype(np.float32)


def _reference(a_lanes: np.ndarray, b_lanes: np.ndarray) -> np.ndarray:
    """Per-lane expected result for the documented 4x4x4 lane map."""
    out = np.zeros((64, 4), dtype=np.float32)
    for blk in range(16):
        A = a_lanes[blk * 4 : blk * 4 + 4]  # A[i][k]
        Bt = b_lanes[blk * 4 : blk * 4 + 4]  # Bt[j][k] == B[k][j]
        Dm = A @ Bt.T  # D[i][j]
        for j in range(4):
            out[blk * 4 + j, :] = Dm[:, j]
    return out


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_lowering_emits_expected_intrinsic(dtype):
    ll = lower_kernel_to_llvm(_build(dtype))
    assert _INTRIN[dtype] in ll
    if dtype == "bf16":
        assert "bitcast <4 x bfloat>" in ll
        assert "declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16>" in ll


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_hip_lowering_emits_expected_builtin(dtype):
    from rocke.core.lower_hip import lower_kernel_to_hip

    builtin = {
        "f16": "__builtin_amdgcn_mfma_f32_4x4x4f16(",
        "bf16": "__builtin_amdgcn_mfma_f32_4x4x4bf16_1k(",
    }[dtype]
    assert builtin in lower_kernel_to_hip(_build(dtype))


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_numerics_on_device(dtype):
    try:
        from rocke import compile_kernel
        from rocke.runtime import KernelLauncher, LaunchConfig, Runtime

        rt = Runtime()
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"no HIP runtime: {e}")
    try:
        from rocke.runtime import get_device_info

        arch = get_device_info().base_arch
    except (ImportError, OSError, RuntimeError) as e:  # pragma: no cover
        pytest.skip(f"no visible GPU: {e}")
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"4x4x4 MFMA check needs gfx942/gfx950 (got {arch})")
    art = compile_kernel(_build(dtype), arch=arch, capture_ir_text=False)

    rng = np.random.default_rng(1234)
    a = _from_bits(_to_bits(rng.uniform(-2, 2, (64, 4)), dtype), dtype)
    bm = _from_bits(_to_bits(rng.uniform(-2, 2, (64, 4)), dtype), dtype)
    a_bits = np.ascontiguousarray(_to_bits(a, dtype))
    b_bits = np.ascontiguousarray(_to_bits(bm, dtype))
    out = np.zeros((64, 4), dtype=np.float32)

    def u8(x):
        return (ctypes.c_uint8 * x.nbytes).from_address(x.ctypes.data)

    dA, dB, dD = rt.alloc(a_bits.nbytes), rt.alloc(b_bits.nbytes), rt.alloc(out.nbytes)
    rt.memcpy_h2d(dA, u8(a_bits), a_bits.nbytes)
    rt.memcpy_h2d(dB, u8(b_bits), b_bits.nbytes)
    pt = "bf16" if dtype == "bf16" else "f16"
    sig = [
        {"name": "A", "type": f"ptr<{pt}, global>", "size_bytes": 8},
        {"name": "B", "type": f"ptr<{pt}, global>", "size_bytes": 8},
        {"name": "D", "type": "ptr<f32, global>", "size_bytes": 8},
    ]
    launcher = KernelLauncher(
        hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig
    )
    launcher(
        {"A": dA, "B": dB, "D": dD},
        config=LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
    )
    rt.sync()
    rt.memcpy_d2h(u8(out), dD, out.nbytes)
    for ptr in (dA, dB, dD):
        rt.free(ptr)
    ref = _reference(a, bm)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
