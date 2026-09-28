# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-device OCP E4M3FN decoding and arbitrary-scale rounding boundaries."""

import ctypes

import numpy as np
import pytest

from rocke.runtime.hip_module import get_device_arch


@pytest.mark.gpu
@pytest.mark.skipif(get_device_arch() != "gfx1151", reason="needs a gfx1151 GPU")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("scale", [0.3, 0.011])
def test_all_fp8_bytes_and_non_power_of_two_scales(dtype, scale):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from rocke.core.ir import BF16, F16, FP8E4M3, F32, IRBuilder, PtrType
    from rocke.helpers import compile_kernel
    from rocke.helpers.attention import decode_fp8e4m3fn_to_f32
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.packing import pack_args

    target = BF16 if dtype == "bf16" else F16
    np_target = ml_dtypes.bfloat16 if dtype == "bf16" else np.float16
    b = IRBuilder(f"wmma_fp8_decode_{dtype}")
    b.kernel.attrs["max_workgroup_size"] = 32
    source = b.param("source", PtrType(FP8E4M3, "global"), readonly=True, align=1)
    decoded = b.param("decoded", PtrType(F32, "global"), writeonly=True, align=4)
    converted = b.param("converted", PtrType(target, "global"), writeonly=True, align=2)
    factor = b.param("scale", F32)
    index = b.add(b.mul(b.block_id_x(), b.const_i32(32)), b.thread_id_x())
    value = decode_fp8e4m3fn_to_f32(b, b.global_load(source, index, FP8E4M3, align=1))
    b.global_store(decoded, index, value, align=4)
    b.global_store(converted, index, b.cast_f32_to(b.fmul(value, factor), target), align=2)
    b.ret()

    raw = np.arange(256, dtype=np.uint8)
    expected = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scaled = (expected * np.float32(scale)).astype(np_target).astype(np.float32)
    arrays = {
        "source": raw,
        "decoded": np.full(256, np.nan, np.float32),
        "converted": np.full(256, np.nan, np_target),
    }
    rt = Runtime()
    pointers = {}
    module = None
    try:
        for name, array in arrays.items():
            pointers[name] = rt.alloc(array.nbytes)
            host = (ctypes.c_char * array.nbytes).from_address(array.ctypes.data)
            rt.memcpy_h2d(pointers[name], host, array.nbytes)
        artifact = compile_kernel(b.kernel, arch="gfx1151", backend="python")
        module = rt.load_module(artifact.hsaco)
        signature = [{"name": param.name, "type": param.type.name} for param in b.kernel.params]
        args = pack_args(signature, dict(pointers, scale=scale))
        rt.launch(module.get_function(artifact.kernel_name), (8, 1, 1), (32, 1, 1), args)
        rt.sync()
        for name in ("decoded", "converted"):
            array = arrays[name]
            host = (ctypes.c_char * array.nbytes).from_address(array.ctypes.data)
            rt.memcpy_d2h(host, pointers[name], array.nbytes)
        finite = np.isfinite(expected)
        actual = arrays["decoded"]
        np.testing.assert_array_equal(actual[finite].view(np.uint32), expected[finite].view(np.uint32))
        np.testing.assert_array_equal(np.isnan(actual), ~finite)
        rounded = arrays["converted"].astype(np.float32)
        # The existing gfx1151 FP16 fptrunc normalizes -0 to +0. Decode is
        # bit-exact above; scale/cast must preserve every numeric value.
        np.testing.assert_array_equal(rounded[finite], scaled[finite])
        np.testing.assert_array_equal(np.isnan(rounded), ~finite)
    finally:
        rt.sync()
        if module is not None:
            module.unload()
        for pointer in pointers.values():
            rt.free(pointer)
