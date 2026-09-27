# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real GPU tests for scalar bit preservation and a separate FP32 rounding step."""

import ctypes
import os
import struct

import numpy as np
import pytest

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
)
from rocke.helpers import compile_kernel
from rocke.helpers.compile import compile_kernel_via_hipcc
from rocke.runtime.hip_module import Runtime, get_device_arch


@pytest.fixture(scope="module")
def arch():
    target = get_device_arch(0)
    if not target:
        if os.environ.get("ROCKE_REQUIRE_GFX1250") == "1":
            pytest.fail("required gfx1250 GPU unavailable")
        pytest.skip("requires an AMD GPU")
    if os.environ.get("ROCKE_REQUIRE_GFX1250") == "1":
        assert target == "gfx1250"
    return target


def _run(kernel, data, arch, route):
    compile_fn = compile_kernel if route == "comgr" else compile_kernel_via_hipcc
    art = compile_fn(kernel, arch=arch)
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    ptr = rt.alloc(len(data))
    try:
        host = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        rt.memcpy_h2d(ptr, host, len(data))
        rt.launch(
            module.get_function(art.kernel_name),
            (1, 1, 1),
            (64, 1, 1),
            struct.pack("<Q", ptr),
        )
        rt.sync()
        rt.memcpy_d2h(host, ptr, len(data))
        return bytes(host)
    finally:
        rt.free(ptr)
        module.unload()


@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize(
    "dtype,width",
    [
        (I8, 1),
        (I16, 2),
        (I32, 4),
        (I64, 8),
        (BF16, 2),
        (F16, 2),
        (F32, 4),
        (FP8E4M3, 1),
        (BF8E5M2, 1),
    ],
)
def test_barrier_preserves_scalar_bits(arch, route, dtype, width):
    b = IRBuilder("barrier_bits_" + dtype.name)
    p = b.param("p", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    value = b.global_load(p, tid, dtype)
    b.global_store(p, tid, b.optimization_barrier(value))
    b.ret()
    # Arbitrary bit patterns, including signed values and floating special values.
    raw = np.random.default_rng(42).integers(0, 256, 64 * width, dtype=np.uint8)
    raw[:width] = 0
    raw[width : 2 * width] = 255
    data = raw.tobytes()
    assert _run(b.kernel, data, arch, route) == data


@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_barrier_rounds_product_before_add(arch, route):
    b = IRBuilder("barrier_product")
    p = b.param("p", PtrType(F32, "global"))
    tid = b.thread_id_x()
    a = b.global_load(p, tid, F32)
    other = b.add(tid, b.const_i32(64))
    bb = b.global_load(p, other, F32)
    product = b.optimization_barrier(b.fmul(a, bb))
    value = b.fadd(product, b.const_f32(-1.0))
    b.global_store(p, tid, value)
    b.ret()
    # (1+2^-13)*(1-2^-13) rounds to FP32 1.0; a fused FMA returns -2^-26.
    a = np.full(64, 1 + 2**-13, dtype=np.float32)
    bb = np.full(64, 1 - 2**-13, dtype=np.float32)
    data = np.concatenate((a, bb)).tobytes()
    result = np.frombuffer(_run(b.kernel, data, arch, route), dtype=np.float32)
    np.testing.assert_array_equal(result[:64], np.zeros(64, dtype=np.float32))


@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_barrier_preserves_lane_predicates(arch, route):
    b = IRBuilder("barrier_predicate")
    p = b.param("p", PtrType(I32, "global"))
    tid = b.thread_id_x()
    value = b.global_load(p, tid, I32)
    predicate = b.cmp_lt(value, b.const_i32(0))
    result = b.optimization_barrier(predicate)
    b.global_store(p, tid, b.zext(result, I32))
    b.ret()
    values = np.random.default_rng(13).integers(-16, 16, 64, dtype=np.int32)
    result = np.frombuffer(
        _run(b.kernel, values.tobytes(), arch, route), dtype=np.int32
    )
    np.testing.assert_array_equal(result, (values < 0).astype(np.int32))


@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_barrier_rounds_product_before_fp16(arch, route):
    b = IRBuilder("barrier_fp16")
    p = b.param("p", PtrType(F32, "global"))
    tid = b.thread_id_x()
    a = b.global_load(p, tid, F32)
    scale = b.global_load(p, b.add(tid, b.const_i32(64)), F32)
    product = b.optimization_barrier(b.fmul(a, scale))
    rounded = b.cast_f32_to(product, F16)
    b.global_store(p, tid, b.cast_to_f32(rounded))
    b.ret()
    # The FP32 product is exactly a half-precision midpoint. Directly rounding
    # the unrounded product to FP16 instead selects the adjacent value.
    a = np.full(64, -412.5, dtype=np.float32)
    scale = np.full(64, np.float32(1.3) * np.float32(0.7), dtype=np.float32)
    expected = (a * scale).astype(np.float16).astype(np.float32)
    assert np.all(expected == -375.5)
    assert np.all((a.astype(np.float64) * scale).astype(np.float16) == -375.25)
    result = np.frombuffer(
        _run(b.kernel, np.concatenate((a, scale)).tobytes(), arch, route),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result[:64], expected)
