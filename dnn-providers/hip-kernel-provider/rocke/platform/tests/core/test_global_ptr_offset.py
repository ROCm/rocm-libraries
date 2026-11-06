# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``global_ptr_offset`` and the HIP lowering of ``global_ptr_to_i64``.

``global_ptr_offset(ptr, byte_off)`` moves a global pointer by an i64 byte
offset without promising the compiler that the result stays inside the
allocation: the LLVM lowering is a ``getelementptr i8`` without ``inbounds``,
and the HIP lowering adds in integer arithmetic, so clang has no pointer step
to mark ``inbounds`` either. ``global_ptr_to_i64`` is the ``ptrtoint`` of a
global pointer; the HIP route lowers it to an integer cast.

Covers the LLVM IR and HIP source of both engines (the same text at every LLVM
flavor), the builder's, the verifier's and the lowerings' refusals (including a
parameter whose ``addr_space`` override makes it a constant pointer), the device
IR hipcc makes from the HIP source (no ``inbounds`` step; a ``global_ptr_add``
control shows the check can see one), and, when a GPU is present, addresses
and loads through offset pointers -- one of them outside the allocation -- on
the LLVM and the hipcc route.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import struct
import subprocess

import numpy as np
import pytest
from rocke.core.ir import F32, I64, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)

FLAVORS = ("llvm20", "llvm22", "llvm23")
X_ELEMS = 8


def _address_kernel(name):
    """Store ``X`` moved by the i64 parameter ``off`` bytes, as an address, to
    ``Z[0]``, and the f32 one element past that pointer to ``Y[0]``."""
    b = IRBuilder(name)
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    z = b.param("Z", PtrType(I64, "global"))
    off = b.param("off", I64)
    moved = b.global_ptr_offset(x, off)
    b.global_store(z, b.const_i32(0), b.global_ptr_to_i64(moved))
    b.global_store(y, b.const_i32(0), b.global_load(moved, b.const_i32(1), F32))
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------
# LLVM IR and HIP source
# --------------------------------------------------------------------------


def test_llvm_ir_offsets_without_inbounds_and_casts_the_pointer_to_i64():
    text = _lower_kernel_to_llvm_python(_address_kernel("gpo_llvm"), arch="gfx942")
    steps = [line for line in text.splitlines() if "%X, i64 %off" in line]
    assert len(steps) == 1
    assert re.fullmatch(
        r"  %gptr\d+ = getelementptr i8, ptr addrspace\(1\) %X, i64 %off", steps[0]
    ), steps[0]
    assert re.search(r" = ptrtoint ptr addrspace\(1\) %gptr\d+ to i64\n", text)


def test_both_engines_emit_the_same_llvm_ir():
    kernel = _address_kernel("gpo_engines")
    for flavor in FLAVORS:
        text = lower_with_both_engines(kernel, "gfx942", llvm_flavor=flavor)
        assert "getelementptr i8, ptr addrspace(1) %X, i64 %off" in text


def test_hip_offsets_in_integer_arithmetic_and_casts_the_pointer_to_int64():
    hip = lower_hip_with_both_engines(_address_kernel("gpo_hip"), "gfx942")
    # The result keeps the pointer's element type, so indexing it counts
    # elements, as indexing X does.
    assert re.search(
        r"float\* (gptr\d+) = \(float\*\)\(\(uintptr_t\)X \+ \(uint64_t\)off\);", hip
    ), hip
    assert re.search(r"int64_t gaddr\d+ = \(int64_t\)\(uintptr_t\)gptr\d+;", hip), hip


# --------------------------------------------------------------------------
# hipcc: the compiled device IR has no inbounds step
# --------------------------------------------------------------------------


def _device_ir(tmp_path, kernel):
    """hipcc's optimized device LLVM IR of the HIP source of ``kernel``."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    source = tmp_path / f"{kernel.name}.hip"
    source.write_text(lower_kernel_to_hip(kernel, arch="gfx942"))
    output = tmp_path / f"{kernel.name}.ll"
    subprocess.run(
        [hipcc, "--offload-arch=gfx942", "--cuda-device-only", "-O3", "-S",
         "-emit-llvm", str(source), "-o", str(output)],
        check=True,
        capture_output=True,
        timeout=240,
    )  # fmt: skip
    return output.read_text()


def _step_kernel(name, *, inbounds):
    """Load ``X`` moved by ``off`` bytes into ``Y[0]``, through
    ``global_ptr_add`` (inbounds) or ``global_ptr_offset``."""
    b = IRBuilder(name)
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    off = b.param("off", I64)
    moved = (b.global_ptr_add if inbounds else b.global_ptr_offset)(x, off)
    b.global_store(y, b.const_i32(0), b.global_load(moved, b.const_i32(0), F32))
    b.ret()
    return b.kernel


def test_hipcc_device_ir_has_no_inbounds_step(tmp_path):
    offset_ir = _device_ir(tmp_path, _step_kernel("gpo_step", inbounds=False))
    assert "getelementptr inbounds" not in offset_ir
    # The control: the same kernel through global_ptr_add steps inbounds, so
    # the check above can see an inbounds step when there is one.
    add_ir = _device_ir(tmp_path, _step_kernel("gpa_step", inbounds=True))
    assert "getelementptr inbounds" in add_ir


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_builder_refuses_a_pointer_that_is_not_global():
    b = IRBuilder("gpo_private")
    flat = b.param("flat", PtrType(F32, "private"))
    with pytest.raises(TypeError) as error:
        b.global_ptr_offset(flat, b.const_i64(4))
    assert str(error.value) == (
        "global_ptr_offset ptr must be a global pointer, got ptr<f32,private>"
    )


def test_builder_refuses_an_offset_that_is_not_i64():
    b = IRBuilder("gpo_i32")
    x = b.param("X", PtrType(F32, "global"))
    with pytest.raises(TypeError) as error:
        b.global_ptr_offset(x, b.const_i32(4))
    assert str(error.value) == (
        "global_ptr_offset byte_off must be i64, got i32; widen it with sext "
        "(a signed offset) or zext (an unsigned one)"
    )


def _raw_i32_offset_kernel():
    """A global_ptr_offset with an i32 offset, written past the builder's
    check, as serialized IR can reach the verifier and the lowerers."""
    b = IRBuilder("gpo_raw")
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    moved = b._op(
        "tile.global_ptr_offset",
        [x, b.const_i32(4)],
        [x.type],
        result_name_hint="gptr",
    ).result
    b.global_store(y, b.const_i32(0), b.global_load(moved, b.const_i32(0), F32))
    b.ret()
    return b.kernel


RAW_MESSAGE = (
    "global_ptr_offset byte_off must be i64, got i32; widen it with sext "
    "(a signed offset) or zext (an unsigned one)"
)


def test_verifier_refuses_an_offset_that_is_not_i64():
    messages = [d.message for d in verify(_raw_i32_offset_kernel())]
    assert RAW_MESSAGE in messages


def test_both_lowerings_refuse_an_offset_that_is_not_i64():
    kernel = _raw_i32_offset_kernel()
    assert_both_engines_reject(kernel, "gfx942", ValueError, RAW_MESSAGE)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, RAW_MESSAGE)


def _raw_global_ptr_to_i64_kernel(name, pointer_is_global, result_type):
    """A global_ptr_to_i64 written past the builder's check, as serialized IR
    can reach the verifier and the lowerers: its operand is ``X`` or an i64
    constant, its result has ``result_type``, and the result is stored."""
    b = IRBuilder(name)
    x = b.param("X", PtrType(F32, "global"))
    z = b.param("Z", PtrType(result_type, "global"))
    operand = x if pointer_is_global else b.const_i64(64)
    address = b._op(
        "tile.global_ptr_to_i64", [operand], [result_type], result_name_hint="gaddr"
    ).result
    b.global_store(z, b.const_i32(0), address)
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    ("pointer_is_global", "result_type", "message"),
    [
        (False, I64, "global_ptr_to_i64 ptr must be a global pointer, got i64"),
        (True, F32, "global_ptr_to_i64 result must be i64, got f32"),
    ],
    ids=["integer_operand", "f32_result"],
)
def test_verifier_and_every_lowering_refuse_a_malformed_global_ptr_to_i64(
    pointer_is_global, result_type, message
):
    """An integer in the pointer's place, or an f32 result the lowerings would
    otherwise emit as an i64 under the f32 value's name."""
    kernel = _raw_global_ptr_to_i64_kernel("p2i_raw", pointer_is_global, result_type)
    assert message in [d.message for d in verify(kernel)]
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, message)


@pytest.mark.parametrize(
    "op_name", ["global_ptr_offset", "global_ptr_add", "global_ptr_to_i64"]
)
def test_both_llvm_lowerings_refuse_a_constant_space_parameter(op_name):
    """A global-pointer parameter declared with ``addr_space="constant"`` is
    ``ptr addrspace(4)`` in the kernel header, so an instruction that names it
    ``ptr addrspace(1)`` would not match the header: every global-pointer op
    is refused instead, in the LLVM lowering of both engines."""
    b = IRBuilder(f"{op_name}_constant")
    x = b.param("X", PtrType(F32, "global"), addr_space="constant")
    z = b.param("Z", PtrType(I64, "global"))
    if op_name == "global_ptr_to_i64":
        address = b.global_ptr_to_i64(x)
    else:
        moved = getattr(b, op_name)(x, b.const_i64(4))
        address = b.global_ptr_to_i64(moved)
    b.global_store(z, b.const_i32(0), address)
    b.ret()
    message = (
        f"{op_name} ptr must be a global pointer, got ptr addrspace(4) "
        "(its parameter's addr_space)"
    )
    assert_both_engines_reject(b.kernel, "gfx942", TypeError, message)


# --------------------------------------------------------------------------
# GPU: addresses and loads through offset pointers, on both compile routes
# --------------------------------------------------------------------------


def _gpu_kernel(name):
    """One lane. ``Z[0]`` = the address of ``X`` + 2 bytes; ``Z[1]`` = the
    address of ``X`` - 8 bytes, outside the allocation; ``Y[0]``, ``Y[1]`` =
    the f32 elements 0 and 1 of ``X`` + 8 bytes (``X[2]``, ``X[3]``); ``Y[2]``
    = element 1 of ``X`` - 8 bytes moved back by 12 bytes, i.e. of ``X`` + 4
    bytes (``X[2]``)."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 1
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    z = b.param("Z", PtrType(I64, "global"))
    i = [b.const_i32(k) for k in range(3)]
    plus2 = b.global_ptr_offset(x, b.const_i64(2))
    before = b.global_ptr_offset(x, b.const_i64(-8))
    plus8 = b.global_ptr_offset(x, b.const_i64(8))
    back = b.global_ptr_offset(before, b.const_i64(12))
    b.global_store(z, i[0], b.global_ptr_to_i64(plus2))
    b.global_store(z, i[1], b.global_ptr_to_i64(before))
    b.global_store(y, i[0], b.global_load(plus8, i[0], F32))
    b.global_store(y, i[1], b.global_load(plus8, i[1], F32))
    b.global_store(y, i[2], b.global_load(back, i[1], F32))
    b.ret()
    return b.kernel


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run(blob, kernel_name, x):
    """Launch one lane; return the device address of X, Y and Z."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    y = np.zeros(3, dtype=np.float32)
    z = np.zeros(2, dtype=np.int64)
    buffers = [runtime.alloc(a.nbytes) for a in (x, y, z)]
    try:
        runtime.memcpy_h2d(
            buffers[0], (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes
        )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (1, 1, 1),
            struct.pack("3Q", *buffers),
        )
        for buffer, out in zip(buffers[1:], (y, z)):
            runtime.memcpy_d2h(
                (ctypes.c_ubyte * out.nbytes).from_buffer(out), buffer, out.nbytes
            )
    finally:
        for buffer in buffers:
            runtime.free(buffer)
        module.unload()
    return buffers[0], y, z


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_offset_pointers(route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    kernel = _gpu_kernel(f"gpo_gpu_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        if shutil.which("hipcc") is None:
            pytest.skip("hipcc not found")
        from rocke.helpers.compile import compile_kernel_via_hipcc

        blob = compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    x = np.arange(1, X_ELEMS + 1, dtype=np.float32) * np.float32(1.5)
    x_address, y, z = _run(blob, kernel.name, x)
    # hipMalloc aligns an allocation to far more than 4 bytes, so X + 2 has
    # low bits 2.
    assert int(z[0]) & 3 == 2
    assert int(z[0]) == x_address + 2
    assert int(z[1]) == x_address - 8
    np.testing.assert_array_equal(y, np.array([x[2], x[3], x[2]], dtype=np.float32))
