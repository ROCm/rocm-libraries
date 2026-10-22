# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Buffer loads and stores for every element width: ``buffer_load_vN`` and
``buffer_store_vN``.

Both take any scalar whose width is whole bytes and move 1, 2, 4, 8 or 16 bytes
per access. Covers the LLVM IR each width lowers to, that both engines emit the
same LLVM IR and the same HIP source, the builder's, the verifier's and the
lowerers' refusals, and, when a GPU is present, a bit-exact copy through both compile routes (LLVM
through COMGR, HIP through hipcc). The C++ builder's refusal texts are checked
in the ``rocke_future_intrinsic_lowering`` ctest, the same texts as here.
"""

from __future__ import annotations

import ctypes
import shutil
import struct

import numpy as np
import pytest
from rocke.core.ir import (
    BF8E5M2,
    BF16,
    E5M3,
    E8M0,
    F16,
    F32,
    F64,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    IRBuilder,
    PtrType,
    VectorType,
)
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

# (element type, count) for every access width each element width can make.
WIDTHS = (
    (I8, 1),
    (I8, 2),
    (I8, 4),
    (I8, 8),
    (I8, 16),
    (FP8E4M3, 1),
    (FP8E4M3, 4),
    (BF8E5M2, 16),
    (I16, 1),
    (F16, 1),
    (F16, 2),
    (F16, 8),
    (BF16, 1),
    (BF16, 4),
    (F32, 1),
    (F32, 4),
    (I32, 2),
    (F64, 1),
    (F64, 2),
)

# Access bytes -> (LLVM integer type, intrinsic suffix, HIP builtin suffix).
FORMS = {
    1: ("i8", "i8", "_b8"),
    2: ("i16", "i16", "_b16"),
    4: ("i32", "i32", "_b32"),
    8: ("<2 x i32>", "v2i32", "_b64"),
    16: ("<4 x i32>", "v4i32", "_b128"),
}

_LLVM_ELEM = {
    "i8": "i8",
    "fp8e4m3": "i8",
    "bf8e5m2": "i8",
    "i16": "i16",
    "f16": "half",
    "bf16": "bfloat",
    "f32": "float",
    "i32": "i32",
    "f64": "double",
}


def _nbytes(dtype, n):
    return n * {"i8": 1, "fp8e4m3": 1, "bf8e5m2": 1, "i16": 2, "f16": 2,
                "bf16": 2, "f32": 4, "i32": 4, "f64": 8}[dtype.name]  # fmt: skip


def _id(width):
    dtype, n = width
    return f"{n}x{dtype.name}"


def _copy_kernel(name, widths, *, lanes=64):
    """Lane ``l`` of section ``s`` copies one access of ``widths[s]`` from
    ``X`` to ``Y`` at byte ``base_s + l * bytes_s``: one buffer load, one
    buffer store. Returns the kernel and the bytes it covers."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = lanes
    x = b.param("X", PtrType(I8, "global"))
    y = b.param("Y", PtrType(I8, "global"))
    total = sum(lanes * _nbytes(dtype, n) for dtype, n in widths)
    src = b.buffer_rsrc(x, b.const_i32(total))
    dst = b.buffer_rsrc(y, b.const_i32(total))
    lane = b.thread_id_x()
    zero = b.const_i32(0)
    base = 0
    for dtype, n in widths:
        nbytes = _nbytes(dtype, n)
        offset = b.add(b.mul(lane, b.const_i32(nbytes)), b.const_i32(base))
        value = b.buffer_load_vN(src, offset, zero, dtype, n)
        b.buffer_store_vN(dst, offset, zero, value)
        base += lanes * nbytes
    b.ret()
    return b.kernel, total


# --------------------------------------------------------------------------
# LLVM IR
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS, ids=_id)
def test_load_lowers_to_one_load_of_its_width(width):
    dtype, n = width
    kernel, _ = _copy_kernel("buffer_load_width", [width])
    text = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    llvm_int, suffix, _ = FORMS[_nbytes(dtype, n)]
    assert f" = call {llvm_int} @llvm.amdgcn.raw.ptr.buffer.load.{suffix}(" in text
    assert f" to <{n} x {_LLVM_ELEM[dtype.name]}>" in text
    assert text.count("@llvm.amdgcn.raw.ptr.buffer.load.") == 2  # declare + call


@pytest.mark.parametrize("width", WIDTHS, ids=_id)
def test_store_lowers_to_one_store_of_its_width(width):
    dtype, n = width
    kernel, _ = _copy_kernel("buffer_store_width", [width])
    text = _lower_kernel_to_llvm_python(kernel, arch="gfx942")
    llvm_int, suffix, _ = FORMS[_nbytes(dtype, n)]
    assert f"bitcast <{n} x {_LLVM_ELEM[dtype.name]}> " in text
    assert f"call void @llvm.amdgcn.raw.ptr.buffer.store.{suffix}({llvm_int} " in text
    assert text.count("@llvm.amdgcn.raw.ptr.buffer.store.") == 2  # declare + call


def test_load_records_dwords_only_for_whole_dword_accesses():
    b = IRBuilder("buffer_load_attrs")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    narrow = b.buffer_load_vN(rsrc, zero, zero, I8, 2)
    wide = b.buffer_load_vN(rsrc, zero, zero, F64, 2)
    assert narrow.op.attrs == {"elem_type": "i8"}
    assert wide.op.attrs == {"dwords": 4, "elem_type": "f64"}
    assert narrow.type == VectorType(I8, 2)


def test_both_engines_emit_the_same_llvm_ir():
    kernel, _ = _copy_kernel("buffer_every_width", WIDTHS)
    for flavor in FLAVORS:
        lower_with_both_engines(kernel, "gfx942", llvm_flavor=flavor)


# --------------------------------------------------------------------------
# HIP source
# --------------------------------------------------------------------------


@pytest.mark.parametrize("width", WIDTHS, ids=_id)
def test_hip_uses_the_builtin_of_its_width(width):
    dtype, n = width
    kernel, _ = _copy_kernel("buffer_hip_width", [width])
    hip = lower_hip_with_both_engines(kernel, "gfx942")
    nbytes = _nbytes(dtype, n)
    _, _, builtin = FORMS[nbytes]
    assert f"= __builtin_amdgcn_raw_buffer_load{builtin}(" in hip
    assert f"__builtin_memcpy(&_bs, &bl{n}" in hip
    assert f", {nbytes}); __builtin_amdgcn_raw_buffer_store{builtin}(_bs, " in hip


def test_hip_scopes_the_store_temporary():
    # Two stores of one value in one block must not redeclare a temporary.
    b = IRBuilder("buffer_store_twice")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    value = b.buffer_load_vN(rsrc, zero, zero, I8, 4)
    b.buffer_store_vN(rsrc, zero, zero, value)
    b.buffer_store_vN(rsrc, b.const_i32(4), zero, value)
    b.ret()
    hip = lower_hip_with_both_engines(b.kernel, "gfx942")
    stores = [line for line in hip.splitlines() if "raw_buffer_store_b32" in line]
    assert len(stores) == 2
    assert all(line.strip().startswith("{ int _bs;") for line in stores), stores


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,n,error_type,message",
    [
        (F16, 3, ValueError, "buffer_load_vN: 3 x f16 is 6 bytes; a buffer access moves 1, 2, 4, 8 or 16 bytes"),
        (F32, 3, ValueError, "buffer_load_vN: 3 x f32 is 12 bytes; a buffer access moves 1, 2, 4, 8 or 16 bytes"),
        (F64, 3, ValueError, "buffer_load_vN: 3 x f64 is 24 bytes; a buffer access moves 1, 2, 4, 8 or 16 bytes"),
        (F16, 16, ValueError, "buffer_load_vN: 16 x f16 is 32 bytes; a buffer access moves 1, 2, 4, 8 or 16 bytes"),
        (F32, 0, ValueError, "buffer_load_vN: n must be at least 1, got 0"),
        (F32, 1.1, TypeError, "buffer_load_vN: n must be an int, got 1.1"),
        (F32, 4.0, TypeError, "buffer_load_vN: n must be an int, got 4.0"),
        (F32, True, TypeError, "buffer_load_vN: n must be an int, got True"),
        (I8, 536870913, ValueError, "buffer_load_vN: 536870913 x i8 is 536870913 bytes; a buffer access moves 1, 2, 4, 8 or 16 bytes"),
        (I1, 8, ValueError, "buffer_load_vN: i1 elements are 1 bits, not whole bytes; a buffer access moves whole bytes"),
        (VectorType(F16, 2), 1, TypeError, "buffer_load_vN: the element type must be a scalar, got vec<f16x2>"),
        (E8M0, 1, TypeError, "buffer_load_vN: e8m0 is an MMA scale encoding with no value form; move its bytes as u8"),
        (E5M3, 16, TypeError, "buffer_load_vN: e5m3 is an MMA scale encoding with no value form; move its bytes as u8"),
    ],
)  # fmt: skip
def test_load_builder_refuses(dtype, n, error_type, message):
    b = IRBuilder("buffer_load_refused")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    with pytest.raises(error_type) as error:
        b.buffer_load_vN(rsrc, zero, zero, dtype, n)
    assert str(error.value) == message


def test_store_builder_refuses_a_scalar_and_a_wrong_width():
    b = IRBuilder("buffer_store_refused")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    with pytest.raises(TypeError) as scalar:
        b.buffer_store_vN(rsrc, zero, zero, b.const_i32(1))
    assert str(scalar.value) == (
        "buffer_store_vN: the value must be a vector, got i32; "
        "a one-element vector is vector_splat(x, 1)"
    )
    wide = b.vector_splat(b.const_f32(1.0), 8)
    with pytest.raises(ValueError) as width:
        b.buffer_store_vN(rsrc, zero, zero, wide)
    assert str(width.value) == (
        "buffer_store_vN: 8 x f32 is 32 bytes; a buffer access moves 1, 2, 4, 8 "
        "or 16 bytes"
    )


def _raw_load_kernel(result_type):
    """A load the builder would refuse, written as a raw op, as serialized IR
    can reach the lowerers without passing through the builder."""
    b = IRBuilder("buffer_load_raw")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    b._op(
        "tile.buffer_load_vN",
        [rsrc, zero, zero],
        [result_type],
        attrs={"dwords": 3, "elem_type": "f32"},
    )
    b.ret()
    return b.kernel


def _raw_store_kernel(value_type_count):
    b = IRBuilder("buffer_store_raw")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    value = b.vector_splat(b.const_f32(1.0), value_type_count)
    b._op("tile.buffer_store_vN", [rsrc, zero, zero, value])
    b.ret()
    return b.kernel


def test_both_lowerings_refuse_a_load_of_a_wrong_width():
    kernel = _raw_load_kernel(VectorType(F32, 3))
    message = (
        "tile.buffer_load_vN: 3 x f32 is 12 bytes; a buffer access moves 1, 2, 4, "
        "8 or 16 bytes"
    )
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, message)


def test_both_lowerings_refuse_a_store_of_a_wrong_width():
    kernel = _raw_store_kernel(8)
    message = (
        "tile.buffer_store_vN: 8 x f32 is 32 bytes; a buffer access moves 1, 2, 4, "
        "8 or 16 bytes"
    )
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, message)


@pytest.mark.parametrize("scale", [E8M0, E5M3], ids=lambda t: t.name)
def test_every_check_refuses_a_scale_encoding_access(scale):
    """e8m0 and e5m3 are whole bytes but have no LLVM or HIP value type, so a
    load of them, written past the builder, is refused by the verifier and by
    every lowering of both engines rather than failing on the missing type."""
    kernel = _raw_load_kernel(VectorType(scale, 4))
    message = (
        f"tile.buffer_load_vN: {scale.name} is an MMA scale encoding with no value "
        "form; move its bytes as u8"
    )
    assert_both_engines_reject(kernel, "gfx942", TypeError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", TypeError, message)
    assert message in [d.message for d in verify(kernel)]


def _raw_buffer_op_kernel(name, operand_count, result_types):
    """A buffer op ``name`` with the first ``operand_count`` of (rsrc, voffset,
    soffset, a 4-byte vector) and ``result_types``, written past the builder."""
    b = IRBuilder("buffer_op_raw")
    x = b.param("X", PtrType(I8, "global"))
    rsrc = b.buffer_rsrc(x, b.const_i32(64))
    zero = b.const_i32(0)
    operands = [rsrc, zero, zero, b.vector_splat(b.const_f32(1.0), 1)]
    b._op(name, operands[:operand_count], result_types)
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "name,operand_count,result_types,message",
    [
        ("tile.buffer_load_vN", 2, [VectorType(F32, 1)], "tile.buffer_load_vN: expected 3 operands / 1 result, got 2 / 1"),
        ("tile.buffer_load_vN", 3, [], "tile.buffer_load_vN: expected 3 operands / 1 result, got 3 / 0"),
        ("tile.buffer_store_vN", 3, [], "tile.buffer_store_vN: expected 4 operands / 0 results, got 3 / 0"),
        ("tile.buffer_store_vN", 4, [VectorType(F32, 1)], "tile.buffer_store_vN: expected 4 operands / 0 results, got 4 / 1"),
    ],
)  # fmt: skip
def test_both_lowerings_and_the_verifier_refuse_a_wrong_operand_count(
    name, operand_count, result_types, message
):
    """The counts are checked before any operand is read, so a short op is a
    refusal, not a crash, in every lowering of both engines."""
    kernel = _raw_buffer_op_kernel(name, operand_count, result_types)
    assert_both_engines_reject(kernel, "gfx942", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx942", ValueError, message)
    assert message in [d.message for d in verify(kernel)]


def test_verifier_refuses_a_load_and_a_store_of_a_wrong_width():
    load = _raw_load_kernel(VectorType(F32, 3))
    assert (
        "tile.buffer_load_vN: 3 x f32 is 12 bytes; a buffer access moves 1, 2, 4, "
        "8 or 16 bytes"
    ) in [d.message for d in verify(load)]
    store = _raw_store_kernel(8)
    assert (
        "tile.buffer_store_vN: 8 x f32 is 32 bytes; a buffer access moves 1, 2, 4, "
        "8 or 16 bytes"
    ) in [d.message for d in verify(store)]


# --------------------------------------------------------------------------
# GPU: a bit-exact copy through both compile routes
# --------------------------------------------------------------------------


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run_copy(blob, kernel_name, source, lanes):
    """Launch one workgroup of ``lanes`` lanes copying ``source``; return Y."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    out = np.full(source.size + 64, 0xA5, dtype=np.uint8)
    src = np.ascontiguousarray(source).copy()
    x = runtime.alloc(src.nbytes)
    y = runtime.alloc(out.nbytes)
    try:
        runtime.memcpy_h2d(
            x, (ctypes.c_ubyte * src.nbytes).from_buffer(src), src.nbytes
        )
        runtime.memcpy_h2d(
            y, (ctypes.c_ubyte * out.nbytes).from_buffer(out), out.nbytes
        )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (lanes, 1, 1),
            struct.pack("2Q", x, y),
        )
        runtime.memcpy_d2h(
            (ctypes.c_ubyte * out.nbytes).from_buffer(out), y, out.nbytes
        )
    finally:
        runtime.free(x)
        runtime.free(y)
        module.unload()
    return out


# The widths the GPU copy covers: i8 / fp8 / f16 / bf16 / f32 / i32 at 1, 2
# and 4 dwords, and the 1-, 2- and 8-byte-element forms.
GPU_WIDTHS = (
    (I8, 4), (I8, 8), (I8, 16),
    (FP8E4M3, 4), (FP8E4M3, 8), (FP8E4M3, 16),
    (F16, 2), (F16, 4), (F16, 8),
    (BF16, 2), (BF16, 4), (BF16, 8),
    (F32, 1), (F32, 2), (F32, 4),
    (I32, 1), (I32, 2), (I32, 4),
    (I8, 1), (I8, 2), (BF8E5M2, 1), (F16, 1), (BF16, 1), (I16, 1),
    (F64, 1), (F64, 2),
)  # fmt: skip


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_copy_is_bit_exact(route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    lanes = 64
    kernel, total = _copy_kernel(f"buffer_copy_{route}", GPU_WIDTHS, lanes=lanes)
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        if shutil.which("hipcc") is None:
            pytest.skip("hipcc not found")
        from rocke.helpers.compile import compile_kernel_via_hipcc

        blob = compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    source = np.random.default_rng(7).integers(0, 256, total, dtype=np.uint8)
    out = _run_copy(blob, kernel.name, source, lanes)
    np.testing.assert_array_equal(out[:total], source)
    # Bytes past the descriptor's range are never written.
    assert (out[total:] == 0xA5).all()


def test_hip_source_of_every_width_compiles(tmp_path):
    """The HIP lowering of every width is accepted by hipcc (no GPU needed)."""
    import subprocess

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    kernel, _ = _copy_kernel("buffer_every_width_hip", WIDTHS)
    source = tmp_path / "buffer_every_width.hip"
    source.write_text(lower_kernel_to_hip(kernel, arch="gfx942"))
    subprocess.run(
        [hipcc, "--offload-arch=gfx942", "--genco", "-O3", str(source), "-o",
         str(tmp_path / "out.hsaco")],
        check=True,
        capture_output=True,
        timeout=240,
    )  # fmt: skip
