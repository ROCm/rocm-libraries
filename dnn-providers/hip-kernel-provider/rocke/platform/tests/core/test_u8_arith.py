# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unsigned arithmetic, compares and conversions of U8.

U8 renders as LLVM i8, and LLVM integers carry no sign, so every op whose
result depends on signedness must emit its unsigned form for a u8 operand:
udiv / urem, icmp ult / ule / ugt / uge, no nsw flag on add / sub / mul, zext,
lshr, llvm.umax / llvm.umin, uitofp and fptoui. The ops that are signed by
definition (smax, smin, sext and their vector forms) refuse a u8 operand. The
C++ builders are reached through the ``rocke_ir_types`` test binary
(``ROCKE_IR_TYPES_TEST``), because the serialized-IR seam lowers kernels the
Python builder built. The numeric test runs on a visible gfx90a only, through
the active engine, so a strict both-engines run
(``ROCKE_BACKEND=both ROCKE_CPP_STRICT=1``) checks both engines on the GPU.
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from rocke.core.ir import F32, I8, I32, U8, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts

# Output buffer -> its element type, in kernel parameter order after the inputs.
_OUTPUTS = {
    "DIV": U8,
    "REM": U8,
    "LT": U8,
    "LE": U8,
    "GT": U8,
    "GE": U8,
    "MAX": U8,
    "MIN": U8,
    "SHR": U8,
    "ADD": U8,
    "SUB": U8,
    "MUL": U8,
    "TO_F32": F32,
    "WIDE": I32,
    "SAT": U8,
}

_ALIGN = {U8: 1, F32: 4, I32: 4}


def _u8_arith_kernel():
    """Per thread, one result of every u8 op on A[i] and B[i], and the saturating
    f32 -> u8 conversion of F[i]. Compares store 0 or 1; SHR shifts by B & 7.

    REM divides by C[i], not B[i]: LLVM's AMDGPU backend at -O0 merges a division
    and a remainder of the same i8 operands into one i16 divrem it cannot select,
    in either signedness, and the IR-validity check compiles at -O0."""
    b = IRBuilder("u8_arith")
    a_ptr = b.param("A", PtrType(U8, "global"))
    b_ptr = b.param("B", PtrType(U8, "global"))
    c_ptr = b.param("C", PtrType(U8, "global"))
    f_ptr = b.param("F", PtrType(F32, "global"))
    out = {name: b.param(name, PtrType(t, "global")) for name, t in _OUTPUTS.items()}
    tid = b.thread_id_x()
    x = b.global_load(a_ptr, tid, U8)
    y = b.global_load(b_ptr, tid, U8)
    z = b.global_load(c_ptr, tid, U8)
    f = b.global_load(f_ptr, tid, F32, align=4)
    seven = b.trunc(b.const_i32(7), U8)
    results = {
        "DIV": b.div(x, y),
        "REM": b.mod(x, z),
        "LT": b.zext(b.cmp_lt(x, y), U8),
        "LE": b.zext(b.cmp_le(x, y), U8),
        "GT": b.zext(b.cmp_gt(x, y), U8),
        "GE": b.zext(b.cmp_ge(x, y), U8),
        "MAX": b.umax(x, y),
        "MIN": b.umin(x, y),
        "SHR": b.lshr(x, b.land(y, seven)),
        "ADD": b.add(x, y),
        "SUB": b.sub(x, y),
        "MUL": b.mul(x, y),
        "TO_F32": b.uitofp_f32(x),
        "WIDE": b.zext(x, I32),
        "SAT": b.cvt_f32_to_u8_sat(f),
    }
    for name, value in results.items():
        b.global_store(out[name], tid, value, align=_ALIGN[value.type])
    return b.kernel


def test_u8_ops_lower_to_their_unsigned_llvm_forms():
    text = lower_with_both_engines(_u8_arith_kernel(), "gfx90a")
    for form in (
        r"= udiv i8 ",
        r"= urem i8 ",
        r"= icmp ult i8 ",
        r"= icmp ule i8 ",
        r"= icmp ugt i8 ",
        r"= icmp uge i8 ",
        r"= call i8 @llvm\.umax\.i8\(i8 ",
        r"= call i8 @llvm\.umin\.i8\(i8 ",
        r"= lshr i8 ",
        r"= add i8 ",
        r"= sub i8 ",
        r"= mul i8 ",
        r"= uitofp i8 \S+ to float",
        r"= zext i8 \S+ to i32",
        r"= fptoui float \S+ to i8",
    ):
        assert len(re.findall(form, text)) == 1, form
    # No signed form of any i8 op, and no signed-overflow flag on one.
    assert not re.search(
        r"\b(sdiv|srem|ashr|sext|sitofp) i8 |icmp s\w+ i8 |nsw i8 ", text
    )
    assert "declare i8 @llvm.umax.i8(i8, i8)" in text
    assert "declare i8 @llvm.umin.i8(i8, i8)" in text
    assert_target_accepts(text, "gfx90a")


def test_u8_ops_lower_to_uint8_t_in_hip():
    hip = lower_hip_with_both_engines(_u8_arith_kernel(), "gfx90a")
    # Every u8 result is a uint8_t, so C's own operators are the unsigned ones.
    for hint in ("div", "mod", "umax", "umin", "add", "sub", "mul", "qu8"):
        assert re.search(rf"\buint8_t \w*{hint}\w* = ", hip), hint
    assert re.search(r"\bfloat \w*uitof\w* = \(float\)", hip)


def test_u8_hip_source_compiles(tmp_path):
    if not shutil.which("hipcc"):
        pytest.skip("hipcc not in PATH")
    path = tmp_path / "kernel.hip"
    path.write_text(lower_kernel_to_hip(_u8_arith_kernel(), arch="gfx90a"))
    subprocess.run(
        [
            "hipcc",
            "--offload-arch=gfx90a",
            "--cuda-device-only",
            "-fsyntax-only",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


# One builder call on two u8 parameters X and Y, its expected refusal (None when
# it builds one op), as the rows of k_unsigned_cases in test_ir_types.cpp.
_UNSIGNED_CASES = {
    "umax": (lambda b, x, y: b.umax(x, y), None),
    "umin": (lambda b, x, y: b.umin(x, y), None),
    "uitofp_f32": (lambda b, x, y: b.uitofp_f32(x), None),
    "zext": (lambda b, x, y: b.zext(x, I32), None),
    "smax": (
        lambda b, x, y: b.smax(x, y),
        (
            TypeError,
            "smax treats its operands as signed, but u8 is unsigned -- use umax",
        ),
    ),
    "smin": (
        lambda b, x, y: b.smin(x, y),
        (
            TypeError,
            "smin treats its operands as signed, but u8 is unsigned -- use umin",
        ),
    ),
    "sext": (
        lambda b, x, y: b.sext(x, I32),
        (
            TypeError,
            "sext treats its operands as signed, but u8 is unsigned -- use zext",
        ),
    ),
    "vector_smax": (
        lambda b, x, y: b.vector_smax(b.vector_splat(x, 4), b.vector_splat(y, 4)),
        (TypeError, "vector_smax treats its operands as signed, but u8 is unsigned"),
    ),
    "vector_smin": (
        lambda b, x, y: b.vector_smin(b.vector_splat(x, 4), b.vector_splat(y, 4)),
        (TypeError, "vector_smin treats its operands as signed, but u8 is unsigned"),
    ),
    "vector_sext": (
        lambda b, x, y: b.vector_sext(b.vector_splat(x, 4), I32),
        (TypeError, "vector_sext treats its operands as signed, but u8 is unsigned"),
    ),
    "umax_of_i32": (
        lambda b, x, y: b.umax(b.const_i32(1), b.const_i32(1)),
        (
            ValueError,
            "umax expects two operands of one unsigned integer type, got i32 and i32",
        ),
    ),
    "uitofp_f32_of_i32": (
        lambda b, x, y: b.uitofp_f32(b.const_i32(1)),
        (ValueError, "uitofp_f32 expects an unsigned integer input, got i32"),
    ),
    "cvt_f32_to_u8_sat_of_u8": (
        lambda b, x, y: b.cvt_f32_to_u8_sat(x),
        (ValueError, "cvt_f32_to_u8_sat expects f32 input, got u8"),
    ),
}


def _python_unsigned_case(name: str) -> str:
    """Run one case with the Python builder; ``ok`` or ``error: <text>``."""
    build, _ = _UNSIGNED_CASES[name]
    b = IRBuilder("unsigned_case")
    x = b.param("X", U8)
    y = b.param("Y", U8)
    try:
        build(b, x, y)
    except (TypeError, ValueError) as error:
        return f"error: {error}"
    return "ok"


@pytest.mark.parametrize("name", sorted(_UNSIGNED_CASES))
def test_unsigned_builders(name):
    build, refusal = _UNSIGNED_CASES[name]
    b = IRBuilder("unsigned_case")
    x = b.param("X", U8)
    y = b.param("Y", U8)
    if refusal is None:
        ops_before = len(b.kernel.body.ops)
        build(b, x, y)
        assert len(b.kernel.body.ops) == ops_before + 1
        return
    error_type, message = refusal
    with pytest.raises(error_type) as error:
        build(b, x, y)
    assert str(error.value) == message


@pytest.mark.parametrize("name", sorted(_UNSIGNED_CASES))
def test_native_unsigned_builders_match_the_python_builder(name):
    executable = os.environ.get("ROCKE_IR_TYPES_TEST")
    if not executable:
        pytest.skip("set ROCKE_IR_TYPES_TEST to the built rocke_ir_types test")
    native = subprocess.run(
        [executable, "--unsigned", name],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n")
    assert native == _python_unsigned_case(name)


@pytest.mark.parametrize("op_name", ["smax", "smin"])
def test_vector_binary_refuses_a_signed_op_on_u8(op_name):
    # vector_binary builds every vector.<op>, so it refuses a signed op itself
    # rather than leaving the refusal to the vector_smax / vector_smin wrappers.
    b = IRBuilder("vector_binary_case")
    v = b.vector_splat(b.param("X", U8), 4)
    with pytest.raises(TypeError) as error:
        b.vector_binary(op_name, v, v)
    assert str(error.value) == (
        f"vector_{op_name} treats its operands as signed, but u8 is unsigned"
    )


# Signed op -> (builds it on an i8 value x, the unsigned op it names, if any).
_SIGNED_OPS = {
    "arith.smax": (lambda b, x: b.smax(x, x), "arith.umax"),
    "arith.smin": (lambda b, x: b.smin(x, x), "arith.umin"),
    "arith.sext": (lambda b, x: b.sext(x, I32), "arith.zext"),
    "vector.smax": (
        lambda b, x: b.vector_smax(b.vector_splat(x, 4), b.vector_splat(x, 4)),
        None,
    ),
    "vector.smin": (
        lambda b, x: b.vector_smin(b.vector_splat(x, 4), b.vector_splat(x, 4)),
        None,
    ),
    "vector.sext": (lambda b, x: b.vector_sext(b.vector_splat(x, 4), I32), None),
}


@pytest.mark.parametrize("op_name", sorted(_SIGNED_OPS))
def test_lowerings_refuse_a_signed_op_on_u8(op_name):
    # The builders refuse the op on u8, so the kernel is built on i8 and its
    # serialized text retyped to u8: IR that did not come through a builder,
    # which each lowering must refuse with the builders' text under the op's
    # name. HIP has no u8 vector type, so the HIP lowerings refuse a u8 vector
    # before they reach a vector op.
    build, instead = _SIGNED_OPS[op_name]
    b = IRBuilder("signed_op")
    build(b, b.param("X", I8))
    b.ret()
    kernel = parse(serialize(b.kernel).replace("i8", "u8"))
    message = f"{op_name} treats its operands as signed, but u8 is unsigned"
    if instead is not None:
        message += f" -- use {instead}"
    assert_both_engines_reject(kernel, "gfx90a", TypeError, message)
    if op_name.startswith("vector."):
        assert_both_engines_reject_hip(
            kernel, "gfx90a", NotImplementedError, "no HIP mapping for type vec<u8x4>"
        )
    else:
        assert_both_engines_reject_hip(kernel, "gfx90a", TypeError, message)


# ---- on GPU (gfx90a) ---------------------------------------------------------

_LANES = 64


def _inputs():
    """A, B and C cover both halves of the u8 range (B and C are never 0, so DIV
    and REM are defined); F holds the rounding ties, the clamp edges, the infinities and
    a NaN, then ordinary values."""
    rng = np.random.default_rng(0)
    a = np.array(
        [0, 1, 2, 127, 128, 129, 200, 254, 255, 255, 128, 3, 250, 17, 255, 0]
        + list(rng.integers(0, 256, _LANES - 16)),
        dtype=np.uint8,
    )
    b = np.array(
        [1, 255, 2, 128, 127, 129, 3, 255, 1, 2, 255, 200, 7, 17, 254, 9]
        + list(rng.integers(1, 256, _LANES - 16)),
        dtype=np.uint8,
    )
    c = np.array(
        [255, 1, 128, 129, 127, 2, 200, 3, 254, 255, 5, 128, 251, 1, 255, 10]
        + list(rng.integers(1, 256, _LANES - 16)),
        dtype=np.uint8,
    )
    f_edges = [
        -3.5, -0.4, 0.5, 1.5, 2.5, 127.5, 128.5, 254.5,
        255.4, 255.6, 300.0, np.inf, -np.inf, np.nan, 0.0, -0.0,
    ]  # fmt: skip
    f = np.array(
        f_edges + list(rng.uniform(-20.0, 280.0, _LANES - len(f_edges))),
        dtype=np.float32,
    )
    return a, b, c, f


def _expected(a, b, c, f):
    """numpy's uint8 results, which wrap modulo 256 like the u8 ops."""
    with np.errstate(over="ignore", invalid="ignore"):
        saturated = np.nan_to_num(np.clip(np.rint(f), 0.0, 255.0), nan=0.0)
        return {
            "DIV": a // b,
            "REM": a % c,
            "LT": (a < b).astype(np.uint8),
            "LE": (a <= b).astype(np.uint8),
            "GT": (a > b).astype(np.uint8),
            "GE": (a >= b).astype(np.uint8),
            "MAX": np.maximum(a, b),
            "MIN": np.minimum(a, b),
            "SHR": a >> (b & np.uint8(7)),
            "ADD": a + b,
            "SUB": a - b,
            "MUL": a * b,
            "TO_F32": a.astype(np.float32),
            "WIDE": a.astype(np.int32),
            "SAT": saturated.astype(np.uint8),
        }


def test_u8_ops_on_gpu_match_numpy_uint8():
    a, b, c, f = _inputs()
    host_dtype = {U8: np.uint8, F32: np.float32, I32: np.int32}
    outputs = {
        name: np.zeros(_LANES, dtype=host_dtype[t]) for name, t in _OUTPUTS.items()
    }
    run_on_device(
        _u8_arith_kernel(),
        {"A": a, "B": b, "C": c, "F": f, **outputs},
        threads=_LANES,
        arches={"gfx90a"},
    )
    for name, expected in _expected(a, b, c, f).items():
        np.testing.assert_array_equal(
            outputs[name].view(np.uint8), expected.view(np.uint8), err_msg=name
        )
