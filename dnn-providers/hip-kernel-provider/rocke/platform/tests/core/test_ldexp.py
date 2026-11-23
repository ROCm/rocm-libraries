# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``ldexp`` op: ``value * 2^exponent``, rounded once.

``ldexp(value, exponent)`` takes an f32 value and an i32 exponent and lowers
to ``call float @llvm.ldexp.f32.i32`` (HIP ``__builtin_ldexpf``), one
``v_ldexp_f32``. The power of two is never formed as a float, so the exponent
may be any i32: the product is correct where ``value * exp2(exponent)`` would
overflow or flush the factor.

Covers the LLVM IR and HIP source of both engines (the same text at every LLVM
flavor), the refusals of the builder, the verifier and every lowering with
their exact texts, the instruction in the compiled code object, and, when a GPU
is present, on the LLVM and the hipcc route, the results for values and
exponents at and beyond the f32 range against numpy's ldexp.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.helpers.spec import sig_param
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)
from rocke.tests.gpu import current_device_arch, launch_with_buffers
from rocke.tests.ir_validity import target_mnemonics


def _ldexp_kernel(name):
    """Y[i] = ldexp(X[i], E[i]) for one lane per element."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 64
    x = b.param("X", PtrType(F32, "global"))
    e = b.param("E", PtrType(I32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    i = b.thread_id_x()
    b.global_store(
        y, i, b.ldexp(b.global_load_f32(x, i), b.global_load_i32(e, i)), align=4
    )
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------
# LLVM IR and HIP source
# --------------------------------------------------------------------------


def test_both_engines_emit_one_ldexp_call_and_its_declaration():
    kernel = _ldexp_kernel("ldexp_llvm")
    for flavor in LLVM_FLAVORS:
        text = lower_with_both_engines(kernel, "gfx90a", llvm_flavor=flavor)
        assert text.count(" = call float @llvm.ldexp.f32.i32(float ") == 1, flavor
        assert (
            text.count("declare float @llvm.ldexp.f32.i32(float, i32)\n") == 1
        ), flavor


def test_both_engines_emit_the_hip_ldexp_builtin():
    hip = lower_hip_with_both_engines(_ldexp_kernel("ldexp_hip"), "gfx90a")
    assert hip.count(" = __builtin_ldexpf(") == 1


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value_type, exponent_type",
    [(F32, F32), (F16, I32)],
    ids=["f32_exponent", "f16_value"],
)
def test_builder_refuses_operands_that_are_not_an_f32_value_and_an_i32_exponent(
    value_type, exponent_type
):
    b = IRBuilder("ldexp_bad")
    value = b.param("v", value_type)
    exponent = b.param("e", exponent_type)
    message = (
        "ldexp takes an f32 value and an i32 exponent and gives an f32, got "
        f"{value_type.name}, {exponent_type.name} -> f32"
    )
    with pytest.raises(ValueError, match=message):
        b.ldexp(value, exponent)


def _raw_ldexp_of_an_i32_kernel():
    """A math.ldexp of an i32 value, written past the builder (as serialized IR
    can reach the verifier and the lowerers)."""
    b = IRBuilder("ldexp_i32_value")
    y = b.param("Y", PtrType(F32, "global"))
    r = b._op("math.ldexp", [b.const_i32(1), b.const_i32(2)], [F32]).result
    b.global_store(y, b.const_i32(0), r)
    b.ret()
    return b.kernel


def _raw_ldexp_with_one_operand_kernel():
    """A math.ldexp with one operand, written past the builder."""
    b = IRBuilder("ldexp_one_operand")
    y = b.param("Y", PtrType(F32, "global"))
    r = b._op("math.ldexp", [b.const_f32(1.0)], [F32]).result
    b.global_store(y, b.const_i32(0), r)
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "make_kernel, message",
    [
        (
            _raw_ldexp_of_an_i32_kernel,
            (
                "ldexp takes an f32 value and an i32 exponent and gives an f32, got "
                "i32, i32 -> f32"
            ),
        ),
        (
            _raw_ldexp_with_one_operand_kernel,
            "ldexp takes two operands and gives one result",
        ),
    ],
    ids=["i32_value", "one_operand"],
)
def test_verifier_and_every_lowering_refuse_a_malformed_ldexp(make_kernel, message):
    kernel = make_kernel()
    assert message in [d.message for d in verify(kernel)]
    assert_both_engines_reject(kernel, "gfx90a", ValueError, message)
    assert_both_engines_reject_hip(kernel, "gfx90a", ValueError, message)


# --------------------------------------------------------------------------
# The instruction in the code object
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
def test_code_object_holds_one_v_ldexp_f32(arch):
    ir = lower_kernel_to_llvm(_ldexp_kernel(f"ldexp_isa_{arch}"), arch=arch)
    mnemonics = target_mnemonics(ir, arch)
    assert mnemonics.count("v_ldexp_f32") == 1, mnemonics


# --------------------------------------------------------------------------
# GPU: values at and beyond the f32 range
# --------------------------------------------------------------------------

# (value, exponent) pairs: ordinary products, products that overflow to an
# infinity of the value's sign, products far below the f32 range that round to
# a zero of the value's sign, exponents a float factor 2^e could not hold
# (2^200 * 2^-190), and NaN, infinities and zeros, which pass through.
_CASES = [
    (1.5, 3),
    (-1.5, -3),
    (3.0, 127),
    (1.0, 128),
    (-1.0, 128),
    (2.0**-100, 200),
    (2.0**100, -190),
    (1.0, -200),
    (-1.0, -200),
    (1.0, 2**30),
    (-1.0, 2**30),
    (1.0, -(2**30)),
    (float("nan"), 5),
    (float("inf"), -300),
    (float("-inf"), 7),
    (0.0, 300),
    (-0.0, -300),
]


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_results_equal_numpy_ldexp(route):
    arch = current_device_arch()
    if route == "hipcc" and shutil.which("hipcc") is None:
        pytest.skip("hipcc not found")
    kernel = _ldexp_kernel(f"ldexp_gpu_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        artifact = compile_kernel(kernel, arch=arch)
    else:
        from rocke.helpers.compile import compile_kernel_via_hipcc

        artifact = compile_kernel_via_hipcc(kernel, arch=arch)
    x = np.array([v for v, _ in _CASES], dtype=np.float32)
    e = np.array([k for _, k in _CASES], dtype=np.int32)
    with np.errstate(over="ignore", under="ignore"):
        want = np.ldexp(x, e)
    got = np.zeros(len(_CASES), dtype=np.float32)
    launch_with_buffers(
        artifact,
        [sig_param("X", "f32"), sig_param("E", "i32"), sig_param("Y", "f32")],
        {"X": x, "E": e, "Y": got},
        grid=(1, 1, 1),
        block=(len(_CASES), 1, 1),
    )
    # NaN payloads may differ; every other result must match bit for bit,
    # which also checks the sign of each infinity and zero.
    nan = np.isnan(want)
    assert np.array_equal(np.isnan(got), nan), list(zip(_CASES, got.tolist()))
    assert np.array_equal(got[~nan].view(np.uint32), want[~nan].view(np.uint32)), list(
        zip(_CASES, got.tolist(), want.tolist())
    )
