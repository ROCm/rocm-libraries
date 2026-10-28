# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Two values never get the same name.

The builder names every value it makes ``%<hint>_<n>``: the hint the op chose,
then ``_``, then a counter that never repeats inside one builder. Before the
``_`` was there, a hint that ends in digits ran into the counter: the
16-element zero vector made as value 5 (hint ``cz16``) and the 1-element zero
vector made as value 65 (hint ``cz1``) were both ``%cz165``, and the lowered
module defined that name twice.

Because builder names always end in ``_`` and digits, a name the author picks
(a kernel parameter, a loop induction variable, a loop-carried value) is
refused when it ends that way, or when it is not an identifier. A hint that is
empty or starts with a digit is refused too.

These tests cover the Python builder and check that both engines lower the
kernel to the same LLVM IR. tests/core/test_value_names.cpp checks the C++
builder, with the same names and the same messages.
"""

from __future__ import annotations

import os
import re

import pytest
from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCH = "gfx90a"


def _engine():
    """The C++ engine binding; a strict both-engine run fails without it."""
    try:
        import rocke_engine
    except ImportError as exc:
        if os.environ.get("ROCKE_CPP_STRICT") == "1":
            raise
        pytest.skip(f"rocke_engine extension not built: {exc}")
    return rocke_engine


def _not_identifier(what: str, name: str) -> str:
    return (
        f"{what} name '{name}' is not an identifier: use letters, digits and '_', "
        "not starting with a digit"
    )


def _reserved(what: str, name: str) -> str:
    return (
        f"{what} name '{name}' ends in '_' and digits, the form the builder gives "
        "the values it names; choose a name that does not end in '_<digits>'"
    )


def _hint(hint: str) -> str:
    return f"value name hint '{hint}' must be non-empty and must not start with a digit"


def _bounds(b: IRBuilder):
    return b.const_i32(0), b.const_i32(2), b.const_i32(1)


def _two_zero_vectors() -> IRBuilder:
    """A kernel whose 16-element zero vector is value 5 and whose 1-element
    zero vector is value 65: the pair whose names used to be equal."""
    b = IRBuilder("value_names")
    b.param("Out", PtrType(F32, "global"))
    for _ in range(4):
        b.const_i32(0)
    b.zero_vec_f32(16)
    for _ in range(59):
        b.const_i32(0)
    b.zero_vec_f32(1)
    b.ret()
    return b


def test_a_hint_ending_in_digits_does_not_run_into_the_counter():
    body = _two_zero_vectors().kernel.body.ops
    names = [op.results[0].name for op in body if op.name == "arith.constant_vec"]
    assert names == ["%cz16_5", "%cz1_65"]


def test_both_engines_define_each_name_once():
    kernel = _two_zero_vectors().kernel
    python_llvm = _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor="llvm20")
    cpp_llvm = _engine().lower_serialized_ir(
        serialize(kernel), arch=_ARCH, flavor="llvm20"
    )
    assert cpp_llvm == python_llvm
    defined = re.findall(r"^\s*(%[\w.]+) =", python_llvm, flags=re.MULTILINE)
    assert len(defined) == len(set(defined))
    assert "%cz16_5" in defined and "%cz1_65" in defined


@pytest.mark.parametrize("name", ["stride_0", "x_12"])
def test_a_parameter_name_ending_in_underscore_digits_is_refused(name):
    b = IRBuilder("param_name")
    with pytest.raises(
        ValueError, match=re.escape(_reserved("kernel parameter", name))
    ):
        b.param(name, I32)


@pytest.mark.parametrize("name", ["", "0x", "a.b", "a-b"])
def test_a_parameter_name_that_is_not_an_identifier_is_refused(name):
    b = IRBuilder("param_name")
    with pytest.raises(
        ValueError, match=re.escape(_not_identifier("kernel parameter", name))
    ):
        b.param(name, I32)


@pytest.mark.parametrize("name", ["stride0", "x_y", "_", "acc_m0_n1"])
def test_other_parameter_names_are_accepted(name):
    b = IRBuilder("param_name")
    assert b.param(name, I32).name == f"%{name}"


def test_an_induction_variable_name_ending_in_underscore_digits_is_refused():
    b = IRBuilder("iv_name")
    what = "loop induction variable"
    with pytest.raises(ValueError, match=re.escape(_reserved(what, "k_0"))):
        b.scf_for(*_bounds(b), iv_name="k_0")
    with pytest.raises(ValueError, match=re.escape(_reserved(what, "k_0"))):
        b.scf_for_iter(*_bounds(b), [("acc", b.const_i32(0))], iv_name="k_0")


def test_a_loop_carried_value_name_ending_in_underscore_digits_is_refused():
    b = IRBuilder("iter_arg_name")
    zero = b.const_i32(0)
    what = "loop-carried value"
    with pytest.raises(ValueError, match=re.escape(_reserved(what, "acc_1"))):
        b.scf_for_iter(*_bounds(b), [("acc", zero), ("acc_1", zero)])
    with pytest.raises(ValueError, match=re.escape(_not_identifier(what, "acc.1"))):
        b.scf_for_iter(*_bounds(b), [("acc.1", zero)])


@pytest.mark.parametrize("hint", ["", "16x"])
def test_an_empty_or_digit_led_hint_is_refused(hint):
    b = IRBuilder("hint")
    with pytest.raises(ValueError, match=re.escape(_hint(hint))):
        b._op("arith.constant", result_types=[I32], result_name_hint=hint)
