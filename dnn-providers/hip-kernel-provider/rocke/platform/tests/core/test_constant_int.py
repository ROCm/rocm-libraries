# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``constant_int``: the compile-time integer a builder value or a Python int is.

Builder-level code (the tiling verbs' packed-run checks, the folder) asks this one
function whether a window origin, a bound or an operand is known at compile time.
"""

import pytest
from rocke.core.ir import IRBuilder
from rocke.core.passes import constant_int


def test_a_python_int_is_itself():
    assert constant_int(48) == 48
    assert constant_int(-3) == -3


def test_an_integer_constant_is_read_from_its_defining_op():
    b = IRBuilder("integer_constants")
    assert constant_int(b.const_i32(16)) == 16
    assert constant_int(b.const_i64(1 << 40)) == 1 << 40


def test_an_i32_constant_is_read_at_its_width():
    # The folder holds an i32 as the register does: these bits are -2**31.
    assert constant_int(IRBuilder("wrapped").const_i32(0x80000000)) == -(2**31)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda b: b.const_f32(16.0), id="float constant"),
        pytest.param(lambda b: b.thread_id_x(), id="runtime value"),
        pytest.param(lambda b: True, id="bool"),
    ],
)
def test_anything_else_is_not_a_compile_time_int(build):
    assert constant_int(build(IRBuilder("not_constant"))) is None
