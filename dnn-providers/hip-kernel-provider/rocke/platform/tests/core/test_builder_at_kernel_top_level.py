# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``IRBuilder.at_kernel_top_level``: whether the builder emits into the kernel body itself.

It is True at the start of a kernel and again after every loop and ``if`` region has been left,
and False inside a loop body, an ``if`` body or either branch of an ``if``/``else``, nested or
not. A helper that must emit at the kernel's top level (an early return of the whole workgroup)
asks this instead of reading the builder's region stack.
"""

from __future__ import annotations

from rocke.core.ir import I32, IRBuilder


def _builder() -> IRBuilder:
    b = IRBuilder("top_level")
    b.param("n", I32)
    return b


def test_a_new_builder_is_at_the_kernel_top_level():
    assert _builder().at_kernel_top_level


def test_a_loop_body_is_not_the_top_level_and_leaving_it_returns_there():
    b = _builder()
    with b.scf_for(b.const_i32(0), b.get_param("n"), b.const_i32(1)):
        assert not b.at_kernel_top_level
    assert b.at_kernel_top_level


def test_a_loop_body_carrying_values_is_not_the_top_level():
    b = _builder()
    zero = b.const_i32(0)
    with b.scf_for_iter(zero, b.get_param("n"), b.const_i32(1), [("acc", zero)]) as (
        _,
        (acc,),
    ):
        assert not b.at_kernel_top_level
        b.scf_yield(acc)
    assert b.at_kernel_top_level


def test_an_if_body_is_not_the_top_level_and_leaving_it_returns_there():
    b = _builder()
    with b.scf_if(b.cmp_lt(b.get_param("n"), b.const_i32(4))):
        assert not b.at_kernel_top_level
    assert b.at_kernel_top_level


def test_both_branches_of_an_if_else_are_not_the_top_level():
    b = _builder()
    with b.scf_if_else(b.cmp_lt(b.get_param("n"), b.const_i32(4))) as (then, other):
        with then:
            assert not b.at_kernel_top_level
        with other:
            assert not b.at_kernel_top_level
    assert b.at_kernel_top_level


def test_an_if_nested_in_a_loop_is_not_the_top_level_at_any_depth():
    b = _builder()
    with b.scf_for(b.const_i32(0), b.get_param("n"), b.const_i32(1)) as i:
        with b.scf_if(b.cmp_lt(i, b.const_i32(4))):
            assert not b.at_kernel_top_level
        assert not b.at_kernel_top_level
    assert b.at_kernel_top_level
