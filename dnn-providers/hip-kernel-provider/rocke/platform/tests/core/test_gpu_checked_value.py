# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``IRBuilder.gpu_checked_value``: facts on a value only the GPU can see,
checked on the GPU.

What is pinned here:

- what the check emits (a compare per stated side, one unsigned compare for
  ``0 <= v < n``, ``scf.if`` on the failure, ``tile.trap``) and the range marker
  it returns, read back by ``stated_facts_of``;
- every refusal, with its exact text: a value the host already checks (a
  parameter, a value computed from parameters, constants or thread ids),
  arithmetic on an unchecked load before the check, and a malformed call;
- that a load reaching the value only through a condition is accepted, and that
  the walks see through loops, branches and selects;
- the range marker's forms (a ``ge`` operand, the ``by`` attr) and the
  verifier's texts for a bad one;
- that both engines lower a checked kernel to the same bytes, and that
  ``optimize_kernel`` keeps the check;
- through ``ROCKE_GPU_CHECKED_VALUE_CLI`` (the ``rocke_gpu_checked_value``
  ctest), that the C++ builder ``rocke_b_gpu_checked`` emits the same IR and
  refuses with the same texts for every case in ``CASES``, and that the C++
  verifier gives the same messages for every marker in ``MARKERS``;
- when a GPU is present, on the LLVM and the hipcc route, each in a child
  process: a launch whose loaded value meets the facts runs to its store, and
  one whose value breaks them takes the trap, so the launch fails, the store
  never happens and that process cannot use the GPU afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest

from rocke.core.facts import StatedFacts, stated_facts_of
from rocke.core.intexpr import product_of
from rocke.core.ir import (
    I32,
    I64,
    IRBuilder,
    PtrType,
    Value,
    one_unsigned_compare_checks,
)
from rocke.core.ir_print import print_ir
from rocke.core.ir_serialize import canonicalize, parse, serialize
from rocke.core.passes import optimize_kernel
from rocke.core.verify import verify
from rocke.tests.engines import lower_hip_with_both_engines, lower_with_both_engines

INT64_MAX = (1 << 63) - 1
INT32_MAX = (1 << 31) - 1


class Params(NamedTuple):
    T: Value
    M: Value
    N64: Value
    K: Value
    cu: Value


def _declare(b: IRBuilder) -> Params:
    """The parameters every case starts from (``declare`` in the C++ test)."""
    return Params(
        b.param("T", I32, ge=0, le=4096),
        b.param("M", I32, ge=1, le=1 << 20, multiple_of=64),
        b.param("N64", I64, ge=1),
        b.param("K", I32),
        b.param("cu", PtrType(I32, "global"), readonly=True),
    )


def _loaded(b: IRBuilder, p: Params, index: Value) -> Value:
    """One wave-uniform i32 loaded from ``cu``: unchecked, GPU-only."""
    return b.readfirstlane(b.global_load(p.cu, index, I32))


def _loop(b: IRBuilder, p: Params, *, through_data: bool) -> Value:
    """A loop over eight loads. ``through_data=False`` is the binary-search
    shape: ``lo += load < T``, so the loads reach the result only through a
    select's condition. ``through_data=True`` sums the loads."""
    lo, hi, step = b.const_i32(0), b.const_i32(8), b.const_i32(1)
    loop = b.scf_for_iter(lo, hi, step, [("lo", b.const_i32(0))])
    with loop as (k, (acc,)):
        x = b.global_load(p.cu, k, I32)
        if through_data:
            nxt = b.add(acc, x)
        else:
            below = b.cmp_lt(x, p.T)
            nxt = b.select(below, b.add(acc, b.const_i32(1)), acc)
        b.scf_yield(nxt)
    return loop.results[0]


def _varlen_prologue(b, p):
    s = b.block_id_x()
    start = b.gpu_checked_value(_loaded(b, p, s), ge=0, le=p.T)
    end = b.gpu_checked_value(
        _loaded(b, p, b.add(s, b.const_i32(1))), ge=0, lt=4096, multiple_of=4
    )
    b.gpu_checked_value(b.sub(end, start), ge=0, le=p.T)


def _every_end_form(b, p):
    s = b.block_id_x()
    x = _loaded(b, p, s)
    y = b.gpu_checked_value(x, gt=p.T, multiple_of=8)
    b.gpu_checked_value(_loaded(b, p, y), lt=y)
    b.gpu_checked_value(_loaded(b, p, s), ge=-5, lt=100)
    b.gpu_checked_value(_loaded(b, p, s), gt=-1, le=99)
    wide = b.sext(s, I64)
    b.gpu_checked_value(wide, ge=0, le=p.N64)
    b.gpu_checked_value(wide, ge=0, le=INT64_MAX)
    b.gpu_checked_value(wide, ge=0, lt=1000)


def _branch_yields_a_load(b, p):
    s = b.block_id_x()
    branch = b.scf_if_else(b.cmp_lt(s, p.T), result_types=[I32])
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(_loaded(b, p, s))
        with else_ctx:
            b.scf_yield(b.const_i32(0))
    b.gpu_checked_value(branch.results[0], ge=0)


def _subtraction_on_a_load(b, p):
    raw = _loaded(b, p, b.block_id_x())
    start = b.gpu_checked_value(raw, ge=0, le=p.T)
    b.gpu_checked_value(b.sub(raw, start), ge=0)


def _select_of_a_load(b, p):
    s = b.block_id_x()
    below = b.cmp_lt(s, p.T)
    x = _loaded(b, p, s)
    picked = b.select(below, x, b.const_i32(0))
    b.gpu_checked_value(b.mul(picked, b.const_i32(4)), ge=0)


def _on_a_load(**facts):
    return lambda b, p: b.gpu_checked_value(_loaded(b, p, b.block_id_x()), **facts)


# The cases the C++ test builds with the C API, in its order. A name that starts
# with "refuse_" must be refused.
CASES = {
    "varlen_prologue": _varlen_prologue,
    "every_end_form": _every_end_form,
    "condition_only_load": lambda b, p: b.gpu_checked_value(
        _loop(b, p, through_data=False), ge=0, lt=p.T
    ),
    "branch_yields_a_load": _branch_yields_a_load,
    "param_value_bound": lambda b, p: b.param("ldc", I32, ge=p.T),
    "refuse_param_bound_computed": lambda b, p: b.param("X", I32, ge=b.add(p.T, p.T)),
    "refuse_parameter": lambda b, p: b.gpu_checked_value(p.M, le=4096),
    "refuse_parameter_two_facts": lambda b, p: b.gpu_checked_value(
        p.K, ge=0, le=p.T, multiple_of=8
    ),
    "refuse_computed_from_one_parameter": lambda b, p: b.gpu_checked_value(
        b.sub(p.T, b.const_i32(1)), le=4095
    ),
    "refuse_computed_from_parameters": lambda b, p: b.gpu_checked_value(
        b.add(p.T, p.M), le=4095
    ),
    "refuse_thread_id": lambda b, p: b.gpu_checked_value(b.thread_id_x(), lt=64),
    "refuse_thread_id_and_parameter": lambda b, p: b.gpu_checked_value(
        b.add(b.thread_id_x(), p.T), lt=64
    ),
    "refuse_constants": lambda b, p: b.gpu_checked_value(
        b.add(b.const_i32(2), b.const_i32(3)), lt=8
    ),
    "refuse_subtraction_on_a_load": _subtraction_on_a_load,
    "refuse_loop_sum_of_loads": lambda b, p: b.gpu_checked_value(
        _loop(b, p, through_data=True), ge=0
    ),
    "refuse_select_of_a_load": _select_of_a_load,
    "refuse_no_fact": _on_a_load(),
    "refuse_two_lower_bounds": _on_a_load(ge=0, gt=-1),
    "refuse_empty_range": _on_a_load(ge=5, lt=5),
    "refuse_gt_at_the_maximum": _on_a_load(gt=INT32_MAX),
    "refuse_bound_without_facts": lambda b, p: b.gpu_checked_value(
        _loaded(b, p, b.block_id_x()), le=p.K
    ),
    "refuse_bound_of_another_type": lambda b, p: b.gpu_checked_value(
        _loaded(b, p, b.block_id_x()), le=p.N64
    ),
    "refuse_multiple_of_zero": _on_a_load(multiple_of=0),
    "refuse_bound_outside_the_type": _on_a_load(le=1 << 31),
    "refuse_an_i1_value": lambda b, p: b.gpu_checked_value(
        b.cmp_lt(b.block_id_x(), p.T), ge=0
    ),
}


def _build(name: str) -> IRBuilder:
    b = IRBuilder(name)
    CASES[name](b, _declare(b))
    b.ret()
    return b


def _report(name: str) -> str:
    """What the C++ program prints for one case with ``--cases``."""
    try:
        return print_ir(_build(name).kernel)
    except ValueError as error:
        return f"refused: {error}"


def _refused_text(name: str) -> str:
    with pytest.raises(ValueError) as refused:
        _build(name)
    return str(refused.value)


ACCEPTED = [name for name in CASES if not name.startswith("refuse_")]
REFUSED = [name for name in CASES if name.startswith("refuse_")]


@pytest.mark.parametrize("name", ACCEPTED)
def test_accepted_cases_verify_clean(name):
    kernel = _build(name).kernel
    assert verify(kernel) == []
    round_tripped = parse(serialize(kernel))
    assert verify(round_tripped) == []
    assert canonicalize(round_tripped) == canonicalize(kernel)


@pytest.mark.parametrize("name", REFUSED)
def test_refused_cases_are_refused(name):
    _refused_text(name)


# --------------------------------------------------------------------------
# what it emits and what it states
# --------------------------------------------------------------------------


def _ops(kernel):
    return [op.name for op in kernel.body.ops]


def test_each_stated_side_takes_one_compare_and_a_trap():
    b = IRBuilder("sides")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    start = b.gpu_checked_value(raw, ge=0, le=p.T)
    ops = [op for op in b.kernel.body.ops]
    *_, low, high, either, branch, marker = ops
    assert (low.name, low.attrs["pred"], low.operands) == (
        "arith.cmp",
        "lt",
        [raw, ops[3].result],
    )
    assert (high.name, high.attrs["pred"], high.operands) == (
        "arith.cmp",
        "gt",
        [raw, p.T],
    )
    assert either.name == "arith.or"
    assert branch.name == "scf.if" and branch.operands == [either.result]
    assert [op.name for op in branch.regions[0].ops] == ["tile.trap"]
    assert marker.result is start
    assert marker.attrs["by"] == "gpu_checked_value"


def test_zero_to_an_int_end_is_one_unsigned_compare():
    b = IRBuilder("one_compare")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    b.gpu_checked_value(raw, ge=0, lt=100)
    compare = [op for op in b.kernel.body.ops if op.name == "arith.cmp"]
    assert len(compare) == 1
    (ult,) = compare
    assert ult.attrs["pred"] == "ult"
    assert ult.operands[0].op.attrs["value"] == 99 and ult.operands[1] is raw


def test_only_the_stated_sides_are_checked():
    b = IRBuilder("one_side")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    b.gpu_checked_value(raw, multiple_of=4)
    assert [op.name for op in b.kernel.body.ops][-6:] == [
        "arith.constant",
        "arith.mod",
        "arith.constant",
        "arith.cmp",
        "scf.if",
        "arith.assume_range",
    ]


def test_stated_facts_of_reads_the_checked_value():
    b = IRBuilder("facts")
    p = _declare(b)
    s = b.block_id_x()
    start = b.gpu_checked_value(_loaded(b, p, s), ge=0, le=p.T)
    end = b.gpu_checked_value(_loaded(b, p, s), gt=p.T, lt=1 << 20, multiple_of=8)
    seqlen = b.gpu_checked_value(b.sub(end, start), ge=0, lt=p.T)
    assert stated_facts_of(start) == StatedFacts(("ge", "lt"), 0, 4096, 1, 0)
    # gt=T resolves through T's own lower bound: end > T >= 0.
    assert stated_facts_of(end) == StatedFacts(
        ("ge", "lt", "multiple_of"), 1, (1 << 20) - 1, 8, 0
    )
    assert stated_facts_of(seqlen) == StatedFacts(("ge", "lt"), 0, 4095, 1, 0)


def test_an_end_at_the_type_maximum_states_no_upper_end():
    b = IRBuilder("type_max")
    wide = b.sext(b.block_id_x(), I64)
    checked = b.gpu_checked_value(wide, ge=0, le=INT64_MAX)
    assert "lt" not in checked.op.attrs
    assert stated_facts_of(checked) == StatedFacts(("ge",), 0, None, 1, 0)


def test_optimize_kernel_keeps_the_check():
    kernel = _build("varlen_prologue").kernel
    optimize_kernel(kernel)
    traps = [
        inner.name
        for op in kernel.body.ops
        if op.name == "scf.if"
        for inner in op.regions[0].ops
    ]
    assert traps == ["tile.trap"] * 3


@pytest.mark.parametrize("name", ACCEPTED)
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22"])
def test_both_engines_lower_a_checked_kernel_the_same(name, flavor):
    ir = lower_with_both_engines(_build(name).kernel, "gfx942", llvm_flavor=flavor)
    if name != "param_value_bound":
        assert "@llvm.trap()" in ir


@pytest.mark.parametrize("name", ACCEPTED)
def test_both_engines_lower_a_checked_kernel_to_the_same_hip(name):
    lower_hip_with_both_engines(_build(name).kernel, "gfx942")


# --------------------------------------------------------------------------
# refusals, with the exact texts
# --------------------------------------------------------------------------


def test_a_parameter_is_checked_by_the_host():
    assert _refused_text("refuse_parameter") == (
        "gpu_checked_value: M is a kernel parameter. The host checks every fact of a "
        "kernel parameter before every launch and refuses the launch if one fails, so "
        "a GPU check here would only repeat that check.\n"
        "Fix: state the fact where M is declared:\n"
        '    b.param("M", I32, le=4096)\n'
        'M is declared today as: b.param("M", I32, ge=1, le=1048576, multiple_of=64)'
    )


def test_a_parameter_with_no_facts_gets_every_stated_word():
    assert _refused_text("refuse_parameter_two_facts").endswith(
        "Fix: state the facts where K is declared:\n"
        '    b.param("K", I32, ge=0, le=T, multiple_of=8)\n'
        'K is declared today as: b.param("K", I32)'
    )


def test_a_value_computed_from_parameters_is_checked_by_the_host():
    assert _refused_text("refuse_computed_from_one_parameter") == (
        "gpu_checked_value: T - 1 is computed only from kernel parameters (T), which "
        "the host checks before every launch. Fix: state the bound on the parameter: "
        'b.param("T", I32, ...). The IR range walk derives the facts of T - 1 from it.'
    )
    assert _refused_text("refuse_computed_from_parameters") == (
        "gpu_checked_value: T + M is computed only from kernel parameters (T, M), which "
        "the host checks before every launch. Fix: state the bounds on the parameters: "
        'b.param("T", I32, ...), b.param("M", I32, ...). The IR range walk derives the '
        "facts of T + M from them."
    )


def test_a_thread_id_is_bounded_by_the_workgroup_size():
    assert _refused_text("refuse_thread_id") == (
        "gpu_checked_value: thread_id_x is bounded by the kernel's workgroup size, "
        "which the host checks.\n"
        'Fix: b.kernel.attrs["workgroup_size"] = 64'
    )
    assert _refused_text("refuse_thread_id_and_parameter") == (
        "gpu_checked_value: thread_id_x + T is computed only from thread ids and kernel "
        "parameters (T), which the host bounds before every launch.\n"
        'Fix: state the workgroup size, b.kernel.attrs["workgroup_size"] = '
        "<threads per workgroup>, and the parameters' facts on their declarations; the "
        "IR range walk derives the facts of thread_id_x + T from them."
    )


def test_a_constant_has_nothing_to_check():
    assert _refused_text("refuse_constants") == (
        "gpu_checked_value: 2 + 3 is computed only from constants, so it is known when "
        "the kernel is built -- there is nothing to check on the GPU"
    )


def test_a_block_id_value_is_gpu_only():
    b = IRBuilder("split_k")
    k = b.param("K", I32, ge=1, le=1 << 20)
    k_len = b.sub(k, b.mul(b.block_id_z(), b.const_i32(256)))
    checked = b.gpu_checked_value(k_len, ge=1)
    assert stated_facts_of(checked).lower == 1


def test_arithmetic_on_an_unchecked_load_comes_too_late():
    assert _refused_text("refuse_subtraction_on_a_load") == (
        "gpu_checked_value: %ufm3 is loaded from memory and unchecked, and sub uses it "
        "before this check (the subtraction can wrap or become poison first). Fix: check "
        "%ufm3 with gpu_checked_value first, then subtract the Values gpu_checked_value "
        "returned."
    )


def test_the_taint_walk_follows_loops_and_selects():
    assert _refused_text("refuse_loop_sum_of_loads") == (
        "gpu_checked_value: %gl6 is loaded from memory and unchecked, and add uses it "
        "before this check (the addition can wrap or become poison first). Fix: check "
        "%gl6 with gpu_checked_value first, then add the Values gpu_checked_value "
        "returned."
    )
    assert _refused_text("refuse_select_of_a_load") == (
        "gpu_checked_value: %sel6 is computed from a value loaded from memory and "
        "unchecked, and mul uses it before this check (the multiplication can wrap or "
        "become poison first). Fix: check %sel6 with gpu_checked_value first, then "
        "multiply the Values gpu_checked_value returned."
    )


def test_a_load_that_reaches_only_a_condition_is_accepted():
    # The binary-search shape: the loads feed compares, the loop carries plain counts.
    kernel = _build("condition_only_load").kernel
    assert verify(kernel) == []


@pytest.mark.parametrize(
    "name,message",
    [
        (
            "refuse_no_fact",
            "gpu_checked_value: state at least one fact: ge, gt, le, lt or multiple_of",
        ),
        (
            "refuse_two_lower_bounds",
            "gpu_checked_value: states two lower bounds (gt, ge) -- state one",
        ),
        ("refuse_empty_range", "gpu_checked_value: ge=5 and lt=5 leave no i32 value"),
        (
            "refuse_gt_at_the_maximum",
            "gpu_checked_value: gt=2147483647 leaves no i32 value",
        ),
        (
            "refuse_bound_without_facts",
            "gpu_checked_value: le=K has no stated facts -- a Value bound is a kernel "
            "parameter whose declaration states facts, or a Value gpu_checked_value "
            "returned",
        ),
        (
            "refuse_bound_of_another_type",
            "gpu_checked_value: le=N64 is i64 and the value is i32 -- a bound has the "
            "value's type",
        ),
        (
            "refuse_multiple_of_zero",
            "gpu_checked_value: multiple_of must be >= 1, got 0",
        ),
        (
            "refuse_bound_outside_the_type",
            "gpu_checked_value: le=2147483648 is outside the range of i32",
        ),
        (
            "refuse_an_i1_value",
            "gpu_checked_value: the value is i1 -- a GPU check takes an i32 or i64 value",
        ),
        (
            "refuse_param_bound_computed",
            "param X: ge=add1 uses add1, which is computed in the kernel -- the host can "
            "only evaluate kernel parameters and ints",
        ),
    ],
)
def test_a_malformed_call_is_refused(name, message):
    assert _refused_text(name) == message


def test_a_host_expression_is_not_a_gpu_bound():
    b = IRBuilder("host_bound")
    p = _declare(b)
    with pytest.raises(ValueError) as refused:
        b.gpu_checked_value(_loaded(b, p, b.block_id_x()), le=product_of(p.M, p.T))
    assert str(refused.value) == (
        "gpu_checked_value: le=product_of(M, T) is a host expression -- on the GPU a "
        "bound is an int or one Value; compute it with b.add / b.mul, where the "
        "instruction can be seen"
    )


def test_a_misspelled_or_mistyped_word_is_a_type_error():
    b = IRBuilder("typed")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    with pytest.raises(TypeError):
        b.gpu_checked_value(raw, el=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        b.gpu_checked_value(raw, eq=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="le is an int or a Value"):
        b.gpu_checked_value(raw, le=4.0)
    with pytest.raises(TypeError, match="multiple_of is an int"):
        b.gpu_checked_value(raw, multiple_of=True)


def test_a_refused_call_emits_nothing():
    b = IRBuilder("nothing")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    before = len(b.kernel.body.ops)
    with pytest.raises(ValueError):
        b.gpu_checked_value(raw, ge=5, lt=5)
    assert len(b.kernel.body.ops) == before


# --------------------------------------------------------------------------
# the wrap-free compare rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "least,most,t,decides",
    [
        (0, 0, I32, True),
        (1, INT32_MAX, I32, True),
        (0, INT32_MAX + 1, I32, False),
        (-1, 10, I32, False),
        (0, INT64_MAX, I64, True),
        (0, INT32_MAX + 1, I64, True),
    ],
)
def test_one_unsigned_compare_decides_only_a_nonnegative_signed_end(
    least, most, t, decides
):
    assert one_unsigned_compare_checks(least, most, t) is decides


# --------------------------------------------------------------------------
# the range marker's forms
# --------------------------------------------------------------------------


def test_the_marker_takes_a_value_at_either_end():
    b = IRBuilder("ends")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    marked = b._assume_range(raw, ge=p.T, ge_plus=1, lt=p.M, lt_plus=-1)
    assert marked.op.operands == [raw, p.T, p.M]
    assert stated_facts_of(marked) == StatedFacts(("ge", "lt"), 1, (1 << 20) - 2, 1, 0)
    b.ret()
    assert verify(b.kernel) == []


def _marker_kernel(operands, attrs, result_type=I32):
    b = IRBuilder("marker")
    p = _declare(b)
    values = {"a": _loaded(b, p, b.block_id_x()), "T": p.T, "M": p.M, "N64": p.N64}
    b._op(
        "arith.assume_range",
        [values[name] for name in operands],
        [result_type],
        attrs={"multiple_of": 1, "residue": 0, **attrs},
    )
    b.ret()
    return b.kernel


# (operands, attrs, the verifier's messages)
MARKERS = {
    "ge_operand": (["a", "T"], {"ge_plus": 0}, []),
    "both_operands": (
        ["a", "T", "M"],
        {"ge_plus": 1, "lt_plus": 0, "by": "gpu_checked_value"},
        [],
    ),
    "ge_twice": (
        ["a", "T"],
        {"ge": 0, "ge_plus": 0},
        [
            "arith.assume_range: ge is given both as an attr and as an operand; "
            "state one lower end"
        ],
    ),
    "ge_plus_alone": (
        ["a"],
        {"ge_plus": 1},
        ["arith.assume_range: ge_plus is given without a ge operand"],
    ),
    "two_ends_without_ge_plus": (
        ["a", "T", "M"],
        {"lt_plus": 0},
        [
            "arith.assume_range: two Value ends need ge_plus (the operands are the "
            "value, the ge operand, then the lt operand)"
        ],
    ),
    "ge_plus_not_an_int": (
        ["a", "T"],
        {"ge_plus": "1"},
        ["arith.assume_range: ge_plus must be an integer"],
    ),
    "by_not_a_string": (["a"], {"by": 1}, ["arith.assume_range: by must be a string"]),
    "by_unknown": (
        ["a"],
        {"by": "someone"},
        [
            "arith.assume_range: by names no marker writer: 'someone' (one of: "
            "gpu_checked_value)"
        ],
    ),
    "ge_operand_type": (
        ["a", "N64"],
        {"ge_plus": 0},
        ["arith.assume_range: ge operand type i64 != operand type i32"],
    ),
    "four_operands": (
        ["a", "T", "M", "T"],
        {"ge_plus": 0, "lt_plus": 0},
        ["arith.assume_range: expected 1 to 3 operands / 1 result, got 4 / 1"],
    ),
}


@pytest.mark.parametrize("name", sorted(MARKERS))
def test_the_verifier_reads_every_marker_form(name):
    operands, attrs, messages = MARKERS[name]
    assert [d.message for d in verify(_marker_kernel(operands, attrs))] == messages


def test_the_marker_builder_refuses_with_the_verifier_text():
    b = IRBuilder("marker_builder")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    with pytest.raises(ValueError) as refused:
        b._assume_range(raw, ge=0, ge_plus=1)
    assert (
        str(refused.value)
        == "arith.assume_range: ge_plus is given without a ge operand"
    )
    with pytest.raises(ValueError) as refused:
        b._assume_range(raw, ge=p.N64)
    assert str(refused.value) == "_assume_range: ge type i64 != v type i32"
    with pytest.raises(ValueError) as refused:
        b._assume_range(raw, by="someone")
    assert "by names no marker writer" in str(refused.value)


# --------------------------------------------------------------------------
# the C++ engine: same IR, same texts
# --------------------------------------------------------------------------


def _cli() -> Path:
    """``ROCKE_GPU_CHECKED_VALUE_CLI`` names the built ``rocke_gpu_checked_value``:
    ``tests/run_all.py`` resolves it from a configured build and the installed
    ``rocke_pytest`` ctest sets it. A value that names no file fails."""
    executable = os.environ.get("ROCKE_GPU_CHECKED_VALUE_CLI")
    if not executable:
        pytest.skip(
            "Set ROCKE_GPU_CHECKED_VALUE_CLI to the built rocke_gpu_checked_value"
        )
    path = Path(executable).resolve()
    assert path.is_file(), f"ROCKE_GPU_CHECKED_VALUE_CLI names {path}, not a file"
    return path


def test_cpp_builds_and_refuses_every_case_the_same():
    printed = subprocess.run(
        [str(_cli()), "--cases"], capture_output=True, text=True, check=True, timeout=30
    ).stdout
    reports = {}
    for block in printed.split("== ")[1:]:
        name, _, body = block.partition("\n")
        reports[name] = body[:-1]
    assert list(reports) == list(CASES)
    for name in CASES:
        assert reports[name] == _report(name), name


def test_cpp_verifies_every_marker_form_the_same():
    path = _cli()
    for name in sorted(MARKERS):
        operands, attrs, messages = MARKERS[name]
        printed = subprocess.run(
            [str(path), "--verify"],
            input=serialize(_marker_kernel(operands, attrs)),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        assert printed.splitlines() == messages, name


# --------------------------------------------------------------------------
# GPU: the check passing and failing, each in a child process
# --------------------------------------------------------------------------


def _checked_store_kernel(name: str):
    """One lane: load ``cu[0]``, check ``0 <= value <= T`` on the GPU, and store
    the checked value to ``Y[0]``."""
    b = IRBuilder(name)
    b.kernel.attrs["workgroup_size"] = 1
    y = b.param("Y", PtrType(I32, "global"))
    cu = b.param("cu", PtrType(I32, "global"), readonly=True)
    t = b.param("T", I32, ge=0, le=4096)
    start = b.gpu_checked_value(
        b.readfirstlane(b.global_load(cu, b.const_i32(0), I32)), ge=0, le=t
    )
    b.global_store(y, b.const_i32(0), start)
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


# Runs in the child: compiles the kernel on ``route``, launches it with
# cu[0] = ``loaded`` and T = 10, and prints the stored value. A child, because a
# taken trap ends the launch with a hardware exception after which the process
# cannot use the GPU, or aborts it outright.
_CHILD = textwrap.dedent(
    """
    import ctypes, struct, sys
    import numpy as np
    from rocke.runtime.hip_module import Runtime
    sys.path.insert(0, sys.argv[4])
    from test_gpu_checked_value import _checked_store_kernel

    route, arch, loaded = sys.argv[1], sys.argv[2], int(sys.argv[3])
    kernel = _checked_store_kernel(f"gpu_checked_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel
        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        from rocke.helpers.compile import compile_kernel_via_hipcc
        blob = compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    runtime = Runtime()
    function = runtime.load_module(blob).get_function(kernel.name)
    y_dev = runtime.alloc(4)
    runtime.memset(y_dev, 0xFF, 4)
    cu = np.array([loaded], dtype=np.int32)
    cu_dev = runtime.alloc(4)
    runtime.memcpy_h2d(cu_dev, (ctypes.c_ubyte * 4).from_buffer(cu), 4)
    print("launching", flush=True)
    runtime.launch_blocking(
        function, (1, 1, 1), (1, 1, 1), struct.pack("QQi", y_dev, cu_dev, 10)
    )
    y = np.zeros(1, dtype=np.int32)
    runtime.memcpy_d2h((ctypes.c_ubyte * 4).from_buffer(y), y_dev, 4)
    print(f"stored: {int(y[0])}", flush=True)
    """
)


def _run_child(route: str, loaded: int):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    if route == "hipcc" and shutil.which("hipcc") is None:
        pytest.skip("hipcc not found")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            route,
            arch,
            str(loaded),
            str(Path(__file__).parent),
        ],
        check=False,  # the child's output is the result, read below
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_a_value_meeting_its_facts_runs_to_the_store(route):
    child = _run_child(route, 7)
    assert child.returncode == 0, child.stdout + child.stderr
    assert "stored: 7" in child.stdout.splitlines(), child.stdout + child.stderr


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_a_value_breaking_its_facts_traps_the_launch(route):
    """cu[0] = 11 > T = 10: the check fails and the trap ends the kernel before
    the store. The launch reports an error or the runtime aborts the child;
    either way the child does not finish and never reads back a stored value."""
    child = _run_child(route, 11)
    lines = child.stdout.splitlines()
    assert "launching" in lines, child.stdout + child.stderr
    assert child.returncode != 0, child.stdout + child.stderr
    assert not any(line.startswith("stored:") for line in lines), child.stdout
