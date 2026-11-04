# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The carried values of an ``scf.for`` and the ``scf.yield`` that ends its body.

A loop's ``num_iter_args``, init operands, ``iter_args`` metadata and results
agree in count, and the loop takes its carried values from the one
``scf.yield`` that ends its body: the body yields only at its end, and ends in a
yield of one value per carried value, of that value's type, whenever the loop
carries any. A loop that carries none may end in an empty yield or in none.
The verifier and the LLVM and HIP lowerings of both engines refuse any other
loop with the same text and a ``ValueError``, also in an unrolled loop of zero
trips, which the LLVM lowering does not lower; each loop is held to its own
counts, and a loop or an if/else nested in a body keeps its own yields. Each
refusal runs the Python verifier and the Python lowerings everywhere, and the
C++ lowerings where the ``rocke_engine`` binding is built.
``tests/core/test_yield_rules.cpp`` covers the same rules in the C++ engine,
its verifier included.
"""

from __future__ import annotations

import pytest
from rocke.core.ir import I32, IRBuilder
from rocke.tests.engines import assert_accepted_everywhere, assert_refused_everywhere

ARCH = "gfx942"
# A concrete LLVM flavor, so the checks need no COMGR to pick one.
HOST_FLAVOR = "llvm22"
LOOP_TRIPS = 4
EARLY_YIELD_MESSAGE = (
    "scf.for body yields before its last op; a loop takes its values only from "
    "the scf.yield that ends its body"
)


def _loop_kernel_with_body(name, body, trips=LOOP_TRIPS, unroll=False):
    """A kernel with one loop of ``trips`` trips (unrolled when ``unroll``)
    that carries the lane id; its body holds only what ``body(b, k, acc)``
    emits, its yields included."""
    b = IRBuilder(name)
    loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(trips),
        b.const_i32(1),
        [("acc", b.thread_id_x())],
        unroll=unroll,
    )
    with loop as (k, (acc,)):
        body(b, k, acc)
    b.ret()
    return b.kernel


def _refused(kernel, message):
    assert_refused_everywhere(kernel, ARCH, ValueError, message, HOST_FLAVOR)


def _yield_then_op(b, k, acc):
    """A yield, then an op, and no closing yield."""
    b.scf_yield(acc)
    b.add(k, b.const_i32(1))


def _yield_twice(b, k, acc):
    """Two yields in a row, the second closing the body."""
    b.scf_yield(acc)
    b.scf_yield(acc)


def _yield_early(b, k, acc):
    """A yield, then an op, then the closing yield."""
    _yield_then_op(b, k, acc)
    b.scf_yield(acc)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_yield_early, id="yield_op_closing_yield"),
        pytest.param(_yield_twice, id="two_yields"),
        pytest.param(_yield_then_op, id="yield_then_op"),
    ],
)
def test_a_loop_body_that_yields_before_its_end_is_refused(body):
    """Only the yield that ends the body gives the next trip its values, so
    any yield before the body's last op is refused rather than counted toward
    the carried values or dropped."""
    _refused(_loop_kernel_with_body("loop_early_yield", body), EARLY_YIELD_MESSAGE)


def test_an_unrolled_loop_of_zero_trips_refuses_two_yields():
    """The LLVM lowering does not lower the body of an unrolled loop of zero
    trips, yet it refuses the loop as the verifier and the HIP lowering do."""
    kernel = _loop_kernel_with_body(
        "zero_trip_two_yields", _yield_twice, trips=0, unroll=True
    )
    _refused(kernel, EARLY_YIELD_MESSAGE)


def test_a_loop_that_carries_nothing_yields_only_at_its_end():
    b = IRBuilder("plain_loop_two_yields")
    with b.scf_for(b.const_i32(0), b.const_i32(LOOP_TRIPS), b.const_i32(1)):
        b.scf_yield()
        b.scf_yield()
    b.ret()
    _refused(b.kernel, EARLY_YIELD_MESSAGE)


def test_a_loop_that_carries_a_value_must_end_in_its_yield():
    _refused(
        _loop_kernel_with_body("loop_missing_yield", lambda b, k, acc: b.add(k, acc)),
        "scf.for carries 1 values, so its body must end in scf.yield of 1 values",
    )


def test_a_loop_yield_of_the_wrong_count_is_refused():
    _refused(
        _loop_kernel_with_body(
            "loop_yield_count", lambda b, k, acc: b.scf_yield(acc, acc)
        ),
        "scf.for carries 1 values, but its body yields 2 values",
    )


def test_a_loop_yield_of_the_wrong_type_is_refused():
    _refused(
        _loop_kernel_with_body(
            "loop_yield_type", lambda b, k, acc: b.scf_yield(b.const_f32(1.0))
        ),
        "scf.for iter_arg 0 is i32, but its body yields f32",
    )


def test_each_loop_is_held_to_its_own_carried_values():
    """A well-formed outer loop does not hide an inner loop that carries two
    values and yields one: the inner loop's own counts are refused."""

    def body(b, k, acc):
        inner = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(2),
            b.const_i32(1),
            [("x", k), ("y", acc)],
            iv_name="inner",
        )
        with inner as (_, (x, _y)):
            b.scf_yield(x)
        b.scf_yield(acc)

    _refused(
        _loop_kernel_with_body("inner_loop_yield_count", body),
        "scf.for carries 2 values, but its body yields 1 values",
    )


def _set_num_iter_args_to_zero(loop):
    loop.attrs["num_iter_args"] = 0


def _add_a_metadata_entry(loop):
    # A name of its own: the IR parser refuses a repeated name before the loop
    # rule runs.
    loop.attrs["iter_args"] = loop.attrs["iter_args"] + [
        {"name": "%extra", "type": "i32"}
    ]


def _drop_the_result(loop):
    loop.results = []


def _make_the_metadata_a_string(loop):
    loop.attrs["iter_args"] = "invalid"
    # Without metadata the body's carried value has no name the C++ parser can
    # resolve; yield the init value so the refusal reaches the lowerer.
    loop.regions[0].ops[-1].operands = loop.operands[3:]


@pytest.mark.parametrize(
    "edit,message",
    [
        pytest.param(
            _set_num_iter_args_to_zero,
            "scf.for declares 0 iter_args but has 1 init operands",
            id="num_iter_args",
        ),
        pytest.param(
            _add_a_metadata_entry,
            "scf.for declares 1 iter_args but has 2 metadata entries",
            id="metadata_entries",
        ),
        pytest.param(
            _drop_the_result,
            "scf.for declares 1 iter_args but has 0 results",
            id="results",
        ),
        pytest.param(
            _make_the_metadata_a_string,
            "scf.for iter_args metadata must be a list",
            id="metadata_not_a_list",
        ),
    ],
)
def test_a_loop_whose_carried_counts_disagree_is_refused(edit, message):
    """Serialized IR can give a loop a ``num_iter_args`` or an ``iter_args``
    list that disagrees with its init operands; the verifier and every
    lowering refuse it, the HIP lowering included, instead of reading the
    disagreeing count."""
    kernel = _loop_kernel_with_body("loop_counts", lambda b, k, acc: b.scf_yield(acc))
    (loop,) = [op for op in kernel.body.ops if op.name == "scf.for"]
    edit(loop)
    _refused(kernel, message)


def test_well_formed_loops_lower():
    """A loop that carries a value and ends in its yield lowers, and so does
    one whose body holds a loop and an if/else with results, whose own yields
    are theirs; a loop that carries nothing may end in an empty yield or in
    none."""
    b = IRBuilder("loops_well_formed")
    lane = b.thread_id_x()
    loop = b.scf_for_iter(
        b.const_i32(0), b.const_i32(LOOP_TRIPS), b.const_i32(1), [("acc", lane)]
    )
    with loop as (k, (acc,)):
        with b.scf_for(b.const_i32(0), b.const_i32(2), b.const_i32(1), iv_name="inner"):
            b.scf_yield()
        branch = b.scf_if_else(b.cmp_lt(k, lane), result_types=[I32])
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(b.add(acc, k))
            with else_ctx:
                b.scf_yield(acc)
        b.scf_yield(b.add(acc, b.const_i32(1)))
    for iv_name, closing_yield in (("i", False), ("j", True)):
        with b.scf_for(
            b.const_i32(0), b.const_i32(LOOP_TRIPS), b.const_i32(1), iv_name=iv_name
        ):
            b.add(b.const_i32(1), b.const_i32(2))
            if closing_yield:
                b.scf_yield()
    b.ret()
    assert_accepted_everywhere(b.kernel, ARCH, HOST_FLAVOR)
