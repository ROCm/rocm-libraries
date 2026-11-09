# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An unrolled ``scf.for`` whose body holds nested regions.

``scf_for_iter(..., unroll=True)`` emits the loop body once per trip. Each trip
must read that trip's induction variable and loop-carried values, and every
value the body defines must get a name of its own in each trip -- including the
uses and definitions inside regions nested in the body: an ``scf.if`` body, an
``scf.if_else`` branch and the ``scf.yield`` that ends it, and an inner
``scf.for``, unrolled or not.

Each kernel below puts one of those shapes in a loop of ``TRIPS`` unrolled
trips. The tests lower each kernel with both engines (same text) at every LLVM
flavor, compile and link it, and, when a GPU is present, run it and compare
every lane with the value worked out on the host.
``tests/core/test_unrolled_for_nested_regions.cpp`` lowers the same kernels with
the C++ engine alone.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import LLVM_FLAVORS, lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts

LANES = 64
TRIPS = 4
# The GPUs the run test has run on; on any other it skips until it has.
GPU_ARCHES_RUN_ON = ("gfx90a", "gfx942")
# Lanes below SPLIT take the then side of a branch.
SPLIT = 24


def _kernel(name):
    """Return a builder with params ``Y`` and ``Z`` (both i32) and the lane id."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = LANES
    y = b.param("Y", PtrType(I32, "global"))
    z = b.param("Z", PtrType(I32, "global"))
    return b, y, z, b.thread_id_x()


def _unrolled(b, init, trips=TRIPS, name="acc", iv_name="k"):
    return b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(trips),
        b.const_i32(1),
        [(name, init)],
        iv_name=iv_name,
        unroll=True,
    )


def _iv_used_only_in_nested_if():
    """Trip ``k`` stores ``10 * k`` to ``Y[lane]`` when ``lane == k``: the
    induction variable is read only inside the nested ``scf.if``."""
    b, y, z, lane = _kernel("unroll_iv_in_nested_if")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        with b.scf_if(b.cmp_eq(lane, k)):
            b.global_store(y, lane, b.mul(k, b.const_i32(10)))
        b.scf_yield(b.add(acc, lane))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _iv_used_only_in_nested_if_expected(lane):
    y = np.where(lane < TRIPS, 10 * lane, 0)
    return y, TRIPS * lane


def _result_defined_in_nested_if():
    """Each trip defines ``100 * k + lane`` inside an ``scf.if``, so every trip
    defines a value of its own there."""
    b, y, z, lane = _kernel("unroll_result_in_nested_if")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        with b.scf_if(b.cmp_lt(lane, b.const_i32(SPLIT))):
            b.global_store(y, lane, b.add(b.mul(k, b.const_i32(100)), lane))
        b.scf_yield(b.add(acc, k))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _result_defined_in_nested_if_expected(lane):
    y = np.where(lane < SPLIT, 100 * (TRIPS - 1) + lane, 0)
    return y, np.full_like(lane, sum(range(TRIPS)))


def _carried_value_used_only_in_branch_yield():
    """An if/else gives the carried value on its then branch and the lane on
    its else branch; the carried value is read only by the then branch's
    ``scf.yield``. The loop then adds the induction variable."""
    b, _, z, lane = _kernel("unroll_carried_in_branch_yield")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        branch = b.scf_if_else(b.cmp_lt(lane, b.const_i32(SPLIT)), result_types=[I32])
        with branch as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(acc)
            with else_ctx:
                b.scf_yield(lane)
        b.scf_yield(b.add(branch.results[0], k))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _carried_value_used_only_in_branch_yield_expected(lane):
    z = np.where(lane < SPLIT, sum(range(TRIPS)), lane + TRIPS - 1)
    return np.zeros_like(lane), z


def _nested_if_in_if():
    """An ``scf.if`` inside an ``scf.if`` inside the unrolled body; the
    innermost one reads the induction variable and the carried value and
    defines a value."""
    b, y, z, lane = _kernel("unroll_if_in_if")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        with (
            b.scf_if(b.cmp_lt(lane, b.const_i32(SPLIT))),
            b.scf_if(b.cmp_eq(lane, k)),
        ):
            b.global_store(y, lane, b.add(acc, b.const_i32(1000)))
        b.scf_yield(b.add(acc, b.const_i32(1)))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _nested_if_in_if_expected(lane):
    # On trip k the carried value is k, so lane k (< TRIPS) stores 1000 + k.
    y = np.where(lane < TRIPS, 1000 + lane, 0)
    return y, np.full_like(lane, TRIPS)


def _rolled_loop_in_unrolled_loop():
    """A rolled ``scf.for`` inside the unrolled body: trip ``k`` sums
    ``j`` over ``0 <= j <= k`` on top of the lane. The inner loop's induction
    variable and carried value are defined once per trip."""
    b, _, z, lane = _kernel("unroll_rolled_inner_loop")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        inner = b.scf_for_iter(
            b.const_i32(0),
            b.add(k, b.const_i32(1)),
            b.const_i32(1),
            [("s", lane)],
            iv_name="j",
        )
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, j))
        b.scf_yield(b.add(acc, inner.results[0]))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _rolled_loop_in_unrolled_loop_expected(lane):
    triangle = sum(k * (k + 1) // 2 for k in range(TRIPS))
    return np.zeros_like(lane), TRIPS * lane + triangle


def _rolled_loops_side_by_side_in_unrolled_loop():
    """Two rolled ``scf.for`` loops one after the other in the unrolled body,
    both naming their values ``j`` and ``s``: trip ``k`` adds ``j`` over
    ``0 <= j <= k`` twice. Every trip lowers both loops, so the kernel holds
    ``2 * TRIPS`` loops of the same names."""
    b, _, z, lane = _kernel("unroll_rolled_loops_side_by_side")
    loop = _unrolled(b, lane)
    with loop as (k, (acc,)):
        carried = acc
        for _ in range(2):
            inner = b.scf_for_iter(
                b.const_i32(0),
                b.add(k, b.const_i32(1)),
                b.const_i32(1),
                [("s", carried)],
                iv_name="j",
            )
            with inner as (j, (s,)):
                b.scf_yield(b.add(s, j))
            carried = inner.results[0]
        b.scf_yield(carried)
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _rolled_loops_side_by_side_in_unrolled_loop_expected(lane):
    triangle = sum(k * (k + 1) // 2 for k in range(TRIPS))
    return np.zeros_like(lane), lane + 2 * triangle


def _shared_memory_in_nested_if():
    """Inside an ``scf.if`` in each trip, lanes below ``SPLIT`` allocate shared
    memory, store ``100 * k + lane`` to it and copy it back to ``Y[lane]``.
    The allocation names one buffer of the kernel, which every trip shares."""
    b, y, z, lane = _kernel("unroll_shared_memory_in_nested_if")
    loop = _unrolled(b, b.const_i32(0))
    with loop as (k, (acc,)):
        with b.scf_if(b.cmp_lt(lane, b.const_i32(SPLIT))):
            lds = b.smem_alloc(I32, [LANES], name_hint="lds")
            b.smem_store_vN(lds, [lane], b.add(b.mul(k, b.const_i32(100)), lane), 1)
            stored = b.smem_load_vN(lds, lane, dtype=I32, n=1)
            b.global_store(y, lane, b.vec_extract(stored, 0))
        b.scf_yield(b.add(acc, k))
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _unrolled_loop_in_unrolled_loop():
    """An unrolled ``scf.for`` of three trips inside the unrolled body; its body
    reads the outer induction variable and defines a value."""
    b, _, z, lane = _kernel("unroll_unrolled_inner_loop")
    loop = _unrolled(b, lane)
    with loop as (k, (acc,)):
        inner = _unrolled(b, acc, trips=3, name="s", iv_name="j")
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, b.mul(k, j)))
        b.scf_yield(inner.results[0])
    b.global_store(z, lane, loop.results[0])
    b.ret()
    return b.kernel


def _unrolled_loop_in_unrolled_loop_expected(lane):
    return np.zeros_like(lane), lane + sum(
        k * j for k in range(TRIPS) for j in range(3)
    )


CASES = {
    "iv_used_only_in_nested_if": (
        _iv_used_only_in_nested_if,
        _iv_used_only_in_nested_if_expected,
    ),
    "result_defined_in_nested_if": (
        _result_defined_in_nested_if,
        _result_defined_in_nested_if_expected,
    ),
    "carried_value_used_only_in_branch_yield": (
        _carried_value_used_only_in_branch_yield,
        _carried_value_used_only_in_branch_yield_expected,
    ),
    "nested_if_in_if": (_nested_if_in_if, _nested_if_in_if_expected),
    "rolled_loop_in_unrolled_loop": (
        _rolled_loop_in_unrolled_loop,
        _rolled_loop_in_unrolled_loop_expected,
    ),
    "unrolled_loop_in_unrolled_loop": (
        _unrolled_loop_in_unrolled_loop,
        _unrolled_loop_in_unrolled_loop_expected,
    ),
    "rolled_loops_side_by_side_in_unrolled_loop": (
        _rolled_loops_side_by_side_in_unrolled_loop,
        _rolled_loops_side_by_side_in_unrolled_loop_expected,
    ),
    # The allocation copies one lane's value per trip, as the nested if does.
    "shared_memory_in_nested_if": (
        _shared_memory_in_nested_if,
        _result_defined_in_nested_if_expected,
    ),
}


_LOCAL_NAME = re.compile(r"%[-\w.$]+")


def _local_names(text):
    """The local names of the one function in the LLVM IR ``text``, as
    ``(values, bound, read)``: ``values`` lists every value definition in order
    (a name defined twice appears twice), ``bound`` is the set of parameter and
    block-label names, and ``read`` is the set of names the body reads."""
    lines = text.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith("define ")
    )
    end = lines.index("}", start)
    values = []
    bound = set(_LOCAL_NAME.findall(lines[start]))
    read = set()
    for line in lines[start + 1 : end]:
        if line.endswith(":"):
            bound.add("%" + line[:-1])
            continue
        target, assigns, operands = line.partition(" = ")
        if assigns and target.strip().startswith("%"):
            values.append(target.strip())
        else:
            operands = line
        read.update(_LOCAL_NAME.findall(operands))
    return values, bound, read


def _names_defined_more_than_once(text):
    """The SSA names the LLVM IR ``text`` defines more than once, sorted."""
    values, _, _ = _local_names(text)
    return sorted(name for name in set(values) if values.count(name) > 1)


def _names_read_but_never_defined(text):
    """The local names the one function in the LLVM IR ``text`` reads but
    defines nowhere in it -- not as a parameter, a block label or a value --
    sorted. (Whether each definition dominates its reads is LLVM's check, which
    the compile-and-link test makes.)"""
    values, bound, read = _local_names(text)
    return sorted(read - set(values) - bound)


def _reuse_message(name):
    return (
        f"scf.for: the name '{name}' is already taken by an induction variable or "
        "carried value of this loop or of a loop it is nested in; give it a name "
        "of its own"
    )


def _inner_loop(b, iv_name, carried, init, unroll):
    """A loop over [0, 2) with induction variable ``iv_name`` and, unless
    ``carried`` is None, one carried value of that name starting at ``init``
    (a plain ``scf_for`` has no carried values and is never unrolled)."""
    if carried is None:
        return b.scf_for(b.const_i32(0), b.const_i32(2), b.const_i32(1), iv_name)
    return b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(2),
        b.const_i32(1),
        [(carried, init)],
        iv_name=iv_name,
        unroll=unroll,
    )


@pytest.mark.parametrize(
    "iv_name, carried, unroll, through_if, taken",
    [
        ("k", "s", False, False, "%k"),
        ("k", "s", True, False, "%k"),
        ("j", "acc", False, False, "%acc"),
        ("j", "acc", True, False, "%acc"),
        ("k", None, False, False, "%k"),
        ("k", "s", False, True, "%k"),
    ],
    ids=[
        "rolled_reuses_induction_variable",
        "unrolled_reuses_induction_variable",
        "rolled_reuses_carried_value",
        "unrolled_reuses_carried_value",
        "plain_loop_reuses_induction_variable",
        "reused_from_inside_an_if",
    ],
)
def test_nested_loop_may_not_reuse_an_enclosing_loop_name(
    iv_name, carried, unroll, through_if, taken
):
    """A loop nested in an unrolled body may not reuse the name of the outer
    loop's induction variable or carried value, also from inside an ``scf.if``
    in that body. The serialized IR refers to a value by its name, so a read of
    the reused name in the inner body would be ambiguous: the C++ engine would
    read one value and the Python engine the other. The C++ builder refuses
    with the same text (``tests/core/test_scf_for_name_reuse.cpp``)."""
    b, _, _, lane = _kernel("reuse_" + taken.lstrip("%"))
    loop = _unrolled(b, lane)
    with loop as (_, (acc,)):
        if through_if:
            with (
                b.scf_if(b.cmp_lt(lane, b.const_i32(SPLIT))),
                pytest.raises(ValueError) as refused,
            ):
                _inner_loop(b, iv_name, carried, acc, unroll)
        else:
            with pytest.raises(ValueError) as refused:
                _inner_loop(b, iv_name, carried, acc, unroll)
    assert str(refused.value) == _reuse_message(taken)


@pytest.mark.parametrize("followed", ["outer", "inner"])
def test_reuse_is_refused_when_an_op_follows_a_loop_handle(followed):
    """A loop handle may be created and other ops emitted after it before its
    body is entered, so the loop is not the last op of its region. The names of
    its values are still taken inside its body; ``followed`` is the loop whose
    handle an op follows, and the innermost loop reuses one of its names."""
    b, _, _, lane = _kernel("reuse_after_op_" + followed)
    outer = _unrolled(b, lane)
    if followed == "outer":
        b.add(lane, lane)
    with outer as (k, (acc,)):
        inner = _inner_loop(b, "j", "s", acc, unroll=False)
        if followed == "inner":
            b.add(acc, k)
        with inner as (_, (s,)):
            taken = "k" if followed == "outer" else "j"
            with pytest.raises(ValueError) as refused:
                _inner_loop(b, taken, "t", s, unroll=False)
    assert str(refused.value) == _reuse_message("%" + taken)


# The inner loop's induction-variable attrs, as _nested_loops_text serializes them.
_INNER_IV_ATTRS = 'iv = s:"%j", iv_type = s:"i32"'


def _nested_loops_text():
    """The serialized IR of an unrolled loop ``k`` over ``acc`` whose body
    holds a rolled loop ``j`` over ``s``, and the name of the lane id."""
    b, _, z, lane = _kernel("parse_reuse")
    outer = _unrolled(b, lane)
    with outer as (_, (acc,)):
        inner = _inner_loop(b, "j", "s", acc, unroll=False)
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, j))
        b.scf_yield(inner.results[0])
    b.global_store(z, lane, outer.results[0])
    b.ret()
    return serialize(b.kernel), lane.name


def _edit_once(text, old, new):
    """``text`` with its one occurrence of ``old`` replaced by ``new``."""
    assert text.count(old) == 1, f"expected one {old!r} in the serialized IR"
    return text.replace(old, new)


def _assert_both_parsers_refuse(text, message):
    """Both parsers refuse ``text`` with ``message``: the Python parser with a
    ValueError, the C++ one (through the binding, when built) with a
    RuntimeError ending in it."""
    with pytest.raises(ValueError) as refused:
        parse(text)
    assert str(refused.value) == message
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError) as native:
        engine.lower_serialized_ir(text, arch="gfx942", flavor="llvm22")
    assert str(native.value).endswith(": " + message), str(native.value)


@pytest.mark.parametrize(
    "reused, inner_name, inner_iv_type",
    [
        ("%k", "%j", 's:"i32"'),
        ("%acc", "%s", 's:"i32"'),
        ("%j", "%s", 's:"i32"'),
        ("%k", "%j", "i:0"),
    ],
    ids=[
        "outer_induction_variable",
        "outer_carried_value",
        "same_loop",
        "outer_induction_variable_by_one_whose_type_is_not_named",
    ],
)
def test_both_parsers_refuse_a_reused_loop_name(reused, inner_name, inner_iv_type):
    """Serialized IR can name a value of a nested loop like a value of the loop
    it is in, or two values of one loop alike, past the builder. Both parsers
    refuse it with the builder's message rather than bind the inner body's
    reads of that name to one of the values and unbind the other. An induction
    variable whose ``iv_type`` is not a type name is never bound, but its name
    is the loop's all the same: unrefused, the inner body's reads of it would
    read the outer induction variable."""
    text, _ = _nested_loops_text()
    text = _edit_once(text, _INNER_IV_ATTRS, f'iv = s:"%j", iv_type = {inner_iv_type}')
    text = re.sub(re.escape(inner_name) + r"\b", reused, text)
    _assert_both_parsers_refuse(text, _reuse_message(reused))


@pytest.mark.parametrize(
    "value_kind",
    ["induction_variable", "carried_value"],
)
def test_both_parsers_refuse_a_loop_value_without_a_type_name(value_kind):
    """A parser binds a loop's induction variable or carried value with the
    type its attr names. When the attr is not a type name, the value cannot be
    bound, and the body's reads of its name would read whatever other value
    holds that name: here the inner induction variable takes the name of the
    lane id, which no loop value holds, so the reused-name rule passes it.
    Both parsers refuse it instead, with one message."""
    text, lane_name = _nested_loops_text()
    if value_kind == "induction_variable":
        text = _edit_once(text, _INNER_IV_ATTRS, 'iv = s:"%j", iv_type = i:0')
        text = re.sub(r"%j\b", lane_name, text)
        untyped = lane_name
    else:
        text = _edit_once(
            text, '{ name = s:"%s", type = s:"i32" }', '{ name = s:"%s", type = i:0 }'
        )
        untyped = "%s"
    _assert_both_parsers_refuse(
        text,
        f"scf.for: the loop value '{untyped}' has no type name; its iv_type attr "
        "or iter_args entry must name its type",
    )


def _nested_loops_renaming(renames, iv_type_attr=None):
    """A rolled loop ``k`` over ``acc`` whose body holds a rolled loop ``j``
    over ``s``, built by the builder; then, past it, each value of the inner
    loop named in ``renames`` takes the name it maps to, in the loop's attrs
    and in the value every read of it sees, and the inner loop's ``iv_type``
    attr becomes ``iv_type_attr`` when that is given."""
    b, _, z, lane = _kernel("verify_reuse")
    outer = _inner_loop(b, "k", "acc", lane, unroll=False)
    with outer as (_, (acc,)):
        inner = _inner_loop(b, "j", "s", acc, unroll=False)
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, j))
        b.scf_yield(inner.results[0])
    b.global_store(z, lane, outer.results[0])
    b.ret()
    loop = inner.op
    if iv_type_attr is not None:
        loop.attrs["iv_type"] = iv_type_attr
    loop.attrs["iv"] = renames.get(loop.attrs["iv"], loop.attrs["iv"])
    for meta in loop.attrs["iter_args"]:
        meta["name"] = renames.get(meta["name"], meta["name"])
    for value in (j, s):
        value.name = renames.get(value.name, value.name)
    return b.kernel


@pytest.mark.parametrize(
    "renames, reported",
    [
        ({"%j": "%k"}, "%k"),
        ({"%s": "%acc"}, "%acc"),
        ({"%s": "%j"}, "%j"),
        ({"%j": "%k", "%s": "%acc"}, "%k"),
    ],
    ids=[
        "outer_induction_variable",
        "outer_carried_value",
        "same_loop",
        "both_names_reported_once_by_the_first",
    ],
)
def test_the_verifier_reports_a_reused_loop_name(renames, reported):
    """IR built past the builder can reuse a loop name the builder and the
    parsers refuse. The verifier reports the loop's first reused name (its
    induction variable before its carried values) with their message, once,
    and nothing else; ``tests/core/test_scf_for_name_reuse.cpp`` checks that
    the C++ verifier reports the same."""
    diagnostics = verify(_nested_loops_renaming(renames))
    assert [(d.severity, d.message) for d in diagnostics] == [
        ("error", _reuse_message(reported))
    ]


def test_the_verifier_counts_an_induction_variable_whose_type_is_not_named():
    """The induction variable is a loop name whenever the loop names it, also
    when its ``iv_type`` attr is not a type name, as in the C++ verifier
    (``tests/core/test_scf_for_name_reuse.cpp``)."""
    diagnostics = verify(_nested_loops_renaming({"%j": "%k"}, iv_type_attr=0))
    assert [(d.severity, d.message) for d in diagnostics] == [
        ("error", _reuse_message("%k"))
    ]


def test_the_verifier_accepts_the_names_the_builder_gave():
    assert verify(_nested_loops_renaming({})) == []


def test_one_loop_may_not_name_two_of_its_values_alike():
    b, _, _, lane = _kernel("reuse_in_one_loop")
    with pytest.raises(ValueError) as refused:
        _unrolled(b, lane, name="k", iv_name="k")
    assert str(refused.value) == _reuse_message("%k")


def test_loops_side_by_side_may_share_names():
    """Names are only taken inside the loop that defines them: two loops one
    after the other may use the same names, and so may each loop nested in
    them. The kernel still defines each name once, compiles and links."""
    b, _, z, lane = _kernel("names_side_by_side")
    start = lane
    for _ in range(2):
        rolled = _inner_loop(b, "j", "s", start, unroll=False)
        with rolled as (j, (s,)):
            b.scf_yield(b.add(s, j))
        start = rolled.results[0]
    first = _unrolled(b, start)
    with first as (k, (acc,)):
        inner = _inner_loop(b, "j", "s", acc, unroll=False)
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, j))
        b.scf_yield(inner.results[0])
    second = _unrolled(b, first.results[0])
    with second as (k, (acc,)):
        inner = _inner_loop(b, "j", "s", acc, unroll=False)
        with inner as (j, (s,)):
            b.scf_yield(b.add(s, k))
        b.scf_yield(inner.results[0])
    b.global_store(z, lane, second.results[0])
    b.ret()
    text = lower_with_both_engines(b.kernel, "gfx942")
    assert _names_defined_more_than_once(text) == []
    assert_target_accepts(text, "gfx942")


def _rolled_loops_side_by_side(first_carried):
    """Two rolled loops one after the other, both with induction variable
    ``j``; the first carries a value named ``first_carried``, the second one
    named ``s``."""
    b, _, z, lane = _kernel("rolled_loops_side_by_side")
    carried = lane
    for name in (first_carried, "s"):
        rolled = _inner_loop(b, "j", name, carried, unroll=False)
        with rolled as (j, (s,)):
            b.scf_yield(b.add(s, j))
        carried = rolled.results[0]
    b.global_store(z, lane, carried)
    b.ret()
    return b.kernel


def _loop_named_like_a_renamed_loop_value():
    """When two loops share their names, the second renames its values and
    carries ``%s.for.header.<n>``. In this kernel the first loop gives its own
    carried value that name, so the second loop's renamed name is taken too.
    The builder refuses a value name that is not an identifier, so the first
    loop is built with a placeholder name and the renamed name is put in its
    serialized IR: serialized IR is the way such a name reaches the lowering."""
    plain = lower_kernel_to_llvm(_rolled_loops_side_by_side("s"), arch="gfx942")
    renamed = re.search(r"%(s\.for\.header\.\d+) = phi", plain)
    assert renamed is not None, "the second loop did not rename its carried value"
    serialized = serialize(_rolled_loops_side_by_side("first"))
    edited, count = re.subn(r"%first\b", "%" + renamed[1], serialized)
    assert count, "the serialized IR has no %first to rename"
    return parse(edited)


def test_a_loop_named_like_a_renamed_loop_value_defines_each_name_once():
    """Checks the text ``lower_kernel_to_llvm`` emits, so it runs without the
    C++ engine binding too; ``tests/core/test_unrolled_for_nested_regions.cpp``
    checks the C++ engine's text."""
    text = lower_kernel_to_llvm(_loop_named_like_a_renamed_loop_value(), arch="gfx942")
    assert _names_defined_more_than_once(text) == []
    assert _names_read_but_never_defined(text) == []
    assert_target_accepts(text, "gfx942")


def test_both_engines_rename_a_loop_named_like_a_renamed_loop_value_alike():
    lower_with_both_engines(_loop_named_like_a_renamed_loop_value(), "gfx942")


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_both_engines_emit_the_same_text(case, flavor):
    lower_with_both_engines(CASES[case][0](), "gfx942", llvm_flavor=flavor)


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_trip_defines_its_own_values(case):
    """Every SSA name is defined once, and every name read is defined. Checks
    the text ``lower_kernel_to_llvm`` emits, so it also runs where the C++
    engine binding is not built (the installed test payload), on the Python
    engine's text there."""
    text = lower_kernel_to_llvm(CASES[case][0](), arch="gfx942")
    assert _names_defined_more_than_once(text) == []
    assert _names_read_but_never_defined(text) == []


@pytest.mark.parametrize("case", sorted(CASES))
def test_kernel_compiles_and_links(case):
    assert_target_accepts(
        lower_kernel_to_llvm(CASES[case][0](), arch="gfx942"), "gfx942"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_gpu_every_lane_matches_the_host(case):
    build, expected = CASES[case]
    y, z = (np.zeros(LANES, dtype=np.int32) for _ in range(2))
    run_on_device(build(), {"Y": y, "Z": z}, LANES, GPU_ARCHES_RUN_ON)
    expected_y, expected_z = expected(np.arange(LANES, dtype=np.int32))
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(z, expected_z)
