# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``scf_if_else`` that yields values.

With ``result_types`` an if/else gives one value per type: each branch ends in
``scf_yield`` of values of those types, and the results are the values of the
branch that ran. The LLVM lowering joins them with one ``phi`` per result; the
HIP lowering declares the results before the ``if`` and assigns them at the end
of both branches. Covers both lowerings in both engines (same text), an if/else
nested in a loop that carries a value, the builder's, the verifier's and the
lowerers' refusals of a branch that does not yield the results or yields a
buffer descriptor, of a yield in a branch of an if/else without results (alone
and in a loop) or before the end of a branch of one with results, directly or
in an scf.if within the branch, of an LDS allocation or a packed-only type as a
result, of an if/else without one condition and two branches, of a yield in
the kernel body or in an scf.if in a loop body, and of a malformed if/else in
an unrolled loop of zero trips, directly in its body or in an scf.if there; a
loop in a branch keeps its own yield; and,
when a GPU is present, a kernel whose lanes take both branches on the LLVM and
the hipcc route. The refusals run the verifier and the Python lowerings
everywhere, and the C++ lowerings where the ``rocke_engine`` binding is built.
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess

import numpy as np
import pytest
from rocke.core.ir import (
    F32,
    I4,
    I32,
    IRBuilder,
    Op,
    PtrType,
    Region,
    SmemType,
    Value,
    VectorType,
)
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.tests.engines import (
    assert_accepted_everywhere,
    assert_refused_everywhere,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)

FLAVORS = ("llvm20", "llvm22", "llvm23")
# The flavor the host-only checks lower for: a concrete one, so they need no
# COMGR to pick it.
HOST_FLAVOR = "llvm22"
LANES = 64
# Lanes below SPLIT take the then branch, the rest the else branch.
SPLIT = 24
LOOP_TRIPS = 4


def _branch_kernel(name):
    """Lane ``l`` stores what an if/else on ``l < SPLIT`` yields: ``X[l]`` and
    ``(l, l)`` on the then branch, ``7.5`` and ``(-1, -1)`` on the else branch,
    into ``Y[l]`` and ``Z[2l], Z[2l+1]``. ``W[l]`` sums, over ``LOOP_TRIPS``
    trips of a loop, an if/else on ``k < l % LOOP_TRIPS`` that yields 1 or 10,
    so the loop carries the if/else's result."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = LANES
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    z = b.param("Z", PtrType(I32, "global"))
    w = b.param("W", PtrType(I32, "global"))
    lane = b.thread_id_x()

    branch = b.scf_if_else(
        b.cmp_lt(lane, b.const_i32(SPLIT)), result_types=[F32, VectorType(I32, 2)]
    )
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            b.scf_yield(b.global_load(x, lane, F32), b.vector_splat(lane, 2))
        with else_ctx:
            b.scf_yield(b.const_f32(7.5), b.vector_splat(b.const_i32(-1), 2))
    value, pair = branch.results
    b.global_store(y, lane, value)
    twice = b.mul(lane, b.const_i32(2))
    b.global_store(z, twice, b.vec_extract(pair, 0))
    b.global_store(z, b.add(twice, b.const_i32(1)), b.vec_extract(pair, 1))

    taken = b.mod(lane, b.const_i32(LOOP_TRIPS))
    loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(LOOP_TRIPS),
        b.const_i32(1),
        [("acc", b.const_i32(0))],
    )
    with loop as (k, (acc,)):
        step = b.scf_if_else(b.cmp_lt(k, taken), result_types=[I32])
        with step as (then_ctx, else_ctx):
            with then_ctx:
                b.scf_yield(b.const_i32(1))
            with else_ctx:
                b.scf_yield(b.const_i32(10))
        b.scf_yield(b.add(acc, step.results[0]))
    b.global_store(w, lane, loop.results[0])
    b.ret()
    return b.kernel


def _expected(x):
    lanes = np.arange(LANES)
    then = lanes < SPLIT
    y = np.where(then, x, np.float32(7.5)).astype(np.float32)
    z = np.repeat(np.where(then, lanes, -1), 2).astype(np.int32)
    taken = lanes % LOOP_TRIPS
    w = (taken * 1 + (LOOP_TRIPS - taken) * 10).astype(np.int32)
    return y, z, w


# --------------------------------------------------------------------------
# LLVM IR
# --------------------------------------------------------------------------


def test_results_lower_to_one_phi_each_in_the_join_block():
    text = _lower_kernel_to_llvm_python(_branch_kernel("if_else_phi"), arch="gfx942")
    lines = text.splitlines()
    end = next(i for i, line in enumerate(lines) if line.startswith("if.end"))
    then_label = next(line[:-1] for line in lines if line.startswith("if.then"))
    else_label = next(line[:-1] for line in lines if line.startswith("if.else"))
    # The two phis open the join block, one per result, each taking the value
    # its branch yields.
    assert lines[end + 1].startswith("  %if")
    assert " = phi float [ %gl" in lines[end + 1]
    assert f", %{then_label} ], [ 0x401E000000000000, %{else_label} ]" in lines[end + 1]
    assert " = phi <2 x i32> [ " in lines[end + 2]
    # The loop's if/else feeds the loop-carried sum through its own phi.
    assert text.count(" = phi i32 [ 1, %if.then") == 1


def test_an_if_else_without_results_has_no_phi():
    b = IRBuilder("if_else_no_results")
    y = b.param("Y", PtrType(I32, "global"))
    lane = b.thread_id_x()
    with b.scf_if_else(b.cmp_lt(lane, b.const_i32(SPLIT))) as (then_ctx, else_ctx):
        with then_ctx:
            b.global_store(y, lane, b.const_i32(1))
        with else_ctx:
            b.global_store(y, lane, b.const_i32(2))
    b.ret()
    text = lower_with_both_engines(b.kernel, "gfx942")
    assert " = phi " not in text


def test_both_engines_emit_the_same_llvm_ir():
    kernel = _branch_kernel("if_else_engines")
    for flavor in FLAVORS:
        lower_with_both_engines(kernel, "gfx942", llvm_flavor=flavor)


# --------------------------------------------------------------------------
# HIP source
# --------------------------------------------------------------------------


def test_hip_declares_the_results_before_the_if_and_assigns_them_in_both_branches():
    hip = lower_hip_with_both_engines(_branch_kernel("if_else_hip"), "gfx942")
    lines = [line.strip() for line in hip.splitlines()]
    head = lines.index("float if4;")
    assert lines[head + 1] == "i32x2 if5;"
    assert lines[head + 2].startswith("if(") and lines[head + 2].endswith(") {")
    middle = lines.index("} else {", head)
    end = lines.index("}", middle)
    then_lines = lines[head + 3 : middle]
    else_lines = lines[middle + 1 : end]
    assert then_lines[-2].startswith("if4 = gl") and then_lines[-1].startswith("if5 = ")
    assert else_lines[-2].startswith("if4 = c") and else_lines[-1].startswith("if5 = ")


def test_hip_source_compiles(tmp_path):
    """The HIP lowering is accepted by hipcc (no GPU needed)."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found")
    source = tmp_path / "if_else.hip"
    source.write_text(
        lower_kernel_to_hip(_branch_kernel("if_else_hipcc"), arch="gfx942")
    )
    subprocess.run(
        [hipcc, "--offload-arch=gfx942", "--genco", "-O3", str(source), "-o",
         str(tmp_path / "out.hsaco")],
        check=True,
        capture_output=True,
        timeout=240,
    )  # fmt: skip


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def _open_if_else(result_types):
    b = IRBuilder("if_else_refused")
    lane = b.thread_id_x()
    branch = b.scf_if_else(
        b.cmp_lt(lane, b.const_i32(SPLIT)), result_types=result_types
    )
    return b, branch


def _assert_refused(kernel, error_type, message):
    """``assert_refused_everywhere`` for gfx942 at the host flavor."""
    assert_refused_everywhere(kernel, "gfx942", error_type, message, HOST_FLAVOR)


def _assert_lowered(kernel):
    """``assert_accepted_everywhere`` for gfx942 at the host flavor."""
    assert_accepted_everywhere(kernel, "gfx942", HOST_FLAVOR)


@pytest.mark.parametrize(
    "then_values,message",
    [
        (None, "scf.if_else has 2 results, so its then branch must end in scf.yield of 2 values"),
        (("f32",), "scf.if_else has 2 results, but its then branch yields 1 values"),
        (("i32", "i32"), "scf.if_else result 0 is f32, but its then branch yields i32"),
    ],
)  # fmt: skip
def test_builder_refuses_a_branch_that_does_not_yield_the_results(then_values, message):
    b, branch = _open_if_else([F32, I32])
    constants = {"f32": lambda: b.const_f32(1.0), "i32": lambda: b.const_i32(1)}
    with pytest.raises(ValueError) as error, branch as (then_ctx, else_ctx):
        with then_ctx:
            if then_values is not None:
                b.scf_yield(*(constants[name]() for name in then_values))
        with else_ctx:
            b.scf_yield(b.const_f32(2.0), b.const_i32(2))
    assert str(error.value) == message


def test_builder_refuses_an_else_branch_that_does_not_yield():
    b, branch = _open_if_else([I32])
    # The else branch is never entered, so it ends in nothing.
    with pytest.raises(ValueError) as error, branch as (then_ctx, _), then_ctx:
        b.scf_yield(b.const_i32(1))
    assert str(error.value) == (
        "scf.if_else has 1 results, so its else branch must end in scf.yield of 1 values"
    )


def _raw_if_else_kernel():
    """An if/else whose else branch yields the wrong type, written past the
    builder's check, as serialized IR can reach the lowerers."""
    b, branch = _open_if_else([F32])
    then_ctx, else_ctx = branch.__enter__()
    with then_ctx:
        b.scf_yield(b.const_f32(1.0))
    with else_ctx:
        b.scf_yield(b.const_i32(1))
    b.ret()
    return b.kernel


def test_both_lowerings_and_the_verifier_refuse_a_branch_that_yields_the_wrong_type():
    kernel = _raw_if_else_kernel()
    message = "scf.if_else result 0 is f32, but its else branch yields i32"
    _assert_refused(kernel, ValueError, message)


def _descriptor_if_else_kernel():
    """An if/else whose branches yield buffer descriptors, the result typed as
    the descriptor's nominal ``<4 x i32>``."""
    b = IRBuilder("if_else_descriptor")
    x = b.param("X", PtrType(F32, "global"))
    lane = b.thread_id_x()
    branch = b.scf_if_else(
        b.cmp_lt(lane, b.const_i32(SPLIT)), result_types=[VectorType(I32, 4)]
    )
    then_ctx, else_ctx = branch.__enter__()
    with then_ctx:
        b.scf_yield(b.buffer_rsrc(x, b.const_i32(64)))
    with else_ctx:
        b.scf_yield(b.buffer_rsrc(x, b.const_i32(32)))
    return b, branch


DESCRIPTOR_MESSAGE = (
    "scf.if_else result 0: its then branch yields a buffer descriptor, which an "
    "if/else cannot carry; yield the values the descriptor is built from and build "
    "it after the if/else"
)


def test_builder_refuses_a_branch_that_yields_a_buffer_descriptor():
    """A descriptor is <4 x i32> in the IR but an opaque ``ptr addrspace(8)``
    (HIP ``rsrc_t``) once lowered: a phi or variable of the nominal type would
    not compile."""
    _, branch = _descriptor_if_else_kernel()
    with pytest.raises(ValueError) as error:
        branch.__exit__(None, None, None)
    assert str(error.value) == DESCRIPTOR_MESSAGE


def test_both_lowerings_and_the_verifier_refuse_a_branch_that_yields_a_buffer_descriptor():
    b, _ = _descriptor_if_else_kernel()
    b.ret()
    _assert_refused(b.kernel, ValueError, DESCRIPTOR_MESSAGE)


LDS_RESULT_MESSAGE = (
    "scf.if_else result 0 is an LDS allocation, which an if/else cannot carry; "
    "allocate before the if/else and yield an offset into the allocation instead"
)


def test_builder_refuses_an_lds_allocation_result():
    """An LDS allocation is a name the lowerings resolve to its base, not a
    value, so neither a phi nor a HIP variable can hold one."""
    b = IRBuilder("if_else_lds_builder")
    cond = b.cmp_lt(b.thread_id_x(), b.const_i32(SPLIT))
    with pytest.raises(ValueError) as error:
        b.scf_if_else(cond, result_types=[SmemType(F32, (64,))])
    assert str(error.value) == LDS_RESULT_MESSAGE


def _raw_lds_result_if_else(elem):
    """An if/else with one result, an LDS tile of ``elem``, its branches
    yielding two such allocations, written past the builder."""
    b = IRBuilder(f"if_else_lds_{elem.name}")
    lane = b.thread_id_x()
    first = b.smem_alloc(F32, (64,), name_hint="first")
    second = b.smem_alloc(F32, (64,), name_hint="second")
    result = Value("%if_lds0", SmemType(elem, (64,)))
    then_r, else_r = Region("then"), Region("else")
    op = Op(
        name="scf.if_else",
        operands=[b.cmp_lt(lane, b.const_i32(SPLIT))],
        results=[result],
        regions=[then_r, else_r],
    )
    result.op = op
    b._emit(op)
    for region, value in ((then_r, first), (else_r, second)):
        b.push_region(region)
        b.scf_yield(value)
        b.pop_region()
    b.ret()
    return b.kernel


def test_both_lowerings_and_the_verifier_refuse_an_lds_allocation_result():
    """The same if/else written past the builder, its branches yielding two
    allocations, is refused by the verifier and every lowering of both
    engines instead of lowering to a phi of undefined names."""
    kernel = _raw_lds_result_if_else(F32)
    _assert_refused(kernel, ValueError, LDS_RESULT_MESSAGE)


def test_a_packed_only_lds_result_is_refused_as_packed_only_first():
    """Both engines check an if/else result's type in one order: the
    packed-only rule (TypeError) before the LDS rule (ValueError), so an LDS
    tile of i4 gets the packed-only refusal from the verifier and both
    engines' lowerings. ``test_packed_only.py`` checks the same order in both
    builders."""
    kernel = _raw_lds_result_if_else(I4)
    message = (
        "scf.if_else: i4 values travel packed in 32-bit words and have no scalar "
        "or vector form -- keep them in i32 storage and unpack explicitly"
    )
    _assert_refused(kernel, TypeError, message)


STRAY_YIELD_MESSAGE = (
    "scf.if_else has no results, so its then branch cannot contain scf.yield"
)

# Where a test puts the extra scf.yield it writes: directly in the region being
# built, or in an scf.if within it. An scf.if takes no yields, so a yield in one
# belongs to the region around it and is refused as such.
PLACEMENTS = ("direct", "in_scf_if")


def _emit_placed(b, placement, emit):
    """Call ``emit()`` to emit into ``b``'s current region, at ``placement``
    (one of ``PLACEMENTS``)."""
    if placement == "in_scf_if":
        with b.scf_if(b.cmp_lt(b.thread_id_x(), b.const_i32(SPLIT))):
            emit()
    else:
        emit()


def _extra_yield(b, placement):
    """Emit an ``scf.yield`` of one i32 into ``b``'s current region, at
    ``placement`` (one of ``PLACEMENTS``)."""
    _emit_placed(b, placement, lambda: b.scf_yield(b.const_i32(1)))


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_builder_refuses_a_yield_in_an_if_else_without_results(placement):
    """Nothing takes the values of a yield in a branch of an if/else without
    results, so leaving the ``with`` refuses one."""
    b, branch = _open_if_else([])
    with pytest.raises(ValueError) as error, branch as (then_ctx, else_ctx):
        with then_ctx:
            _extra_yield(b, placement)
        with else_ctx:
            pass
    assert str(error.value) == STRAY_YIELD_MESSAGE


def _stray_yield_if_else(b, cond, placement):
    """Build an if/else without results whose then branch yields one value at
    ``placement``, past the builder's check, as serialized IR can reach the
    lowerers."""
    then_ctx, _ = b.scf_if_else(cond).__enter__()
    with then_ctx:
        _extra_yield(b, placement)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_both_lowerings_and_the_verifier_refuse_a_yield_in_an_if_else_without_results(
    placement,
):
    """The same if/else written past the builder is refused by the verifier
    and every lowering of both engines, not with a stale loop error."""
    b = IRBuilder("if_else_stray_yield")
    _stray_yield_if_else(b, b.cmp_lt(b.thread_id_x(), b.const_i32(SPLIT)), placement)
    b.ret()
    _assert_refused(b.kernel, ValueError, STRAY_YIELD_MESSAGE)


def _carrying_loop_body(b, trips, fill, unroll=False, carry_name="acc", iv_name="k0"):
    """Emit into ``b`` a loop of ``trips`` trips (unrolled when ``unroll``)
    with induction variable ``iv_name`` that carries the lane id as
    ``carry_name``; its body holds what ``fill(b, k)`` emits, then yields the
    carried value."""
    loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(trips),
        b.const_i32(1),
        [(carry_name, b.thread_id_x())],
        unroll=unroll,
        iv_name=iv_name,
    )
    with loop as (k, (acc,)):
        fill(b, k)
        b.scf_yield(acc)


def _carrying_loop_kernel(name, trips, fill, unroll=False):
    """A kernel holding one :func:`_carrying_loop_body` loop."""
    b = IRBuilder(name)
    _carrying_loop_body(b, trips, fill, unroll)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_yield_in_an_if_else_without_results_in_a_loop_is_refused_as_such(placement):
    """In a loop body the stray value is not counted toward the loop's yields:
    the if/else rule refuses it before the loop's yield count is checked."""
    kernel = _carrying_loop_kernel(
        "if_else_stray_yield_loop",
        LOOP_TRIPS,
        lambda b, k: _stray_yield_if_else(b, b.cmp_lt(k, b.thread_id_x()), placement),
    )
    _assert_refused(kernel, ValueError, STRAY_YIELD_MESSAGE)


EARLY_YIELD_MESSAGE = (
    "scf.if_else has 1 results, so its then branch must yield only at its end"
)


def _early_yield_if_else_kernel(placement):
    """An if/else with one result whose then branch yields at ``placement``
    before the ``scf.yield`` that ends it, written past the builder's check."""
    b, branch = _open_if_else([I32])
    then_ctx, else_ctx = branch.__enter__()
    with then_ctx:
        _extra_yield(b, placement)
        b.scf_yield(b.const_i32(5))
    with else_ctx:
        b.scf_yield(b.const_i32(2))
    b.ret()
    return b, branch


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_builder_refuses_a_yield_before_the_end_of_a_branch(placement):
    """Only the yield that ends a branch gives the results their values, so
    an earlier one is refused rather than silently dropped."""
    _, branch = _early_yield_if_else_kernel(placement)
    with pytest.raises(ValueError) as error:
        branch.__exit__(None, None, None)
    assert str(error.value) == EARLY_YIELD_MESSAGE


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_both_lowerings_and_the_verifier_refuse_a_yield_before_the_end_of_a_branch(
    placement,
):
    b, _ = _early_yield_if_else_kernel(placement)
    _assert_refused(b.kernel, ValueError, EARLY_YIELD_MESSAGE)


ORPHAN_YIELD_MESSAGE = (
    "scf.yield must be directly in an scf.for body or in a branch of an "
    "scf.if_else with results"
)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_a_yield_in_the_kernel_body_is_refused(placement):
    """Nothing takes the values of a yield in the kernel body, directly or in
    an scf.if there; the verifier and both lowerings of both engines refuse it
    (the C++ LLVM lowering used to accept it)."""
    b = IRBuilder("orphan_yield")
    _extra_yield(b, placement)
    b.ret()
    _assert_refused(b.kernel, ValueError, ORPHAN_YIELD_MESSAGE)


def test_a_yield_in_an_scf_if_in_a_loop_body_is_refused():
    """A loop takes only the yields directly in its body: one in an scf.if
    there would add a value to the loop's yields on the trips that take the
    branch."""
    kernel = _carrying_loop_kernel(
        "orphan_yield_loop", LOOP_TRIPS, lambda b, k: _extra_yield(b, "in_scf_if")
    )
    _assert_refused(kernel, ValueError, ORPHAN_YIELD_MESSAGE)


def test_an_if_else_in_an_scf_if_in_a_loop_keeps_its_own_yields():
    """The search for a yield nothing takes stops at an if/else with results:
    one in an scf.if in a loop body is accepted by the verifier and both
    lowerings of every available engine."""

    def fill(b, k):
        with b.scf_if(b.cmp_lt(k, b.thread_id_x())):
            then_ctx, else_ctx = b.scf_if_else(
                b.cmp_lt(k, b.const_i32(SPLIT)), result_types=[I32]
            ).__enter__()
            with then_ctx:
                b.scf_yield(b.const_i32(1))
            with else_ctx:
                b.scf_yield(b.const_i32(2))

    kernel = _carrying_loop_kernel("if_else_in_scf_if_loop", LOOP_TRIPS, fill)
    _assert_lowered(kernel)


def test_a_loop_in_a_branch_keeps_its_own_yield():
    """A loop nested in a branch of an if/else without results owns the yield
    in its body: the verifier and both lowerings of every available engine
    accept it."""
    b, branch = _open_if_else([])
    with branch as (then_ctx, else_ctx):
        with then_ctx:
            lane = b.thread_id_x()
            loop = b.scf_for_iter(
                b.const_i32(0), b.const_i32(LOOP_TRIPS), b.const_i32(1), [("acc", lane)]
            )
            with loop as (_, (acc,)):
                b.scf_yield(b.add(acc, b.const_i32(1)))
        with else_ctx:
            pass
    b.ret()
    _assert_lowered(b.kernel)


def _zero_trip_if_else_kernel(stray_yield, placement="direct"):
    """An unrolled loop of zero trips whose body holds, at ``placement`` (one
    of ``PLACEMENTS``), an if/else without results, its then branch holding a
    yield when ``stray_yield``."""

    def emit_if_else(b, k):
        then_ctx, else_ctx = b.scf_if_else(b.cmp_lt(k, b.thread_id_x())).__enter__()
        with then_ctx:
            if stray_yield:
                _extra_yield(b, "direct")
        with else_ctx:
            pass

    def fill(b, k):
        _emit_placed(b, placement, lambda: emit_if_else(b, k))

    return _carrying_loop_kernel("zero_trip_if_else", 0, fill, unroll=True)


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_an_unrolled_loop_of_zero_trips_refuses_a_malformed_if_else(placement):
    """An unrolled loop of zero trips lowers no body, yet the LLVM lowering
    refuses the body's malformed if/else, directly in the body or in an scf.if
    within it, as the verifier and the HIP lowering do."""
    kernel = _zero_trip_if_else_kernel(True, placement)
    _assert_refused(kernel, ValueError, STRAY_YIELD_MESSAGE)


def test_an_unrolled_loop_of_zero_trips_refuses_a_malformed_nested_loop():
    """The same holds for a loop nested in that body: its own rule applies."""

    def fill(b, _k):
        _carrying_loop_body(
            b,
            LOOP_TRIPS,
            lambda inner_b, _inner_k: _extra_yield(inner_b, "in_scf_if"),
            carry_name="inner_acc",
            iv_name="inner_k",
        )

    kernel = _carrying_loop_kernel("zero_trip_nested_loop", 0, fill, unroll=True)
    _assert_refused(kernel, ValueError, ORPHAN_YIELD_MESSAGE)


def test_an_unrolled_loop_of_zero_trips_accepts_a_well_formed_body():
    kernel = _zero_trip_if_else_kernel(False)
    _assert_lowered(kernel)


@pytest.mark.parametrize("operands,regions", [(0, 2), (1, 1), (1, 0), (2, 2)])
@pytest.mark.parametrize("result_types", [(), (I32,)])
def test_a_malformed_if_else_is_refused_not_lowered(operands, regions, result_types):
    """An scf.if_else without one condition and two branches, written as a raw
    op, is refused before any lowering reads it, with or without results."""
    b = IRBuilder("if_else_malformed")
    lane = b.thread_id_x()
    cond = b.cmp_lt(lane, b.const_i32(SPLIT))
    results = [Value(f"%if_raw{i}", t) for i, t in enumerate(result_types)]
    op = Op(
        name="scf.if_else",
        operands=[cond] * operands,
        results=results,
        regions=[Region(name) for name in ("then", "else")[:regions]],
    )
    for result in results:
        result.op = op
    b._emit(op)
    b.ret()
    message = (
        f"scf.if_else: expected 1 condition / 2 branches, got {operands} / {regions}"
    )
    _assert_refused(b.kernel, ValueError, message)


# --------------------------------------------------------------------------
# GPU: lanes take both branches, on both compile routes
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


def _run(blob, kernel_name, x):
    """Launch one workgroup of ``LANES`` lanes; return Y, Z and W."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    outputs = [
        np.zeros(LANES, dtype=np.float32),
        np.zeros(2 * LANES, dtype=np.int32),
        np.zeros(LANES, dtype=np.int32),
    ]
    buffers = [runtime.alloc(a.nbytes) for a in [x, *outputs]]
    try:
        runtime.memcpy_h2d(
            buffers[0], (ctypes.c_ubyte * x.nbytes).from_buffer(x), x.nbytes
        )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (LANES, 1, 1),
            struct.pack("4Q", *buffers),
        )
        for buffer, out in zip(buffers[1:], outputs):
            runtime.memcpy_d2h(
                (ctypes.c_ubyte * out.nbytes).from_buffer(out), buffer, out.nbytes
            )
    finally:
        for buffer in buffers:
            runtime.free(buffer)
        module.unload()
    return outputs


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_lanes_take_both_branches(route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    kernel = _branch_kernel(f"if_else_gpu_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel

        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        if shutil.which("hipcc") is None:
            pytest.skip("hipcc not found")
        from rocke.helpers.compile import compile_kernel_via_hipcc

        blob = compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    x = np.random.default_rng(3).standard_normal(LANES).astype(np.float32)
    y, z, w = _run(blob, kernel.name, x)
    expected_y, expected_z, expected_w = _expected(x)
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(w, expected_w)
