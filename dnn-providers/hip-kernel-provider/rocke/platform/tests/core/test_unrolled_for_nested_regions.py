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
flavor, compile it, and, when a GPU is present, run it and compare every lane
with the value worked out on the host.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.tests.engines import lower_with_both_engines

FLAVORS = ("llvm20", "llvm22", "llvm23")
LANES = 64
TRIPS = 4
# The GPUs the run test has run on; on any other it skips until it has.
GPU_ARCHES_RUN_ON = ("gfx90a",)
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
    b, y, z, lane = _kernel("unroll_carried_in_branch_yield")
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
        with b.scf_if(b.cmp_lt(lane, b.const_i32(SPLIT))):
            with b.scf_if(b.cmp_eq(lane, k)):
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
    b, y, z, lane = _kernel("unroll_rolled_inner_loop")
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


def _unrolled_loop_in_unrolled_loop():
    """An unrolled ``scf.for`` of three trips inside the unrolled body; its body
    reads the outer induction variable and defines a value."""
    b, y, z, lane = _kernel("unroll_unrolled_inner_loop")
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
}


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_both_engines_emit_the_same_text(case, flavor):
    lower_with_both_engines(CASES[case][0](), "gfx942", llvm_flavor=flavor)


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_trip_defines_its_own_values(case):
    """Every SSA name is defined once."""
    text = lower_with_both_engines(CASES[case][0](), "gfx942")
    defined = [
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith("  %") and "=" in line
    ]
    assert len(defined) == len(set(defined)), sorted(
        name for name in set(defined) if defined.count(name) > 1
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_kernel_compiles(case):
    from rocke.helpers.compile import compile_kernel

    assert compile_kernel(CASES[case][0](), arch="gfx942").hsaco


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


def _run(blob, kernel_name):
    """Launch one workgroup of ``LANES`` lanes on zeroed ``Y`` and ``Z``."""
    from rocke.runtime.hip_module import Runtime

    runtime = Runtime()
    module = runtime.load_module(blob)
    outputs = [np.zeros(LANES, dtype=np.int32) for _ in range(2)]
    buffers = [runtime.alloc(out.nbytes) for out in outputs]
    try:
        for buffer, out in zip(buffers, outputs):
            runtime.memcpy_h2d(
                buffer, (ctypes.c_ubyte * out.nbytes).from_buffer(out), out.nbytes
            )
        runtime.launch_blocking(
            module.get_function(kernel_name),
            (1, 1, 1),
            (LANES, 1, 1),
            struct.pack("2Q", *buffers),
        )
        for buffer, out in zip(buffers, outputs):
            runtime.memcpy_d2h(
                (ctypes.c_ubyte * out.nbytes).from_buffer(out), buffer, out.nbytes
            )
    finally:
        for buffer in buffers:
            runtime.free(buffer)
        module.unload()
    return outputs


@pytest.mark.parametrize("case", sorted(CASES))
def test_gpu_every_lane_matches_the_host(case):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    if arch not in GPU_ARCHES_RUN_ON:
        pytest.skip(f"has not run on {arch} hardware yet; enable once it has")
    from rocke.helpers.compile import compile_kernel

    build, expected = CASES[case]
    kernel = build()
    y, z = _run(compile_kernel(kernel, arch=arch).hsaco, kernel.name)
    expected_y, expected_z = expected(np.arange(LANES, dtype=np.int32))
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(z, expected_z)
