# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``IRBuilder.gpu_checked_value``: facts on a value only the GPU can see,
checked on the GPU.

What is pinned here:

- what the check emits (a compare per stated side, one unsigned compare for
  ``0 <= v < n``, ``scf.if`` on the failure holding ``tile.gpu_check_failed``)
  and the range marker it returns, read back by ``stated_facts_of``;
- every refusal, with its exact text: a value the host already checks (a
  parameter, a value computed from parameters, constants or thread ids),
  arithmetic on an unchecked load before the check, and a malformed call;
- that a load reaching the value only through a condition is accepted, and that
  the walks see through loops, branches and selects;
- that an op that reads memory is a load to both walks even when it is pure
  (``av_load_b128``, an LDS transposed read): its value is GPU-only, and
  arithmetic on it before the check is refused;
- the range marker's forms (a ``ge`` operand, the ``by`` attr) and the
  verifier's texts for a bad one;
- that the LLVM and HIP lowerings of a checked kernel make the failure path
  store 1 to the kernel's GPU-check word and return from the kernel, with no
  trap; that a ``tile.gpu_check_failed`` that is not bare is refused by the
  verifier and every lowering; and that ``optimize_kernel`` keeps the check;
- through ``ROCKE_GPU_CHECKED_VALUE_CLI`` (the ``rocke_gpu_checked_value``
  ctest, so no ``rocke_engine`` binding is needed), that the C++ builder
  ``rocke_b_gpu_checked_value`` emits the same IR and refuses with the same texts for
  every case in ``CASES``, that the C++ verifier gives the same messages for
  every marker in ``MARKERS``, that the C++ walks count exactly the ops of
  ``value_sources.MEMORY_READ_OPS`` as memory reads, and that the C++ engine
  lowers every accepted case to the same LLVM IR and HIP source and refuses a
  failure path that is not bare with the same text;
- when a GPU that they have run on is present (``_RUN_ON``), on the LLVM and
  the hipcc route, each in a child process that loads a code object compiled
  once in the test process: a launch whose loaded value meets the facts runs
  to its store; one whose value breaks them raises ``GpuCheckFailed`` with
  the store not done, and the same process then launches again and stores;
  and with two waves of which one breaks the facts (a wave-uniform value),
  that wave stops while the other passes a barrier and stores, and the launch
  still ends with ``GpuCheckFailed``; and with a per-lane value, the lanes
  that break the facts stop while the rest of their wave passes the barrier
  and stores;
- how the runtime reads the word, on the GPU: two streams with unread
  launches of one kernel both see a failure; a word set by a launch the
  runtime never saw is not blamed on the next; ``check_gpu_checks`` and
  ``release_pending_for_stream`` read it after a wait outside the runtime;
  unloading a module drops its unread words; a kernel without a check has no
  word and leaves no HIP error behind; a checked launch on a stream that is
  being captured into a graph is refused; and a wait with nothing noted asks
  HIP nothing;
- that the failure path comes before a gfx1250 cluster barrier, the same in
  both engines (the GPU case is skipped until it has run on gfx1250 hardware).
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

from rocke.core.arch import ArchTarget
from rocke.core.facts import StatedFacts, stated_facts_of
from rocke.core.intexpr import product_of
from rocke.core.ir import (
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    one_unsigned_compare_checks,
)
from rocke.core.ir_print import print_ir
from rocke.core.ir_serialize import canonicalize, parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.passes import optimize_kernel
from rocke.core.value_sources import MEMORY_READ_OPS
from rocke.core.verify import verify
from rocke.helpers.compile import compile_kernel, compile_kernel_via_hipcc
from rocke.tests.built_programs import assert_refuses_arguments, built_program
from rocke.tests.gpu import probe_device_arch
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

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


def _av_loaded(b, p):
    """One wave-uniform i32 of an ``av_load_b128``: a pure op that reads memory."""
    return b.readfirstlane(b.vec_extract(b.av_load_b128(p.cu), 0))


def _lds_read(b):
    """One i32 of an LDS transposed read (``ds_read_tr_b8``): a pure op that
    reads memory."""
    tile = b.smem_alloc(I8, [16, 16])
    zero = b.const_i32(0)
    return b.sext(b.vec_extract(b.ds_read_tr_b8(tile, zero, zero, dtype=I8), 0), I32)


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
    "av_load_value": lambda b, p: b.gpu_checked_value(_av_loaded(b, p), ge=0, le=p.T),
    "lds_transposed_read_value": lambda b, p: b.gpu_checked_value(
        _lds_read(b), ge=0, le=p.T
    ),
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
    "refuse_addition_on_an_av_load": lambda b, p: b.gpu_checked_value(
        b.add(_av_loaded(b, p), b.const_i32(1)), ge=0
    ),
    "refuse_addition_on_an_lds_transposed_read": lambda b, p: b.gpu_checked_value(
        b.add(_lds_read(b), b.const_i32(1)), ge=0
    ),
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
    # The range-marker builder itself, refusing a Value end of another type.
    "refuse_marker_end_of_another_type": lambda b, p: b._assume_range(
        _loaded(b, p, b.block_id_x()), ge=p.N64
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
LLVM_FLAVORS = ["llvm20", "llvm22"]
REFUSED = [name for name in CASES if name.startswith("refuse_")]
# The HIP lowering has no av_load_b128 or ds_read_tr_b8.
HIP_LOWERED = [
    name
    for name in ACCEPTED
    if name not in ("av_load_value", "lds_transposed_read_value")
]


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


def test_each_stated_side_takes_one_compare_and_one_failure_path():
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
    assert [op.name for op in branch.regions[0].ops] == ["tile.gpu_check_failed"]
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
    assert stated_facts_of(start) == StatedFacts(0, 4096, 1, 0)
    # gt=T resolves through T's own lower bound: end > T >= 0.
    assert stated_facts_of(end) == StatedFacts(1, (1 << 20) - 1, 8, 0)
    assert stated_facts_of(seqlen) == StatedFacts(0, 4095, 1, 0)


def test_an_end_at_the_type_maximum_states_no_upper_end():
    b = IRBuilder("type_max")
    wide = b.sext(b.block_id_x(), I64)
    checked = b.gpu_checked_value(wide, ge=0, le=INT64_MAX)
    assert "lt" not in checked.op.attrs
    assert stated_facts_of(checked) == StatedFacts(0, None, 1, 0)


def test_optimize_kernel_keeps_the_check():
    kernel = _build("varlen_prologue").kernel
    optimize_kernel(kernel)
    failure_paths = [
        inner.name
        for op in kernel.body.ops
        if op.name == "scf.if"
        for inner in op.regions[0].ops
    ]
    assert failure_paths == ["tile.gpu_check_failed"] * 3


@pytest.mark.parametrize("name", ACCEPTED)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_a_false_fact_stores_to_the_word_and_returns(name, flavor):
    ir = lower_kernel_to_llvm(_build(name).kernel, arch="gfx942", llvm_flavor=flavor)
    assert "llvm.trap" not in ir
    if name != "param_value_bound":
        word = f"@rocke_gpu_check_failed_{name}"
        assert ir.count(f"{word} = protected addrspace(1) externally_initialized ") == 1
        store = (
            f"  store atomic i32 1, ptr addrspace(1) {word} "
            'syncscope("agent") monotonic, align 4\n'
        )
        failures = ir.count(f"{store} ret void\n")
        assert failures >= 1 and failures == ir.count(store), name


@pytest.mark.parametrize("name", HIP_LOWERED)
def test_in_hip_a_false_fact_stores_to_the_word_and_returns(name):
    hip = lower_kernel_to_hip(_build(name).kernel, arch="gfx942")
    assert "__builtin_trap" not in hip
    if name != "param_value_bound":
        word = f"rocke_gpu_check_failed_{name}"
        assert hip.count(f"__device__ int {word};\n") == 1
        store = (
            f"__hip_atomic_store(&{word}, 1, __ATOMIC_RELAXED, "
            "__HIP_MEMORY_SCOPE_AGENT);"
        )
        failures = hip.count(f"{store}\n  return;\n")
        assert failures >= 1 and failures == hip.count(store), name


def _failure_path_with_an_operand_kernel():
    """A tile.gpu_check_failed that takes an i32 operand, written past the
    builder (as serialized IR can reach the verifier and the lowerers)."""
    b = IRBuilder("check_failed_operand")
    b.param("Y", PtrType(I32, "global"))
    b._op("tile.gpu_check_failed", [b.const_i32(1)], [])
    b.ret()
    return b.kernel


NOT_BARE = "gpu_check_failed takes no operands, results or regions"


def test_verifier_and_every_lowering_refuse_a_failure_path_that_is_not_bare():
    kernel = _failure_path_with_an_operand_kernel()
    assert NOT_BARE in [d.message for d in verify(kernel)]
    with pytest.raises(ValueError, match=f"^{NOT_BARE}$"):
        lower_kernel_to_llvm(kernel, arch="gfx942", llvm_flavor="llvm20")
    with pytest.raises(ValueError, match=f"^{NOT_BARE}$"):
        lower_kernel_to_hip(kernel, arch="gfx942")


# --------------------------------------------------------------------------
# refusals, with the exact texts
# --------------------------------------------------------------------------


def test_a_parameter_is_checked_by_the_host():
    assert _refused_text("refuse_parameter") == (
        "gpu_checked_value: M is a kernel parameter. The host checks every fact of a "
        "kernel parameter before each launch through KernelLauncher.from_artifact and "
        "refuses the launch if one fails, so a GPU check here would only repeat that "
        "check.\n"
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
        "the host checks before each launch through KernelLauncher.from_artifact. "
        "Fix: state the bound on the parameter: "
        'b.param("T", I32, ...). The IR range walk derives the facts of T - 1 from it.'
    )
    assert _refused_text("refuse_computed_from_parameters") == (
        "gpu_checked_value: T + M is computed only from kernel parameters (T, M), which "
        "the host checks before each launch through KernelLauncher.from_artifact. "
        "Fix: state the bounds on the parameters: "
        'b.param("T", I32, ...), b.param("M", I32, ...). The IR range walk derives the '
        "facts of T + M from them."
    )


def test_a_thread_id_is_bounded_by_the_workgroup_size():
    assert _refused_text("refuse_thread_id") == (
        "gpu_checked_value: thread_id_x is bounded by the kernel's workgroup size, "
        "which the host checks before each launch through KernelLauncher.from_artifact.\n"
        'Fix: b.kernel.attrs["workgroup_size"] = 64'
    )
    assert _refused_text("refuse_thread_id_and_parameter") == (
        "gpu_checked_value: thread_id_x + T is computed only from thread ids and kernel "
        "parameters (T), which the host bounds before each launch through "
        "KernelLauncher.from_artifact.\n"
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
            "gpu_checked_value: states two lower bounds (ge, gt) -- state one",
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
        (
            "refuse_marker_end_of_another_type",
            "arith.assume_range: ge operand type i64 != operand type i32",
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


def test_one_unsigned_compare_refuses_a_value_other_than_i32_or_i64():
    with pytest.raises(ValueError) as refused:
        one_unsigned_compare_checks(0, 10, I8)
    assert str(refused.value) == (
        "one_unsigned_compare_checks failed - t=i8, expected i32 or i64"
    )


# --------------------------------------------------------------------------
# the range marker's forms
# --------------------------------------------------------------------------


def test_the_marker_takes_a_value_at_either_end():
    b = IRBuilder("ends")
    p = _declare(b)
    raw = _loaded(b, p, b.block_id_x())
    marked = b._assume_range(raw, ge=p.T, ge_plus=1, lt=p.M, lt_plus=-1)
    assert marked.op.operands == [raw, p.T, p.M]
    assert stated_facts_of(marked) == StatedFacts(1, (1 << 20) - 2, 1, 0)
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
    assert (
        str(refused.value)
        == "arith.assume_range: ge operand type i64 != operand type i32"
    )
    with pytest.raises(ValueError) as refused:
        b._assume_range(raw, by="someone")
    assert "by names no marker writer" in str(refused.value)


# --------------------------------------------------------------------------
# the C++ engine: same IR, same texts
# --------------------------------------------------------------------------


def _cli() -> Path:
    """``ROCKE_GPU_CHECKED_VALUE_CLI`` names the built ``rocke_gpu_checked_value``:
    ``tests/run_all.py`` resolves it from a configured build and the installed
    ``rocke_pytest_*`` ctest entries set it. A value that names no file fails."""
    return built_program("ROCKE_GPU_CHECKED_VALUE_CLI", "rocke_gpu_checked_value")


@pytest.mark.parametrize(
    "arguments",
    [
        ("--lower-lvm", "gfx942", "llvm20"),
        ("--lower-llvm", "gfx942"),
        ("--lower-hip",),
        ("--cases", "extra"),
        ("--memory-read-ops", "extra"),
    ],
)
def test_cpp_program_refuses_a_mistyped_mode(arguments):
    """A mistyped mode or a wrong argument count exits 2 with the usage, instead
    of running the self-test and passing."""
    assert_refuses_arguments(_cli(), *arguments)


def test_cpp_counts_the_same_ops_as_memory_reads():
    """The C++ walks' memory reads (``rocke_i_opcode_reads_memory``) are exactly
    ``MEMORY_READ_OPS``: each engine lists them by hand, so this pins the two
    lists to each other."""
    printed = subprocess.run(
        [str(_cli()), "--memory-read-ops"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    assert sorted(printed.split()) == sorted(MEMORY_READ_OPS)


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


def _cpp_lowering(kernel, *mode: str) -> subprocess.CompletedProcess:
    """The C++ engine's lowering of ``kernel``: ``mode`` is ``--lower-llvm ARCH
    FLAVOR`` or ``--lower-hip ARCH``. Exit 1 with the refusal on stderr."""
    return subprocess.run(
        [str(_cli()), *mode],
        input=serialize(kernel),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("name", ACCEPTED)
@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_cpp_lowers_every_checked_kernel_to_the_same_llvm(name, flavor):
    kernel = _build(name).kernel
    lowered = _cpp_lowering(kernel, "--lower-llvm", "gfx942", flavor)
    assert lowered.returncode == 0, lowered.stderr
    assert lowered.stdout == lower_kernel_to_llvm(
        kernel, arch="gfx942", llvm_flavor=flavor
    )


@pytest.mark.parametrize("name", HIP_LOWERED)
def test_cpp_lowers_every_checked_kernel_to_the_same_hip(name):
    kernel = _build(name).kernel
    lowered = _cpp_lowering(kernel, "--lower-hip", "gfx942")
    assert lowered.returncode == 0, lowered.stderr
    assert lowered.stdout == lower_kernel_to_hip(kernel, arch="gfx942")


def test_cpp_lowerings_refuse_a_failure_path_that_is_not_bare():
    kernel = _failure_path_with_an_operand_kernel()
    for mode in (("--lower-llvm", "gfx942", "llvm20"), ("--lower-hip", "gfx942")):
        lowered = _cpp_lowering(kernel, *mode)
        assert lowered.returncode == 1, mode
        assert lowered.stderr.rstrip("\n").endswith(NOT_BARE), lowered.stderr


def test_cpp_lowers_a_check_before_a_cluster_barrier_the_same():
    kernel = _cluster_kernel("gpu_checked_cluster")
    lowered = _cpp_lowering(kernel, "--lower-llvm", "gfx1250", "llvm23")
    assert lowered.returncode == 0, lowered.stderr
    assert lowered.stdout == lower_kernel_to_llvm(
        kernel, arch="gfx1250", llvm_flavor="llvm23"
    )


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


def _two_wave_kernel(name: str, wave: int, *, per_lane: bool):
    """Two waves, then a workgroup barrier, then every lane ``i`` stores its
    checked value to ``Y[i]``. Each lane checks ``0 <= value <= T`` on:

    - ``per_lane=False``: ``cu[w]`` for its wave ``w``, made wave-uniform with
      ``readfirstlane``, so a wave whose value breaks the facts returns whole
      and must not hold the other wave at the barrier;
    - ``per_lane=True``: ``cu[i % 2]``, so in each wave the even lanes return
      and the odd lanes go on to the barrier and their stores.
    """
    b = IRBuilder(name)
    b.kernel.attrs["workgroup_size"] = 2 * wave
    y = b.param("Y", PtrType(I32, "global"))
    cu = b.param("cu", PtrType(I32, "global"), readonly=True)
    t = b.param("T", I32, ge=0, le=4096)
    lane = b.thread_id_x()
    if per_lane:
        loaded = b.global_load(cu, b.mod(lane, b.const_i32(2)), I32)
    else:
        loaded = b.readfirstlane(b.global_load(cu, b.div(lane, b.const_i32(wave)), I32))
    value = b.gpu_checked_value(loaded, ge=0, le=t)
    b.sync()
    b.global_store(y, lane, value)
    b.ret()
    return b.kernel


def _cluster_kernel(name: str):
    """A cluster of two one-lane workgroups: workgroup ``g`` checks
    ``0 <= cu[g] <= T``, then every workgroup of the cluster meets at the
    cluster barrier and stores its checked value to ``Y[g]``. A workgroup whose
    value breaks the facts returns before the barrier and must not hold the
    other one there."""
    b = IRBuilder(name)
    b.kernel.attrs["workgroup_size"] = 1
    b.set_cluster_dims(2)
    y = b.param("Y", PtrType(I32, "global"))
    cu = b.param("cu", PtrType(I32, "global"), readonly=True)
    t = b.param("T", I32, ge=0, le=4096)
    group = b.block_id_x()
    value = b.gpu_checked_value(
        b.readfirstlane(b.global_load(cu, group, I32)), ge=0, le=t
    )
    b.cluster_barrier()
    b.global_store(y, group, value)
    b.ret()
    return b.kernel


def _plain_kernel(name: str) -> KernelDef:
    """A kernel with no check: it only returns."""
    b = IRBuilder(name)
    b.ret()
    return b.kernel


def _kernel_for(case: str, route: str, wave: int) -> KernelDef:
    """The kernel the child launches for ``case`` on ``route``: the two-wave,
    cluster or unchecked kernel for those cases, and the checked-store kernel
    for every other case (a list of ``cu`` values). Its name carries the
    route, as the failure texts the tests read back name the kernel."""
    if case in ("two_waves", "per_lane"):
        return _two_wave_kernel(
            f"gpu_checked_{case}_{route}", wave, per_lane=case == "per_lane"
        )
    if case == "cluster":
        return _cluster_kernel(f"gpu_checked_cluster_{route}")
    if case == "unchecked":
        return _plain_kernel(f"plain_{route}")
    return _checked_store_kernel(f"gpu_checked_{route}")


# How each route compiles a kernel: rocKE's LLVM lowering and COMGR, or its
# HIP lowering and hipcc.
_COMPILE_ON_ROUTE = {"llvm": compile_kernel, "hipcc": compile_kernel_via_hipcc}


def _code_object(kernel: KernelDef, route: str, arch: str) -> bytes:
    """``kernel`` compiled for ``arch`` on ``route``."""
    return _COMPILE_ON_ROUTE[route](kernel, arch=arch).hsaco


# Runs in the child: loads the code object the test compiled, launches its
# kernel with T = 10 and the ``cu`` values of ``case``, and prints how each
# launch ended and what it stored (-1 where nothing was stored). A child, so
# that a failure path that ever stops ending the launch fails this test at the
# time bound below instead of hanging the run, and one child per case, so that
# no case starts from HIP or runtime state another case left behind. The child
# compiles nothing and imports neither numpy nor this file: it imports the
# rocke package for its runtime, loads the code object, and runs the launches
# and reads under test.
_CHILD = textwrap.dedent(
    """
    import ctypes, struct, sys
    from rocke.runtime import hip_module
    from rocke.runtime.hip_module import GpuCheckCaptureRefused, GpuCheckFailed, Runtime

    case, code_object, kernel_name = sys.argv[1:4]
    wave = int(sys.argv[4])
    runtime = Runtime()
    with open(code_object, "rb") as code:
        module = runtime.load_module(code.read())
    function = module.get_function(kernel_name)


    def buffers(lanes, loaded):
        y_dev = runtime.alloc(4 * lanes)
        runtime.memset(y_dev, 0xFF, 4 * lanes)
        cu = (ctypes.c_int32 * len(loaded))(*loaded)
        cu_dev = runtime.alloc(ctypes.sizeof(cu))
        runtime.memcpy_h2d(cu_dev, cu, ctypes.sizeof(cu))
        return y_dev, cu_dev


    def stored(y_dev, lanes):
        y = (ctypes.c_int32 * lanes)()
        runtime.memcpy_d2h(y, y_dev, ctypes.sizeof(y))
        return list(y)


    def waited(what, wait):
        try:
            wait()
            print(f"{what}: completed", flush=True)
        except (GpuCheckFailed, GpuCheckCaptureRefused) as error:
            print(f"{what}: {type(error).__name__}: {error}", flush=True)


    def launch(lanes, loaded, grid=(1, 1, 1), cluster=None):
        y_dev, cu_dev = buffers(lanes, loaded)
        args = struct.pack("QQi", y_dev, cu_dev, 10)
        waited(
            "launch",
            lambda: runtime.launch_blocking(
                function, grid, (lanes, 1, 1), args, cluster=cluster
            ),
        )
        return stored(y_dev, lanes * grid[0])


    def foreign_launch(loaded):
        # A launch the runtime never sees, as a graph replay or another
        # library's launch would be, and a wait outside the runtime.
        y_dev, cu_dev = buffers(1, [loaded])
        values = [ctypes.c_uint64(y_dev), ctypes.c_uint64(cu_dev), ctypes.c_int32(10)]
        params = (ctypes.c_void_p * 3)(
            *[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values]
        )
        status = hip_module._hipModuleLaunchKernel(
            function, 1, 1, 1, 1, 1, 1, 0, None, params, None
        )
        assert status == 0, status
        assert hip_module._hipDeviceSynchronize() == 0


    if case in ("two_waves", "per_lane"):
        y = launch(2 * wave, [11, 7])
        if case == "per_lane":
            print(f"even lanes stored: {sorted(set(y[0::2]))}", flush=True)
            print(f"odd lanes stored: {sorted(set(y[1::2]))}", flush=True)
        else:
            print(f"first wave stored: {sorted(set(y[:wave]))}", flush=True)
            print(f"second wave stored: {sorted(set(y[wave:]))}", flush=True)
    elif case == "cluster":
        y = launch(1, [11, 7], grid=(2, 1, 1), cluster=(2, 1, 1))
        print(f"workgroups stored: {y}", flush=True)
    elif case == "two_streams":
        # One failing launch on stream a, one passing launch on stream b, both
        # queued before either wait: the shared word cannot say which failed,
        # so both waits raise. Then a launch on a alone is checked afresh.
        a, b = runtime.create_stream(), runtime.create_stream()
        for stream, loaded in ((a, 11), (b, 7)):
            y_dev, cu_dev = buffers(1, [loaded])
            runtime.launch(function, (1, 1, 1), (1, 1, 1),
                           struct.pack("QQi", y_dev, cu_dev, 10), stream=stream)
        waited("stream a", lambda: runtime.wait_stream(a))
        waited("stream b", lambda: runtime.wait_stream(b))
        y_dev, cu_dev = buffers(1, [7])
        args = struct.pack("QQi", y_dev, cu_dev, 10)
        waited("stream a again",
               lambda: runtime.launch_blocking(function, (1, 1, 1), (1, 1, 1), args, stream=a))
        runtime.destroy_stream(a)
        runtime.destroy_stream(b)
    elif case == "destroy_failing":
        # Destroying a stream whose unread launch failed its check raises that
        # failure, and HIP destroys the stream all the same. HIP has no query
        # that is safe on a destroyed handle, so the child records the
        # hipStreamDestroy calls the runtime makes, each still made for real.
        stream = runtime.create_stream()
        y_dev, cu_dev = buffers(1, [11])
        runtime.launch(function, (1, 1, 1), (1, 1, 1),
                       struct.pack("QQi", y_dev, cu_dev, 10), stream=stream)
        hip_stream_destroy = hip_module._hipStreamDestroy
        destroyed = []

        def recording_stream_destroy(handle):
            status = hip_stream_destroy(handle)
            destroyed.append((handle.value, status))
            return status

        hip_module._hipStreamDestroy = recording_stream_destroy
        waited("destroy", lambda: runtime.destroy_stream(stream))
        calls = [(handle == stream, status) for handle, status in destroyed]
        print(f"hipStreamDestroy calls (this stream, status): {calls}", flush=True)
    elif case == "stale":
        # A launch the runtime never saw sets the word; the next runtime
        # launch, whose value meets the facts, must not be blamed for it.
        foreign_launch(11)
        print(f"stored: {launch(1, [7])[0]}", flush=True)
    elif case == "explicit":
        # Waits outside the runtime read nothing: check_gpu_checks after a
        # device synchronize, and release_pending_for_stream, do.
        for name, read in (
            ("check_gpu_checks", lambda: runtime.check_gpu_checks()),
            ("release_pending_for_stream", lambda: runtime.release_pending_for_stream(0)),
        ):
            y_dev, cu_dev = buffers(1, [11])
            runtime.launch(function, (1, 1, 1), (1, 1, 1), struct.pack("QQi", y_dev, cu_dev, 10))
            assert hip_module._hipDeviceSynchronize() == 0
            waited(name, read)
    elif case == "unload":
        # Unloading the code object drops its unread words: the next wait
        # reads nothing of the freed memory.
        y_dev, cu_dev = buffers(1, [11])
        runtime.launch(function, (1, 1, 1), (1, 1, 1), struct.pack("QQi", y_dev, cu_dev, 10))
        assert hip_module._hipDeviceSynchronize() == 0
        module.unload()
        waited("sync after unload", runtime.sync)
    elif case == "unchecked":
        # A kernel without a check has no word, and looking for it leaves no
        # error behind for hipGetLastError.
        print(f"gpu_check_word: {module.gpu_check_word(kernel_name)}", flush=True)
        print(f"hipGetLastError: {hip_module._hipGetLastError()}", flush=True)
    elif case == "capture":
        # A checked kernel launched on a stream that is being captured is
        # refused before anything is queued, and the capture then ends.
        begin = hip_module._b("hipStreamBeginCapture", ctypes.c_void_p, ctypes.c_int)
        end = hip_module._b(
            "hipStreamEndCapture", ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        destroy = hip_module._b("hipGraphDestroy", ctypes.c_void_p)
        stream = runtime.create_stream()
        y_dev, cu_dev = buffers(1, [7])
        relaxed = 2  # hipStreamCaptureModeRelaxed
        assert begin(ctypes.c_void_p(stream), relaxed) == 0
        waited(
            "launch during capture",
            lambda: runtime.launch(function, (1, 1, 1), (1, 1, 1),
                                   struct.pack("QQi", y_dev, cu_dev, 10),
                                   stream=stream),
        )
        graph = ctypes.c_void_p(0)
        print(f"end capture: {end(ctypes.c_void_p(stream), ctypes.byref(graph))}", flush=True)
        if graph.value:
            assert destroy(graph) == 0
        runtime.destroy_stream(stream)
    else:
        for loaded in [int(v) for v in case.split(",")]:
            print(f"stored: {launch(1, [loaded])[0]}", flush=True)
    """
)


# The child's own bound, below the 60 s per-test limit of the CI lane that runs
# this file, so a child that never finishes fails its own test instead of ending
# the whole run. A case's time also holds any compile of its code object, which
# counts toward the same per-test limit; prepare_before_tests normally keeps the
# slow hipcc compiles out of every case's time.
_CHILD_TIMEOUT_S = 45


# The processors the GPU cases below have run on. On any other they are skipped
# until they have.
_RUN_ON = ("gfx90a", "gfx942", "gfx1100", "gfx1201")

# The routes a code object is built by: rocKE's LLVM lowering and COMGR, or its
# HIP lowering and hipcc. The cases whose result depends on the lowering run on
# both: a true fact stores, a false one stops its lane and fails the launch, and
# a wave or lane that returns at its check does not hold the rest at a barrier.
# Each test parametrized on _ROUTES names its case in a ``case`` parameter,
# which prepare_before_tests reads to compile the hipcc code objects.
_ROUTES = ("llvm", "hipcc")

# The route of the cases that test how rocke.runtime reads the GPU-check word
# (streams, a stale word, explicit reads, unload, a kernel without a check,
# stream capture). They do not depend on which route built the code object, so
# they run on the LLVM route only.
_RUNTIME_ROUTE = "llvm"

# Code objects compiled before any test ran, keyed by (route, kernel name): the
# bytes, or the error the compile raised, which the case that needs it re-raises.
_PREBUILT: dict[tuple[str, str], bytes | Exception] = {}


def _gpu_case_target(route: str) -> tuple[str, int] | str:
    """``(arch, wave size)`` of the GPU the cases run on with ``route``, or why
    they are skipped here."""
    arch, absence = probe_device_arch()
    if arch is None:
        return absence
    from rocke.runtime.hip_module import current_device, get_device_target_id

    processor = (get_device_target_id(current_device()) or arch).split(":")[0]
    if processor not in _RUN_ON:
        return f"has not run on {processor} hardware yet; enable once it has"
    if route == "hipcc" and shutil.which("hipcc") is None:
        return "hipcc not found"
    return arch, ArchTarget.from_gfx(arch).wave_size


def prepare_before_tests(items: list[pytest.Item]) -> None:
    """Compile the hipcc route's kernels once, before any test runs, when a
    collected GPU case that is not skipped will launch one (tests/conftest.py
    calls this after collection).

    hipcc parses the HIP headers for every kernel, which on a slow host takes a
    large share of a case's per-test limit, so no case's time should hold it.
    A skipped target compiles nothing, and a compile error is kept for the case
    that needs the code object, which raises it. Only a routed test without a
    ``case`` parameter raises here (a ``KeyError``), which the conftest hook
    reports as a warning.
    """
    # The cases the hipcc route will launch, read off the collected items: each
    # test parametrized on _ROUTES names its case in its ``case`` parameter.
    cases = sorted(
        {
            item.callspec.params["case"]
            for item in items
            if item.name.startswith("test_gpu_")
            and item.get_closest_marker("skip") is None
            and hasattr(item, "callspec")
            and item.callspec.params.get("route") == "hipcc"
        }
    )
    if not cases:
        return
    try:
        target = _gpu_case_target("hipcc")
        if isinstance(target, str):
            return
        arch, wave = target
        kernels = {
            kernel.name: kernel
            for kernel in (_kernel_for(case, "hipcc", wave) for case in cases)
        }
    except Exception:  # noqa: BLE001 - each case looks up its target and reports this
        return
    for name, kernel in kernels.items():
        try:
            _PREBUILT[("hipcc", name)] = _code_object(kernel, "hipcc", arch)
        except Exception as error:  # noqa: BLE001 - re-raised by the case that needs it
            _PREBUILT[("hipcc", name)] = error


@pytest.fixture(scope="module")
def run_gpu_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str, str], list[str]]:
    """Run one GPU case, ``run_gpu_case(route, case)``, in a child process and
    return the lines it printed.

    The code object is compiled in this process, once per kernel and route (or
    taken from _PREBUILT), so the child only loads it and its time bound covers
    no compile."""
    code_object_dir = tmp_path_factory.mktemp("gpu_checked_code_objects")
    paths: dict[tuple[str, str], Path] = {}

    def code_object_path(route: str, kernel: KernelDef, arch: str) -> Path:
        key = (route, kernel.name)
        if key not in paths:
            code = _PREBUILT.get(key)
            if isinstance(code, Exception):
                raise code
            if code is None:
                code = _code_object(kernel, route, arch)
            path = code_object_dir / f"{route}_{kernel.name}.hsaco"
            path.write_bytes(code)
            paths[key] = path
        return paths[key]

    def run(route: str, case: str) -> list[str]:
        target = _gpu_case_target(route)
        if isinstance(target, str):
            pytest.skip(target)
        arch, wave = target
        kernel = _kernel_for(case, route, wave)
        child = subprocess.run(
            rocke_python_command(
                "-c",
                _CHILD,
                case,
                code_object_path(route, kernel, arch),
                kernel.name,
                str(wave),
            ),
            check=False,  # the child's output is the result, read below
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            env=rocke_subprocess_env(),
        )
        assert child.returncode == 0, child.stdout + child.stderr
        return child.stdout.splitlines()

    return run


@pytest.mark.parametrize("case", ["7"])
@pytest.mark.parametrize("route", _ROUTES)
def test_gpu_a_value_meeting_its_facts_runs_to_the_store(run_gpu_case, case, route):
    lines = run_gpu_case(route, case)
    assert lines == ["launch: completed", "stored: 7"]


@pytest.mark.parametrize("case", ["11,7"])
@pytest.mark.parametrize("route", _ROUTES)
def test_gpu_a_value_breaking_its_facts_fails_the_launch(run_gpu_case, case, route):
    """cu[0] = 11 > T = 10: the lane stops at the check, so the store never
    happens, and the launch ends with GpuCheckFailed. The same process then
    launches the kernel with cu[0] = 7, which stores: the GPU is still usable
    and the word was cleared."""
    lines = run_gpu_case(route, case)
    failed = f"launch: GpuCheckFailed: gpu_checked_{route}: a gpu_checked_value fact"
    assert len(lines) == 4 and lines[0].startswith(failed), lines
    assert lines[1:] == ["stored: -1", "launch: completed", "stored: 7"]


@pytest.mark.parametrize("case", ["two_waves"])
@pytest.mark.parametrize("route", _ROUTES)
def test_gpu_one_wave_breaking_its_facts_does_not_hold_the_other(
    run_gpu_case, case, route
):
    """cu = [11, 7]: the first wave stops at its check; the second passes the
    barrier after it and stores 7 in every lane; the launch ends with
    GpuCheckFailed."""
    lines = run_gpu_case(route, case)
    assert len(lines) == 3 and lines[0].startswith("launch: GpuCheckFailed: "), lines
    assert lines[1:] == ["first wave stored: [-1]", "second wave stored: [7]"]


@pytest.mark.parametrize("case", ["per_lane"])
@pytest.mark.parametrize("route", _ROUTES)
def test_gpu_lanes_breaking_their_facts_stop_and_the_rest_of_the_wave_goes_on(
    run_gpu_case, case, route
):
    """Lane i checks cu[i % 2], cu = [11, 7]: the even lanes return at their
    check, the odd lanes of the same waves pass the barrier and store 7, and
    the launch ends with GpuCheckFailed."""
    lines = run_gpu_case(route, case)
    assert len(lines) == 3 and lines[0].startswith("launch: GpuCheckFailed: "), lines
    assert lines[1:] == ["even lanes stored: [-1]", "odd lanes stored: [7]"]


def test_gpu_two_streams_with_unread_launches_both_see_the_failure(run_gpu_case):
    """A failing launch on stream a and a passing one on stream b, both queued
    before either wait: the kernel's one word cannot say which failed, so
    both waits raise and say so. Once both are read, the next launch on a
    clears the word first and is checked on its own."""
    lines = run_gpu_case(_RUNTIME_ROUTE, "two_streams")
    shared = (
        f"GpuCheckFailed: gpu_checked_{_RUNTIME_ROUTE}: a gpu_checked_value fact was false "
        f"in a launch of gpu_checked_{_RUNTIME_ROUTE} on this stream or on another stream"
    )
    assert len(lines) == 3, lines
    assert lines[0].startswith(f"stream a: {shared}"), lines
    assert lines[1].startswith(f"stream b: {shared}"), lines
    assert lines[2] == "stream a again: completed", lines


def test_gpu_destroying_a_stream_whose_check_failed_raises_and_still_destroys_it(
    run_gpu_case,
):
    """Runtime.destroy_stream drains the stream first; when that drain raises
    GpuCheckFailed, hipStreamDestroy still runs for the stream, once and
    successfully, before the error leaves."""
    lines = run_gpu_case(_RUNTIME_ROUTE, "destroy_failing")
    assert len(lines) == 2, lines
    assert lines[0].startswith("destroy: GpuCheckFailed: "), lines
    assert (
        lines[1] == "hipStreamDestroy calls (this stream, status): [(True, 0)]"
    ), lines


def test_gpu_a_word_set_by_a_launch_the_runtime_never_saw_is_not_blamed_on_the_next(
    run_gpu_case,
):
    """A launch outside the runtime (as a graph replay would be) breaks the
    facts and sets the word; the next runtime launch, whose value meets them,
    completes and stores."""
    lines = run_gpu_case(_RUNTIME_ROUTE, "stale")
    assert lines == ["launch: completed", "stored: 7"]


def test_gpu_after_a_wait_outside_the_runtime_the_explicit_reads_raise(run_gpu_case):
    """After a device synchronize outside the runtime, check_gpu_checks and
    release_pending_for_stream each read the word of the failing launch."""
    lines = run_gpu_case(_RUNTIME_ROUTE, "explicit")
    assert len(lines) == 2, lines
    assert lines[0].startswith("check_gpu_checks: GpuCheckFailed: "), lines
    assert lines[1].startswith("release_pending_for_stream: GpuCheckFailed: "), lines


def test_gpu_unloading_a_module_drops_its_unread_words(run_gpu_case):
    lines = run_gpu_case(_RUNTIME_ROUTE, "unload")
    assert lines == ["sync after unload: completed"]


def test_gpu_a_kernel_without_a_check_has_no_word_and_leaves_no_error(run_gpu_case):
    lines = run_gpu_case(_RUNTIME_ROUTE, "unchecked")
    assert lines == ["gpu_check_word: None", "hipGetLastError: 0"]


def test_gpu_a_checked_launch_during_stream_capture_is_refused(run_gpu_case):
    """Nothing reads the word after a graph replay, so the runtime refuses to
    queue a checked kernel on a stream being captured, and queues nothing."""
    lines = run_gpu_case(_RUNTIME_ROUTE, "capture")
    refused = (
        f"launch during capture: GpuCheckCaptureRefused: gpu_checked_{_RUNTIME_ROUTE}: this "
        "kernel checks gpu_checked_value facts, and its stream is being captured "
        "into a graph; nothing reads the kernel's GPU-check word after a replay, so "
        "a false fact would go unseen. Launch it outside the capture."
    )
    assert lines == [refused, "end capture: 0"]


def test_a_wait_with_nothing_noted_asks_hip_nothing():
    """With no launch of a checked kernel noted, a wait reads no word and does
    not even ask for the current device: a kernel without a GPU-check word
    adds nothing to a wait."""
    from rocke.runtime.hip_module import GpuCheckLedger, _HipFunctionHandle

    asked = []
    ledger = GpuCheckLedger(current_device=lambda: asked.append("device") or 0)
    ledger.read(None)
    ledger.read(0)
    # A function handle with no GPU-check word registered for it, as
    # Module.get_function returns for a kernel without a check.
    unchecked = _HipFunctionHandle(0x1000)
    assert ledger.word_of(unchecked) is None
    ledger.note(unchecked, 0)
    assert asked == []


def test_a_check_comes_before_a_cluster_barrier():
    """gfx1250: the failure path (store, return) comes before the cluster
    barrier, so a workgroup that fails leaves before it."""
    ir = lower_kernel_to_llvm(
        _cluster_kernel("gpu_checked_cluster"), arch="gfx1250", llvm_flavor="llvm23"
    )
    store = (
        "  store atomic i32 1, ptr addrspace(1) @rocke_gpu_check_failed_gpu_checked_"
        'cluster syncscope("agent") monotonic, align 4\n ret void\n'
    )
    barrier = "  call void @llvm.amdgcn.s.cluster.barrier()\n"
    assert ir.count(store) == 1 and ir.count(barrier) == 1
    assert ir.index(store) < ir.index(barrier)


@pytest.mark.skip(reason="has not run on gfx1250 hardware yet; enable once it has")
@pytest.mark.parametrize("case", ["cluster"])
@pytest.mark.parametrize("route", _ROUTES)
def test_gpu_one_workgroup_breaking_its_facts_does_not_hold_its_cluster(
    run_gpu_case, case, route
):
    """gfx1250, cu = [11, 7] in a cluster of two workgroups: the first returns
    at its check, the second passes the cluster barrier and stores 7, and the
    launch ends with GpuCheckFailed."""
    if not (probe_device_arch()[0] or "").startswith("gfx1250"):
        pytest.skip("needs a gfx1250 GPU (workgroup clusters)")
    lines = run_gpu_case(route, case)
    assert len(lines) == 2 and lines[0].startswith("launch: GpuCheckFailed: "), lines
    assert lines[1] == "workgroups stored: [-1, 7]"
