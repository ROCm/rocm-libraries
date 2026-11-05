# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both engines resolve the SSA names of a serialized kernel the same way.

A loop's induction variable and iter-args are defined by the loop and go out of
scope when it ends, so sibling loops can define the same names again. The C++
parser keeps the names in a hash table that removes a loop's names at its end,
probes past and reuses the slots they leave, and rebuilds itself as it grows;
Python keeps a dict. A kernel with many sibling loops that reuse their names,
and over a thousand values, takes those paths.
"""

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.ir_serialize import parse, serialize
from rocke.tests.engines import lower_hip_with_both_engines, lower_with_both_engines

# Enough sibling loops that the C++ table grows several times.
_SIBLING_LOOPS = 200


def _kernel_with_reused_loop_names():
    builder = IRBuilder("reused_loop_names")
    data = builder.param("X", PtrType(F32, "global"))
    total = builder.const_f32(0.0)
    for _ in range(_SIBLING_LOOPS):
        loop = builder.scf_for_iter(
            builder.const_i32(0),
            builder.const_i32(4),
            builder.const_i32(1),
            [("acc", total)],
            iv_name="k",
        )
        with loop as (index, (acc,)):
            builder.scf_yield(builder.fadd(acc, builder.global_load_f32(data, index)))
        total = loop.results[0]
    builder.global_store(data, builder.const_i32(0), total)
    builder.ret()
    return builder.kernel


def test_sibling_loops_define_the_same_names():
    loops = [
        op for op in _kernel_with_reused_loop_names().body.ops if op.name == "scf.for"
    ]
    assert len(loops) == _SIBLING_LOOPS
    assert len({loop.attrs["iv"] for loop in loops}) == 1
    assert len({str(loop.attrs["iter_args"]) for loop in loops}) == 1


def test_both_engines_lower_a_kernel_that_reuses_loop_names_to_llvm():
    kernel = parse(serialize(_kernel_with_reused_loop_names()))
    llvm_ir = lower_with_both_engines(kernel, "gfx942")
    assert llvm_ir.count("= load float") == _SIBLING_LOOPS


def test_both_engines_lower_a_kernel_that_reuses_loop_names_to_hip():
    kernel = parse(serialize(_kernel_with_reused_loop_names()))
    hip = lower_hip_with_both_engines(kernel, "gfx942")
    # One load in each loop body, and the final store.
    assert hip.count("X[") == _SIBLING_LOOPS + 1
