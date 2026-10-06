#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dense full-rank MFMA workload for the rocGDB snapshot demo.

The logical 16x16 operands are A = J + I and B = J + 2I, where J is the
all-ones matrix. Both are dense and full rank. Their product is 19J + 2I, so
the expected tile has 21 on the diagonal and 19 everywhere else. That pattern
exposes row/column and lane/slot mapping mistakes which an all-ones tile hides.
"""

from __future__ import annotations

import argparse
import ctypes
import struct

from rocke.core.arch import ArchTarget
from rocke.core.ir import F16, F32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.runtime.hip_module import Runtime

_M = 16
_N = 16
_OP_ID = "mfma_f32_16x16x16_f16"


def build_kernel(arch: str):
    builder = IRBuilder("rocke_full_rank_snapshot_demo", capture_loc=True)
    output = builder.param("output", PtrType(F32, "global"))
    lane = builder.thread_id_x()

    operation = ArchTarget.from_gfx(arch).mma.by_op_id(_OP_ID)
    if operation is None:
        raise ValueError(f"{arch} does not provide {_OP_ID}")

    one = builder.trunc_f32_to_f16(builder.const_f32(1.0))
    two = builder.trunc_f32_to_f16(builder.const_f32(2.0))
    three = builder.trunc_f32_to_f16(builder.const_f32(3.0))

    a_elements = []
    b_elements = []
    for slot in range(operation.a_frag_len):
        row, k = operation.a_layout().coord(builder, lane, slot)
        a_elements.append(builder.select(builder.cmp_eq(row, k), two, one))
    for slot in range(operation.b_frag_len):
        k, column = operation.b_layout().coord(builder, lane, slot)
        b_elements.append(builder.select(builder.cmp_eq(k, column), three, one))

    a_fragment = builder.vec_pack(a_elements, F16)
    b_fragment = builder.vec_pack(b_elements, F16)
    accumulator = builder.mfma_f32_16x16x16_f16(
        a_fragment,
        b_fragment,
        builder.zero_vec_f32(operation.c_frag_len),
    )
    builder.debug_value(accumulator)

    accumulator_slots = []
    for slot in range(operation.c_frag_len):
        value = builder.vec_extract(accumulator, slot)
        accumulator_slots.append(value)

    width = builder.const_i32(operation.n)
    for slot, value in enumerate(accumulator_slots):
        row, column = operation.c_layout().coord(builder, lane, slot)
        index = builder.add(builder.mul(row, width), column)
        builder.global_store(output, index, value)
    builder.ret()
    return builder.kernel


def expected_product() -> list[float]:
    return [
        21.0 if row == column else 19.0 for row in range(_M) for column in range(_N)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()

    artifact = compile_kernel(build_kernel(args.arch), arch=args.arch, backend="python")
    runtime = Runtime()
    output = runtime.alloc(_M * _N * ctypes.sizeof(ctypes.c_float))
    module = None
    try:
        module = runtime.load_module(artifact.hsaco)
        function = module.get_function(artifact.kernel_name)
        runtime.launch(function, (1, 1, 1), (64, 1, 1), struct.pack("<Q", output))
        runtime.sync()

        host_output = (ctypes.c_float * (_M * _N))()
        runtime.memcpy_d2h(host_output, output, ctypes.sizeof(host_output))
        observed = list(host_output)
        expected = expected_product()
        if observed != expected:
            raise AssertionError(f"matrix product mismatch: {observed}")

        print("C diagonal:", [observed[index * _N + index] for index in range(4)])
        print("C row 0:", observed[:_N])
    finally:
        if module is not None:
            module.unload()
        runtime.free(output)


if __name__ == "__main__":
    main()
