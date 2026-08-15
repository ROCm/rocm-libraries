# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Child process for the device-print GPU functional test."""

from __future__ import annotations

import argparse
import struct
import sys

from rocke.core.ir import F32, IRBuilder, PrintValue, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.runtime.hip_module import Runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()

    builder = IRBuilder("device_print_gpu_functional")
    pointer = builder.param("p", PtrType(F32, "global"))
    signed = builder.const_i32(-5)
    seven = builder.const_i32(7)
    eight = builder.const_i32(8)
    floating = builder.const_f32(6.5)
    truth = builder.cmp_eq(seven, seven)
    falsehood = builder.cmp_ne(seven, seven)

    builder.device_print(
        "DEVICE_PRINT_FALSE_SENTINEL",
        predicate=falsehood,
    )
    builder.device_print(
        "DEVICE_PRINT_GPU ",
        truth,
        " ",
        falsehood,
        " ",
        signed,
        " ",
        PrintValue(signed, "u32"),
        " ",
        floating,
        " ",
        pointer,
        " ",
        seven,
        " ",
        eight,
        predicate=truth,
    )
    builder.ret()

    artifact = compile_kernel(builder.kernel, arch=args.arch, backend="python")
    runtime = Runtime()
    allocation = runtime.alloc(4)
    module = None
    print(f"DEVICE_PRINT_EXPECTED_PTR=0x{allocation:x}", file=sys.stderr, flush=True)
    try:
        module = runtime.load_module(artifact.hsaco)
        function = module.get_function(artifact.kernel_name)
        runtime.launch(
            function,
            (1, 1, 1),
            (1, 1, 1),
            struct.pack("<Q", allocation),
        )
        runtime.sync()
    finally:
        if module is not None:
            module.unload()
        runtime.free(allocation)


if __name__ == "__main__":
    main()
