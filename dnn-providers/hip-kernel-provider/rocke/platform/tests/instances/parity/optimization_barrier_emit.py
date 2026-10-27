# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C value-barrier parity, independent of matrix kernels."""

import sys

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
)

from _emit_common import run_emit
from rocke.core.lower_hip import lower_kernel_to_hip

ARCHES = (
    "gfx1250",
    "gfx950",
    "gfx90a",
    "gfx942",
    "gfx1151",
    "gfx1201",
    "gfx11-generic",
)


def build_optimization_barriers(b: IRBuilder) -> None:
    tid = b.thread_id_x()
    for dtype in (I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2):
        ptr = b.param(f"p_{dtype.name}", PtrType(dtype, "global"))
        value = b.global_load(ptr, tid, dtype, align=1)
        value = b.optimization_barrier(value)
        b.global_store(ptr, tid, value, align=1)
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return None, ARCHES[idx]


def _build(spec, *, arch):
    b = IRBuilder("optimization_barrier")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_optimization_barriers(b)
    return b.kernel


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[2] == "hip":
        idx = int(sys.argv[1])
        if not 0 <= idx < 8:
            raise SystemExit(f"unknown config index {idx}")
        arch = ("gfx950", "gfx1250")[idx // 4]
        dtype = (F16, BF16)[(idx // 2) % 2]
        b = IRBuilder("optimization_barrier_hip")
        b.kernel.attrs["max_workgroup_size"] = 64
        output = b.param("output", PtrType(dtype, "global"))
        acc = b.param("acc", F32)
        factor_a = b.param("factor_a", F32)
        factor_b = b.param("factor_b", F32)
        scaled = b.fmul(acc, b.fmul(factor_a, factor_b))
        opaque = b.optimization_barrier(scaled)
        if idx % 2:
            opaque.op.attrs["clobber"] = ""
        tid = b.thread_id_x()
        b.global_store(output, tid, b.cast_f32_to(opaque, dtype), align=2)
        b.ret()
        sys.stdout.write(lower_kernel_to_hip(b.kernel, arch=arch))
        return 0
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
