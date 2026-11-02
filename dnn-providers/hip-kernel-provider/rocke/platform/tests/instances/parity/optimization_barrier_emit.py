# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C value-barrier parity, independent of matrix kernels."""

import sys

from rocke.core.lower_hip import lower_kernel_to_hip
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


HIP_CONFIGS = tuple(
    (arch, dtype, clobber)
    for arch in ("gfx950", "gfx1250")
    for dtype in (BF16, F16)
    for clobber in (False, True)
)


def build_hip_epilogue(dtype, explicit_clobber):
    b = IRBuilder("tensor_epilogue")
    b.kernel.attrs["max_workgroup_size"] = 64
    output = b.param("output", PtrType(dtype, "global"))
    acc = b.param("acc", F32)
    factor_a = b.param("factor_a", F32)
    factor_b = b.param("factor_b", F32)
    value = b.fmul(acc, b.fmul(factor_a, factor_b))
    opaque = b.optimization_barrier(value)
    if explicit_clobber:
        opaque.op.attrs["clobber"] = ""
    b.global_store(output, b.thread_id_x(), b.cast_f32_to(opaque, dtype), align=2)
    b.ret()
    return b.kernel


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[2] == "hip":
        idx = int(sys.argv[1])
        if not 0 <= idx < len(HIP_CONFIGS):
            raise SystemExit(f"unknown config index {idx}")
        arch, dtype, clobber = HIP_CONFIGS[idx]
        print(
            lower_kernel_to_hip(build_hip_epilogue(dtype, clobber), arch=arch), end=""
        )
        return 0
    from _emit_common import run_emit

    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
