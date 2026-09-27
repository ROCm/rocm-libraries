# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C value-barrier parity, independent of matrix kernels."""

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


def build_optimization_barriers(b: IRBuilder) -> None:
    tid = b.thread_id_x()
    for dtype in (I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2):
        ptr = b.param(f"p_{dtype.name}", PtrType(dtype, "global"))
        value = b.global_load(ptr, tid, dtype, align=1)
        value = b.optimization_barrier(value)
        b.global_store(ptr, tid, value, align=1)
    b.ret()


def _spec(idx):
    if idx not in (0, 1):
        raise SystemExit(f"unknown config index {idx}")
    return None, ("gfx1250", "gfx950")[idx]


def _build(spec, *, arch):
    b = IRBuilder("optimization_barrier")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_optimization_barriers(b)
    return b.kernel


if __name__ == "__main__":
    raise SystemExit(run_emit(_spec, _build))
