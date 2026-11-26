#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/f64_math_emit.py -- Python reference emitter for the f64 math
# ops (``rocke.core.f64_math`` expansions, f64 fma / fmax3 / fmin3 /
# vector.fma) and the f32 forms of the natural ``math.exp`` / ``math.log``.
# Each config is one kernel ``Y[i] = op(X[i])``; f64_math_emit.c builds the
# same kernels so run_diff.py can byte-compare the two engines.
#
# The f64 configs lower for gfx950 (the default), gfx942 and gfx90a: the
# programs use only generic LLVM intrinsics, so every CDNA target takes them.
from rocke.core.ir import F32, F64, IRBuilder, PtrType

from _emit_common import run_emit

# (op, dtype, arch)
_CONFIGS = (
    ("exp", "f64", "gfx950"),
    ("exp2", "f64", "gfx950"),
    ("log", "f64", "gfx950"),
    ("log2", "f64", "gfx950"),
    ("tanh", "f64", "gfx950"),
    ("sqrt", "f64", "gfx950"),
    ("rsqrt", "f64", "gfx950"),
    ("fma", "f64", "gfx950"),
    ("fmax3", "f64", "gfx950"),
    ("fmin3", "f64", "gfx950"),
    ("vector_fma", "f64", "gfx950"),
    ("exp", "f32", "gfx950"),
    ("log", "f32", "gfx950"),
    ("exp", "f64", "gfx942"),
    ("tanh", "f64", "gfx90a"),
)


def build_f64_math_kernel(op: str, dtype: str):
    """``Y[i] = op(X[i])`` for one thread per element (no bounds check)."""
    ty = F64 if dtype == "f64" else F32
    align = 8 if dtype == "f64" else 4
    b = IRBuilder(f"math_{op}_{dtype}")
    x_ptr = b.param("X", PtrType(ty, "global"), align=16)
    y_ptr = b.param("Y", PtrType(ty, "global"), align=16)
    tid = b.thread_id_x()
    if op == "vector_fma":
        two = b.const_i32(2)
        idx = b.mul(tid, two)
        v = b.global_load_vN(x_ptr, idx, ty, 2)
        y = b.vector_fma(v, v, v)
        b.global_store_vN(y_ptr, idx, y, 2)
        b.ret()
        return b.kernel
    x = b.global_load(x_ptr, tid, ty, align=align)
    if op in ("fma", "fmax3", "fmin3"):
        x2 = b.fmul(x, x)
        y = getattr(b, op)(x, x2, x)
    else:
        y = getattr(b, op)(x)
    b.global_store(y_ptr, tid, y, align=align)
    b.ret()
    return b.kernel


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    op, dtype, arch = _CONFIGS[idx]
    return (op, dtype), arch


def _build(spec, arch=None):
    return build_f64_math_kernel(*spec)


def main() -> int:
    return run_emit(_spec, _build, usage="usage: f64_math_emit.py <config_index>\n")


if __name__ == "__main__":
    raise SystemExit(main())
