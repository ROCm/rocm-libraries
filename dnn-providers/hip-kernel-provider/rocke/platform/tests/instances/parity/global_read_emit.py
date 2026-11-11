# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone copy author using only the public helper front door.

Compare this emitter with global_read_emit.c using the normal parity harness.
Runtime metadata, uniform expressions, divergent fallback, signed deltas,
scalar/vector types and rejected widths all exercise the real load primitives.
"""
from rocke.core.ir import BF16, F16, F32, F64, I32, I64, IRBuilder, PtrType
from rocke.helpers import GlobalReadSpec, emit_global_read_paths, make_global_view
from _emit_common import run_emit

ARCHES = ("gfx950", "gfx1151", "gfx1201")
CASES = (
    ("auto", "f16", 2),
    ("auto", "f16", 4),
    ("auto", "f16", 8),
    ("auto", "bf16", 8),
    ("auto", "f32", 2),
    ("auto", "f32", 4),
    ("auto", "i32", 4),
    ("auto", "f16", 0),
    ("auto", "bf16", 0),
    ("auto", "f32", 0),
    ("auto", "i32", 0),
    ("global64", "f16", 8),
    ("global64", "f32", 4),
    ("negative", "f16", 4),
    ("divergent", "f32", 4),
    ("wide", "f16", 8),
    ("expr", "f16", 8),
    ("auto", "f64", 2),
)
TYPES = {"f16": F16, "bf16": BF16, "f32": F32, "i32": I32, "f64": F64}


def build_global_read(b, case):
    mode, dtype_name, n = case
    dtype = TYPES[dtype_name]
    x = b.param("X", PtrType(dtype, "global"))
    y = b.param("Y", PtrType(dtype, "global"))
    meta_type = I64 if mode == "wide" else I32
    rows = b.param("rows", meta_type)
    stride = b.param("stride", meta_type)
    tid = b.thread_id_x()
    if mode == "divergent":
        rows = b.add(rows, tid)
    if mode == "expr":
        rows = b.add(rows, b.block_id_x())
    row_stride = -512 if mode == "negative" else stride
    view = make_global_view(x, (1, 1), dtype, strides=(row_stride, 1))
    spec = GlobalReadSpec(view, (rows, 512))

    def body(plans):
        row = b.const_i32(1)
        col = b.mul(tid, b.const_i32(max(n, 1)))
        cursor = plans[0].bind(b, (row, col))
        delta_row = b.const_i32(-1)
        delta_col = b.const_i32(0)
        if n:
            value = cursor.load_vec(b, (delta_row, delta_col), n=n)
            b.global_store_vN(y, col, value, n)
        else:
            value = cursor.load_scalar(b, (delta_row, delta_col))
            b.global_store(y, col, value)
        b.sync()

    emit_global_read_paths(
        b, (spec,), body, addressing="global64" if mode == "global64" else "auto"
    )
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    b = IRBuilder("global_read")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_global_read(b, spec)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
