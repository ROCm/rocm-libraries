# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the buffer address space of rocke.helpers.tensor_view.

Covers BufferResource / make_buffer_resource / make_buffer_view and the
``addr_space == "buffer"`` branches of TensorView.load_scalar, store_scalar,
load_vec, store_vec, load_vec_at and store_vec_at (with and without ``mask=``),
reached directly and through a TileWindow. Byte-compared against
tensor_view_buffer_emit.c, which drives rocke_tv_make_buffer_resource,
rocke_tv_make_buffer_view and the rocke_tensor_view_* / rocke_tile_window_*
entry points.

Config index = case * len(ARCHES) + arch_index, with CASES (op, dtype, n)
below. The buffer branches emit IR for f16 only (the one dtype the Python
helper wires), so the accepted cases are f16. The rejected cases hit each raise
of the Python helper: an unwired dtype (NotImplementedError, after or inside
``_dtype_elem_bytes``), a width outside {2, 4, 8} (ValueError), a mask on a
non-buffer view (NotImplementedError) and a buffer view whose base is not a
BufferResource (TypeError). A rejected case emits nothing and exits non-zero on
both sides, which the differential harness records as BOTH_REJECTED. That
verdict does not compare the errors; tests/core/test_tensor_view_buffer_errors
(.py and .cpp) pin the same exception kind and message on both engines.

Every builder call is its own statement in the same order the C emitter issues
it: each call consumes an SSA id, so a reordering emits the same IR under
different names.
"""

from rocke.core.ir import BF16, F16, F32, F64, I32, I64, IRBuilder, PtrType
from rocke.helpers.tensor_view import (
    make_buffer_resource,
    make_buffer_view,
    make_global_view,
)

from _emit_common import run_emit

ARCHES = ("gfx950", "gfx1151", "gfx1201")
_TYPES = {"f16": F16, "bf16": BF16, "f32": F32, "i32": I32, "i64": I64, "f64": F64}
SHAPE = (4, 64)
CASES = (
    # accepted (f16)
    ("scalar", "f16", 1),
    ("vec", "f16", 2),
    ("vec", "f16", 4),
    ("vec", "f16", 8),
    ("vec_at", "f16", 2),
    ("vec_at", "f16", 4),
    ("vec_at", "f16", 8),
    ("vec_at_mask", "f16", 2),
    ("vec_at_mask", "f16", 4),
    ("vec_at_mask", "f16", 8),
    ("window", "f16", 2),
    ("window", "f16", 4),
    ("window", "f16", 8),
    ("window_f32", "f16", 1),
    ("window_f32", "f16", 2),
    ("window_f32", "f16", 4),
    ("window_f32", "f16", 8),
    # rejected: dtype not wired on the buffer path
    ("scalar", "bf16", 1),
    ("vec", "bf16", 4),
    ("vec_at", "bf16", 4),
    ("vec_at_mask", "bf16", 4),
    ("scalar", "f32", 1),
    ("vec", "f32", 4),
    ("scalar", "i32", 1),
    ("scalar", "i64", 1),
    ("scalar", "f64", 1),
    ("vec_at", "f64", 2),
    # rejected: f16 width outside {2, 4, 8}
    ("vec", "f16", 1),
    ("vec", "f16", 3),
    ("vec", "f16", 16),
    ("vec_at", "f16", 1),
    ("vec_at", "f16", 6),
    ("vec_at_mask", "f16", 3),
    # rejected: mask on a global view; buffer view without a BufferResource
    ("mask_global", "f16", 4),
    ("no_rsrc", "f16", 4),
)


def build_tensor_view_buffer(b: IRBuilder, op: str, dtype_name: str, n: int) -> None:
    dtype = _TYPES[dtype_name]
    x = b.param("X", PtrType(dtype, "global"))
    y = b.param("Y", PtrType(dtype, "global"))
    n_bytes = b.param("N_bytes", I32)
    s_row = b.param("S_row", I32)
    s_off = b.param("S_off", I32)
    # Default soffset (const_i32(0)) for X; an explicit one for Y.
    x_rsrc = make_buffer_resource(b, x, num_bytes=n_bytes)
    y_rsrc = make_buffer_resource(b, y, num_bytes=n_bytes, soffset=s_off)
    if op == "no_rsrc":
        # The base is a pointer, not a BufferResource: TypeError on first use.
        x_view = make_buffer_view(x, SHAPE, dtype)
    else:
        # Packed row-major for X; a runtime row stride for Y.
        x_view = make_buffer_view(x_rsrc, SHAPE, dtype)
    y_view = make_buffer_view(y_rsrc, SHAPE, dtype, strides=(s_row, 1))
    tid = b.thread_id_x()
    row = b.const_i32(1)

    if op == "scalar":
        v = x_view.load_scalar(b, [row, tid])
        y_view.store_scalar(b, [row, tid], v)
    elif op in ("vec", "no_rsrc"):
        v = x_view.load_vec(b, [row, tid], n)
        y_view.store_vec(b, [row, tid], v, n)
    elif op == "vec_at":
        c_n = b.const_i32(n)
        off = b.mul(tid, c_n)
        v = x_view.load_vec_at(b, off, n)
        y_view.store_vec_at(b, off, v, n)
    elif op == "vec_at_mask":
        c_n = b.const_i32(n)
        off = b.mul(tid, c_n)
        c_lim = b.const_i32(48)
        mask = b.cmp_lt(tid, c_lim)
        v = x_view.load_vec_at(b, off, n, mask=mask)
        y_view.store_vec_at(b, off, v, n, mask=mask)
    elif op == "mask_global":
        g_view = make_global_view(x, SHAPE, dtype)
        c_n = b.const_i32(n)
        off = b.mul(tid, c_n)
        c_lim = b.const_i32(48)
        mask = b.cmp_lt(tid, c_lim)
        g_view.load_vec_at(b, off, n, mask=mask)
    elif op in ("window", "window_f32"):
        x_win = x_view.tile((1, 8), (row, tid))
        y_win = y_view.tile((1, 8), (row, tid))
        zero = b.const_i32(0)
        if op == "window":
            v = x_win.load_vec(b, zero, zero, n=n)
            y_win.store_vec(b, zero, zero, value=v, n=n)
        else:
            vals = x_win.load_vec_as_f32(b, zero, zero, n=n)
            y_win.store_vec_from_f32(b, zero, zero, values=vals)
    else:
        raise SystemExit(f"unknown op {op}")
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    op, dtype_name, n = spec
    b = IRBuilder("tensor_view_buffer")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_tensor_view_buffer(b, op, dtype_name, n)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
