# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the dtype-generic ``buffer_load_vN`` builder op.

Config index = case * len(ARCHES) + arch_index, with CASES (dtype, n) below.
Cases 0..25 are accepted: every element width (1, 2, 4 and 8 bytes) at every
access width it can make (1, 2, 4, 8 and 16 bytes). Cases 26..34 must be
rejected by both engines: a count of zero, or elements that make an access of
some other size (6, 12, 24 or 32 bytes). A rejected case emits nothing and
exits non-zero on both sides, which the differential harness records as
BOTH_REJECTED.
"""

from rocke.core.ir import (
    BF8E5M2,
    BF16,
    F16,
    F32,
    F64,
    FP8E4M3,
    I8,
    I16,
    I32,
    IRBuilder,
    PtrType,
)

from _emit_common import run_emit

ARCHES = ("gfx950", "gfx1151", "gfx1201")
_TYPES = {
    "i8": I8,
    "fp8e4m3": FP8E4M3,
    "bf8e5m2": BF8E5M2,
    "i16": I16,
    "f16": F16,
    "bf16": BF16,
    "f32": F32,
    "i32": I32,
    "f64": F64,
}
CASES = (
    ("f16", 2),
    ("f16", 4),
    ("f16", 8),
    ("bf16", 2),
    ("bf16", 4),
    ("bf16", 8),
    ("f32", 1),
    ("f32", 2),
    ("f32", 4),
    ("i32", 1),
    ("i32", 4),
    ("i8", 1),
    ("i8", 2),
    ("i8", 4),
    ("i8", 8),
    ("i8", 16),
    ("fp8e4m3", 4),
    ("bf8e5m2", 16),
    ("f16", 1),
    ("bf16", 1),
    ("i16", 1),
    ("i16", 8),
    ("i32", 2),
    ("f64", 1),
    ("f64", 2),
    ("fp8e4m3", 1),
    # rejected
    ("f16", 3),
    ("f16", 5),
    ("f16", 6),
    ("f16", 16),
    ("bf16", 3),
    ("f32", 3),
    ("f32", 8),
    ("f32", 0),
    ("f64", 3),
)


def build_buffer_load_vN(b: IRBuilder, dtype_name: str, n: int) -> None:
    dtype = _TYPES[dtype_name]
    x = b.param("X", PtrType(dtype, "global"))
    y = b.param("Y", PtrType(dtype, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(x, n_bytes)
    tid = b.thread_id_x()
    voffset = b.mul(tid, b.const_i32(16))
    soffset = b.const_i32(0)
    v = b.buffer_load_vN(rsrc, voffset, soffset, dtype, n)
    for i in range(n):
        b.global_store(
            y, b.add(b.mul(tid, b.const_i32(n)), b.const_i32(i)), b.vec_extract(v, i)
        )
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    dtype_name, n = spec
    b = IRBuilder("buffer_load_vN")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_buffer_load_vN(b, dtype_name, n)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
