# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the ``buffer_store_vN`` and ``buffer_store_vN_f32`` ops.

Config index = case * len(ARCHES) + arch_index, with CASES (dtype, n) below
followed by F32_CASES (n).
Each lane splats one loaded element to an ``n``-element vector and stores it.
Cases 0..15 are accepted: every element width (1, 2, 4 and 8 bytes) at every
access width it can make (1, 2, 4, 8 and 16 bytes). Cases 16..19 must be
rejected by both engines: vectors of 6, 12, 24 or 32 bytes. A rejected case
emits nothing and exits non-zero on both sides, which the differential harness
records as BOTH_REJECTED. Cases 20..22 store through ``buffer_store_vN_f32``
with 1, 2 and 4 dwords; n = 1 is its single-dword path.
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
    ("i8", 1),
    ("i8", 2),
    ("i8", 4),
    ("i8", 8),
    ("i8", 16),
    ("fp8e4m3", 4),
    ("bf8e5m2", 16),
    ("i16", 1),
    ("f16", 1),
    ("f16", 2),
    ("bf16", 8),
    ("f32", 1),
    ("f32", 4),
    ("i32", 2),
    ("f64", 1),
    ("f64", 2),
    # rejected
    ("f16", 3),
    ("f32", 3),
    ("f64", 3),
    ("f32", 8),
)
F32_CASES = (1, 2, 4)


def build_buffer_store_vN(b: IRBuilder, dtype_name: str, n: int) -> None:
    dtype = _TYPES[dtype_name]
    x = b.param("X", PtrType(dtype, "global"))
    y = b.param("Y", PtrType(dtype, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(y, n_bytes)
    tid = b.thread_id_x()
    value = b.vector_splat(b.global_load(x, tid, dtype), n)
    voffset = b.mul(tid, b.const_i32(16))
    soffset = b.const_i32(0)
    b.buffer_store_vN(rsrc, voffset, soffset, value)
    b.ret()


def build_buffer_store_vN_f32(b: IRBuilder, n: int) -> None:
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(y, n_bytes)
    tid = b.thread_id_x()
    value = b.vector_splat(b.global_load(x, tid, F32), n)
    voffset = b.mul(tid, b.const_i32(16))
    soffset = b.const_i32(0)
    b.buffer_store_vN_f32(rsrc, voffset, soffset, value, n)
    b.ret()


def _spec(idx):
    if not 0 <= idx < (len(CASES) + len(F32_CASES)) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    case = idx // len(ARCHES)
    if case < len(CASES):
        spec = CASES[case]
    else:
        spec = ("vN_f32", F32_CASES[case - len(CASES)])
    return spec, ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    dtype_name, n = spec
    b = IRBuilder("buffer_store_vN")
    b.kernel.attrs["max_workgroup_size"] = 64
    if dtype_name == "vN_f32":
        build_buffer_store_vN_f32(b, n)
    else:
        build_buffer_store_vN(b, dtype_name, n)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
