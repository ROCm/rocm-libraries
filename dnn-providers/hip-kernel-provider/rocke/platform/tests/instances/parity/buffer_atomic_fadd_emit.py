# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the ``buffer_atomic_fadd_f32`` builder op.

Config index = case * len(ARCHES) + arch_index, with CASES (cachepolicy) below.
Each lane loads one f32 and adds it into its dword of a buffer resource. Cases
0..3 are accepted: no policy, the low and high ends of the five-bit immediate,
and CPOL_ATOMIC_CASCADE_DEV. Cases 4..5 must be rejected by both engines: a
cache policy outside 0..31. A rejected case emits nothing and exits non-zero on
both sides, which the differential harness records as BOTH_REJECTED.
"""

from rocke.core.ir import CPOL_ATOMIC_CASCADE_DEV, F32, I32, IRBuilder, PtrType

from _emit_common import run_emit

ARCHES = ("gfx950", "gfx1250")
CASES = (
    0,
    1,
    CPOL_ATOMIC_CASCADE_DEV,
    31,
    # rejected
    32,
    -1,
)


def build_buffer_atomic_fadd(b: IRBuilder, cachepolicy: int) -> None:
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    n_bytes = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(y, n_bytes)
    tid = b.thread_id_x()
    value = b.global_load(x, tid, F32)
    voffset = b.mul(tid, b.const_i32(4))
    soffset = b.const_i32(0)
    b.buffer_atomic_fadd_f32(rsrc, voffset, soffset, value, cachepolicy=cachepolicy)
    b.ret()


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(spec, *, arch):
    b = IRBuilder("buffer_atomic_fadd")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_buffer_atomic_fadd(b, spec)
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
