# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for the two block-size kernel attrs.

Config index = case * len(ARCHES) + arch_index, with CASES below. Each kernel
stores its thread id to ``Y[tid]``; the case picks the attr that states the
block size:
- "workgroup_size": the exact block size, 64.
- "max_workgroup_size": an upper bound, 64.
Both lower through the one accessor, so both cases emit the same function
attributes; the wave32 arch shows the lower end of the range following the
wave size.
"""

from rocke.core.ir import I32, IRBuilder, PtrType

from _emit_common import run_emit

ARCHES = ("gfx942", "gfx1201")
CASES = ("workgroup_size", "max_workgroup_size")


def _spec(idx):
    if not 0 <= idx < len(CASES) * len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return CASES[idx // len(ARCHES)], ARCHES[idx % len(ARCHES)]


def _build(attr, *, arch):
    b = IRBuilder("workgroup_size_probe")
    b.kernel.attrs[attr] = 64
    y = b.param("Y", PtrType(I32, "global"))
    tid = b.thread_id_x()
    b.global_store(y, tid, tid)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
