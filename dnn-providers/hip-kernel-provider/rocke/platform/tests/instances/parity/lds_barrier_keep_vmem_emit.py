# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Native Python/C parity for ``rocke.helpers.asm.lds_barrier_keep_vmem``.

Each config builds the smallest kernel that puts the barrier between a global
load staged through LDS and the LDS read that consumes it, on one arch. The
gfx12 RDNA targets take the inline-asm barrier; every other target takes
``sync_lds_only``; an unknown arch must be refused by both engines.

Config index = arch index. Every builder call is its own statement in the
order the C emitter issues it: each call consumes an SSA id, so a reordering
emits the same IR under other names.
"""

from rocke.core.ir import F16, IRBuilder, PtrType
from rocke.helpers.asm import lds_barrier_keep_vmem

from _emit_common import run_emit

ARCHES = ("gfx1201", "gfx1200", "gfx12-generic", "gfx950", "gfx11-generic", "gfx0000")
BLOCK = 256


def _spec(idx):
    if not 0 <= idx < len(ARCHES):
        raise SystemExit(f"unknown config index {idx}")
    return ARCHES[idx], ARCHES[idx]


def _build(target, *, arch):
    b = IRBuilder("lds_barrier_keep_vmem")
    b.kernel.attrs["max_workgroup_size"] = BLOCK
    x = b.param("X", PtrType(F16, "global"))
    y = b.param("Y", PtrType(F16, "global"))
    tid = b.thread_id_x()
    tile = b.smem_alloc(F16, [BLOCK], name_hint="tile")
    v = b.global_load(x, tid, F16)
    b.smem_store_vN(tile, [tid], v, 1)
    lds_barrier_keep_vmem(b, target)
    c_last = b.const_i32(BLOCK - 1)
    flip = b.sub(c_last, tid)
    u = b.smem_load_vN_f16(tile, flip, n=1)
    h = b.vec_extract(u, 0)
    b.global_store(y, tid, h)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
