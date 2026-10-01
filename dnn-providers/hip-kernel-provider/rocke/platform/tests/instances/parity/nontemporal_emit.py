#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/nontemporal_emit.py -- Python reference emitter for the
# `nontemporal=` keyword of global_load_vN / global_store_vN.
#
# The flag records an op attr only when set and lowers to `!nontemporal !5`
# plus one module-level `!5 = !{i32 1}` node. Each config is a copy kernel
# (load src -> store dst) that pins one combination, byte-compared against
# nontemporal_emit.c. The all-default config pins that an unset flag adds
# nothing; the mixed configs pin that the attr stays on the op that set it.
from rocke.core.ir import BF16, F32, IRBuilder, KernelDef, PtrType

from _emit_common import run_emit

# (elem, n, load_nt, store_nt, arch)
CONFIGS = [
    (BF16, 8, True, True, "gfx950"),
    (BF16, 8, False, False, "gfx950"),
    (BF16, 8, True, False, "gfx942"),
    (F32, 4, False, True, "gfx942"),
]


def _spec(idx: int):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return CONFIGS[idx][:4], CONFIGS[idx][4]


def _build(spec, *, arch: str = "gfx950") -> KernelDef:
    elem, n, load_nt, store_nt = spec
    b = IRBuilder("nontemporal")
    b.kernel.attrs["max_workgroup_size"] = 64
    src = b.param("S", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    dst = b.param("D", PtrType(elem, "global"), noalias=True, align=16)
    tid = b.thread_id_x()
    off = b.mul(tid, b.const_i32(n))
    v = b.global_load_vN(src, off, elem, n, nontemporal=load_nt)
    b.global_store_vN(dst, off, v, n, nontemporal=store_nt)
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: nontemporal_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
