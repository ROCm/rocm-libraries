#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/attention_bwd_lds_emit.py -- Python reference emitter for the
# IR-emitting LDS helper of the attention backward:
# kernels.common._attention_bwd_caps.emit_lds_dma (the per-arch LDS-DMA width
# check, then b.async_buffer_load_lds). The layout classes of
# _attention_bwd_lds.py emit no IR (attention_bwd_layout_emit covers them).
# The helper has no kernel of its own, so this builds a small probe kernel
# that issues one LDS DMA per (dwords, coherency) of the config into
# consecutive 1 KiB pieces of one LDS slab; the C twin attention_bwd_lds_emit.c
# builds the same kernel through rocke_attn_bwd_emit_lds_dma and the two .ll
# must match. A width the arch refuses (or a coherency the builder refuses)
# fails the build on both sides.
#
# Config index map (keep in sync with attention_bwd_lds_emit.c):
#   0  gfx942   1 dword
#   1  gfx942   1 dword, coherency 1, 2, 3
#   2  gfx950   1, 3 and 4 dwords
#   3  gfx950   4 dwords coherency 3, 3 dwords coherency 2, 1 dword coherency 1
#   4  gfx942   rejected: 1 dword, then 4 dwords
#   5  gfx950   rejected: 2 dwords
#   6  gfx1151  rejected: no LDS DMA path
#   7  gfx1201  rejected: no LDS DMA path
#   8  gfx950   rejected: 0 dwords
#   9  gfx950   rejected: coherency 4 (builder)
#
# argv[2] selects the output: ll (default), ir, verify (see _emit_common.py).
from kernels.common._attention_bwd_caps import emit_lds_dma
from rocke.core.ir import F16, I32, I64, IRBuilder, PtrType

from _emit_common import run_emit

_CONFIGS = (
    ("gfx942", ((1, 0),)),
    ("gfx942", ((1, 1), (1, 2), (1, 3))),
    ("gfx950", ((1, 0), (3, 0), (4, 0))),
    ("gfx950", ((4, 3), (3, 2), (1, 1))),
    ("gfx942", ((1, 0), (4, 0))),
    ("gfx950", ((2, 0),)),
    ("gfx1151", ((1, 0),)),
    ("gfx1201", ((4, 0),)),
    ("gfx950", ((0, 0),)),
    ("gfx950", ((4, 4),)),
)


def _spec(idx: int):
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    arch, loads = _CONFIGS[idx]
    return loads, arch


def build_attention_bwd_lds_kernel(loads, *, arch: str):
    """The probe kernel: one emit_lds_dma per (dwords, coherency)."""
    b = IRBuilder("attn_bwd_lds_probe")
    src = b.param("src", PtrType(F16, "global"))
    nbytes = b.param("nbytes", I32)
    soff = b.param("soff", I32)
    tid = b.thread_id_x()
    rsrc = b.buffer_rsrc(src, nbytes)
    slab = b.smem_alloc(F16, [len(loads), 512], name_hint="slab")
    addr = b.smem_addr_of(slab)
    for i, (dwords, coherency) in enumerate(loads):
        piece = b.const_i32(i * 1024)
        piece64 = b.zext(piece, I64)
        lds = b.smem_ptr_add(addr, piece64)
        width = b.const_i32(dwords * 4)
        voff = b.mul(tid, width)
        emit_lds_dma(
            b,
            arch=arch,
            rsrc=rsrc,
            lds_ptr=lds,
            voffset=voff,
            soffset=soff,
            dwords=dwords,
            coherency=coherency,
        )
    b.ret()
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        build_attention_bwd_lds_kernel,
        usage="usage: attention_bwd_lds_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
