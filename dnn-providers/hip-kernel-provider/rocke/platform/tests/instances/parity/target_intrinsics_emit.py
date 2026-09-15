#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/target_intrinsics_emit.py -- Python reference emitter for the
# LLVM 23 / future-operator intrinsic surface.
#
# The per-intrinsic smoke tests (tests/test_rocke.py TestNewTargetIntrinsics and
# tests/core/future_intrinsic_lowering.cpp) assert substrings against each
# engine's OWN output, so they cannot catch an engine that emits plausible-looking
# but different IR. This family closes that hole: run_diff.py byte-compares this
# output against target_intrinsics_emit.c, so the two engines have to agree
# exactly.
#
# Config 0 is deliberately the two-LDS-allocation async copy: the C++ engine used
# the raw smem allocation name as the destination GEP base instead of the pooled
# base pointer, emitting a reference to a global that is never defined. Every
# assertion-style test passed anyway because both engines named *a* pointer;
# only a byte comparison (or an LLVM verifier run) shows it.
#
# arch = gfx950, llvm_flavor = AUTO, matching the C side.
from rocke.core.ir import CACHE_STREAM, F16, I32, IRBuilder, KernelDef, PtrType

from _emit_common import run_emit


def build_async_lds_two_allocs(b: IRBuilder) -> None:
    """Async global->LDS copy into each of two LDS allocations.

    The second destination sits at a non-zero offset in the unified smem pool,
    so a lowering that skips the pool base-pointer step cannot accidentally
    produce the right address. Both allocations are copied into because an
    allocation with no use is dropped from the pool: leaving stageA dead would
    put stageB back at offset 0, where _emit_smem_base_ptr returns the pool
    name directly and the base-pointer hop this config exists to cover never
    runs.
    """
    src = b.param("src", PtrType(I32, "global"), align=16)
    stage_a = b.smem_alloc(I32, [64], name_hint="stageA")
    stage_b = b.smem_alloc(I32, [64, 4], name_hint="stageB")
    tid = b.thread_id_x()
    zero = b.const_i32(0)
    b.global_load_async_to_lds(
        src,
        tid,
        stage_a,
        [tid],
        width_bytes=16,
        coherency=CACHE_STREAM,
    )
    b.global_load_async_to_lds(
        src,
        tid,
        stage_b,
        [tid, zero],
        width_bytes=16,
        coherency=CACHE_STREAM,
    )
    b.s_wait_asynccnt(0)
    b.ret()


def build_async_lds_b8(b: IRBuilder) -> None:
    src = b.param("src", PtrType(I32, "global"), align=4)
    lds = b.smem_alloc(I32, [64], name_hint="stage")
    zero = b.const_i32(0)
    b.global_load_async_to_lds(src, zero, lds, [zero], width_bytes=1)
    b.ret()


def build_prefetch_inst(b: IRBuilder) -> None:
    """``s_prefetch_inst`` on a non-flat pointer.

    The operand is ``llvm_anyptr_ty``, so the address space is part of the
    overload; naming a bare ``ptr`` for an ``addrspace(1)`` value is a type
    mismatch the LLVM parser rejects outright.
    """
    code = b.param("code", PtrType(I32, "global"), align=4)
    length = b.const_i32(64)
    b.s_prefetch_inst(code, length)
    b.ret()


def build_buffer_load_lds_async(b: IRBuilder) -> None:
    X = b.param("X", PtrType(F16, "global"))
    N = b.param("N_bytes", I32)
    rsrc = b.buffer_rsrc(X, N)
    lds = b.smem_alloc(F16, [64, 8], name_hint="stage")
    lds_addr = b.smem_addr_of(lds)
    voffset = b.const_i32(0)
    soffset = b.const_i32(0)
    b.buffer_load_lds_async(
        rsrc,
        lds_addr,
        voffset,
        soffset,
        dwords=4,
        coherency=CACHE_STREAM,
    )
    b.ret()


def build_permlane(b: IRBuilder) -> None:
    tid = b.thread_id_x()
    old = b.const_i32(0)
    src1 = b.const_i32(2)
    src2 = b.const_i32(3)
    b.permlane16(old, tid, src1, src2)
    b.permlane16(old, tid, src1, src2, fi=True, bound_ctrl=True)
    b.permlane64(tid)
    b.ret()


def build_av_b128(b: IRBuilder) -> None:
    p = b.param("p", PtrType(I32, "global"), align=16)
    data = b.av_load_b128(p)
    b.av_store_b128(p, data)
    b.ret()


def build_scheduler_hints(b: IRBuilder) -> None:
    b.s_alloc_vgpr(8)
    b.asyncmark()
    b.wait_asyncmark(3)
    b.s_wait_event(1)
    b.ret()


BUILDERS = [
    build_async_lds_two_allocs,
    build_async_lds_b8,
    build_prefetch_inst,
    build_buffer_load_lds_async,
    build_permlane,
    build_av_b128,
    build_scheduler_hints,
]


def _spec(idx: int):
    """Config selector: the builder for `idx`, as the shared driver expects."""
    if not 0 <= idx < len(BUILDERS):
        raise SystemExit(f"unknown config index {idx}")
    return BUILDERS[idx]


def _build(build_fn, *, arch: str = "gfx950") -> KernelDef:
    b = IRBuilder("target_intrinsics")
    build_fn(b)
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: target_intrinsics_emit.py <config_index> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
