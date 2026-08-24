#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_lowering_emit.py -- Python reference emitter for the
# gfx1250 lowering surface.
#
# gfx1250 kernels are authored in Python (there is no cpp/instances/gfx1250/),
# so no instance-builder family in this gate exercises the gfx1250 *lowerer*.
# That left the six places Gfx1250Backend diverges from its Gfx12RdnaBackend
# parent covered only by the ROCKE_BACKEND=both pytest lane, and invisible to
# check_byte_identity.py -- the tree's stated definition-of-done. This family
# closes that: one config per divergence, built from the smallest kernel that
# reaches it, byte-compared against gfx1250_lowering_emit.c.
#
# The gfx950 twins are deliberately included. Four of the six divergences are a
# *choice between two encodings*, so a lowering that ignored the backend and
# always picked the gfx1250 form would still pass a gfx1250-only family. Pairing
# each with its gfx950 counterpart pins both sides of the choice.
#
# arch is per-config (see _spec), llvm_flavor = AUTO, matching the C side.
from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType

from _emit_common import run_emit


def _frag_operands(b, elem, n):
    """Load the three MMA fragments: A and B as <n x elem>, C as <8 x float>.

    Loading them (rather than materializing constants) keeps the operands
    opaque to the lowerer, so the WMMA call site is the only thing the config
    is testing.
    """
    a_ptr = b.param("A", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    b_ptr = b.param("B", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    c_ptr = b.param("C", PtrType(F32, "global"), noalias=True, align=16)
    tid = b.thread_id_x()
    a = b.global_load_vN(a_ptr, tid, dtype=elem, n=n)
    bb = b.global_load_vN(b_ptr, tid, dtype=elem, n=n)
    c = b.global_load_vN(c_ptr, tid, dtype=F32, n=8)
    return tid, c_ptr, a, bb, c


def _wmma_k32(elem):
    """K=32 f16/bf16 WMMA: the gfx1250 8-operand signature.

    bf16 is the interesting half: gfx11/gfx12 bitcast the operands to
    <16 x i16> before the call, while gfx1250 takes <16 x bfloat> directly.
    """

    def build(b: IRBuilder) -> None:
        tid, c_ptr, a, bb, c = _frag_operands(b, elem, 16)
        suffix = "f16" if elem is F16 else "bf16"
        d = b.mma(f"wmma_gfx1250_f32_16x16x32_{suffix}", a, bb, c)
        b.global_store(c_ptr, tid, d)
        b.ret()

    return build


def _wmma_k64(a_kind, b_kind):
    """K=64 fp8/bf8 WMMA: the gfx1250 6-operand signature.

    The fragments arrive as <8 x i32> (32 packed bytes per lane), so the
    dtype pair lives only in the op_id and the mangled intrinsic name.
    """

    def build(b: IRBuilder) -> None:
        tid, c_ptr, a, bb, c = _frag_operands(b, I32, 8)
        d = b.mma(f"wmma_gfx1250_f32_16x16x64_{a_kind}_{b_kind}", a, bb, c)
        b.global_store(c_ptr, tid, d)
        b.ret()

    return build


def _tr16_b128(elem):
    """ds_read_b128_tr_b16.

    gfx950 has one type-agnostic opcode returning <8 x i16> that the handler
    reinterprets; gfx1250 has per-element-type opcodes (.v8f16 / .v8bf16) that
    land in the right type with no reinterpret. Run on both arches so the
    element-typed selection cannot be hardcoded either way.
    """

    def build(b: IRBuilder) -> None:
        out = b.param("out", PtrType(elem, "global"), noalias=True, align=16)
        smem = b.smem_alloc(elem, [64, 8], name_hint="tile")
        tid = b.thread_id_x()
        zero = b.const_i32(0)
        v = b.ds_read_tr16_b128(smem, tid, zero, dtype=elem)
        b.global_store(out, tid, v)
        b.ret()

    return build


def build_barrier_drains(b: IRBuilder) -> None:
    """The two LDS workgroup barriers.

    An s_barrier does not drain outstanding LDS traffic, so each barrier is
    preceded by a wait. gfx9/10/11 spend one monolithic s_waitcnt on it;
    gfx1250 emits split s_wait_loadcnt / s_wait_dscnt (and for the LDS-only
    barrier, dscnt alone -- the VMEM chain deliberately stays in flight).
    Both barriers are here so the drain_vmem=True and =False paths are
    distinguished.
    """
    out = b.param("out", PtrType(F16, "global"), noalias=True, align=16)
    smem = b.smem_alloc(F16, [64, 8], name_hint="tile")
    tid = b.thread_id_x()
    zero = b.const_i32(0)
    b.sync()
    v = b.smem_load_vN(smem, tid, zero, dtype=F16, n=8)
    b.sync_lds_only()
    b.global_store(out, tid, v)
    b.ret()


def build_wait_counters(b: IRBuilder) -> None:
    """The two wait-counter facts.

    ``s_wait_asynccnt`` lowers to nothing on a backend with no async-DMA
    counter and to the intrinsic on gfx1250; ``s_waitcnt`` is the mirror
    image, since llvm.amdgcn.s.waitcnt is not selectable on gfx1250. A config
    that emitted both, or neither, on one arch would be wrong on the other.
    """
    b.s_wait_asynccnt(0)
    b.s_waitcnt(vmcnt=0, lgkmcnt=0)
    b.s_wait_asynccnt(3)
    b.s_waitcnt(lgkmcnt=0)
    b.ret()


# (builder, arch). Each gfx1250 config that tests a *choice* of encoding is
# followed by its gfx950 twin, so the pair pins both branches.
CONFIGS = [
    (_wmma_k32(F16), "gfx1250"),
    (_wmma_k32(BF16), "gfx1250"),
    (_wmma_k64("fp8", "fp8"), "gfx1250"),
    (_wmma_k64("fp8", "bf8"), "gfx1250"),
    (_wmma_k64("bf8", "fp8"), "gfx1250"),
    (_wmma_k64("bf8", "bf8"), "gfx1250"),
    (_tr16_b128(F16), "gfx1250"),
    (_tr16_b128(F16), "gfx950"),
    (_tr16_b128(BF16), "gfx1250"),
    (_tr16_b128(BF16), "gfx950"),
    (build_barrier_drains, "gfx1250"),
    (build_barrier_drains, "gfx950"),
    (build_wait_counters, "gfx1250"),
    (build_wait_counters, "gfx950"),
]


def _spec(idx: int):
    """Config selector: the (builder, arch) pair the shared driver expects."""
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return CONFIGS[idx]


def _build(build_fn, *, arch: str = "gfx1250") -> KernelDef:
    b = IRBuilder("gfx1250_lowering")
    b.kernel.attrs["max_workgroup_size"] = 64
    build_fn(b)
    return b.kernel


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_lowering_emit.py <config_index> [ll|ir|verify]\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
