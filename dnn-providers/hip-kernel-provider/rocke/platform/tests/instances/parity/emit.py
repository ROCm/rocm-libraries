#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/emit.py -- Python-side emitter for the rocke parity harness.
#
# Builds one of four kernels (selected by argv[1]) identically to the C emitter
# in emit.c and prints the lowered AMDGPU LLVM .ll to stdout, so the two outputs
# can be byte-compared.
#
#   scalar  : c = const(1); r = c + c
#   memory  : params -> tid -> 2x global_load_f32 -> fadd -> global_store
#   forloop : scf.for accumulating loop (exercises scf.for port)
#   vector  : vector splat + vector binop + vector fptrunc/extract
#
# arch = gfx950, llvm_flavor = None (AUTO), matching the C side.
import sys

from rocke.core.ir import IRBuilder, F16, F32, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def build_scalar(b: IRBuilder) -> None:
    c = b.const_i32(1)
    _ = b.add(c, c)
    b.ret()


def build_memory(b: IRBuilder) -> None:
    pf32 = PtrType(F32, "global")
    A = b.param("A", pf32)
    B = b.param("B", pf32)
    C = b.param("C", pf32)
    tid = b.thread_id_x()
    a = b.global_load_f32(A, tid, align=4)
    bb = b.global_load_f32(B, tid, align=4)
    s = b.fadd(a, bb)
    b.global_store(C, tid, s, align=4)
    b.ret()


def build_forloop(b: IRBuilder) -> None:
    pf32 = PtrType(F32, "global")
    C = b.param("C", pf32)
    lo = b.const_i32(0)
    hi = b.const_i32(16)
    step = b.const_i32(1)
    acc0 = b.const_f32(0.0)
    loop = b.scf_for_iter(
        lo,
        hi,
        step,
        [("acc", acc0)],
        iv_name="k0",
        unroll=False,
        elide_trailing_barrier=True,
    )
    with loop as (k0, iters):
        acc = iters[0]
        one = b.const_f32(1.0)
        nacc = b.fadd(acc, one)
        b.scf_yield(nacc)
    tid = b.thread_id_x()
    b.global_store(C, tid, loop.results[0], align=4)
    b.ret()


def build_vector(b: IRBuilder) -> None:
    pf16 = PtrType(F16, "global")
    C = b.param("C", pf16)
    s = b.const_f32(2.0)
    v = b.vector_splat(s, 4)  # <4 x f32>
    w = b.vector_add(v, v)  # <4 x f32>
    h = b.vec_trunc_f32_to_f16(w)  # <4 x f16>
    e = b.vec_extract(h, 0)  # f16
    tid = b.thread_id_x()
    b.store_f16(C, tid, e)
    b.ret()


BUILDERS = {
    "scalar": build_scalar,
    "memory": build_memory,
    "forloop": build_forloop,
    "vector": build_vector,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in BUILDERS:
        sys.stderr.write("usage: emit.py <scalar|memory|forloop|vector>\n")
        return 2
    which = sys.argv[1]
    b = IRBuilder("parity_kernel")
    BUILDERS[which](b)
    text = lower_kernel_to_llvm(b.kernel, llvm_flavor=None, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
