#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# qk_block.py -- the head_size-PARAMETRIC crux of the unified-attention 2D
# kernel, on a COPY (production attention_unified.py is untouched).
#
# This mirrors the production QK vec8 dot-product (attention_unified.py lines
# ~2696-2721): the reason head_size is "structural" is the
# `for d8 in b.unroll(head_size // 8)` compile-time unroll PLUS head_size baked
# into the descriptor stride (`head * head_size`). Both are made parametric here
# over the spec `D`, so ONE recorded recipe covers D64 / D128 / D256:
#
#   - `build_qk_block(D)`  : the Python reference (idiomatic, Python-time unroll)
#   - `record_qk_block()`  : authored against the recorder with a rolled
#                            `static_for_acc` over `spec(D)//8` (one artifact)
#
# Goal: the VM-expanded recipe is HSACO byte-identical to the reference for
# every D. Replay it with `rocke_portable_ir_replay_cli --recipe <file> --int D=...`,
# or in-process via rocke.portable_ir.src.online.
import argparse
import json
import sys

from rocke.core.ir import F16, F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm

VEC = 8


def kernel_name(D: int, dtype: str) -> str:
    return f"rocke_qk_block_d{D}_{dtype}"


def build_qk_block(D: int, dtype: str = "f16"):
    """Python reference: head_size baked at D (unroll count AND head stride)."""
    DT = F16
    b = IRBuilder(kernel_name(D, dtype))
    b.kernel.attrs["max_workgroup_size"] = 64
    Q = b.param("Q", PtrType(DT, "global"), noalias=True, readonly=True, align=16)
    K = b.param("K", PtrType(DT, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    head = b.param("head", I32)
    tid = b.thread_id_x()
    hs = b.const_i32(D)  # head_size -> descriptor stride (parametric)
    q_off_base = b.mul(head, hs)
    k_off_base = b.mul(head, hs)
    score = b.const_f32(0.0)
    for d8 in range(D // VEC):  # compile-time unroll over head_size
        d_base = b.const_i32(d8 * VEC)
        qv = b.global_load_vN(Q, b.add(q_off_base, d_base), DT, VEC, align=16)
        kv = b.global_load_vN(K, b.add(k_off_base, d_base), DT, VEC, align=16)
        for i in range(VEC):
            score = b.fadd(
                score,
                b.fmul(
                    b.cast_to_f32(b.vec_extract(qv, i)),
                    b.cast_to_f32(b.vec_extract(kv, i)),
                ),
            )
    b.global_store(C, tid, score, align=4)
    b.ret()
    return b.kernel


def record_qk_block():
    from rocke.portable_ir.src.recipe_recorder import RecipeRecorder, IExpr

    rec = RecipeRecorder(
        "rocke_qk_block_d{D}_{dtype}",
        spec=[{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
        kattrs={"max_workgroup_size": 64},
    )
    ptr_f16 = {"kind": "ptr", "pointee": "f16", "space": "global"}
    Q = rec.param("Q", ptr_f16, noalias=True, readonly=True, align=16)
    K = rec.param("K", ptr_f16, noalias=True, readonly=True, align=16)
    C = rec.param(
        "C",
        {"kind": "ptr", "pointee": "f32", "space": "global"},
        noalias=True,
        writeonly=True,
        align=16,
    )
    head = rec.param("head", "i32")
    tid = rec.thread_id_x()
    hs = rec.const_i32(rec.spec("D"))  # head_size = spec D (parametric stride)
    q_off_base = rec.mul(head, hs)
    k_off_base = rec.mul(head, hs)
    score0 = rec.const_f32(0.0)
    loop = rec.static_for_acc(
        "d8", 0, rec.idiv(rec.spec("D"), VEC), [("score", score0)]
    )
    with loop as (d8, (score,)):
        d_base = rec.const_i32(rec.imul(d8, VEC))
        qv = rec.load_vN(Q, rec.add(q_off_base, d_base), "f16", VEC)
        kv = rec.load_vN(K, rec.add(k_off_base, d_base), "f16", VEC)
        for i in range(VEC):
            score = rec.fadd(
                score,
                rec.fmul(
                    rec.cast_to_f32(rec.vec_extract(qv, i)),
                    rec.cast_to_f32(rec.vec_extract(kv, i)),
                ),
            )
        loop.set_carry("score", score)
    score = loop.results[0]
    rec.store(C, tid, score, elem="f32", align=4)
    rec.ret()
    return rec.recipe()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll", "name"], default="recipe")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    if args.emit == "recipe":
        sys.stdout.write(json.dumps(record_qk_block(), indent=2))
    elif args.emit == "name":
        sys.stdout.write(kernel_name(args.D, args.dtype))
    else:
        sys.stdout.write(
            lower_kernel_to_llvm(build_qk_block(args.D, args.dtype), arch=args.arch)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
