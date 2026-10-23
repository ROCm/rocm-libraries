#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# mini_attn.py -- a representative reduction kernel that exercises the SAME
# structural features the real unified-attention 2D scalar kernel uses (per the
# IR-surface analysis): a RUNTIME scf.for online-softmax accumulation over a
# runtime length `n` (3 iter-args m/l/acc), a RUNTIME scf.if store guard, and a
# COMPILE-TIME static_if on the spec flag `use_norm` (present-or-not normalize).
#
# This is the evaluation vehicle for "build the VM ISA to support the attention
# kernel": it proves the recipe VM's ISA can express attention's control
# structure (runtime scf.for/scf.if + compile-time spec branch + the
# exp2/fmax/rcp reduction math) and that ONE recipe specializes on `use_norm`
# at JIT time, matching the Python reference HSACO byte-for-byte.
#
#   mini_attn.py --emit recipe
#   mini_attn.py --emit ll   --use-norm 1
#   mini_attn.py --emit name --use-norm 1
import argparse
import json
import sys

from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def kernel_name(use_norm: int, dtype: str) -> str:
    return f"rocke_mini_attn_norm{int(use_norm)}_{dtype}"


def build_mini_attn(use_norm: int, dtype: str = "f32"):
    b = IRBuilder(kernel_name(use_norm, dtype))
    b.kernel.attrs["max_workgroup_size"] = 64
    Q = b.param("Q", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    O = b.param("O", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    n = b.param("n", I32)
    tid = b.thread_id_x()
    m0 = b.const_f32(-1e30)
    l0 = b.const_f32(0.0)
    acc0 = b.const_f32(0.0)
    lo = b.const_i32(0)
    step = b.const_i32(1)
    loop = b.scf_for_iter(
        lo,
        n,
        step,
        [("m", m0), ("l", l0), ("acc", acc0)],
        iv_name="k",
        unroll=False,
        elide_trailing_barrier=True,
    )
    with loop as (k, iters):
        m, l, acc = iters
        idx = b.add(tid, k)
        x = b.global_load_f32(Q, idx, align=4)
        mnew = b.fmax(m, x)
        p = b.exp2(b.fsub(x, mnew))
        c = b.exp2(b.fsub(m, mnew))
        lnew = b.fadd(b.fmul(l, c), p)
        accnew = b.fadd(b.fmul(acc, c), b.fmul(p, x))
        b.scf_yield(mnew, lnew, accnew)
    accf = loop.results[2]
    lf = loop.results[1]
    if use_norm:  # COMPILE-TIME branch on the spec
        out = b.fmul(accf, b.rcp(lf))
    else:
        out = accf
    z = b.const_i32(0)
    cond = b.cmp_gt(n, z)
    guard = b.scf_if(cond)
    with guard:
        b.global_store(O, tid, out, align=4)
    b.ret()
    return b.kernel


def _i(v):
    return {"t": "i", "v": v}


def _f(v):
    return {"t": "f", "v": v}


def _s(v):
    return {"t": "s", "v": v}


def _const(bind, ity, value):
    return {
        "op": "emit",
        "opcode": "arith.constant",
        "out": {"bind": bind, "type": ity},
        "attrs": {"ity": _s(ity), "value": value},
    }


def make_recipe() -> dict:
    ptr_f32 = {"kind": "ptr", "pointee": "f32", "space": "global"}
    f32load = {"align": _i(4), "elem_type": _s("f32")}
    body = [
        {
            "op": "emit",
            "opcode": "arith.add",
            "in": ["tid", "k"],
            "out": {"bind": "idx", "type": "i32"},
        },
        {
            "op": "emit",
            "opcode": "memref.global_load_typed",
            "in": ["Q", "idx"],
            "out": {"bind": "x", "type": "f32"},
            "attrs": f32load,
        },
        {
            "op": "emit",
            "opcode": "arith.fmax",
            "in": ["m", "x"],
            "out": {"bind": "mnew", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fsub",
            "in": ["x", "mnew"],
            "out": {"bind": "d1", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "math.exp2",
            "in": ["d1"],
            "out": {"bind": "p", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fsub",
            "in": ["m", "mnew"],
            "out": {"bind": "d2", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "math.exp2",
            "in": ["d2"],
            "out": {"bind": "c", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fmul",
            "in": ["l", "c"],
            "out": {"bind": "lc", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fadd",
            "in": ["lc", "p"],
            "out": {"bind": "lnew", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fmul",
            "in": ["acc", "c"],
            "out": {"bind": "ac", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fmul",
            "in": ["p", "x"],
            "out": {"bind": "px", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fadd",
            "in": ["ac", "px"],
            "out": {"bind": "accnew", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "scf.yield",
            "in": ["mnew", "lnew", "accnew"],
            "attrs": {"num": _i(3)},
        },
    ]
    program = [
        {
            "op": "param",
            "name": "Q",
            "type": ptr_f32,
            "bind": "Q",
            "attrs": {"noalias": True, "readonly": True, "align": 16},
        },
        {
            "op": "param",
            "name": "O",
            "type": ptr_f32,
            "bind": "O",
            "attrs": {"noalias": True, "writeonly": True, "align": 16},
        },
        {"op": "param", "name": "n", "type": "i32", "bind": "n"},
        {"op": "thread_id_x", "bind": "tid"},
        _const("m0", "f32", _f(-1e30)),
        _const("l0", "f32", _f(0.0)),
        _const("acc0", "f32", _f(0.0)),
        _const("lo", "i32", _i(0)),
        _const("step", "i32", _i(1)),
        {
            "op": "scf_for",
            "iv": "k",
            "lo": "lo",
            "hi": "n",
            "step": "step",
            "iter": [
                {"name": "m", "init": "m0"},
                {"name": "l", "init": "l0"},
                {"name": "acc", "init": "acc0"},
            ],
            "results": ["mf", "lf", "accf"],
            "unroll": False,
            "elide_trailing_barrier": True,
            "body": body,
        },
        {
            "op": "static_if",
            "pred": {"spec": "use_norm"},
            "then": [
                {
                    "op": "emit",
                    "opcode": "math.rcp",
                    "in": ["lf"],
                    "out": {"bind": "rl", "type": "f32"},
                },
                {
                    "op": "emit",
                    "opcode": "arith.fmul",
                    "in": ["accf", "rl"],
                    "out": {"bind": "out", "type": "f32"},
                },
            ],
            "else": [{"op": "alias", "bind": "out", "from": "accf"}],
        },
        _const("z", "i32", _i(0)),
        {
            "op": "emit",
            "opcode": "arith.cmp",
            "in": ["n", "z"],
            "out": {"bind": "cond", "type": "i1"},
            "attrs": {"pred": _s("gt")},
        },
        {
            "op": "scf_if",
            "cond": "cond",
            "then": [
                {
                    "op": "emit",
                    "opcode": "memref.global_store_typed",
                    "in": ["O", "tid", "out"],
                    "attrs": f32load,
                }
            ],
        },
        {"op": "ret"},
    ]
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": "rocke_mini_attn_norm{use_norm}_{dtype}",
        "spec": [{"name": "use_norm", "kind": "int"}, {"name": "dtype", "kind": "str"}],
        "attrs": {"max_workgroup_size": _i(64)},
        "program": program,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll", "name"], default="recipe")
    ap.add_argument("--use-norm", type=int, default=1)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    if args.emit == "recipe":
        sys.stdout.write(json.dumps(make_recipe(), indent=2))
    elif args.emit == "name":
        sys.stdout.write(kernel_name(args.use_norm, args.dtype))
    else:
        sys.stdout.write(
            lower_kernel_to_llvm(
                build_mini_attn(args.use_norm, args.dtype), arch=args.arch
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
