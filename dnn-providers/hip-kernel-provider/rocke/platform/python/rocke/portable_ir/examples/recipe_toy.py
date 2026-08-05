#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# recipe_toy.py -- authoring side of the "builder recipe" demo.
#
# Demonstrates the four goals:
#   Python authoring convenience  -- the kernel family is written once in Python;
#   compact per-builder artifact   -- ONE small recipe JSON encodes the whole
#                                     family (the `static_for` over D), regardless
#                                     of which D is JIT'd;
#   runtime shape flexibility      -- the C recipe VM specializes on D at runtime;
#   no CPython in hipDNN           -- the VM is pure C (src/recipe_vm.c).
#
# `build_toy(D, dtype)` is the Python reference kernel (a D-unrolled multiply-
# accumulate); `make_recipe()` is the compact recipe that the C VM expands into
# exactly that kernel for any D. The two are compared (byte-identical .ll) by
# rocke.portable_ir.drivers.parity_matrix; the same recipe is replayed by the
# hermetic ctest tests/portable_ir/recipe_vm_replay.cpp.
#
#   recipe_toy.py --emit recipe                     # print the recipe JSON
#   recipe_toy.py --emit ll   --D 128 --dtype f32   # print the Python-lowered .ll
#   recipe_toy.py --emit name --D 128 --dtype f32   # print the kernel name
import argparse
import json
import sys

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def kernel_name(D: int, dtype: str) -> str:
    return f"rocke_recipe_toy_d{D}_{dtype}"


def build_toy(D: int, dtype: str = "f32") -> "object":
    """Python reference: a D-unrolled multiply-accumulate over A,B -> C."""
    b = IRBuilder(kernel_name(D, dtype))
    b.kernel.attrs["max_workgroup_size"] = 64
    A = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(F32, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    acc = b.const_f32(0.0)
    for d in range(D):  # STRUCTURAL unroll over D (Python-time)
        off = b.const_i32(d)
        i = b.add(tid, off)
        a = b.global_load_f32(A, i, align=4)
        bb = b.global_load_f32(B, i, align=4)
        p = b.fmul(a, bb)
        acc = b.fadd(acc, p)
    b.global_store(C, tid, acc, align=4)
    b.ret()
    return b.kernel


def _i(v):
    return {"t": "i", "v": v}


def _s(v):
    return {"t": "s", "v": v}


def make_recipe() -> dict:
    """The compact, shape-agnostic recipe: one `static_for` over spec D unrolls
    the same body the Python builder does -- but the unroll happens in the C VM
    at JIT time, so this single artifact covers D64/D128/D256/..."""
    ptr_f32 = {"kind": "ptr", "pointee": "f32", "space": "global"}
    pattrs = {"noalias": True, "readonly": True, "align": 16}
    f32load = {"align": _i(4), "elem_type": _s("f32")}
    body = [
        {"op": "const_i32", "bind": "off", "val": {"var": "d"}},
        {
            "op": "emit",
            "opcode": "arith.add",
            "in": ["tid", "off"],
            "out": {"bind": "i", "type": "i32"},
        },
        {
            "op": "emit",
            "opcode": "memref.global_load_typed",
            "in": ["A", "i"],
            "out": {"bind": "a", "type": "f32"},
            "attrs": f32load,
        },
        {
            "op": "emit",
            "opcode": "memref.global_load_typed",
            "in": ["B", "i"],
            "out": {"bind": "bb", "type": "f32"},
            "attrs": f32load,
        },
        {
            "op": "emit",
            "opcode": "arith.fmul",
            "in": ["a", "bb"],
            "out": {"bind": "p", "type": "f32"},
        },
        {
            "op": "emit",
            "opcode": "arith.fadd",
            "in": ["acc", "p"],
            "out": {"bind": "acc", "type": "f32"},
        },
    ]
    program = [
        {"op": "param", "name": "A", "type": ptr_f32, "bind": "A", "attrs": pattrs},
        {"op": "param", "name": "B", "type": ptr_f32, "bind": "B", "attrs": pattrs},
        {
            "op": "param",
            "name": "C",
            "type": ptr_f32,
            "bind": "C",
            "attrs": {"noalias": True, "writeonly": True, "align": 16},
        },
        {"op": "thread_id_x", "bind": "tid"},
        {"op": "const_f32", "bind": "acc", "fval": 0.0},
        {
            "op": "static_for",
            "var": "d",
            "lo": 0,
            "hi": {"spec": "D"},
            "step": 1,
            "body": body,
        },
        {
            "op": "emit",
            "opcode": "memref.global_store_typed",
            "in": ["C", "tid", "acc"],
            "attrs": {"align": _i(4), "elem_type": _s("f32")},
        },
        {"op": "ret"},
    ]
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": "rocke_recipe_toy_d{D}_{dtype}",
        "spec": [{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
        "attrs": {"max_workgroup_size": _i(64)},
        "program": program,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll", "name"], default="recipe")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    if args.emit == "recipe":
        sys.stdout.write(json.dumps(make_recipe(), indent=2))
    elif args.emit == "name":
        sys.stdout.write(kernel_name(args.D, args.dtype))
    else:
        sys.stdout.write(
            lower_kernel_to_llvm(build_toy(args.D, args.dtype), arch=args.arch)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
