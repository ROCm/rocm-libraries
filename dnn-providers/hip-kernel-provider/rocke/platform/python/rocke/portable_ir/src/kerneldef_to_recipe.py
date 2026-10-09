#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# kerneldef_to_recipe.py -- assemble the FULL unified-attention 2D kernel into a
# recipe by recording the PRODUCTION builder's output.
#
# Rather than hand-port build_unified_attention_2d (~3000 LOC of helpers,
# closures, dataclasses, descriptor math) to the recorder, we run the real
# production builder (untouched) to get its KernelDef, then convert that graph
# op-for-op into a recipe (param / generic emit / scf_for / scf_if / ret). The C
# recipe VM then re-emits it and lowers it. Because the recipe carries the exact
# production op stream, the comgr HSACO is byte-identical (SSA value-name
# differences in the .ll don't affect the object).
#
# This is a CONCRETE (per-shape) recipe -- it proves the recipe VM runs the whole
# production kernel byte-identically. The PARAMETRIC (one-recipe-per-head-dim)
# storage win is demonstrated separately on the head_size crux (qk_block.py).
#
#   kerneldef_to_recipe.py --emit recipe --D 128 --dtype fp16
#   kerneldef_to_recipe.py --emit ll     --D 128 --dtype fp16   # production lower
import argparse
import json
import sys

from rocke.core.ir_export import _attrs_to_json, _type_to_json
from rocke.core.lower_llvm import lower_kernel_to_llvm

from rocke.portable_ir.examples import export_mha


def _reg(v):
    return v.name[1:] if v.name.startswith("%") else v.name


def _bare(name):
    return name[1:] if isinstance(name, str) and name.startswith("%") else name


def _op_to_instr(op):
    if op.name == "scf.for":
        lo, hi, step = op.operands[0], op.operands[1], op.operands[2]
        inits = op.operands[3:]
        iter_meta = op.attrs.get("iter_args", [])
        iters = [
            {"name": _bare(m["name"]), "init": _reg(inits[i])}
            for i, m in enumerate(iter_meta)
        ]
        return {
            "op": "scf_for",
            "iv": _bare(op.attrs["iv"]),
            "lo": _reg(lo),
            "hi": _reg(hi),
            "step": _reg(step),
            "iter": iters,
            "results": [_reg(r) for r in op.results],
            "unroll": bool(op.attrs.get("unroll", False)),
            "elide_trailing_barrier": bool(
                op.attrs.get("elide_trailing_barrier", True)
            ),
            "body": [_op_to_instr(o) for o in op.regions[0].ops],
        }
    if op.name == "scf.if":
        return {
            "op": "scf_if",
            "cond": _reg(op.operands[0]),
            "then": [_op_to_instr(o) for o in op.regions[0].ops],
        }
    if op.name == "cf.return":
        return {"op": "ret"}
    instr = {"op": "emit", "opcode": op.name, "in": [_reg(o) for o in op.operands]}
    if op.results:
        from rocke.portable_ir.src.recording_builder import result_pfx

        pfx = result_pfx(op)
        instr["out"] = {
            "bind": _reg(op.results[0]),
            "type": _type_to_json(op.results[0].type),
            **({"pfx": pfx} if pfx else {}),
        }
    if op.attrs:
        instr["attrs"] = _attrs_to_json(op.attrs)
    return instr


def kerneldef_to_recipe(kernel) -> dict:
    params = [
        {
            "op": "param",
            "name": p.name,
            "type": _type_to_json(p.type),
            "bind": p.name,
            "attrs": dict(p.attrs),
        }
        for p in kernel.params
    ]
    body = [_op_to_instr(o) for o in kernel.body.ops]
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": kernel.name,
        "spec": [],
        "attrs": _attrs_to_json(kernel.attrs),
        "program": params + body,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll"], default="recipe")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--gqa", type=int, default=1)
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    kernel = export_mha.build(args.dtype, args.D, args.seqlen, 1, 32, args.gqa)
    if args.emit == "ll":
        sys.stdout.write(lower_kernel_to_llvm(kernel, arch=args.arch))
    else:
        sys.stdout.write(json.dumps(kerneldef_to_recipe(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
