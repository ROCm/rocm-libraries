#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# recipe_multi_result.py -- a tiny kernel that emits a genuine MULTI-RESULT op
# (`inline_asm_multi` -> tile.inline_asm with a {i32,i32} struct return, the same
# shape the production clustered-MFMA helper uses). It exists to EXERCISE the
# recipe VM's multi-result ("outs") lowering all the way to a byte-identical
# HSACO -- one of the VM's previously-untested limits.
#
# The recipe is produced by the live RecordingIRBuilder (concrete record path):
# no hand-written recipe JSON, no spec parameters -- exactly how a production
# kernel would be captured for bundling.
#
#   recipe_multi_result.py --emit recipe   # concrete recipe JSON (recorded)
#   recipe_multi_result.py --emit ll        # Python reference .ll
#   recipe_multi_result.py --emit name
import argparse
import json
import sys

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm


def kernel_name(dtype: str = "i32") -> str:
    return f"rocke_multi_result_{dtype}"


def build_multi_result(dtype: str = "i32"):
    b = IRBuilder(kernel_name(dtype))
    b.kernel.attrs["max_workgroup_size"] = 64
    O = b.param("O", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    tid = b.thread_id_x()
    c = b.const_i32(7)
    # Two outputs ($0,$1) from two inputs ($2,$3): a real {i32,i32} struct asm.
    outs = b.inline_asm_multi(
        "v_add_u32 $0, $2, $3\n\tv_sub_u32 $1, $2, $3",
        "=v,=v,v,v",
        [tid, c],
        result_types=[I32, I32],
    )
    s = b.add(outs[0], outs[1])
    b.global_store(O, tid, s, align=4)
    b.ret()
    return b.kernel


def make_recipe() -> dict:
    from rocke.portable_ir.src.recording_builder import record_kernel

    _, recipe = record_kernel(lambda: build_multi_result("i32"))
    return recipe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe", "ll", "name"], default="recipe")
    ap.add_argument("--dtype", default="i32")
    ap.add_argument("--arch", default="gfx950")
    args = ap.parse_args()
    if args.emit == "recipe":
        sys.stdout.write(json.dumps(make_recipe(), indent=2))
    elif args.emit == "name":
        sys.stdout.write(kernel_name(args.dtype))
    else:
        sys.stdout.write(
            lower_kernel_to_llvm(build_multi_result(args.dtype), arch=args.arch)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
