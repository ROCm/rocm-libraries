#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# roll_recipe.py -- land #2 end-to-end: turn the per-shape concrete recipe of the
# unified-attention 2D kernel into ONE parametric recipe over head_size (D), so a
# single artifact covers D64/D128/D256/... with the storage benefit.
#
# Strategy (two-trace rolling), operating purely on recipe data:
#   1. Build the concrete recipe at D0=64 and D1=128 (kerneldef_to_recipe).
#   2. Find the repeated vec8 head-block run (its period L is fixed by the two
#      traces: L = (len1 - len0) / (D1/8 - D0/8)) and ROLL it into a
#      `static_for` over spec(D)//8, with the loop-carried `score` riding a
#      stable register (leading/trailing alias), and d_base -> mul(d8, 8).
#   3. Parameterize the head_size-scaled integer constants (the descriptor
#      strides: coeff * head_size) as coeff * spec(D), inferred from the two
#      traces.
#   4. Emit one recipe with kernel_name_fmt "...d{D}...".
#
# The C VM expands this per D; verified HSACO byte-identical to production for
# D64/D128/D256 by replaying the rolled recipe per shape (see
# rocke.portable_ir.src.online / the rocke_portable_ir_replay_cli tool).
import argparse
import copy
import json
import sys

from rocke.portable_ir.examples import export_mha
from rocke.portable_ir.src import kerneldef_to_recipe as k2r

VEC = 8


def _concrete_recipe(D):
    kern = export_mha.build("fp16", D, 2048, 1, 32, 1)
    return k2r.kerneldef_to_recipe(kern)


def _is_const_int(instr):
    return (
        instr.get("op") == "emit"
        and instr.get("opcode") == "arith.constant"
        and instr.get("attrs", {}).get("value", {}).get("t") == "i"
    )


def _const_val(instr):
    return instr["attrs"]["value"]["v"]


def _defs(instr):
    """Register(s) this instruction binds."""
    out = []
    if instr.get("op") == "emit" and "out" in instr:
        out.append(instr["out"]["bind"])
    elif instr.get("op") in ("const_i32", "const_f32", "thread_id_x", "param", "alias"):
        if "bind" in instr:
            out.append(instr["bind"])
    elif instr.get("op") == "scf_for":
        out.extend(instr.get("results", []))
    return out


def _operands(instr):
    return instr.get("in", []) if instr.get("op") == "emit" else []


def _replace_operands(instr, mapping):
    if instr.get("op") == "emit" and "in" in instr:
        instr["in"] = [mapping.get(x, x) for x in instr["in"]]


def _block_carry_in(block):
    """The loop-carried accumulator entering a block = the operand of the first
    arith.fadd that is NOT produced earlier within the block (it comes from the
    previous iteration / pre-loop init)."""
    local = set()
    for instr in block:
        if instr.get("op") == "emit" and instr.get("opcode") == "arith.fadd":
            for o in _operands(instr):
                if o not in local:
                    return o
        local.update(_defs(instr))
    return None


def _roll_body(body0, body1):
    """Roll the repeated head block in body1 (using body0 to fix the period).
    Returns a new rolled body list (parametric in spec D for the loop)."""
    L = (len(body1) - len(body0)) // (128 // VEC - 64 // VEC)
    if L <= 0:
        return None
    n1 = 128 // VEC

    # Locate run start: first index s where body1[s] is arith.constant 0 and
    # body1[s+L] is arith.constant 8 and body1[s+2L] is 16 (the d_base ladder).
    run_start = None
    for s in range(0, len(body1) - 2 * L):
        if (
            _is_const_int(body1[s])
            and _const_val(body1[s]) == 0
            and _is_const_int(body1[s + L])
            and _const_val(body1[s + L]) == VEC
            and _is_const_int(body1[s + 2 * L])
            and _const_val(body1[s + 2 * L]) == 2 * VEC
        ):
            run_start = s
            break
    if run_start is None:
        return None
    run_end = run_start + n1 * L

    # Regs defined before the run (loop-invariant + pre-loop accumulator).
    pre_defs = set()
    for instr in body1[:run_start]:
        pre_defs.update(_defs(instr))

    block0 = body1[run_start : run_start + L]
    block1 = body1[run_start + L : run_start + 2 * L]
    last_block = body1[run_start + (n1 - 1) * L : run_end]

    # Accumulator out of a block = last instr's bind (the 8th fadd).
    acc_out_block0 = _defs(block0[-1])[0]
    acc_out_last = _defs(last_block[-1])[0]

    # Carry-in to block1 = operand referencing block0's accumulator out.
    # Carry-in to block0 = operand referencing a pre-run reg (the pre-loop score).
    block1_defs = set()
    for instr in block1:
        block1_defs.update(_defs(instr))

    ACC = "%__score_acc"
    template = copy.deepcopy(block1)
    # d_base const (first instr) -> mul(d8, 8)
    assert _is_const_int(template[0]), "head block must start with d_base const"
    template[0]["attrs"]["value"] = {"t": "i", "v": {"mul": [{"var": "d8"}, VEC]}}
    # Rewrite the carry-in operand (ref to block0 acc out) -> ACC.
    for instr in template:
        _replace_operands(instr, {acc_out_block0: ACC})
    # Trailing alias: ACC <- this block's accumulator out (for next iteration).
    acc_out_template = _defs(template[-1])[0]
    template.append({"op": "alias", "bind": ACC, "from": acc_out_template})

    carry_in0 = _block_carry_in(block0)
    assert carry_in0 is not None, "could not find pre-loop accumulator"

    rolled = []
    rolled.extend(body1[:run_start])
    rolled.append({"op": "alias", "bind": ACC, "from": carry_in0})
    rolled.append(
        {
            "op": "static_for",
            "var": "d8",
            "lo": 0,
            "hi": {"div": [{"spec": "D"}, VEC]},
            "step": 1,
            "body": template,
        }
    )
    # Suffix: replace refs to the last block's accumulator out with ACC.
    suffix = copy.deepcopy(body1[run_end:])
    for instr in suffix:
        _replace_operands(instr, {acc_out_last: ACC})
    rolled.extend(suffix)
    return rolled


def _roll_program(prog0, prog1):
    """Walk prog0/prog1 in parallel; roll inside the scf_for whose body holds
    the head-block run; everything else copied from prog1."""
    out = []
    for i0, i1 in zip(prog0, prog1):
        if i1.get("op") == "scf_for":
            b0, b1 = i0["body"], i1["body"]
            if len(b0) != len(b1):
                rolled = _roll_body(b0, b1)
                if rolled is not None:
                    new = copy.deepcopy(i1)
                    new["body"] = rolled
                    out.append(new)
                    continue
            new = copy.deepcopy(i1)
            new["body"] = _roll_program(b0, b1)
            out.append(new)
        else:
            out.append(copy.deepcopy(i1))
    return out


def _parameterize_consts(prog0, prog1):
    """After rolling both programs, zip and turn head_size-scaled int constants
    into coeff * spec(D)."""
    for i0, i1 in zip(prog0, prog1):
        op = i1.get("op")
        if op == "emit" and i1.get("opcode") == "arith.constant":
            v0a = i0.get("attrs", {}).get("value", {})
            v1a = i1.get("attrs", {}).get("value", {})
            if v0a.get("t") == "i" and v1a.get("t") == "i":
                a, b = v0a["v"], v1a["v"]
                if isinstance(a, int) and isinstance(b, int) and a != b:
                    # scales with head_size: a = coeff*64, b = coeff*128
                    assert (
                        a % 64 == 0 and b == (a // 64) * 128
                    ), f"unexpected const {a},{b}"
                    coeff = a // 64
                    i1["attrs"]["value"] = {
                        "t": "i",
                        "v": {"mul": [{"spec": "D"}, coeff]},
                    }
        for key in ("body", "then", "else"):
            if key in i1 and key in i0:
                _parameterize_consts(i0[key], i1[key])


def make_parametric_recipe():
    r0 = _concrete_recipe(64)
    r1 = _concrete_recipe(128)
    rolled1 = _roll_program(r0["program"], r1["program"])
    # Roll the D0 program the same way so its structure matches rolled1 for
    # zip-based constant parameterization.
    rolled0 = _roll_program_same(r0["program"], r1["program"])
    _parameterize_consts(rolled0, rolled1)
    name_fmt = r1["kernel_name_fmt"].replace("_d128_", "_d{D}_")
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": name_fmt,
        "spec": [{"name": "D", "kind": "int"}, {"name": "dtype", "kind": "str"}],
        "attrs": r1["attrs"],
        "program": rolled1,
    }


def _roll_program_same(prog0, prog1):
    """Roll prog0 (D=64) using the same period inferred from prog0/prog1, so the
    rolled D0 program is structurally identical to rolled D1 for zip-based
    constant parameterization."""
    out = []
    for i0, i1 in zip(prog0, prog1):
        if i0.get("op") == "scf_for":
            b0, b1 = i0["body"], i1["body"]
            if len(b0) != len(b1):
                rolled = _roll_body_d0(b0, b1)
                new = copy.deepcopy(i0)
                new["body"] = rolled
                out.append(new)
                continue
            new = copy.deepcopy(i0)
            new["body"] = _roll_program_same(b0, b1)
            out.append(new)
        else:
            out.append(copy.deepcopy(i0))
    return out


def _roll_body_d0(body0, body1):
    L = (len(body1) - len(body0)) // (128 // VEC - 64 // VEC)
    n0 = 64 // VEC
    run_start = None
    for s in range(0, len(body0) - 2 * L):
        if (
            _is_const_int(body0[s])
            and _const_val(body0[s]) == 0
            and _is_const_int(body0[s + L])
            and _const_val(body0[s + L]) == VEC
        ):
            run_start = s
            break
    run_end = run_start + n0 * L
    pre_defs = set()
    for instr in body0[:run_start]:
        pre_defs.update(_defs(instr))
    block0 = body0[run_start : run_start + L]
    block1 = body0[run_start + L : run_start + 2 * L]
    last_block = body0[run_start + (n0 - 1) * L : run_end]
    acc_out_block0 = _defs(block0[-1])[0]
    acc_out_last = _defs(last_block[-1])[0]
    ACC = "%__score_acc"
    template = copy.deepcopy(block1)
    template[0]["attrs"]["value"] = {"t": "i", "v": {"mul": [{"var": "d8"}, VEC]}}
    for instr in template:
        _replace_operands(instr, {acc_out_block0: ACC})
    template.append({"op": "alias", "bind": ACC, "from": _defs(template[-1])[0]})
    carry_in0 = _block_carry_in(block0)
    rolled = []
    rolled.extend(body0[:run_start])
    rolled.append({"op": "alias", "bind": ACC, "from": carry_in0})
    rolled.append(
        {
            "op": "static_for",
            "var": "d8",
            "lo": 0,
            "hi": {"div": [{"spec": "D"}, VEC]},
            "step": 1,
            "body": template,
        }
    )
    suffix = copy.deepcopy(body0[run_end:])
    for instr in suffix:
        _replace_operands(instr, {acc_out_last: ACC})
    rolled.extend(suffix)
    return rolled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", choices=["recipe"], default="recipe")
    ap.parse_args()
    sys.stdout.write(json.dumps(make_parametric_recipe()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
