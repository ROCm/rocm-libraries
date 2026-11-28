#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compare consumer operand/scale bytes and writes across two authoring trees.

This is a bounded host I/O oracle. Matrix instructions are deliberately opaque:
we record every input byte and supply a deterministic finite FP32 result to
both arms. It proves tested transport/preparation/store equivalence, not GPU
matrix numerics, timing, or compiler scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from rocke.core.arch import ArchTarget
from rocke.core.dtypes import dtype_info
from rocke.core.ir import PtrType, VectorType
from rocke.core.ir_serialize import parse
from compare_fragment_transport_isa import _EMITTER, cases, source_digest


def width(ty) -> int:
    return (
        width(ty.elem) * ty.count
        if isinstance(ty, VectorType)
        else max(1, dtype_info(ty.name).encoded_bits // 8)
    )


def raw(value, ty) -> bytes:
    if isinstance(value, bytes):
        return value
    bits = width(ty) * 8
    return (int(value) & ((1 << bits) - 1)).to_bytes(width(ty), "little")


def integer(value, ty, signed=True) -> int:
    return (
        int.from_bytes(value, "little", signed=signed)
        if isinstance(value, bytes)
        else int(value)
    )


def floating(value, ty) -> np.float32:
    if ty.name == "f16":
        return np.float32(struct.unpack("<e", raw(value, ty))[0])
    if ty.name == "bf16":
        return np.float32(struct.unpack("<f", b"\0\0" + raw(value, ty))[0])
    return np.float32(struct.unpack("<f", raw(value, ty))[0])


def f32(value) -> bytes:
    return struct.pack("<f", np.float32(value))


def run(kernel, case, lane, initial, pointers, block_x, block_y):
    values = {}
    memory = {name: bytearray(data) for name, data in initial.items()}
    calls, writes = [], []
    for param in kernel.params:
        name = "%" + param.name
        if isinstance(param.type, PtrType):
            values[name] = pointers[name]
        elif param.type.name == "f32":
            values[name] = f32(1.2345 if param.name.startswith("A_") else 0.9876)
        else:
            values[name] = case.get(param.name.lower(), case.get("k", 0))
    for op in kernel.body.ops:
        args = [values[v.name] for v in op.operands]
        name, result = op.name, None
        if name == "gpu.thread_id":
            result = lane
        elif name == "gpu.block_id":
            result = block_x if op.attrs["axis"] == "x" else block_y
        elif name == "arith.constant":
            result = (
                f32(op.attrs["value"])
                if op.result.type.name == "f32"
                else int(op.attrs["value"])
            )
        elif name == "arith.constant_vec":
            result = bytes(width(op.result.type))
        elif name in (
            "arith.add",
            "arith.mul",
            "arith.div",
            "arith.mod",
            "arith.and",
            "arith.or",
            "arith.shl",
            "arith.lshr",
        ):
            a, b = [
                integer(value, operand.type)
                for value, operand in zip(args, op.operands)
            ]
            bits = width(op.result.type) * 8
            if name == "arith.add":
                result = a + b
            elif name == "arith.mul":
                result = a * b
            elif name == "arith.div":
                result = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            elif name == "arith.mod":
                result = a % b
            elif name == "arith.and":
                result = a & b
            elif name == "arith.or":
                result = a | b
            elif name == "arith.shl":
                result = a << b
            else:
                result = (a & ((1 << bits) - 1)) >> b
            if op.result.type.name != "i1":
                result = (result + (1 << (bits - 1))) % (1 << bits) - (1 << (bits - 1))
        elif name in ("arith.sext", "arith.zext"):
            result = integer(args[0], op.operands[0].type, signed=name == "arith.sext")
        elif name == "arith.cmp":
            pred = op.attrs["pred"]
            a, b = [
                integer(value, operand.type, signed=not pred.startswith("u"))
                for value, operand in zip(args, op.operands)
            ]
            result = {
                "eq": a == b,
                "ne": a != b,
                "lt": a < b,
                "slt": a < b,
                "ult": a < b,
                "gt": a > b,
                "sgt": a > b,
                "ugt": a > b,
            }.get(pred)
            if result is None:
                raise NotImplementedError(pred)
        elif name == "arith.select":
            result = args[1] if args[0] else args[2]
        elif name == "tile.global_ptr_add":
            result = (args[0][0], args[0][1] + integer(args[1], op.operands[1].type))
        elif name in ("memref.global_load_typed", "memref.global_load_vN"):
            pointer, offset = args
            unit = (
                op.result.type.elem
                if isinstance(op.result.type, VectorType)
                else op.result.type
            )
            start = pointer[1] + integer(offset, op.operands[1].type) * width(unit)
            result = bytes(memory[pointer[0]][start : start + width(op.result.type)])
            if start < 0 or len(result) != width(op.result.type):
                raise AssertionError((name, start))
        elif name == "memref.global_store_typed":
            pointer, offset, value = args
            data = raw(value, op.operands[2].type)
            start = pointer[1] + integer(offset, op.operands[1].type) * len(data)
            if start < 0 or start + len(data) > len(memory[pointer[0]]):
                raise AssertionError((name, start))
            memory[pointer[0]][start : start + len(data)] = data
            writes.append((pointer[0], start, data.hex()))
        elif name == "arith.bitcast":
            result = raw(args[0], op.operands[0].type)
        elif name in ("vector.concat", "vector.pack"):
            result = b"".join(
                raw(value, operand.type) for value, operand in zip(args, op.operands)
            )
        elif name == "vector.splat":
            result = raw(args[0], op.operands[0].type) * op.result.type.count
        elif name == "vector.extract":
            size, index = width(op.result.type), op.attrs["index"]
            result = args[0][index * size : (index + 1) * size]
        elif name == "vector.insert":
            size, index = width(op.operands[1].type), op.attrs["index"]
            result = (
                args[0][: index * size]
                + raw(args[1], op.operands[1].type)
                + args[0][(index + 1) * size :]
            )
        elif name in ("arith.fmul", "arith.fadd"):
            a, b = [
                floating(value, operand.type)
                for value, operand in zip(args, op.operands)
            ]
            result = f32(a * b if name == "arith.fmul" else a + b)
        elif name == "arith.cast_to_f32":
            result = f32(floating(args[0], op.operands[0].type))
        elif name == "arith.cast_f32_to":
            value = floating(args[0], op.operands[0].type)
            if op.result.type.name == "f16":
                result = struct.pack("<e", np.float16(value))
            elif op.result.type.name == "bf16":
                bits = int.from_bytes(f32(value), "little")
                result = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).to_bytes(
                    2, "little"
                )
            else:
                raise NotImplementedError(op.result.type.name)
        elif (
            name == "tile.inline_asm"
            and op.attrs["template"] == ""
            and op.attrs["constraints"] == "=v,0"
        ):
            result = args[0]
        elif name == "tile.mma":
            operands = [
                (operand.type.name, raw(value, operand.type).hex())
                for value, operand in zip(args, op.operands)
            ]
            calls.append((dict(op.attrs), operands))
            seed = int.from_bytes(
                hashlib.sha256(json.dumps(calls[-1], sort_keys=True).encode()).digest()[
                    :4
                ],
                "little",
            )
            result = b"".join(
                f32((seed % 17 - 8) * 0.25 + index * 0.125 + lane * 0.03125)
                for index in range(op.result.type.count)
            )
        elif name == "cf.return":
            pass
        else:
            raise NotImplementedError(name)
        if op.results:
            values[op.result.name] = result
    return calls, writes


def emit(root: Path, case: dict):
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "python"),
        ROCKE_BACKEND="python",
        ROCKE_LLVM_FLAVOR="llvm23",
    )
    text = subprocess.check_output(
        [sys.executable, "-c", _EMITTER, json.dumps(case), "llvm23", "serialized"],
        cwd=root,
        env=env,
        text=True,
        timeout=120,
    )
    return parse(text)


def setup(kernel, case, aliases):
    memory, pointers = {}, {}
    for index, param in enumerate(kernel.params):
        if not isinstance(param.type, PtrType):
            continue
        name = "%" + param.name
        ty = param.type.pointee
        count = max(16384, 65536 // width(ty))
        if ty.name == "f32":
            data = b"".join(f32((i % 37 - 18) * 0.12345) for i in range(count))
        elif ty.name == "f16":
            data = b"".join(
                struct.pack("<e", (i % 7 + 1) * 0.125) for i in range(count)
            )
        else:
            data = bytes((i * 29 + index * 17) % 256 for i in range(count * width(ty)))
        memory[name] = data
        pointers[name] = (name, 0)
    if aliases:
        group = 64 // case["m"]
        pointers["%PA"] = ("%B", group * 4)
        pointers["%PB"] = ("%A", group * 4)
    return memory, pointers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        baseline_source_sha256=source_digest(args.baseline),
        candidate_source_sha256=source_digest(args.candidate),
        evidence="host I/O trace with opaque matrix results; not GPU numerics",
        cases=[],
    )
    failed = False
    for index, original in enumerate(cases()):
        case = dict(original)
        if case["kind"] != "gfx942":
            tile = 32 if case["kind"] == "gfx950" else 16
            case.update(m=tile * 2, n=tile * 3)
        old, new = emit(args.baseline, case), emit(args.candidate, case)
        checks = 0
        try:
            for aliases in (False, True) if case["kind"] == "gfx942" else (False,):
                memory, pointers = setup(old, case, aliases)
                for lane in range(ArchTarget.from_gfx(case["target"]).wave_size):
                    block_x, block_y = (1, 0) if case["kind"] == "gfx942" else (1, 1)
                    before = run(old, case, lane, memory, pointers, block_x, block_y)
                    after = run(new, case, lane, memory, pointers, block_x, block_y)
                    if before != after:
                        raise AssertionError(("trace mismatch", lane, aliases))
                    checks += 1
            item = dict(case=case, status="pass", lane_checks=checks)
        except (AssertionError, NotImplementedError, KeyError, ValueError) as error:
            item = dict(
                case=case, status="failed", error=str(error), lane_checks=checks
            )
            failed = True
        report["cases"].append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            f"case-{index:03d} {case['target']}: {item['status']} ({checks} lanes)",
            flush=True,
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
