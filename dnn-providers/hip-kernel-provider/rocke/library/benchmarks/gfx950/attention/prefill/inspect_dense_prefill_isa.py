#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Disassemble a dense-prefill kernel and report resources + instruction mix.

Used to prove a candidate lever actually changed codegen (and did not spill)
before trusting any paired timing result.

Usage::

    python inspect_dense_prefill_isa.py --shape-json shape.json \
        --output-json isa.json --dump-asm out.s
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_HERE = os.path.dirname(__file__)
_RK = os.path.abspath(os.path.join(_HERE, "../../../../.."))
sys.path.insert(0, _RK + "/platform/python")
sys.path.insert(0, _RK + "/library")

from builders.gfx950.attention.prefill.attention_dense_prefill import (  # noqa: E402
    make_spec_from_shape,
)
from kernels.gfx950.attention_dense import build_attention_dense  # noqa: E402
from rocke.helpers.compile import compile_kernel  # noqa: E402

# Metadata keys the AMDGPU assembler emits in the .amdhsa_kernel / msgpack note.
_RESOURCE_KEYS = (
    "vgpr_count",
    "agpr_count",
    "sgpr_count",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "private_segment_fixed_size",
    "group_segment_fixed_size",
    "occupancy",
)

_CLASSES = {
    "mfma": re.compile(r"^v_mfma"),
    "ds_read": re.compile(r"^ds_read"),
    "ds_write": re.compile(r"^ds_write"),
    "global_load": re.compile(r"^global_load"),
    "buffer_load": re.compile(r"^buffer_load"),
    "async_load_lds": re.compile(r"^buffer_load_.*lds|^global_load_lds"),
    "scratch": re.compile(r"^scratch_"),
    "s_waitcnt": re.compile(r"^s_waitcnt"),
    "barrier": re.compile(r"^s_barrier"),
    "valu": re.compile(r"^v_(?!mfma)"),
    "transcendental": re.compile(r"^v_exp|^v_rcp|^v_log"),
}


def _llvm_tool(name: str) -> str | None:
    for cand in (
        shutil.which(name),
        f"/opt/rocm/llvm/bin/{name}",
        f"/opt/rocm/bin/{name}",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


def _disassemble(hsaco: bytes) -> str:
    objdump = _llvm_tool("llvm-objdump")
    if not objdump:
        raise RuntimeError("llvm-objdump not found")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.hsaco"
        p.write_bytes(hsaco)
        out = subprocess.run(
            [objdump, "-d", "--mcpu=gfx950", str(p)],
            capture_output=True,
            text=True,
            check=True,
        )
    return out.stdout


def _readelf_notes(hsaco: bytes) -> str:
    readelf = _llvm_tool("llvm-readelf")
    if not readelf:
        return ""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.hsaco"
        p.write_bytes(hsaco)
        out = subprocess.run(
            [readelf, "--notes", str(p)], capture_output=True, text=True
        )
    return out.stdout


def _parse_resources(notes: str) -> dict:
    res: dict = {}
    for key in _RESOURCE_KEYS:
        m = re.search(rf"\.{key}:\s*(\d+)", notes)
        if m:
            res[key] = int(m.group(1))
    return res


def _instruction_mix(asm: str) -> dict:
    counts: Counter[str] = Counter()
    total = 0
    for line in asm.splitlines():
        # objdump body lines look like: "\t<mnemonic> <operands> // encoding"
        m = re.match(r"^\s*(?:[0-9a-f]+:\s+)?([a-z][a-z0-9_]+)\s", line)
        if not m:
            continue
        mnem = m.group(1)
        if mnem.startswith((".", "//")):
            continue
        total += 1
        for cls, pat in _CLASSES.items():
            if pat.match(mnem):
                counts[cls] += 1
    counts_d = dict(sorted(counts.items()))
    counts_d["total_instructions"] = total
    return counts_d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--dump-asm", type=Path, default=None)
    args = ap.parse_args()

    shape = json.loads(args.shape_json.read_text())
    spec = make_spec_from_shape(shape)
    art = compile_kernel(
        build_attention_dense(spec),
        arch="gfx950",
        backend="python",
        capture_ir_text=False,
    )

    asm = _disassemble(art.hsaco)
    if args.dump_asm:
        args.dump_asm.parent.mkdir(parents=True, exist_ok=True)
        args.dump_asm.write_text(asm)

    report = {
        "kernel_name": spec.kernel_name(),
        "shape_json": str(args.shape_json),
        "hsaco_bytes": len(art.hsaco),
        "resources": _parse_resources(_readelf_notes(art.hsaco)),
        "instruction_mix": _instruction_mix(asm),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    spills = report["resources"].get("vgpr_spill_count", 0) + report[
        "resources"
    ].get("sgpr_spill_count", 0)
    scratch = report["resources"].get("private_segment_fixed_size", 0)
    if spills or scratch:
        print(f"WARNING spills={spills} scratch={scratch}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
