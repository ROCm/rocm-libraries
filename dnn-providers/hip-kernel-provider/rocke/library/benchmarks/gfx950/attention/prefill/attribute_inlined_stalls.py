#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Attribute ATT stalls to real call sites using the inline_frames.json sidecar.

rocprofv3 flattens DWARF to the innermost frame, so every MFMA in the kernel is
charged to the one-line ``mfma_*_for_dtype`` dispatcher and every LDS read to one
loader. This walks each instruction's recovered inline stack instead, so a stall
lands on the kernel phase that asked for the work (``do_qk`` vs ``do_pv``).

Usage::

    python attribute_inlined_stalls.py <dispatch_dir> --output-json attr.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# code.json columns (see stage4_analyze/parse_kernel_trace.py).
C_ASM, C_PCIDX, C_SRC, C_CODEOBJ, C_VADDR = 0, 2, 3, 4, 5
C_EXEC, C_TOTAL, C_STALL = 6, 7, 8

# Frames at or above the IR builder are authoring scaffolding, not GPU phases;
# the frame just inside the builder is the phase that emitted the instruction.
_BUILDER_FUNCS = {
    "build_attention_dense",
    "_build_attention_dense_persistent",
    "_build_attention_dense_default",
}


def _stall_type(asm: str) -> str:
    a = asm.lower()
    if "s_waitcnt" in a:
        if "vmcnt" in a:
            return "vmem_wait"
        if "lgkmcnt" in a:
            return "lds_wait"
        return "waitcnt"
    if "s_barrier" in a:
        return "barrier_stall"
    if "ds_read" in a:
        return "lds_read"
    if "ds_write" in a:
        return "lds_write"
    if "buffer_load" in a or "global_load" in a:
        return "vmem_load"
    if "buffer_store" in a or "global_store" in a:
        return "vmem_store"
    if "v_mfma" in a:
        return "mfma"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dispatch_dir", type=Path)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    d = args.dispatch_dir
    rows = json.loads((d / "code.json").read_text())["code"]
    sc = json.loads((d / "inline_frames.json").read_text())
    funcs, files, stacks = sc["functions"], sc["files"], sc["stacks"]

    total = 0
    by_type: dict[str, int] = defaultdict(int)
    by_phase: dict[str, int] = defaultdict(int)
    by_phase_type: dict[tuple[str, str], int] = defaultdict(int)
    by_site: dict[str, int] = defaultdict(int)
    unresolved = 0

    for row in rows:
        if not isinstance(row[C_PCIDX], int) or row[C_PCIDX] == 0:
            continue
        stall = row[C_STALL] if isinstance(row[C_STALL], int) else 0
        if stall <= 0:
            continue
        total += stall
        st = _stall_type(row[C_ASM])
        by_type[st] += stall

        stack = stacks.get(f"{row[C_CODEOBJ]}:{row[C_VADDR]}")
        if not stack:
            unresolved += stall
            continue

        # Outermost first; find the first frame inside the IR builder.
        names = [funcs[f[0]] for f in stack]
        phase = "<kernel-body>"
        for i, name in enumerate(names):
            if name in _BUILDER_FUNCS and i + 1 < len(names):
                phase = names[i + 1]
                break
        by_phase[phase] += stall
        by_phase_type[(phase, st)] += stall

        # Innermost frame plus its call line: the precise emission point.
        inner = stack[-1]
        fn = funcs[inner[0]]
        fl = os.path.basename(files[inner[1]]) if inner[1] >= 0 else "?"
        by_site[f"{fn} <- {fl}:{inner[2]}  [{'/'.join(names[-3:])}]"] += stall

    def pct(v: int) -> float:
        return round(100.0 * v / total, 2) if total else 0.0

    report = {
        "dispatch": str(d.resolve()),
        "total_stall_cycles": total,
        "unresolved_stall_cycles": unresolved,
        "stall_by_type": [
            {"stall_type": k, "cycles": v, "pct": pct(v)}
            for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])
        ],
        "stall_by_phase": [
            {"phase": k, "cycles": v, "pct": pct(v)}
            for k, v in sorted(by_phase.items(), key=lambda kv: -kv[1])
        ],
        "stall_by_phase_and_type": [
            {"phase": k[0], "stall_type": k[1], "cycles": v, "pct": pct(v)}
            for k, v in sorted(by_phase_type.items(), key=lambda kv: -kv[1])[
                : args.topk
            ]
        ],
        "stall_by_emission_site": [
            {"site": k, "cycles": v, "pct": pct(v)}
            for k, v in sorted(by_site.items(), key=lambda kv: -kv[1])[: args.topk]
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))

    print(f"total_stall={total:,}  unresolved={unresolved:,}")
    print("\n--- stall by type ---")
    for e in report["stall_by_type"]:
        print(f"  {e['pct']:6.2f}%  {e['cycles']:>12,}  {e['stall_type']}")
    print("\n--- stall by kernel phase ---")
    for e in report["stall_by_phase"]:
        print(f"  {e['pct']:6.2f}%  {e['cycles']:>12,}  {e['phase']}")
    print("\n--- stall by phase x type ---")
    for e in report["stall_by_phase_and_type"]:
        print(
            f"  {e['pct']:6.2f}%  {e['cycles']:>12,}  {e['phase']:<16} {e['stall_type']}"
        )
    print("\n--- stall by emission site (innermost frame) ---")
    for e in report["stall_by_emission_site"]:
        print(f"  {e['pct']:6.2f}%  {e['cycles']:>12,}  {e['site']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
