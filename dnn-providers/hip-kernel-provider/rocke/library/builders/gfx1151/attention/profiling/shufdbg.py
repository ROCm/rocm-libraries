#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Anatomy of the K-loop's lane-broadcast stream (permlanex16 + cndmask).

dual_gather_finish pays 1 permlanex16 + 2 v_cndmask per dword to replicate each V
subtile into both lane halves, ~16% of the loop's issue cycles. VOPD could retire
two selects per cycle -- v_dual_cndmask_b32 :: v_dual_cndmask_b32 assembles fine
on gfx1151 -- so this reports the two things that decide whether pairing is
reachable:

  1. Do the selects share one mask? v_cndmask_b32_e32 reads VCC implicitly, so a
     VOPD pair must agree on it. Histogram the mask operand to find out.
  2. Are they ADJACENT and independent? The pass pairs neighbours, and emitting
     permlane/select/select per dword interleaves each select between the
     permlane it depends on. Measure the run lengths of consecutive selects.

Usage: shufdbg.py [--block-n N] [--qkdo 0|1]
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import tempfile

from rocke.helpers import compile_kernel

from kernels.gfx1151.wmma_fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk

LINE = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-Fa-f]+):")


def kloop(insts):
    """Largest backward-branch body containing WMMAs = the K-loop."""
    base = insts[0][0]
    idx = {rec[0]: i for i, rec in enumerate(insts)}
    best = None
    for i, (addr, op, _ops, ln) in enumerate(insts):
        if not op.startswith(("s_cbranch", "s_branch")):
            continue
        m = re.search(r"\+0x([0-9a-fA-F]+)>", ln)
        if not m:
            continue
        tgt = base + int(m.group(1), 16)
        if tgt >= addr or tgt not in idx:
            continue
        j = idx[tgt]
        nw = sum(1 for r in insts[j : i + 1] if r[1].startswith("v_wmma"))
        if nw and (best is None or nw > best[0]):
            best = (nw, j, i)
    return insts[best[1] : best[2] + 1] if best else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--qkdo", type=int, default=1)
    args = ap.parse_args()

    cfg = SwapQKCfg(
        head_size=128,
        num_query_heads=24,
        num_kv_heads=0,
        mask_mode="none",
        n_waves=2,
        q_block=1,
        o_f16=False,
        block_n=args.block_n,
        qk_ilp=2,
        sched_mode="pingpong",
        buffer_gather=True,
        dual_gather=True,
        fast_exp2=True,
        v_transposed=True,
        qk_douter=bool(args.qkdo),
    )
    art = compile_kernel(build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151")
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(art.hsaco)
        path = f.name
    txt = subprocess.run(
        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx1151", path],
        capture_output=True,
        text=True,
    ).stdout

    insts = []
    for ln in txt.splitlines():
        m = LINE.match(ln)
        if m:
            insts.append((int(m.group(3), 16), m.group(1), m.group(2).strip(), ln))
    body = kloop(insts)
    if not body:
        print("K-loop not found")
        return 1

    sel = [
        (i, ops)
        for i, (_a, op, ops, _l) in enumerate(body)
        if op.startswith("v_cndmask")
    ]
    shuf = [i for i, (_a, op, _o, _l) in enumerate(body) if op.startswith("v_permlane")]
    dual = collections.Counter(op for _a, op, _o, _l in body if op.startswith("v_dual"))

    print(
        f"K-loop: {len(body)} instructions   "
        f"selects={len(sel)}  permlane={len(shuf)}  "
        f"v_dual total={sum(dual.values())}"
    )
    print(f"  v_dual_cndmask_b32 already formed: {dual['v_dual_cndmask_b32']}")

    masks = collections.Counter(ops.split(",")[-1].strip() for _i, ops in sel)
    print(f"\n  select mask operands ({len(masks)} distinct):")
    for m, n in masks.most_common(6):
        print(f"    x{n:<4} {m}")

    # Run lengths of consecutive selects: VOPD pairs neighbours, so a stream of
    # isolated selects (run length 1) can never pair no matter the bank layout.
    runs, cur, prev = [], 0, -2
    for i, _ops in sel:
        if i == prev + 1:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 1
        prev = i
    if cur:
        runs.append(cur)
    hist = collections.Counter(runs)
    print(f"\n  consecutive-select run lengths ({len(runs)} runs):")
    for length in sorted(hist):
        print(
            f"    len {length:<3} x{hist[length]}   "
            f"({length * hist[length]} selects, "
            f"{(length // 2) * hist[length]} pairable)"
        )
    pairable = sum((length // 2) * n for length, n in hist.items())
    print(
        f"\n  adjacency-pairable selects: {pairable} of {len(sel)} "
        f"-> best case saves {pairable} issue slots/iter"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
