#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Group the K-loop's global_loads by address operand to expose duplicates.

qk_douter is supposed to cut the Q loads from n_kv_sub*n_dk down to n_dk, but the
in-loop global_load count does not move. Either the IR never emitted fewer loads,
or the backend duplicated them again. Counting how many DISTINCT address operands
the loads use answers that: 32 K + 8 Q distinct means the reduction happened in
the IR and the backend re-expanded it.

Usage: loaddbg.py <qk_douter 0|1> [block_n]
"""

from __future__ import annotations

import collections
import re
import subprocess
import sys
import tempfile

from rocke.helpers import compile_kernel

from kernels.gfx1151.wmma_fmha_swapqk import SwapQKCfg, build_wmma_fmha_swapqk

LINE = re.compile(r"^\t(\S+)(.*?)//\s*([0-9A-Fa-f]+):")


def main() -> int:
    qkdo = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
    bn = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    cfg = SwapQKCfg(
        head_size=128,
        num_query_heads=24,
        num_kv_heads=0,
        mask_mode="none",
        n_waves=2,
        q_block=1,
        o_f16=False,
        block_n=bn,
        qk_ilp=2,
        sched_mode="pingpong",
        buffer_gather=True,
        dual_gather=True,
        fast_exp2=True,
        v_transposed=True,
        qk_douter=qkdo,
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

    base = insts[0][0]
    idx = {a: i for i, rec in enumerate(insts) for a in (rec[0],)}
    best = None
    for i, (a, op, _ops, ln) in enumerate(insts):
        if not op.startswith(("s_cbranch", "s_branch")):
            continue
        m = re.search(r"\+0x([0-9a-fA-F]+)>", ln)
        if not m:
            continue
        tgt = base + int(m.group(1), 16)
        if tgt >= a or tgt not in idx:
            continue
        j = idx[tgt]
        nw = sum(1 for rec in insts[j : i + 1] if rec[1].startswith("v_wmma"))
        if nw and (best is None or nw > best[0]):
            best = (nw, j, i)
    if best is None:
        print("K-loop not found")
        return 1

    _, j, i = best
    body = insts[j : i + 1]
    sig = collections.Counter()
    for _a, op, ops, _ln in body:
        if op.startswith("global_load"):
            parts = [x.strip() for x in ops.split(",")]
            sig[",".join(parts[1:])] += 1  # address operands, sans destination

    print(
        f"qkdo={int(qkdo)} bn={bn}: {sum(sig.values())} global_loads, "
        f"{len(sig)} distinct address operands"
    )
    for s, n in sig.most_common(10):
        print(f"  x{n}  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
