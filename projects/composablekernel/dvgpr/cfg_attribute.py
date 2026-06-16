#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: CFG attribution of the fused conv kernel. Splits fused_conv into basic
# blocks, builds successors (branch target + fallthrough), computes blocks reachable
# from each sel-path entry, and reports per-path max VGPR + the shared prologue's VGPR
# footprint. Confirms each path occupies [0, path_peak) so per-path s_alloc_vgpr is
# correct with a small creation-time allocation.
import re, sys
from collections import deque
from pathlib import Path

s = Path(sys.argv[1]).read_text().splitlines()
start = next(i for i, l in enumerate(s) if re.match(r'^fused_conv:', l))
end = next(i for i, l in enumerate(s) if i > start and '.Lfunc_end' in l)
body = s[start + 1:end]

vrange = re.compile(r'v\[(\d+):(\d+)\]')
vsingle = re.compile(r'(?<![a-zA-Z0-9_])v(\d+)(?![0-9:])')
def regs_in(line):
    out = set()
    line = line.split('//')[0].split(';')[0]
    for m in vrange.finditer(line):
        out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in vsingle.finditer(line):
        out.add(int(m.group(1)))
    return out

# Split into ordered blocks. Entry block = 'ENTRY'; others are .LBB1_N labels.
labels, order = {}, []
cur, lines = 'ENTRY', []
def flush():
    labels[cur] = lines
    order.append(cur)
for l in body:
    m = re.match(r'^(\.LBB\d+_\d+):', l)
    if m:
        flush(); cur, lines = m.group(1), []
    else:
        lines.append(l)
flush()

# Successors: branch target (+ fallthrough unless unconditional/terminating).
succ = {}
for idx, name in enumerate(order):
    blk = labels[name]
    nexts = set()
    terminates = False
    uncond = False
    for l in blk:
        ls = l.strip()
        mb = re.match(r'^s_branch\s+(\.LBB\d+_\d+)', ls)
        mc = re.match(r'^s_cbranch_\w+\s+(\.LBB\d+_\d+)', ls)
        if mb:
            nexts.add(mb.group(1)); uncond = True
        elif mc:
            nexts.add(mc.group(1))
        elif ls.startswith('s_endpgm') or ls.startswith('s_setpc'):
            terminates = True
    if not uncond and not terminates and idx + 1 < len(order):
        nexts.add(order[idx + 1])
    succ[name] = nexts

def reach(entry):
    seen, q = set(), deque([entry])
    while q:
        n = q.popleft()
        if n in seen or n not in labels:
            continue
        seen.add(n)
        q.extend(succ.get(n, ()))
    return seen

def maxvgpr(blocks):
    mx, total = -1, set()
    for b in blocks:
        for l in labels[b]:
            r = regs_in(l)
            if r:
                mx = max(mx, max(r)); total |= r
    return mx + 1, len(total)  # peak register index+1, distinct count

paths = {'VS1': '.LBB1_11', 'VS2': '.LBB1_235', 'VS4': '.LBB1_3', 'VS8': '.LBB1_234'}
reach_sets = {k: reach(v) for k, v in paths.items()}

# Prologue = blocks NOT in any path's reachable set.
all_path_blocks = set().union(*reach_sets.values())
prologue = [b for b in order if b not in all_path_blocks]
pk, pc = maxvgpr(prologue)
print(f"prologue blocks: {len(prologue)}  max_vgpr(index+1)={pk}  distinct={pc}")
print(f"solo reference: VS1=259 VS2=200 VS4=216 VS8=174\n")

# A common exit block (e.g. shared s_endpgm) may appear in all paths; flag overlaps.
for k in paths:
    others = set().union(*(reach_sets[o] for o in paths if o != k))
    excl = reach_sets[k] - others
    shared = reach_sets[k] & others
    pk2, pc2 = maxvgpr(excl)
    print(f"{k}: blocks={len(reach_sets[k])} exclusive={len(excl)} shared_with_others={len(shared)}  "
          f"max_vgpr(excl)={pk2} distinct={pc2}")
