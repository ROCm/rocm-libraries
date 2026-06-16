#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: per-basic-block VGPR-usage analysis of the synthetic fused kernel.
# Splits the kernel .s into basic blocks by label and reports each block's
# distinct VGPR set, to see whether the four sel-paths have separable, distinct
# register footprints over a fixed shared (accumulator) set.
import re, sys
from pathlib import Path

s = Path(sys.argv[1]).read_text().splitlines()

# Isolate the kernel body: from the global label `fused:` to the function end.
start = next(i for i, l in enumerate(s) if re.match(r'^fused_conv:', l))
end = next(i for i, l in enumerate(s) if i > start and '.Lfunc_end' in l)
body = s[start:end]

vrange = re.compile(r'v\[(\d+):(\d+)\]')
vsingle = re.compile(r'(?<![a-zA-Z0-9_])v(\d+)(?![0-9:])')

def regs_in(line):
    out = set()
    # strip comments
    line = line.split('//')[0].split(';')[0]
    for m in vrange.finditer(line):
        a, b = int(m.group(1)), int(m.group(2))
        out.update(range(a, b + 1))
    # remove ranges before single scan to avoid double count is unnecessary (set)
    for m in vsingle.finditer(line):
        out.add(int(m.group(1)))
    return out

# Split into blocks by label lines `.LBB[0-9]+_N:` (and the entry).
blocks = []
cur_label, cur_lines = 'entry', []
for l in body:
    m = re.match(r'^(\.LBB[0-9]+_\d+):', l)
    if m:
        blocks.append((cur_label, cur_lines))
        cur_label, cur_lines = m.group(1), []
    else:
        cur_lines.append(l)
blocks.append((cur_label, cur_lines))

print(f"{'block':<12} {'instrs':>6} {'#vgpr':>6} {'min':>4} {'max':>4}  vgpr-set (compact)")
for label, lines in blocks:
    regs = set()
    ninstr = 0
    for l in lines:
        ls = l.strip()
        if not ls or ls.startswith('.') or ls.startswith('//'):
            continue
        ninstr += 1
        regs |= regs_in(l)
    if not regs:
        print(f"{label:<12} {ninstr:>6} {0:>6}")
        continue
    rs = sorted(regs)
    # compact ranges
    parts, a = [], rs[0]
    prev = rs[0]
    for r in rs[1:]:
        if r == prev + 1:
            prev = r
        else:
            parts.append(f"{a}-{prev}" if a != prev else f"{a}")
            a = prev = r
    parts.append(f"{a}-{prev}" if a != prev else f"{a}")
    print(f"{label:<12} {ninstr:>6} {len(regs):>6} {min(regs):>4} {max(regs):>4}  {','.join(parts)}")
