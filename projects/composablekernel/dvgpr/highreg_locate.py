#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: locate high-VGPR uses in fused_conv and attribute to the nearest
# preceding basic-block label, and to the sel-path that block belongs to (by the
# program-order band between path entry labels). Registers 217-258 are VS1-only
# territory (VS2/4/8 peak at 200/216/174); 201-216 are VS1/VS4 territory. If the
# high regs sit only in the expected path bands, the paths segregate as required
# for per-path s_alloc_vgpr.
import re, sys
from pathlib import Path

s = Path(sys.argv[1]).read_text().splitlines()
start = next(i for i, l in enumerate(s) if re.match(r'^fused_conv:', l))
end = next(i for i, l in enumerate(s) if i > start and '.Lfunc_end' in l)

vrange = re.compile(r'v\[(\d+):(\d+)\]')
vsingle = re.compile(r'(?<![a-zA-Z0-9_])v(\d+)(?![0-9:])')
def maxreg(line):
    line = line.split('//')[0].split(';')[0]
    m = -1
    for mm in vrange.finditer(line):
        m = max(m, int(mm.group(2)))
    for mm in vsingle.finditer(line):
        m = max(m, int(mm.group(1)))
    return m

# Path entry labels and the sel value they serve.
entries = {'.LBB1_11': 'VS1(sel0)', '.LBB1_235': 'VS2(sel1)', '.LBB1_234': 'VS8(sel3)'}
# VS4 (sel2) is the fallthrough region starting right after the dispatch (bb.3).

cur_label = 'ENTRY'
# Histogram: for register bands, which labels touch them.
band_hi = {}   # >=217  (VS1 only)
band_mid = {}  # 201-216 (VS1 or VS4)
for i in range(start + 1, end):
    l = s[i]
    m = re.match(r'^(\.LBB\d+_\d+):', l)
    if m:
        cur_label = m.group(1)
        continue
    mr = maxreg(l)
    if mr >= 217:
        band_hi[cur_label] = max(band_hi.get(cur_label, 0), mr)
    elif mr >= 201:
        band_mid[cur_label] = max(band_mid.get(cur_label, 0), mr)

def labelnum(lab):
    m = re.search(r'_(\d+)$', lab)
    return int(m.group(1)) if m else -1

print("=== blocks using v>=217 (VS1-exclusive territory) ===")
for lab in sorted(band_hi, key=labelnum):
    print(f"  {lab:<14} max_v={band_hi[lab]}")
print(f"  total distinct blocks touching v>=217: {len(band_hi)}")
print("\n=== blocks using v201..216 (VS1/VS4 territory) ===")
for lab in sorted(band_mid, key=labelnum):
    print(f"  {lab:<14} max_v={band_mid[lab]}")
print(f"  total distinct blocks touching v201..216: {len(band_mid)}")
print("\npath entry labels: VS1=.LBB1_11  VS2=.LBB1_235  VS8=.LBB1_234  VS4=fallthrough after dispatch (low LBB numbers)")
