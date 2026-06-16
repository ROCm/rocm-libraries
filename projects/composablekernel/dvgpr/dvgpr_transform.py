#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: dynamic-VGPR transform of a single kernel in a gfx1250 device .s.
# For the named kernel only:
#   - insert s_alloc_vgpr(peak) at each sel-path entry (or once at entry for a solo),
#   - lower the creation-time .amdhsa_next_free_vgpr to a small block size,
#   - drop MSG_DEALLOC_VGPRS (illegal in dynamic-VGPR mode).
# RSRC3 bit 17 (ENABLE_DYNAMIC_VGPR) is set afterward by patch_dvgpr.py on the ELF.
# Static transform only; functional/occupancy correctness needs MI450.
#
# Usage: dvgpr_transform.py IN.s OUT.s KERNEL
import re, sys
from pathlib import Path

BLOCK_SIZE = 32  # creation-time allocation; the scalar prologue / entry fits

# Per-kernel insert spec. kind: 'entry' (after kernel label), 'label' (after a .LBB
# label), 'after' (after first line matching regex). alloc = VGPRs to s_alloc.
# Peaks are the solo VGPR counts (== each path's in-fusion max, per attribution).
CONFIG = {
    'solo1':      [('entry', None, 259)],
    'solo2':      [('entry', None, 200)],
    'solo4':      [('entry', None, 216)],
    'solo8':      [('entry', None, 174)],
    'fused_conv': [('label', r'\.LBB1_11',  259),   # sel==0 VS1
                   ('label', r'\.LBB1_235', 200),   # sel==1 VS2
                   ('label', r'\.LBB1_234', 174),   # sel==3 VS8
                   ('after', r's_cbranch_scc0 \.LBB1_234', 216)],  # sel==2 VS4 (fallthrough)
}

inp, outp, kernel = sys.argv[1], sys.argv[2], sys.argv[3]
spec = CONFIG[kernel]
s = Path(inp).read_text().splitlines()

start = next(i for i, l in enumerate(s) if re.match(rf'^{kernel}:', l))
kd_start = next(i for i, l in enumerate(s) if i > start and re.match(rf'\s*\.amdhsa_kernel {kernel}\b', l))
kd_end = next(i for i, l in enumerate(s) if i > kd_start and '.end_amdhsa_kernel' in l)
end = kd_start - 1

label_inserts = {a: alloc for (k, a, alloc) in spec if k == 'label'}
after_inserts = [(a, alloc) for (k, a, alloc) in spec if k == 'after']
entry_inserts = [alloc for (k, a, alloc) in spec if k == 'entry']

out, stats = [], dict(label=0, after=0, entry=0, dealloc=0, nfv=0)
for i, l in enumerate(s):
    if start <= i <= end:
        if 'MSG_DEALLOC_VGPRS' in l:
            stats['dealloc'] += 1
            continue
        out.append(l)
        if i == start:  # entry insert right after the kernel label
            for alloc in entry_inserts:
                out.append(f"\ts_alloc_vgpr {alloc}")
                stats['entry'] += 1
        m = re.match(r'^(\.LBB\d+_\d+):', l)
        if m:
            for lab, alloc in label_inserts.items():
                if re.fullmatch(lab, m.group(1)):
                    out.append(f"\ts_alloc_vgpr {alloc}")
                    stats['label'] += 1
        for rgx, alloc in list(after_inserts):
            if re.search(rgx, l):
                out.append(f"\ts_alloc_vgpr {alloc}")
                stats['after'] += 1
                after_inserts = [(r, a) for (r, a) in after_inserts if r != rgx]
                break
        continue
    if kd_start <= i <= kd_end and 'amdhsa_next_free_vgpr' in l:
        indent = l[:len(l) - len(l.lstrip())]
        out.append(f"{indent}.amdhsa_next_free_vgpr {BLOCK_SIZE}")
        stats['nfv'] += 1
        continue
    out.append(l)

Path(outp).write_text('\n'.join(out) + '\n')
print(f"{kernel}: inserts label={stats['label']} after={stats['after']} entry={stats['entry']} "
      f"dealloc_removed={stats['dealloc']} next_free_vgpr->{BLOCK_SIZE}={stats['nfv']}")
