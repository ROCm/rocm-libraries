#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# AICK-1303: static dynamic-VGPR comparison. Reads the vanilla and patched device .s
# (from build_variants.sh: out/vanilla.s, out/patched.s) and reports, per kernel and
# per sel-path, the creation-time VGPR allocation, the steady-state allocation
# (s_alloc_vgpr peak in the patched build), and the estimated waves/SIMD. Shows the
# occupancy plain fusion loses and dynamic VGPR recovers. Static estimate; achieved
# occupancy + latency need MI450 (see run_profile.sh).
#
# Usage: extract_metrics.py out/vanilla.s out/patched.s
import re, sys
from pathlib import Path

VGPR_BUDGET = 1536  # per-SIMD VGPRs, gfx1250 wave32 (waves/SIMD = budget // vgpr, capped 16)
MAX_WAVES = 16
KERNELS = ['solo1', 'solo2', 'solo4', 'solo8', 'fused_conv']

def waves(vgpr):
    return min(MAX_WAVES, VGPR_BUDGET // max(vgpr, 1))

def kernel_span(lines, k):
    start = next(i for i, l in enumerate(lines) if re.match(rf'^{k}:', l))
    end = next(i for i, l in enumerate(lines) if i > start and '.Lfunc_end' in l)
    kd = next(i for i, l in enumerate(lines) if i > start and re.match(rf'\s*\.amdhsa_kernel {k}\b', l))
    kdend = next(i for i, l in enumerate(lines) if i > kd and '.end_amdhsa_kernel' in l)
    return start, end, kd, kdend

def creation_vgpr(lines, kd, kdend):
    for i in range(kd, kdend):
        m = re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)', lines[i])
        if m:
            return int(m.group(1))
    return None

def alloc_peaks(lines, start, end):
    return [int(m.group(1)) for i in range(start, end)
            for m in [re.search(r's_alloc_vgpr\s+(\d+)', lines[i])] if m]

van = Path(sys.argv[1]).read_text().splitlines()
pat = Path(sys.argv[2]).read_text().splitlines()

print(f"VGPR budget/SIMD = {VGPR_BUDGET} (gfx1250 wave32), waves/SIMD = budget // vgpr (max {MAX_WAVES})\n")
hdr = f"{'kernel':<11} {'variant':<8} {'creation':>8} {'steady':>8} {'waves':>6}  note"
print(hdr); print('-' * len(hdr))

for k in KERNELS:
    vs, ve, vkd, vkde = kernel_span(van, k)
    vcre = creation_vgpr(van, vkd, vkde)
    print(f"{k:<11} {'vanilla':<8} {vcre:>8} {vcre:>8} {waves(vcre):>6}  static allocation")

    ps, pe, pkd, pkde = kernel_span(pat, k)
    pcre = creation_vgpr(pat, pkd, pkde)
    peaks = alloc_peaks(pat, ps, pe)
    if len(peaks) <= 1:
        peak = peaks[0] if peaks else pcre
        print(f"{k:<11} {'patched':<8} {pcre:>8} {peak:>8} {waves(peak):>6}  dynamic (1 path)")
    else:
        vs_of = {259: 'VS1', 200: 'VS2', 216: 'VS4', 174: 'VS8'}  # peak -> VectorSize
        for peak in sorted(peaks):
            lab = vs_of.get(peak, '?')
            print(f"{k:<11} {'patched':<8} {pcre:>8} {peak:>8} {waves(peak):>6}  dynamic path {lab}")
print()
print("Read: fused_conv vanilla pins every path to the VS1 budget (lowest waves); fused_conv")
print("patched lets each path run at its own waves/SIMD - the recovered occupancy. Solo patched")
print("vs solo vanilla isolates dynamic-VGPR overhead. Confirm achieved values on MI450.")
