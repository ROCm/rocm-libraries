#!/usr/bin/env python3
"""Occupancy exposure of the navi32 catalogs on navi33 (gfx1102).

navi33 has 1024 VGPR/SIMD vs 1536 on navi31/navi32. But RDNA3 occupancy is
min(VGPR limit, LDS limit, 16 waves/SIMD) -- and for GEMM kernels LDS is usually
what binds. Computing only the VGPR term overstates the exposure enormously.
See occ_lib.py for the model (both terms, Tensile's 256 B LDS granularity).

BOUNDS EXPOSURE, DOES NOT PREDICT PERFORMANCE.
"""
import csv, collections
from pathlib import Path
from occ_lib import parse, occupancy, toks

H = Path.home() / "navi32"
ARMS = [("HHS", "wgm8", "P6_main", "gridcat"), ("BBS", "bbs_wide", "P9_bbs", "bbs_wide"),
        ("AuxH", "aux_wide", "P10_aux", "aux_wide"), ("AuxB", "auxb_wide", "P11_auxb", "auxb_wide")]

print(f"{'PT':<6}{'kern':>6}{'LDS-bound':>11}{'matched':>9}{'% TIME loses':>14}{'% SHAPES loses':>16}")
print("-" * 62)
tot_t = tot_lost = 0.0; agg = collections.Counter(); lim_all = collections.Counter()
for pt, lib, csvf, arm in ARMS:
    co = next((H / f"libs/{lib}/library/gfx1100").glob("TensileLibrary*.co"))
    ks = parse(co)
    kmap = [(toks(k["sym"]), k) for k in ks]
    for k in ks: lim_all[occupancy(k, 1536)[2]] += 1
    ldsb = sum(1 for k in ks if occupancy(k, 1536)[2] == "lds")
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(H / f"results/{csvf}.csv")):
        if r["arm"] != arm or r.get("status") != "ok": continue
        try: per[r["shape_id"]].append((float(r["us"]), r["kernel"]))
        except ValueError: pass
    cache = {}; t_all = t_lost = 0.0; n = nlost = miss = 0
    for sid, reps in per.items():
        us, kn = min(reps, key=lambda x: x[0])
        if kn not in cache:
            hit = [k for t, k in kmap if t <= toks(kn)]
            cache[kn] = hit[0] if len(hit) == 1 else None
        k = cache[kn]
        if k is None: miss += 1; continue
        a, b = occupancy(k, 1536), occupancy(k, 1024)
        t_all += us; n += 1
        if b[1] < a[1]:
            t_lost += us; nlost += 1; agg[(k["v"], k["lds"], a[1], b[1])] += 1
    assert n > 0, f"{pt}: ZERO kernels matched -- join is broken, not a 0% result"
    if miss: print(f"  ({pt}: {miss} unmatched, excluded)")
    print(f"{pt:<6}{len(ks):>6}{ldsb:>11}{n:>9}{100*t_lost/t_all:>13.1f}%{100*nlost/n:>15.1f}%")
    tot_t += t_all; tot_lost += t_lost
print("-" * 62)
print(f"{'ALL':<6}{'':>6}{'':>11}{'':>9}{100*tot_lost/tot_t:>13.1f}%")
print(f"\nWhat limits occupancy on navi31/navi32: {dict(lim_all)}")
print("\nThe kernels that DO lose (VGPR, LDS bytes, waves/CU navi32 -> navi33):")
for (v, lds, a, b), c in sorted(agg.items(), key=lambda x: -x[1])[:10]:
    print(f"  v={v:>4} lds={lds:>6}: {a:>2} -> {b:>2} waves/CU   {c:>5} shapes")
