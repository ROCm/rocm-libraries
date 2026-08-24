#!/usr/bin/env python3
"""All four shipped catalogs: 96-CU vs genuine 60-CU execution, SAME shapes.

The shipped headline (+18.8..+23.9% wall-clock) was measured at 96-CU execution
with 60-CU selection. This re-measures each at real 60 CUs and compares on the
identical shape subset.

Note the 96-CU sweeps ran --reps 1 and these --reps 2; the contrast takes
best-of-reps per shape, so the comparison holds, but it is not rep-matched.
"""
import csv, collections, math, os

H = os.path.expanduser("~/navi32/results")
PTS = [  # label, 96-CU csv+arms,               60-CU csv+arms
    ("HHS",  "P6_main.csv",  "navi32ship", "gridcat",   "P12_masked60.csv",  "ship", "wide"),
    ("BBS",  "P9_bbs.csv",   "bbs_ship",   "bbs_wide",  "P15_bbs_m60.csv",   "ship", "wide"),
    ("AuxH", "P10_aux.csv",  "aux_ship",   "aux_wide",  "P16_auxh_m60.csv",  "ship", "wide"),
    ("AuxB", "P11_auxb.csv", "auxb_ship",  "auxb_wide", "P17_auxb_m60.csv",  "ship", "wide"),
]

def load(path, keep=None):
    per = collections.defaultdict(dict)
    if not os.path.exists(path): return per
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok": continue
        if keep and r["shape_id"] not in keep: continue
        try: us, g = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or g <= 0: continue      # 1 degenerate shape, symmetric across arms
        per[r["shape_id"]].setdefault(r["arm"], []).append((us, g))
    return per

def contrast(per, base, cand):
    rr, tb, tc = [], 0.0, 0.0
    for sid, d in per.items():
        if base not in d or cand not in d: continue
        rr.append(max(x[1] for x in d[cand]) / max(x[1] for x in d[base]))
        tb += min(x[0] for x in d[base]); tc += min(x[0] for x in d[cand])
    if not rr: return 0, float("nan"), float("nan")
    return len(rr), 100*math.exp(sum(map(math.log, rr))/len(rr)), 100*tb/tc

print(f"{'PT':<6}{'n':>5}{'96-CU geo':>11}{'60-CU geo':>11}{'96-CU wall':>12}{'60-CU wall':>12}{'wall d':>9}")
print("-" * 66)
for lab, c96, b96, w96, c60, b60, w60 in PTS:
    p60 = load(f"{H}/{c60}")
    if not p60: print(f"{lab:<6}{'--- not yet measured ---':>40}"); continue
    keep = set(p60)
    p96 = load(f"{H}/{c96}", keep)
    n1, g1, w1 = contrast(p96, b96, w96)
    n2, g2, w2 = contrast(p60, b60, w60)
    n = min(n1, n2)
    print(f"{lab:<6}{n:>5}{g1:>10.1f}%{g2:>10.1f}%{w1:>11.1f}%{w2:>11.1f}%{w2-w1:>+8.1f}")
