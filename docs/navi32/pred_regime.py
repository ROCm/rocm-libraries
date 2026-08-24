#!/usr/bin/env python3
"""Origami Prediction vs GridBased over an IDENTICAL 298-solution pool,
in both execution regimes, restricted to the same shapes."""
import csv, collections, math

def load(path, keep=None):
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok": continue
        if keep and r["shape_id"] not in keep: continue
        # 1 degenerate shape returns 0 GFlops on EVERY arm (2 rows/arm/sweep,
        # perfectly symmetric) -- drop it rather than divide by it.
        if float(r["gflops"]) <= 0 or float(r["us"]) <= 0: continue
        per[r["shape_id"]].setdefault(r["arm"], []).append((float(r["us"]), float(r["gflops"])))
    return per

def contrast(per, base, cand):
    rr, tb, tc = [], 0.0, 0.0
    for sid, d in per.items():
        if base not in d or cand not in d: continue
        rr.append(max(x[1] for x in d[cand]) / max(x[1] for x in d[base]))
        tb += min(x[0] for x in d[base]); tc += min(x[0] for x in d[cand])
    if not rr: return 0, float("nan"), float("nan")
    return len(rr), 100*math.exp(sum(map(math.log, rr))/len(rr)), 100*tb/tc

keep = {r["shape_id"] for r in csv.DictReader(open("results/P14_pred_masked60.csv"))}
a = load("results/P6_main.csv", keep)
b = load("results/P14_pred_masked60.csv")

print(f"{'pred298 vs gridcat (same 298 solutions)':<42}{'n':>5}{'geomean':>10}{'wall':>9}")
print("-" * 66)
for lbl, per in (("96-CU execution  (original rejection)", a),
                 ("60-CU execution  (matched, new)", b)):
    n, g, w = contrast(per, "gridcat", "pred298")
    print(f"{lbl:<42}{n:>5}{g:>9.2f}%{w:>8.2f}%")
print()
for lbl, per, aa in (("  A/A floor, 96-CU", a, "navi32ship_aa"),
                     ("  A/A floor, 60-CU", b, "gridcat_aa")):
    base = "navi32ship" if aa == "navi32ship_aa" else "gridcat"
    n, g, w = contrast(per, base, aa)
    print(f"{lbl:<42}{n:>5}{g:>9.2f}%{w:>8.2f}%")
