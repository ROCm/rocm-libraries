#!/usr/bin/env python3
"""Oracle over every library measured at genuine 60-CU execution.

The campaign rejected extending the catalog on the grounds that an oracle over
all arms built was only +2.8% above what shipped -- measured at 96-CU execution.
Recomputed here at real 60 CUs, free, from sweeps already taken.

CRITICAL CONTROL: an oracle taken ACROSS SEPARATE SWEEPS picks the luckiest
measurement per shape, so it is inflated by run-to-run noise even when every arm
is the SAME library. The A/A arms measure exactly that inflation -- subtract it
before reading the real oracle.
"""
import csv, collections, math, os
H = os.path.expanduser("~/navi32/results")
SRC = {  # csv -> {arm_in_csv: canonical library name}
    "P12_masked60.csv":      {"ship": "navi32ship", "wide": "wgm8_A",  "ship_aa": "navi32ship_aa"},
    "P13_wgm_masked60.csv":  {"wgm8": "wgm8_B", "wgm6": "wgm6", "wgm10": "wgm10", "wgm8_aa": "wgm8_C"},
    "P14_pred_masked60.csv": {"gridcat": "wgm8_D", "pred298": "pred298", "gridcat_aa": "wgm8_E"},
}
per = collections.defaultdict(dict)
for f, amap in SRC.items():
    for r in csv.DictReader(open(f"{H}/{f}")):
        if r["status"] != "ok" or r["arm"] not in amap: continue
        try: us, g = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or g <= 0: continue
        k = amap[r["arm"]]
        d = per[r["shape_id"]]
        d[k] = (min(d[k][0], us), max(d[k][1], g)) if k in d else (us, g)

def oracle(names, base, shapes):
    tb = to = 0.0; n = 0; wins = collections.Counter()
    for sid in shapes:
        d = per[sid]
        if base not in d or not all(x in d for x in names): continue
        best = min(names, key=lambda k: d[k][0])
        tb += d[base][0]; to += d[best][0]; n += 1; wins[best] += 1
    return n, 100*tb/to, wins

shapes = [s for s, d in per.items() if len(d) >= 8]
print(f"shapes with all libraries measured: {len(shapes)}\n")
WGM8 = ["wgm8_A", "wgm8_B", "wgm8_C", "wgm8_D", "wgm8_E"]
rows = [
  ("NOISE FLOOR: oracle over 5x the SAME library (wgm8)", WGM8),
  ("oracle over the 3 WGM variants", ["wgm8_B", "wgm6", "wgm10"]),
  ("oracle over wgm8 + Prediction", ["wgm8_D", "pred298"]),
  ("oracle over EVERYTHING built", WGM8 + ["wgm6", "wgm10", "pred298", "navi32ship"]),
]
print(f"{'oracle set':<52}{'n':>5}{'vs wgm8':>10}")
print("-" * 68)
for lab, names in rows:
    n, w, wins = oracle(names, "wgm8_A", shapes)
    print(f"{lab:<52}{n:>5}{w:>9.2f}%")
n, wfloor, _ = oracle(WGM8, "wgm8_A", shapes)
n, wall, wins = oracle(WGM8 + ["wgm6", "wgm10", "pred298", "navi32ship"], "wgm8_A", shapes)
print(f"\nreal oracle headroom above the noise floor: {wall - wfloor:+.2f} pt")
print("\nwhich library the oracle picks (n shapes):")
for k, v in wins.most_common(): print(f"  {k:<16}{v:>5}")
