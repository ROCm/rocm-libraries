#!/usr/bin/env python3
"""Can a REALIZABLE size gate capture the GridBased/Prediction oracle headroom?

The 60-CU oracle says pairing GridBased with Origami Prediction is worth +1.47 pt.
An oracle assumes perfect per-shape foreknowledge. A shippable selector would be a
simple predicate on problem size. This measures what such a gate actually captures,
OUT OF SAMPLE -- the threshold is fitted on one half and scored on the other, because
fitting and scoring on the same shapes manufactures a result.
"""
import csv, collections, os, statistics
H = os.path.expanduser("~/navi32/results")

rows = {}
for r in csv.DictReader(open(f"{H}/P14_pred_masked60.csv")):
    if r["status"] != "ok" or r["arm"] not in ("gridcat", "pred298"): continue
    try: us = float(r["us"]); g = float(r["gflops"])
    except ValueError: continue
    if us <= 0 or g <= 0: continue
    d = rows.setdefault(r["shape_id"], {"M": int(r["M"]), "N": int(r["N"]), "K": int(r["K"])})
    d[r["arm"]] = min(d.get(r["arm"], 1e18), us)
sh = [d for d in rows.values() if "gridcat" in d and "pred298" in d]
print(f"shapes: {len(sh)}")

FEATS = {"flops (2MNK)": lambda d: 2*d["M"]*d["N"]*d["K"],
         "output M*N":   lambda d: d["M"]*d["N"],
         "min(M,N)":     lambda d: min(d["M"], d["N"]),
         "K":            lambda d: d["K"]}

def score(shapes, feat, thr):
    """total time using pred298 when feat>=thr else gridcat, vs gridcat everywhere"""
    tb = tg = 0.0
    for d in shapes:
        tb += d["gridcat"]
        tg += d["pred298"] if feat(d) >= thr else d["gridcat"]
    return 100*tb/tg

def oracle(shapes):
    tb = to = 0.0
    for d in shapes:
        tb += d["gridcat"]; to += min(d["gridcat"], d["pred298"])
    return 100*tb/to

print(f"\nin-sample oracle ceiling (perfect per-shape): {oracle(sh):.2f}%\n")
print(f"{'gate feature':<16}{'best thr':>14}{'in-sample':>12}{'OUT-OF-SAMPLE':>15}")
print("-" * 57)
half = len(sh)//2
folds = [(sh[:half], sh[half:]), (sh[half:], sh[:half])]
for name, f in FEATS.items():
    vals = sorted({f(d) for d in sh})
    best_in = max(score(sh, f, t) for t in vals)
    oos = []
    for train, test in folds:
        tvals = sorted({f(d) for d in train})
        bt = max(tvals, key=lambda t: score(train, f, t))
        oos.append(score(test, f, bt))
    thr_all = max(vals, key=lambda t: score(sh, f, t))
    print(f"{name:<16}{thr_all:>14.3g}{best_in:>11.2f}%{statistics.mean(oos):>14.2f}%")
print("\n(100.00% = no better than GridBased alone. Gate always-on = pure pred298.)")
print(f"always-pred298: {score(sh, lambda d: 1, 0):.2f}%")
