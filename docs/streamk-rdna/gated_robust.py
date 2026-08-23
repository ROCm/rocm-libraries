#!/usr/bin/env python3
"""
Robustness check on the sign reversal found by gated_policy.py.

Claim under test: turning StreamK off (mode 4) scores 96.95% by per-shape geomean but
~102% by flops-weighted wall-clock. A sign reversal between metrics is exactly the kind
of result that turns out to be three enormous shapes, so this checks:

  1. per-band throughput -- must agree with the campaign's published per-band geomeans
  2. concentration       -- how much of total time sits in the top-k shapes
  3. jackknife           -- drop the top-k largest and see if the reversal survives
  4. paired sign test    -- share of shapes where OFF is faster, per band

No GPU.
"""

import csv
import statistics
from collections import defaultdict

CSV = "/home/vmijovic/hhs_tn_grid_vs_resource_origami_9k/measurements/campaign/p1_modes.csv"
BASE, OFF = "m6_default", "m4_dataparallel"
BANDS = [("<0.1ms", 0.0, 0.1), ("0.1-1ms", 0.1, 1.0), ("1-5ms", 1.0, 5.0), (">=5ms", 5.0, 1e9)]


def load():
    sh = defaultdict(lambda: defaultdict(list))
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok" or r["arm"] not in (BASE, OFF):
                continue
            try:
                g = float(r["gflops"])
            except ValueError:
                continue
            if g <= 0:
                continue
            s = sh[r["shape_id"]]
            s["mnk"] = (int(r["M"]), int(r["N"]), int(r["K"]))
            s["band"] = float(r["ms"])
            s[r["arm"]].append(g)
    return {k: v for k, v in sh.items() if v.get(BASE) and v.get(OFF)}


def med(x):
    return statistics.median(x)


def flops(m):
    return 2.0 * m[0] * m[1] * m[2]


def tput(shapes, keys):
    """flops-weighted total throughput ratio, OFF vs BASE."""
    n = sum(flops(shapes[k]["mnk"]) for k in keys)
    tb = sum(flops(shapes[k]["mnk"]) / med(shapes[k][BASE]) for k in keys)
    to = sum(flops(shapes[k]["mnk"]) / med(shapes[k][OFF]) for k in keys)
    return (n / to) / (n / tb)


def geo(shapes, keys):
    return statistics.geometric_mean([med(shapes[k][OFF]) / med(shapes[k][BASE]) for k in keys])


def main():
    sh = load()
    keys = sorted(sh)
    print(f"shapes: {len(keys)}\n")

    # 1. per band -- geomean should reproduce the campaign's published table
    print(f"{'band':<10} {'n':>5} {'geomean':>9} {'tput-wtd':>9} {'OFF wins':>9} {'% of time':>10}")
    print("-" * 60)
    tot_t = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in keys)
    for name, lo, hi in BANDS:
        ks = [k for k in keys if lo <= sh[k]["band"] < hi]
        if not ks:
            continue
        wins = sum(1 for k in ks if med(sh[k][OFF]) > med(sh[k][BASE])) / len(ks)
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ks) / tot_t
        print(f"{name:<10} {len(ks):5} {geo(sh,ks)*100:8.2f}% {tput(sh,ks)*100:8.2f}% "
              f"{wins*100:8.1f}% {share*100:9.1f}%")

    # 2. concentration
    print()
    by_time = sorted(keys, key=lambda k: flops(sh[k]["mnk"]) / med(sh[k][BASE]), reverse=True)
    for k_ in (1, 5, 10, 50, 100):
        share = sum(flops(sh[x]["mnk"]) / med(sh[x][BASE]) for x in by_time[:k_]) / tot_t
        print(f"top {k_:>3} shapes hold {share*100:5.1f}% of total kernel time")

    # 3. jackknife -- drop the largest consumers
    print(f"\n{'dropped':>8} {'n':>6} {'geomean':>9} {'tput-wtd':>9}")
    print("-" * 36)
    for d in (0, 1, 5, 10, 50, 100, 200):
        ks = by_time[d:]
        print(f"{d:>8} {len(ks):6} {geo(sh,ks)*100:8.2f}% {tput(sh,ks)*100:8.2f}%")


if __name__ == "__main__":
    main()
