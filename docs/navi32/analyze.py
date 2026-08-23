#!/usr/bin/env python3
"""
Analyse a navi32 arm sweep.

Reports, for every arm against the baseline:
  * per-shape geomean AND flops-weighted wall-clock -- on this workload these have
    disagreed IN SIGN, so quoting one alone is not safe;
  * a jackknife over the largest time consumers, because a handful of shapes can own most
    of the wall-clock and flip a verdict;
  * the A/A arm, which is the only thing that measures the in-session noise floor
    (bootstrap CIs resample shapes and cannot see it);
  * per-stratum breakdown by size and geometry.

Runs on a partial CSV so a live sweep can be inspected.
"""

import collections
import csv
import statistics
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/navi32/results/P3_main.csv"
BASE = sys.argv[2] if len(sys.argv) > 2 else "navi32ship"


def load(path):
    sh = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        try:
            g = float(r["gflops"])
        except ValueError:
            continue
        if g <= 0:
            continue
        sh[r["shape_id"]][r["arm"]] = g
        meta[r["shape_id"]] = (int(r["M"]), int(r["N"]), int(r["K"]), r["stratum"])
    return sh, meta


def flops(m):
    return 2.0 * m[0] * m[1] * m[2]


def contrast(sh, meta, keys, arm, base):
    rr, num, ta, tb = [], 0.0, 0.0, 0.0
    for k in keys:
        ga, gb = sh[k][arm], sh[k][base]
        rr.append(ga / gb)
        F = flops(meta[k])
        num += F
        ta += F / ga
        tb += F / gb
    return statistics.geometric_mean(rr) * 100, (num / ta) / (num / tb) * 100


def main():
    sh, meta = load(CSV)
    arms = sorted({a for v in sh.values() for a in v})
    if BASE not in arms:
        sys.exit(f"baseline {BASE} not in {arms}")
    keys = sorted(k for k, v in sh.items() if all(a in v for a in arms))
    print(f"shapes complete on all {len(arms)} arms: {len(keys)}   arms: {arms}\n")
    if len(keys) < 15:
        print("too few to read yet")
        return

    aa = next((a for a in arms if a.endswith("_aa")), None)
    if aa:
        g, t = contrast(sh, meta, keys, aa, BASE)
        print(f"A/A NOISE FLOOR ({aa} vs {BASE}, same library): "
              f"geomean {g:.2f}%  wall-clock {t:.2f}%")
        print("  a perfect A/A is 100.00%; the gap is the floor an arm must clear\n")

    print(f"{'arm':<16} {'geomean':>9} {'wall-clock':>11}   (vs {BASE})")
    print("-" * 52)
    for a in arms:
        if a == BASE:
            continue
        g, t = contrast(sh, meta, keys, a, BASE)
        mark = "  <- A/A control" if a == aa else ""
        print(f"{a:<16} {g:8.2f}% {t:10.2f}%{mark}")

    tot = sum(flops(meta[k]) / sh[k][BASE] for k in keys)
    print("\nCONCENTRATION (wall-clock is only as trustworthy as its biggest shapes)")
    by_t = sorted(keys, key=lambda k: flops(meta[k]) / sh[k][BASE], reverse=True)
    for n in (5, 10, 25, 50):
        share = sum(flops(meta[k]) / sh[k][BASE] for k in by_t[:n]) / tot * 100
        print(f"  top {n:>3} shapes hold {share:5.1f}% of total kernel time")

    print("\nJACKKNIFE -- drop the largest time consumers, wall-clock")
    hdr = "  " + f"{'dropped':>8}" + "".join(f"{a:>14}" for a in arms if a != BASE)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for d in (0, 5, 10, 25, 50):
        ks = by_t[d:]
        if len(ks) < 20:
            continue
        row = f"  {d:>8}"
        for a in arms:
            if a == BASE:
                continue
            _, t = contrast(sh, meta, ks, a, BASE)
            row += f"{t:13.2f}%"
        print(row)

    for level, name in ((1, "size"), (2, "geometry")):
        print(f"\nBY {name.upper()} (wall-clock)")
        groups = collections.defaultdict(list)
        for k in keys:
            parts = meta[k][3].split(":")
            groups[parts[level] if len(parts) > level else "?"].append(k)
        hdr = f"  {'':<10}{'n':>5}" + "".join(f"{a:>14}" for a in arms if a != BASE)
        print(hdr)
        for gname, ks in sorted(groups.items()):
            if len(ks) < 10:
                continue
            row = f"  {gname:<10}{len(ks):>5}"
            for a in arms:
                if a == BASE:
                    continue
                _, t = contrast(sh, meta, ks, a, BASE)
                row += f"{t:13.2f}%"
            print(row)


if __name__ == "__main__":
    main()
