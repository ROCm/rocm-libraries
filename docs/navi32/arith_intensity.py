#!/usr/bin/env python3
"""
Bound the memory-bandwidth fidelity gap WITHOUT clock control.

THE PROBLEM. Every number in this campaign was measured on a 7900 XTX (~960 GB/s,
96 MB Infinity Cache) while the target, navi32, has ~624 GB/s and 48-64 MB. Memory-bound
shapes therefore look better here than they would on real navi32. The plan's intended fix
was to downclock memory and re-measure, but this system has no clock control
(`get_od_volt, Not supported`; `pp_dpm_mclk` absent), so that probe is impossible.

THE SUBSTITUTE. Split the result by **arithmetic intensity** -- flops per byte of traffic:

    AI = 2*M*N*K / (2 * (M*K + K*N + 2*M*N))        [fp16 in/out, 2 bytes/elem]

A compute-bound shape (high AI) is insensitive to the bandwidth difference, so its measured
ratio transfers to navi32 directly. A memory-bound shape (low AI) is where the
overstatement lives. If the catalog win holds in the high-AI band, the conclusion is safe
regardless of bandwidth -- which is a bound on the gap rather than a measurement of it, but
it is obtainable here and the downclock is not.

The roofline crossover for this part is ~960 GB/s / ~120 TFLOP/s ~ 125 flop/byte; navi32's
is ~624 / ~74 ~ 118, i.e. similar. Shapes well above that are compute-bound on BOTH parts.
"""

import collections
import csv
import statistics
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/navi32/results/P6_main.csv"
BASE = sys.argv[2] if len(sys.argv) > 2 else "navi32ship"
BANDS = [("memory-bound  AI<32", 0, 32), ("mixed  32-128", 32, 128),
         ("compute-bound 128-512", 128, 512), ("deep compute  AI>=512", 512, 1e9)]


def ai(m, n, k):
    flops = 2.0 * m * n * k
    bytes_ = 2.0 * (m * k + k * n + 2 * m * n)
    return flops / bytes_


def main():
    sh, meta = collections.defaultdict(dict), {}
    for r in csv.DictReader(open(CSV)):
        if r["status"] != "ok":
            continue
        g = float(r["gflops"])
        if g <= 0:
            continue
        sh[r["shape_id"]][r["arm"]] = g
        meta[r["shape_id"]] = (int(r["M"]), int(r["N"]), int(r["K"]))

    arms = sorted({a for v in sh.values() for a in v})
    keys = [k for k, v in sh.items() if all(a in v for a in arms)]
    print(f"shapes: {len(keys)}   arms: {arms}\n")

    print(f"{'band':<24}{'n':>5}{'%time':>8}" + "".join(f"{a:>15}" for a in arms if a != BASE))
    print("-" * (37 + 15 * (len(arms) - 1)))
    tot = sum(2.0 * meta[k][0] * meta[k][1] * meta[k][2] / sh[k][BASE] for k in keys)
    for name, lo, hi in BANDS:
        ks = [k for k in keys if lo <= ai(*meta[k]) < hi]
        if len(ks) < 8:
            continue
        share = sum(2.0 * meta[k][0] * meta[k][1] * meta[k][2] / sh[k][BASE] for k in ks) / tot * 100
        row = f"{name:<24}{len(ks):>5}{share:>7.1f}%"
        for a in arms:
            if a == BASE:
                continue
            num = sum(2.0 * meta[k][0] * meta[k][1] * meta[k][2] for k in ks)
            ta = sum(2.0 * meta[k][0] * meta[k][1] * meta[k][2] / sh[k][a] for k in ks)
            tb = sum(2.0 * meta[k][0] * meta[k][1] * meta[k][2] / sh[k][BASE] for k in ks)
            row += f"{(num/ta)/(num/tb)*100:14.2f}%"
        print(row)

    print("\nread: a result that holds in the compute-bound bands transfers to navi32")
    print("      regardless of its lower bandwidth; the memory-bound band is the part")
    print("      this card cannot speak for.")


if __name__ == "__main__":
    main()
