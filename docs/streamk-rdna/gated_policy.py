#!/usr/bin/env python3
"""
Is a size-gated StreamK policy worth it on gfx1100 TN HHS?

Background. The 2026-08-19 grid campaign measured, per band, that turning StreamK off
(TENSILE_STREAMK_DYNAMIC_GRID=4, data_parallel) BEATS the shipped predictor (mode 6)
above 1 ms and loses badly below 0.1 ms:

    <0.1ms  93.92%      1-5ms  101.91% [101.25, 102.57]
    0.1-1ms 100.17%     >=5ms  101.66% [100.08, 103.88]

That invites a gated policy: StreamK on when small, off when large. But those bands are
defined by MEASURED DURATION, which a selector cannot observe at selection time. A
shippable predicate must key on problem SIZE. This asks how much of the oracle gain a
size predicate actually recovers.

No GPU. Reads p1_modes.csv (1500 shapes x 2 reps x 6 arms).

COLUMN TRAP: `ms` is a FIXED per-shape banding reference -- identical across every arm and
rep. It is not a measurement. `gflops` is the measurement. Using `ms` makes every policy
score exactly 100.00% and the noise floor 0.00%, which is how this was caught.

Metrics, which disagree here and are meant to:
  - geomean of per-shape gflops ratio: treats every shape as one vote.
  - flops-weighted total throughput  sum(flops) / sum(flops/gflops): the wall-clock
    number. Large shapes dominate, which is exactly where the gate is supposed to pay.

Discipline: threshold FITTED ON TRAIN, scored ON TEST. Reported against the noise floor
measured rep-vs-rep on the same arm -- bootstrap CIs resample shapes and cannot see it.
"""

import csv
import statistics
from collections import defaultdict

CSV = "/home/vmijovic/hhs_tn_grid_vs_resource_origami_9k/measurements/campaign/p1_modes.csv"
BASE, OFF = "m6_default", "m4_dataparallel"


def load():
    shapes = defaultdict(lambda: defaultdict(list))
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok" or row["arm"] not in (BASE, OFF):
                continue
            try:
                g = float(row["gflops"])
            except ValueError:
                continue
            if g <= 0:
                continue
            s = shapes[row["shape_id"]]
            s["mnk"] = (int(row["M"]), int(row["N"]), int(row["K"]))
            s["band"] = float(row["ms"])          # fixed reference, for reporting only
            s[row["arm"]].append(g)
    return {k: v for k, v in shapes.items() if v.get(BASE) and v.get(OFF)}


def med(xs):
    return statistics.median(xs)


def geo(xs):
    return statistics.geometric_mean(xs) if xs else float("nan")


def flops(mnk):
    return 2.0 * mnk[0] * mnk[1] * mnk[2]


def noise_floor(shapes):
    devs = []
    for s in shapes.values():
        for arm in (BASE, OFF):
            r = s[arm]
            if len(r) >= 2 and min(r) > 0:
                devs.append(abs(max(r) / min(r) - 1.0))
    devs.sort()
    return statistics.median(devs), devs[int(0.95 * len(devs))]


def score(shapes, keys, decide):
    """decide(shape) -> True to use the OFF arm. Returns (geomean, flops-weighted tput)."""
    ratios, num, den_pol, den_base = [], 0.0, 0.0, 0.0
    for k in keys:
        s = shapes[k]
        gb, go = med(s[BASE]), med(s[OFF])
        gp = go if decide(s) else gb
        ratios.append(gp / gb)
        F = flops(s["mnk"])
        num += F
        den_pol += F / gp
        den_base += F / gb
    return geo(ratios), (num / den_pol) / (num / den_base)


def fit(shapes, keys, sizef):
    """Pick the size cut maximising TRAIN flops-weighted throughput."""
    best, cut = -1.0, None
    for c in sorted({sizef(shapes[k]["mnk"]) for k in keys}):
        _, tw = score(shapes, keys, lambda s, C=c: sizef(s["mnk"]) >= C)
        if tw > best:
            best, cut = tw, c
    return cut, best


def main():
    shapes = load()
    keys = sorted(shapes)
    fm, fp = noise_floor(shapes)
    print(f"shapes on both arms: {len(keys)}")
    print(f"noise floor (rep-vs-rep, same arm): median {fm*100:.2f}%  p95 {fp*100:.2f}%\n")

    train = [k for i, k in enumerate(keys) if i % 2 == 0]
    test = [k for i, k in enumerate(keys) if i % 2 == 1]

    print(f"{'policy':<36} {'geomean':>9} {'tput-wtd':>9}   scope")
    print("-" * 76)

    g, t = score(shapes, keys, lambda s: True)
    print(f"{'always off (mode 4)':<36} {g*100:8.2f}% {t*100:8.2f}%   all")

    # oracle: unachievable upper bound, picks the winner per shape
    g, t = score(shapes, keys, lambda s: med(s[OFF]) > med(s[BASE]))
    print(f"{'ORACLE per-shape (upper bound)':<36} {g*100:8.2f}% {t*100:8.2f}%   all")

    for thr in (0.1, 0.5, 1.0):
        g, t = score(shapes, keys, lambda s, T=thr: s["band"] >= T)
        print(f"{'duration-gated >=' + str(thr) + 'ms (oracle)':<36} {g*100:8.2f}% {t*100:8.2f}%   all")

    print()
    proxies = {
        "M*N*K": lambda m: m[0] * m[1] * m[2],
        "M*N": lambda m: m[0] * m[1],
        "K": lambda m: m[2],
    }
    for name, f in proxies.items():
        cut, tr = fit(shapes, train, f)
        g, t = score(shapes, test, lambda s, C=cut, ff=f: ff(s["mnk"]) >= C)
        print(f"{'size-gated ' + name:<36} {g*100:8.2f}% {t*100:8.2f}%   TEST "
              f"(cut={cut:.4g}, train tw={tr*100:.2f}%)")

    # honesty check: how much of the wall-clock does the chosen gate actually flip?
    f = proxies["M*N*K"]
    cut, _ = fit(shapes, train, f)
    tot = sum(flops(shapes[k]["mnk"]) / med(shapes[k][BASE]) for k in keys)
    fl = sum(flops(shapes[k]["mnk"]) / med(shapes[k][BASE])
             for k in keys if f(shapes[k]["mnk"]) >= cut)
    n = sum(1 for k in keys if f(shapes[k]["mnk"]) >= cut)
    print(f"\nflipped region: {n}/{len(keys)} shapes ({n/len(keys)*100:.1f}%), "
          f"{fl/tot*100:.1f}% of total kernel time")


if __name__ == "__main__":
    main()
