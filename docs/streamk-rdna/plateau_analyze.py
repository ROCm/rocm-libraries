#!/usr/bin/env python3
"""
Is the M*N gate threshold a plateau, or did the projection flatter it?

GATE_RESULT.md claims any cut in [5e5, 3e6] performs within 0.03pt. That claim is
PROJECTED from the Aug-19 census, and the same projection overstated the headline by
0.7pt -- so it is the most load-bearing unmeasured claim in that report. This measures it.

Arms: gate_off, gate_off_aa (A/A zero point), and gates at 1e6 / 2.867e6 / 1e7.

Every arm is divided by the A/A arm on the SAME shapes. Per-partition, not globally: the
prior run measured the systematic at ~0.2% above the gate and ~0.8% below it, so a single
global correction over-corrects the half that needs none.

Each threshold is also scored on ITS OWN above-gate partition (the shapes it can act on)
and on the common partition (M*N >= 1e7, which every arm gates), so the thresholds are
compared both on their own terms and on identical shapes.

Runs unchanged on a partial CSV.
"""

import csv
import statistics
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/sk_modes/results/gate_plateau.csv"
BASE, AA = "gate_off", "gate_off_aa"
GATES = [("g_1e6", 1e6), ("g_2867k", 2.867e6), ("g_1e7", 1e7)]


def load(path):
    sh = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "ok":
                continue
            try:
                g = float(r["gflops"])
            except (ValueError, KeyError):
                continue
            if g <= 0:
                continue
            s = sh[r["shape_id"]]
            s["mnk"] = (int(r["M"]), int(r["N"]), int(r["K"]))
            s[r["arm"]].append(g)
    return sh


def med(x):
    return statistics.median(x)


def flops(m):
    return 2.0 * m[0] * m[1] * m[2]


def contrast(sh, keys, a, b):
    rr = []
    n = ta = tb = 0.0
    for k in keys:
        ga, gb = med(sh[k][a]), med(sh[k][b])
        rr.append(ga / gb)
        F = flops(sh[k]["mnk"])
        n += F
        ta += F / ga
        tb += F / gb
    return statistics.geometric_mean(rr) * 100, (n / ta) / (n / tb) * 100


def main():
    sh = load(CSV)
    arms = [BASE, AA] + [g for g, _ in GATES]
    ok = sorted(k for k in sh if all(sh[k].get(a) for a in arms))
    print(f"shapes complete on all {len(arms)} arms: {len(ok)}")
    if len(ok) < 20:
        print("too few to read yet")
        return

    aag, aat = contrast(sh, ok, AA, BASE)
    print(f"A/A zero point (whole suite): geomean {aag:.2f}%  tput {aat:.2f}%\n")

    print("WHOLE SUITE, each gate vs gate_off, divided by the A/A zero point")
    print(f"  {'gate':<10} {'raw geo':>8} {'raw tput':>9} | {'/AA geo':>8} {'/AA tput':>9}")
    print("  " + "-" * 52)
    for name, cut in GATES:
        g, t = contrast(sh, ok, name, BASE)
        print(f"  {name:<10} {g:7.2f}% {t:8.2f}% | {g/aag*100:7.2f}% {t/aat*100:8.2f}%")

    print("\nEACH GATE ON ITS OWN above-gate PARTITION (the shapes it can act on)")
    print(f"  {'gate':<10} {'n':>5} {'%time':>7} | {'/AA geo':>8} {'/AA tput':>9}")
    print("  " + "-" * 48)
    tot = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ok)
    for name, cut in GATES:
        ks = [k for k in ok if sh[k]["mnk"][0] * sh[k]["mnk"][1] >= cut]
        if not ks:
            continue
        g, t = contrast(sh, ks, name, BASE)
        ag, at = contrast(sh, ks, AA, BASE)
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ks) / tot * 100
        print(f"  {name:<10} {len(ks):5} {share:6.1f}% | {g/ag*100:7.2f}% {t/at*100:8.2f}%")

    # Common partition: shapes EVERY gate acts on. Any difference here is the thresholds
    # themselves, not which shapes each one happens to cover.
    common = [k for k in ok if sh[k]["mnk"][0] * sh[k]["mnk"][1] >= max(c for _, c in GATES)]
    if len(common) >= 10:
        print(f"\nCOMMON PARTITION (M*N >= {max(c for _,c in GATES):.3g}, n={len(common)}) — "
              f"all gates act identically here, so these should MATCH")
        ag, at = contrast(sh, common, AA, BASE)
        cal = {}
        for name, _ in GATES:
            g, t = contrast(sh, common, name, BASE)
            cal[name] = (g / ag, t / at)
            print(f"  {name:<10} /AA geo {g/ag*100:7.2f}%   /AA tput {t/at*100:8.2f}%")
        print("  (divergence here would mean the arms differ by something other than the gate)")

        # If those values track the arms' POSITION in the interleave order rather than their
        # thresholds, the spread is drift, not signal -- later-measured arms run warmer. The
        # common partition is then also the calibration: dividing each gate by its OWN value
        # here removes an arm-specific systematic that the A/A arm cannot see, because A/A
        # occupies one fixed position and the gates occupy others.
        order = [n for n, _ in GATES]
        tputs = [cal[n][1] for n in order]
        monotone = all(x < y for x, y in zip(tputs, tputs[1:])) or \
                   all(x > y for x, y in zip(tputs, tputs[1:]))
        print(f"  spread {(max(tputs)-min(tputs))*100:.2f} pt"
              f"{'  <- MONOTONE IN ARM ORDER: position drift, not threshold' if monotone else ''}")

        print("\nWHOLE SUITE, CALIBRATED (each gate / its own common-partition value)")
        print(f"  {'gate':<10} {'geo':>8} {'tput':>9}")
        print("  " + "-" * 30)
        for name, _ in GATES:
            g, t = contrast(sh, ok, name, BASE)
            print(f"  {name:<10} {g/ag/cal[name][0]*100:7.2f}% {t/at/cal[name][1]*100:8.2f}%")
        print("  These are comparable across gates; the uncalibrated table above is not.")

    # The band only 1e6 gates: does gating MORE shapes help or hurt?
    lo = [k for k in ok if 1e6 <= sh[k]["mnk"][0] * sh[k]["mnk"][1] < 2.867e6]
    if len(lo) >= 10:
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in lo) / tot * 100
        ag, at = contrast(sh, lo, AA, BASE)
        g1, t1 = contrast(sh, lo, "g_1e6", BASE)
        g2, t2 = contrast(sh, lo, "g_2867k", BASE)
        print(f"\nTHE CONTESTED BAND  1e6 <= M*N < 2.867e6  (n={len(lo)}, {share:.1f}% of time)")
        print(f"  g_1e6   (gates it)      /AA geo {g1/ag*100:7.2f}%  tput {t1/at*100:8.2f}%")
        print(f"  g_2867k (leaves it on)  /AA geo {g2/ag*100:7.2f}%  tput {t2/at*100:8.2f}%")
        print("  This is the whole question: is the lower cut leaving throughput on the table?")


if __name__ == "__main__":
    main()
