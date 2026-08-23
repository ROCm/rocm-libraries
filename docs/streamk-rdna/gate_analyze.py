#!/usr/bin/env python3
"""
Analyse the ORIGAMI_MN_GATE A/B.

Arms:
  gate_off     TENSILE_STREAMK_DYNAMIC_GRID=6                      (shipped predictor)
  gate_on      ... + ORIGAMI_MN_GATE=2867000                       (StreamK off above M*N)
  gate_off_aa  identical to gate_off                               (A/A -> noise floor)

The A/A arm is the point. Bootstrap CIs resample shapes and cannot see the in-session
floor; only re-running the same configuration can. Nothing below that floor is a result.

Tests the projection from GATED_POLICY.md: 101.30% geomean, 102.08% flops-weighted.
Runs unchanged on a partial CSV, so it can be used to monitor a live run -- but a partial
read is only as representative as the shape ordering (checked and roughly flat here).
"""

import csv
import statistics
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/sk_modes/results/gate_full.csv"
BASE, ON, AA = "gate_off", "gate_on", "gate_off_aa"
BANDS = [("<0.1ms", 0.0, 0.1), ("0.1-1ms", 0.1, 1.0), ("1-5ms", 1.0, 5.0), (">=5ms", 5.0, 1e9)]


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
            s["band"] = float(r["ms"])
            s[r["arm"]].append(g)
            s.setdefault("kern_" + r["arm"], r.get("kernel", ""))
            s.setdefault("grid_" + r["arm"], r.get("sk_grid", ""))
            s.setdefault("tiles_" + r["arm"], r.get("sk_tiles", ""))
    return sh


def med(x):
    return statistics.median(x)


def flops(m):
    return 2.0 * m[0] * m[1] * m[2]


def contrast(sh, keys, a, b):
    """a vs b. geomean of per-shape ratio, and flops-weighted total throughput ratio."""
    ratios = []
    n = ta = tb = 0.0
    for k in keys:
        ga, gb = med(sh[k][a]), med(sh[k][b])
        ratios.append(ga / gb)
        F = flops(sh[k]["mnk"])
        n += F
        ta += F / ga
        tb += F / gb
    return statistics.geometric_mean(ratios), (n / ta) / (n / tb), len(ratios)


def main():
    sh = load(CSV)
    both = sorted(k for k in sh if sh[k].get(BASE) and sh[k].get(ON))
    aa = sorted(k for k in sh if sh[k].get(BASE) and sh[k].get(AA))
    print(f"shapes with gate_off+gate_on: {len(both)}   with A/A pair: {len(aa)}")
    if not both:
        print("no complete pairs yet")
        return

    # 1. noise floor, measured not assumed
    if aa:
        g, t, _ = contrast(sh, aa, AA, BASE)
        devs = sorted(abs(med(sh[k][AA]) / med(sh[k][BASE]) - 1) for k in aa)
        print(f"\nA/A NOISE FLOOR   geomean {g*100:.2f}%  tput {t*100:.2f}%  "
              f"per-shape |dev| median {statistics.median(devs)*100:.2f}%  "
              f"p95 {devs[int(0.95*len(devs))]*100:.2f}%")
        print("  (a perfect A/A is 100.00%; the gap is the floor the gate must clear)")

    # 2. the headline
    g, t, n = contrast(sh, both, ON, BASE)
    print(f"\nGATE ON vs GATE OFF   n={n}")
    print(f"  per-shape geomean      {g*100:7.2f}%   (projected 101.30%)")
    print(f"  flops-weighted tput    {t*100:7.2f}%   (projected 102.08%)")

    # 3. per band
    print(f"\n{'band':<10} {'n':>5} {'geomean':>9} {'tput-wtd':>9} {'ON wins':>9} {'% of time':>10}")
    print("-" * 58)
    tot = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in both)
    for name, lo, hi in BANDS:
        ks = [k for k in both if lo <= sh[k]["band"] < hi]
        if not ks:
            continue
        gg, tt, _ = contrast(sh, ks, ON, BASE)
        wins = sum(1 for k in ks if med(sh[k][ON]) > med(sh[k][BASE])) / len(ks)
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ks) / tot
        print(f"{name:<10} {len(ks):5} {gg*100:8.2f}% {tt*100:8.2f}% {wins*100:8.1f}% {share*100:9.1f}%")

    # 4. did the gate disturb kernel selection? (it must not -- it changes GRID only)
    same_k = sum(1 for k in both if sh[k].get("kern_" + ON) == sh[k].get("kern_" + BASE))
    print(f"\nselection agreement : {same_k/len(both)*100:.1f}% identical kernel "
          f"({len(both)-same_k} differ)  <- must be ~100%, the gate changes GRID not KERNEL")

    # NOTE: sk_grid/sk_tiles are -1 in a perf pass -- TENSILE_DB=0x40 is deliberately not
    # set there, since dumping kernel args would pollute the timing. So "did the grid move"
    # cannot be read from this CSV. The gate PREDICATE is deterministic though, so partition
    # on it instead. Below-gate shapes are a hard negative control: the gate cannot touch
    # them, so any effect there is noise or a side effect, and bounds what we can claim.
    GATE = 2867000.0
    above = [k for k in both if sh[k]["mnk"][0] * sh[k]["mnk"][1] >= GATE]
    below = [k for k in both if sh[k]["mnk"][0] * sh[k]["mnk"][1] < GATE]
    # Report A/A WITHIN each partition next to the gate effect. A single global A/A is not
    # enough: measured here, the systematic is ~0% above the gate and ~0.8% below it, so a
    # global figure would over-correct the half that needs no correction. The A/A column is
    # the zero point its own row must be read against.
    print(f"\npartition on the gate predicate  M*N >= {GATE:.3g}")
    print(f"  {'set':<22} {'n':>4}  {'A/A geo':>8} {'A/A tput':>9} | {'ON geo':>8} {'ON tput':>9} "
          f"| {'ON/AA geo':>9} {'ON/AA tput':>10}  {'%time':>6}")
    print("  " + "-" * 96)
    have_aa = set(aa)
    for label, ks, note in (("ABOVE gate (can act)", above, ""),
                            ("BELOW gate (control)", below, "  must match A/A")):
        if not ks:
            continue
        gg, tt, _ = contrast(sh, ks, ON, BASE)
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ks) / tot
        ks_aa = [k for k in ks if k in have_aa]
        if ks_aa:
            ag, at, _ = contrast(sh, ks_aa, AA, BASE)
            print(f"  {label:<22} {len(ks):4}  {ag*100:7.2f}% {at*100:8.2f}% | "
                  f"{gg*100:7.2f}% {tt*100:8.2f}% | {gg/ag*100:8.2f}% {tt/at*100:9.2f}%  "
                  f"{share*100:5.1f}%{note}")
        else:
            print(f"  {label:<22} {len(ks):4}  {'-':>8} {'-':>9} | "
                  f"{gg*100:7.2f}% {tt*100:8.2f}% | {'-':>9} {'-':>10}  {share*100:5.1f}%{note}")
    print("\n  ON/AA is the estimate: the gate arm divided by the A/A zero point for the SAME\n"
          "  shapes. Projection to beat: geomean 101.30%, tput 102.08%.")


if __name__ == "__main__":
    main()
