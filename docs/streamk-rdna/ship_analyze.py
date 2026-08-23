#!/usr/bin/env python3
"""
Would shipping StreamK for navi31 TN HHS actually be a win?

This is the question the scope finding forced (GATE_RESULT.md "SCOPE"): a default gfx1100
build ships NO StreamK kernels -- 2560 `StreamK: 0` against 22 `StreamK: 3`, and those 22
are in `Experimental/`, which `tasks.py` excludes by default. So the +1.34% gate result is
measured on a library that does not ship. The prior question is whether the StreamK catalog
beats the shipped one at all.

Arms, in interleave order -- the order is the experiment design:
  1 grid_sk0     devlib_stock_grid, 298 SK0 solutions   <- shipped-representative BASELINE
  2 sk3          exp/stock SK3 catalog, no gate
  3 sk3_gate     exp/stock SK3 catalog + ORIGAMI_MN_GATE
  4 grid_sk0_aa  identical to arm 1                     <- A/A, deliberately LAST

Arm 4 is a copy of arm 1 placed at the far end of the cycle, so the pair brackets the whole
interleave and measures the MAXIMUM position drift rather than a fraction of it. The plateau
run showed 0.31 pt of drift that was monotone in arm order and nearly manufactured a ranking.

There is no "definitionally identical" partition here (the arms are different libraries), so
drift is removed by MODEL rather than by control: assume it is linear in interleave position
and interpolate. That assumption is stated, and both raw and corrected numbers are printed
so the reader can see how much work it is doing.

--fixed-iters was mandatory for this run: the libraries differ in size (298 vs 192) and
tiered iteration counts charge one-time library init unevenly -- a known artifact worth
5 points in an earlier campaign.
"""

import csv
import statistics
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/sk_modes/results/ship_test.csv"
BASE, AA = "grid_sk0", "grid_sk0_aa"
POS = {"grid_sk0": 1, "sk3": 2, "sk3_gate": 3, "grid_sk0_aa": 4}
TEST = ["sk3", "sk3_gate"]
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
            s.setdefault("k_" + r["arm"], r.get("kernel", ""))
            s.setdefault("sk_" + r["arm"], r.get("sk_mode", ""))
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
    arms = list(POS)
    ok = sorted(k for k in sh if all(sh[k].get(a) for a in arms))
    print(f"shapes complete on all {len(arms)} arms: {len(ok)}")
    if len(ok) < 20:
        print("too few to read yet")
        return

    dg, dt = contrast(sh, ok, AA, BASE)
    span = POS[AA] - POS[BASE]
    print(f"\nA/A DRIFT across the full interleave (position {POS[BASE]} -> {POS[AA]}):")
    print(f"  geomean {dg:.2f}%   tput {dt:.2f}%   <- the maximum drift this run contains")

    print(f"\nvs shipped baseline '{BASE}'  (>100% = the StreamK arm is FASTER)")
    print(f"  {'arm':<12} {'pos':>4} {'raw geo':>9} {'raw tput':>9} | "
          f"{'corr geo':>9} {'corr tput':>10}")
    print("  " + "-" * 62)
    for a in TEST:
        g, t = contrast(sh, ok, a, BASE)
        # linear-in-position drift model, stated in the docstring
        f = (POS[a] - POS[BASE]) / span
        cg, ct = g / (1 + f * (dg / 100 - 1)), t / (1 + f * (dt / 100 - 1))
        print(f"  {a:<12} {POS[a]:>4} {g:8.2f}% {t:8.2f}% | {cg:8.2f}% {ct:9.2f}%")

    # The cross-library rows above CONFOUND StreamK with catalog: 192 SK3 solutions vs 298
    # SK0 ones, and the SK3 catalog was separately tuned. This contrast does not -- both
    # arms are the same library, differing only by the gate env var. It is the only clean
    # number in this run, and it independently reproduces the gate_full result under a
    # different protocol (--fixed-iters 20 here vs tiered counts there).
    g, t = contrast(sh, ok, "sk3_gate", "sk3")
    f = (POS["sk3_gate"] - POS["sk3"]) / span
    print(f"\nCLEAN within-catalog contrast (same library, gate is the only difference)")
    print(f"  sk3_gate / sk3   raw geo {g:.2f}%  raw tput {t:.2f}%  |  "
          f"corr tput {t/(1+f*(dt/100-1)):.2f}%")
    print(f"  gate_full measured +1.36% tput above the gate; this is an independent check.")

    print(f"\nby duration band (corrected tput)")
    print(f"  {'band':<10} {'n':>5} {'%time':>7} " + "".join(f"{a:>12}" for a in TEST))
    print("  " + "-" * 52)
    tot = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ok)
    for name, lo, hi in BANDS:
        ks = [k for k in ok if lo <= sh[k]["band"] < hi]
        if not ks:
            continue
        share = sum(flops(sh[k]["mnk"]) / med(sh[k][BASE]) for k in ks) / tot * 100
        cells = ""
        for a in TEST:
            _, t = contrast(sh, ks, a, BASE)
            f = (POS[a] - POS[BASE]) / span
            cells += f"{t/(1+f*(dt/100-1)):11.2f}%"
        print(f"  {name:<10} {len(ks):5} {share:6.1f}% {cells}")

    # CONCENTRATION + JACKKNIFE on the cross-library contrast.
    # The cross-library wall-clock number on this suite is set by a handful of huge shapes:
    # between n=338 and n=681 the >=5ms band went from 4 shapes / 15% of time to 13 shapes /
    # 36%, and the headline moved 3.4 pt with it. So the figure is only as trustworthy as its
    # most concentrated band, and it must never be quoted without this table beside it.
    print("\nCONCENTRATION — how much of the wall-clock rides on how few shapes")
    by_t = sorted(ok, key=lambda k: flops(sh[k]["mnk"]) / med(sh[k][BASE]), reverse=True)
    for kk in (1, 5, 10, 25, 50):
        share = sum(flops(sh[x]["mnk"]) / med(sh[x][BASE]) for x in by_t[:kk]) / tot * 100
        print(f"  top {kk:>3} shapes: {share:5.1f}% of total kernel time")

    print("\nJACKKNIFE — drop the largest time consumers, corrected tput")
    print(f"  {'dropped':>8} {'n':>6} " + "".join(f"{a:>12}" for a in TEST))
    print("  " + "-" * 40)
    for d in (0, 1, 5, 10, 25, 50):
        ks = by_t[d:]
        if len(ks) < 20:
            continue
        cells = ""
        for a in TEST:
            _, t = contrast(sh, ks, a, BASE)
            f = (POS[a] - POS[BASE]) / span
            cells += f"{t/(1+f*(dt/100-1)):11.2f}%"
        print(f"  {d:>8} {len(ks):6} {cells}")
    print("  A verdict that survives dropping the top 10 is a verdict; one that does not is\n"
          "  a statement about those 10 shapes.")

    # Did the arms actually differ in what they ran? Two libraries can post the same
    # number while choosing different kernels -- and here they should differ, since one
    # catalog has StreamK kernels and the other does not.
    print("\nwhat actually ran")
    for a in arms:
        modes = defaultdict(int)
        for k in ok:
            modes[sh[k].get("sk_" + a, "?")] += 1
        same = sum(1 for k in ok if sh[k].get("k_" + a) == sh[k].get("k_" + BASE))
        mix = " ".join(f"SK{m}:{c}" for m, c in sorted(modes.items()))
        print(f"  {a:<12} kernel-identical to baseline {same/len(ok)*100:5.1f}%   {mix}")


if __name__ == "__main__":
    main()
