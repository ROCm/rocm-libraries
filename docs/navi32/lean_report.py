#!/usr/bin/env python3
"""Phase 2 reporting for the lean-Grid arms.

Two things analyze.py does not do, and both decide whether a parity claim is real:

  * DURATION BANDING. ~2/3 of this workload sits near a ~25 us host-dispatch floor, where
    a smaller catalog is faster for reasons unrelated to kernel quality. An all-shapes
    geomean therefore reports parity for almost any catalog. Claims are made on the
    ABOVE-FLOOR band.
  * skinny_M vs skinny_N split. The eval strata label both "skinny"; they behave
    differently and the risk was called out up front.

Controls, both mandatory:
  identity ~= full  -> the tooling is neutral; otherwise every lean number is a
                       serialization artifact.
  random  <  lean   -> the metric has power; otherwise "parity" is the floor talking.
"""
import argparse, collections, csv, math, statistics, sys


def geo(v): return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def stratum(m, n):
    lo, hi = min(m, n), max(m, n)
    if lo <= 8:      return "gemv"
    if hi <= 128:    return "tiny"
    if n * 4 <= m:   return "skinny_N"
    if m * 4 <= n:   return "skinny_M"
    if hi >= 4096:   return "large"
    return "med"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--base", default="full")
    ap.add_argument("--floor", type=float, default=40.0)
    a = ap.parse_args()

    per = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(a.csv)):
        if r["status"] != "ok": continue
        try: us, gf = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or gf <= 0: continue
        d = per[r["shape_id"]].setdefault(r["arm"], [])
        d.append((us, gf))
        meta[r["shape_id"]] = (int(r["M"]), int(r["N"]), int(r["K"]))

    arms = sorted({k for v in per.values() for k in v})
    if a.base not in arms: sys.exit(f"baseline {a.base} not in {arms}")
    full = {s: v for s, v in per.items() if set(arms) <= set(v)}
    print(f"shapes complete on all {len(arms)} arms: {len(full)}   arms: {arms}\n")
    if len(full) < 20: print("too few to read yet"); return

    def band(s):
        us = min(x[0] for x in full[s][a.base])
        return "above-floor" if us > a.floor else "at-floor"

    others = [x for x in arms if x != a.base]
    for label, sel in (("ABOVE-FLOOR  (>%.0f us -- where kernel quality is visible)" % a.floor,
                        lambda s: band(s) == "above-floor"),
                       ("AT-FLOOR     (<=%.0f us -- host dispatch dominates)" % a.floor,
                        lambda s: band(s) == "at-floor"),
                       ("ALL SHAPES", lambda s: True)):
        sh = [s for s in full if sel(s)]
        if not sh: continue
        print(f"{label}   n={len(sh)}")
        print(f"  {'arm':<12}{'geomean':>10}{'wall-clock':>12}")
        for arm in others:
            rr, tb, tc = [], 0.0, 0.0
            for s in sh:
                gb = max(x[1] for x in full[s][a.base]); gc = max(x[1] for x in full[s][arm])
                ub = min(x[0] for x in full[s][a.base]); uc = min(x[0] for x in full[s][arm])
                rr.append(gc / gb); tb += ub; tc += uc
            print(f"  {arm:<12}{100*geo(rr):>9.2f}%{100*tb/tc:>11.2f}%")
        print()

    sh = [s for s in full if band(s) == "above-floor"]
    print(f"BY STRATUM, above-floor only (wall-clock vs {a.base})")
    print(f"  {'stratum':<10}{'n':>5}" + "".join(f"{x:>12}" for x in others))
    for st in ("gemv", "tiny", "skinny_M", "skinny_N", "med", "large"):
        rows = [s for s in sh if stratum(meta[s][0], meta[s][1]) == st]
        if not rows: continue
        line = f"  {st:<10}{len(rows):>5}"
        for arm in others:
            tb = sum(min(x[0] for x in full[s][a.base]) for s in rows)
            tc = sum(min(x[0] for x in full[s][arm]) for s in rows)
            line += f"{100*tb/tc:>11.2f}%"
        print(line)

    print("\nCONTROLS")
    # An "_aa" arm is only an A/A control for the library it mirrors. Reporting it against
    # a different baseline yields a meaningless number (e.g. n33ship_aa vs full = 81.9%,
    # which is the port's effect, not a noise floor).
    aa = [x for x in others if x.endswith("_aa") and x[:-3] == a.base]
    if aa:
        tb = sum(min(y[0] for y in full[s][a.base]) for s in sh)
        tc = sum(min(y[0] for y in full[s][aa[0]]) for s in sh)
        print(f"  A/A floor ({aa[0]}): {100*tb/tc:.2f}%  -- an arm must clear this to mean anything")
    else:
        stray = [x for x in others if x.endswith("_aa")]
        if stray:
            print(f"  (no A/A control for baseline '{a.base}'; {stray[0]} mirrors "
                  f"'{stray[0][:-3]}' -- rerun with --base {stray[0][:-3]} for the floor)")
    if "identity" in others:
        tb = sum(min(y[0] for y in full[s][a.base]) for s in sh)
        tc = sum(min(y[0] for y in full[s]["identity"]) for s in sh)
        print(f"  identity vs full: {100*tb/tc:.2f}%  -- must be ~100, else the tooling is not neutral")
    if "lean100" in others and "rand100" in others:
        tl = sum(min(y[0] for y in full[s]["lean100"]) for s in sh)
        tr = sum(min(y[0] for y in full[s]["rand100"]) for s in sh)
        print(f"  lean100 vs rand100: {100*tr/tl:.2f}%  -- must be >100, else the metric has no power")


if __name__ == "__main__":
    main()
