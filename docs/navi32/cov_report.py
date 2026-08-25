#!/usr/bin/env python3
"""Read out the coverage-extension benchmark.

THE CONTROL IS DEFINED FROM THE RUN ITSELF, NOT FROM MY MODEL OF IT.
bench_arms records the kernel each arm actually dispatched. So a query where `shipped` and
`extended` ran the SAME kernel executed an identical catalog, and any difference it shows is
pure noise -- an in-experiment negative control that costs nothing and cannot be wrong about
which rows were touched. Inferring the treated set from the YAMLs instead would re-use my own
reimplementation of the selection rule to validate a result that depends on it.

This matters here specifically: the last time a stratum looked soft in this campaign I asserted
a structural explanation for it, and the check took two minutes and refuted both halves. Of the
10 gemv queries involved, 8 had run an IDENTICAL catalog and still showed a median 0.975 -- the
n~10 noise floor, not a regression. Splitting by measured kernel makes that visible by default.

TWO RUNS, AND WHY THE AGGREGATE IS NOT THE TEST.
A single run cannot separate a real 2% from drift. The decisive statistic for the shipped
re-map was per-shape REPRODUCIBILITY: gains correlated r=0.961 with 90% sign agreement across
independent runs, against an A/A noise reference of r=0.551 computed identically. Same shapes
win twice = structural. That test is repeated here.
"""
import argparse, collections, csv, json, math, statistics


def load(path):
    per, meta = collections.defaultdict(dict), {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            try:
                us, gf = float(r["us"]), float(r["gflops"])
            except ValueError:
                continue
            if us <= 0 or gf <= 0:
                continue
            per[r["shape_id"]][r["arm"]] = (us, gf, r["kernel"])
            meta[r["shape_id"]] = (int(r["M"]), int(r["N"]), int(r["K"]), r["stratum"])
    need = {"shipped", "extended", "shipped_aa"}
    return {s: v for s, v in per.items() if need <= set(v)}, meta


def geo(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")


def block(title, rows, D, key, extras=()):
    """wall-clock of each arm vs `shipped`, plus the A/A floor, per group. >100 = faster."""
    cols = ["shipped_aa", "extended"] + list(extras)
    print(f"\n  {title}")
    print(f"  {'group':<12}{'n':>5}" + "".join(f"{c:>12}" for c in cols) + f"{'ext geo':>10}")
    print("  " + "-" * (17 + 12 * len(cols) + 10))
    for g in sorted({key(s) for s in rows}, key=str):
        sh = [s for s in rows if key(s) == g]
        if not sh:
            continue
        base = sum(D[s]["shipped"][0] for s in sh)
        cells = "".join(f"{100*base/sum(D[s][c][0] for s in sh):>11.2f}%" for c in cols)
        eg = 100 * geo([D[s]["extended"][1] / D[s]["shipped"][1] for s in sh])
        flag = "" if len(sh) >= 15 else "  <- n<15, not claimable"
        print(f"  {str(g):<12}{len(sh):>5}{cells}{eg:>9.2f}%{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1", default="/home/vmijovic/navi32/results/P6_cov_run1.csv")
    ap.add_argument("--run2", default="/home/vmijovic/navi32/results/P6_cov_run2.csv")
    ap.add_argument("--shapes", default="/home/vmijovic/navi32/state/eval_fullcov.json")
    ap.add_argument("--floor-us", type=float, default=25.0,
                    help="cold dispatch floor; below it the arms are comparing launch overhead")
    a = ap.parse_args()

    try:
        A, meta = load(a.run1)
    except FileNotFoundError:
        print(f"run1 not started yet ({a.run1} does not exist)")
        return
    print(f"run1: {len(A)} shapes with all three arms")
    if len(A) < 30:
        print("  too few complete shapes to read yet")
        return
    try:
        B, _ = load(a.run2)
    except FileNotFoundError:
        B = {}
    print(f"run2: {len(B)} shapes")

    info = {s["shape_id"]: s for s in json.load(open(a.shapes))["shapes"]}

    # TREATED vs CONTROL, decided by what actually ran
    treated = [s for s in A if A[s]["shipped"][2] != A[s]["extended"][2]]
    control = [s for s in A if A[s]["shipped"][2] == A[s]["extended"][2]]
    print(f"\n  by MEASURED kernel: {len(treated)} treated (different kernel), "
          f"{len(control)} control (identical kernel)")
    if not treated:
        print("  NOTHING WAS TREATED -- the extension changed no kernel on this eval set.")
        return

    # `full` is optional: it is the best alternative candidate, not the shipped one
    # extended_ship = no holdout (the ship candidate); full = the existing ungated re-map
    ex = [c for c in ("extended_ship", "nogate", "full", "lean") if all(c in A[s] for s in A)]
    if ex:
        print(f"  optional arms present: {ex}")

    above = set(s for s in A if A[s]["shipped"][0] >= a.floor_us)
    print(f"  above the {a.floor_us:.0f}us cold dispatch floor: {len(above)}/{len(A)}")

    tset = set(treated)
    tre_ab = [s for s in treated if s in above]
    con_ab = [s for s in control if s in above]
    block("ALL SHAPES, treated vs control", treated + control, A,
          lambda s: "treated" if s in tset else "control", ex)
    block(f"ABOVE FLOOR ONLY (>{a.floor_us:.0f}us)", tre_ab + con_ab, A,
          lambda s: "treated" if s in tset else "control", ex)
    block("TREATED, by query stratum", tre_ab, A, lambda s: meta[s][3], ex)
    block("TREATED, by serving-row stratum", tre_ab, A,
          lambda s: info[s].get("row_stratum", "?"), ex)

    def band(s):
        u = A[s]["shipped"][0]
        return "<25us" if u < 25 else "25-100us" if u < 100 else \
               "100us-1ms" if u < 1000 else ">=1ms"
    block("TREATED, by kernel duration", treated, A, band, ex)

    # ---- the decisive test: do the same shapes win in BOTH runs? --------------------------
    common = sorted(set(treated) & set(B))
    if len(common) < 30:
        print(f"\n  run 2 not ready ({len(common)} common treated shapes) -- "
              f"reproducibility test pending; the aggregate above is NOT sufficient on its own.")
        return
    def deltas(D, arm):
        return [D[s]["shipped"][0] / D[s][arm][0] - 1 for s in common]
    def corr(x, y):
        n = len(x); mx, my = statistics.mean(x), statistics.mean(y)
        sx, sy = statistics.pstdev(x), statistics.pstdev(y)
        if not sx or not sy:
            return float("nan")
        return sum((p - mx) * (q - my) for p, q in zip(x, y)) / n / (sx * sy)

    da, db = deltas(A, "extended"), deltas(B, "extended")
    aa, bb = deltas(A, "shipped_aa"), deltas(B, "shipped_aa")
    agree = sum(1 for p, q in zip(da, db) if (p > 0) == (q > 0)) / len(common)
    print(f"\n  REPRODUCIBILITY over {len(common)} treated shapes")
    print(f"    per-shape gain correlation run1 vs run2 : r = {corr(da, db):.3f}")
    print(f"    same sign in both runs                  : {100*agree:.0f}%")
    print(f"    same statistic on the A/A arm (noise)   : r = {corr(aa, bb):.3f}  <- reference")
    print("    high r + high agreement = structural; near the A/A reference = noise.")


if __name__ == "__main__":
    main()
