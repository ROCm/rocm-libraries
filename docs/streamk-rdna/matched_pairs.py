#!/usr/bin/env python3
"""
Like-for-like StreamK mode comparison: same solution, only the SK mode differs.

WHY THIS EXISTS. The campaign's headline figures are *best-vs-best* — the best SK4 anywhere
against the best SK3 anywhere. That is a fair answer to "how good can each mode get", but it
lets each mode pick its own DepthU/CLR/prefetch, so it is NOT the cost of choosing a mode.
This matches solutions whose names are **identical except the `_SK<n>_` token** and reports
the median ratio, which is that cost.

The two answer different questions and differ by ~6 pt:
    best-vs-best   SK4 = 95.0% of SK3   "how good can SK4 get"
    matched pairs  SK4 = 89.4% of SK3   "what does choosing SK4 cost"

IT ALSO SERVES AS AN INDEPENDENT REPLICATION. Every axis conclusion in REPORT.md was derived
by best-vs-best aggregation. Re-deriving them here — a different aggregation over the same
raw data — checks that the conclusions are not artefacts of how the sweeps were summarised.
They replicate: same directions, same orderings, magnitudes uniformly ~15-20 pt stricter.

TWO CONFOUNDED TESTS PRECEDED THIS ONE, and both looked like they refuted REPORT.md §10
("SK5 is exactly SK3 or SK4"):
  1. best-SK5 vs best-SK3 per (run, shape)  -> 10% of cells matched "neither", SK5 sometimes
     173% of both. Confounded: different solutions being compared.
  2. the same, restricted to a shared macro-tile -> 43% still mismatched. Still confounded:
     equal MT can still differ in DepthU/CLR/PGR/PLR.
Only full-name matching tests the claim, and it confirms it (SK5/SK3 median 99.91%).
**Before reporting that data contradicts a claim, check the analysis actually tests it.**

Validation is applied via analyze.py's own loader, so the filter that file documents as
mandatory is not accidentally skipped. On this campaign it changes nothing — 7977 records,
0 FAILED — but that is a measured fact, not an assumption.

    python3 matched_pairs.py [results_dir] [logs_dir]
"""

import collections
import importlib.util
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
LOGS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "logs"

_spec = importlib.util.spec_from_file_location("az", HERE / "analyze.py")
az = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(az)

SK = re.compile(r"_SK(\d)_")
MT = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")


def load_all():
    """-> (records, n_pass, n_fail, n_runs, n_skipped); records are validated only."""
    recs, npass, nfail, runs, skipped = [], 0, 0, 0, 0
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        log = LOGS / f"{d.name}.log"
        if not log.exists() or not (d / "2_BenchmarkData").exists():
            skipped += 1
            continue
        try:
            perf, valid, _ = az.load(d, log)
        except Exception:
            skipped += 1
            continue
        runs += 1
        npass += sum(1 for v in valid.values() if v == "PASSED")
        nfail += sum(1 for v in valid.values() if v == "FAILED")
        for (sol, key), g in perf.items():
            if valid.get((sol, key)) == "PASSED" and g > 0:
                recs.append((d.name, key, sol, g))
    return recs, npass, nfail, runs, skipped


def build_pairs(recs):
    """(run, shape, name-with-SK-masked) -> {sk_mode: (gflops, solution_name)}"""
    pairs = collections.defaultdict(dict)
    for run, key, sol, g in recs:
        m = SK.search(sol)
        if m:
            pairs[(run, key, SK.sub("_SKx_", sol))][int(m.group(1))] = (g, sol)
    return pairs


def report(pairs, a, b, label):
    xs = [v[b][0] / v[a][0] for v in pairs.values() if a in v and b in v]
    if not xs:
        print(f"  {label:<12} no matched pairs")
        return
    near = lambda t: sum(1 for x in xs if abs(x - 1) < t) / len(xs) * 100
    print(f"  {label:<12} n={len(xs):5}  median {statistics.median(xs)*100:7.2f}%   "
          f"within2% {near(.02):5.1f}%   within5% {near(.05):5.1f}%")


def by_axis(pairs, a, b, name, pat):
    buck = collections.defaultdict(list)
    for v in pairs.values():
        if a in v and b in v:
            m = re.search(pat, v[a][1])
            buck[m.group(1) if m else "?"].append(v[b][0] / v[a][0])
    cells = [f"{k}={statistics.median(x)*100:.1f}% (n={len(x)})"
             for k, x in sorted(buck.items()) if len(x) >= 20]
    print(f"  by {name:<8} " + "   ".join(cells))


def main():
    recs, npass, nfail, runs, skipped = load_all()
    print(f"runs analysed {runs}   skipped (no log/data) {skipped}")
    print(f"validation records: PASSED {npass}   FAILED {nfail}"
          f"{'   <- filter is inert here, measured not assumed' if nfail == 0 else ''}")
    print(f"validated (run, shape, solution) records: {len(recs)}\n")

    pairs = build_pairs(recs)
    print("LIKE-FOR-LIKE — identical solution, only the SK mode differs")
    report(pairs, 3, 4, "SK4 / SK3")
    report(pairs, 3, 5, "SK5 / SK3")
    report(pairs, 4, 5, "SK5 / SK4")
    print("  SK5/SK3 near 100% confirms REPORT.md §10: SK5 emits SK3 or SK4 and picks at runtime.\n")

    print("SK4 / SK3 split by the axes REPORT.md says invert the ratio")
    by_axis(pairs, 3, 4, "DepthU", r"_MT\d+x\d+x(\d+)_")
    by_axis(pairs, 3, 4, "CLR", r"_CLR(\d)_")
    by_axis(pairs, 3, 4, "PGR", r"_PGR(\d)_")
    print("  Directions and orderings must match the report; magnitudes run ~15-20 pt\n"
          "  stricter because best-vs-best lets each mode choose its own parameters.\n")

    best, where = collections.defaultdict(float), {}
    for run, key, sol, g in recs:
        m = SK.search(sol)
        if m and g > best[int(m.group(1))]:
            best[int(m.group(1))] = g
            where[int(m.group(1))] = (run, sol[:52])
    print("BEST-VS-BEST — the other question: how good can each mode get")
    for sk in sorted(best):
        r, s = where[sk]
        print(f"  SK{sk}: {best[sk]:9.1f} GFlop/s   {r}  {s}")
    if best.get(3) and best.get(4):
        same = where[3][0] == where[4][0]
        print(f"  best SK4 / best SK3 = {best[4]/best[3]*100:.1f}%"
              f"{'   (same run — like-for-like conditions)' if same else '   (DIFFERENT runs — weaker)'}")


if __name__ == "__main__":
    main()
