#!/usr/bin/env python3
"""Parse a Tensile benchmark result set into per-arm, per-band comparisons.

Two sources, both required:
  * 2_BenchmarkData/*.csv  -- WIDE: one row per shape, one COLUMN per solution, GFlops
  * the run log            -- per (solution, shape) `validation` = PASSED/FAILED

Validation is NOT optional. An arm that assembles and computes garbage would look like a
win in the GFlops table alone, so any (solution, shape) whose validation is not PASSED is
excluded from the numbers and reported separately.

Bands come from the CENSUS kernel time of each shape, not from this run, so the banding is
identical across phases and cannot drift with the thing being measured.
"""
from __future__ import annotations
import csv, json, re, sys, math, statistics, pathlib, collections

CENSUS_SHAPES = pathlib.Path("/home/vmijovic/sk_modes/artifacts/shapes.json")


def band(ms: float) -> str:
    return "<0.1ms" if ms < 0.1 else "0.1-1ms" if ms < 1 else ">=1ms"


def decode(name: str) -> dict:
    """Pull the axes we fork on out of a Tensile kernel name."""
    def g(pat, default=None):
        m = re.search(pat, name)
        return m.group(1) if m else default
    return {
        "MT":     g(r"_MT(\d+x\d+x\d+)_"),
        "SK":     g(r"_SK(\d)_"),
        "SKFTR":  g(r"_SKFTR(\d)_"),
        "SKXCCM": g(r"_SKXCCM(\d)_"),
        "SKA":    g(r"_SKA(\d)_"),
    }


def load(results_dir: pathlib.Path, log: pathlib.Path):
    wide = next((results_dir / "2_BenchmarkData").glob("*.csv"))
    rows = list(csv.reader(wide.open()))
    hdr = [h.strip() for h in rows[0]]
    sol_cols = {i: h for i, h in enumerate(hdr) if h.startswith("Cijk_")}
    # shape key -> (M,N,K)
    iI, iJ, iL = hdr.index("SizeI"), hdr.index("SizeJ"), hdr.index("SizeL")

    perf = {}                       # (sol, (M,N,K)) -> gflops
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        key = (int(r[iI]), int(r[iJ]), int(r[iL]))
        for i, sol in sol_cols.items():
            try:
                perf[(sol, key)] = float(r[i])
            except (ValueError, IndexError):
                pass

    # Validation from the log: problem-sizes + PASSED/FAILED, matched to a KNOWN solution.
    #
    # Do NOT regex the solution name out of the line. Every client row begins
    # "Contraction_l_Alik_Bljk_Cijk_Dijk", which contains the substring "Cijk_Dijk" -- a
    # naive r"(Cijk_\w+)" search matches THAT first and silently collapses every record
    # onto one fake solution (192 -> 24). Match against the CSV's solution list instead.
    known = sorted(sol_cols.values(), key=len, reverse=True)
    valid = {}
    for line in log.read_text(errors="ignore").splitlines():
        if "Contraction_" not in line or ("PASSED" not in line and "FAILED" not in line):
            continue
        ms = re.search(r'"\((\d+),(\d+),\d+,(\d+)\)"', line)
        if not ms:
            continue
        sol = next((s for s in known if s in line), None)
        if sol is None:
            continue
        key = (int(ms.group(1)), int(ms.group(2)), int(ms.group(3)))
        valid[(sol, key)] = "PASSED" if "PASSED" in line else "FAILED"
    return perf, valid, list(sol_cols.values())


def geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def main(results_dir, log_path, axis, title):
    results_dir, log_path = pathlib.Path(results_dir), pathlib.Path(log_path)
    perf, valid, sols = load(results_dir, log_path)
    shapes = {(s["M"], s["N"], s["K"]): s for s in json.load(CENSUS_SHAPES.open())}

    bad = [(s, k) for (s, k), v in valid.items() if v != "PASSED"]
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"solutions: {len(sols)}   shapes: {len(shapes)}   "
          f"validation records: {len(valid)}   FAILED: {len(bad)}")
    if bad:
        print("  !! VALIDATION FAILURES (excluded from all numbers below):")
        for s, k in bad[:10]:
            print(f"     {decode(s)}  {k}")

    # group solutions by everything except the axis under test
    groups = collections.defaultdict(dict)
    for s in sols:
        d = decode(s)
        base = tuple((k, v) for k, v in sorted(d.items()) if k != axis)
        groups[base][d[axis]] = s

    for base, arms in sorted(groups.items()):
        if len(arms) < 2:
            continue
        lvls = sorted(arms)
        ref = lvls[0]
        basestr = " ".join(f"{k}={v}" for k, v in base if v is not None)
        print(f"\n--- {basestr}   [{axis}: {' vs '.join(lvls)}]")
        print(f"    {'band':<10} {'n':>3}  " +
              "  ".join(f"{axis}={l:<4}".rjust(12) for l in lvls) + "   ratio")
        for b in ("<0.1ms", "0.1-1ms", ">=1ms", "ALL"):
            keys = [k for k, s in shapes.items() if b == "ALL" or band(s["ms"]) == b]
            per = {}
            for l in lvls:
                sol = arms[l]
                vals = [perf[(sol, k)] for k in keys
                        if (sol, k) in perf and valid.get((sol, k)) == "PASSED"
                        and perf[(sol, k)] > 0]
                per[l] = vals
            n = min((len(v) for v in per.values()), default=0)
            if n == 0:
                continue
            gms = {l: geomean(per[l]) for l in lvls}
            ratio = gms[lvls[-1]] / gms[ref] if gms[ref] > 0 else float("nan")
            print(f"    {b:<10} {n:>3}  " +
                  "  ".join(f"{gms[l]:>10.0f}  " for l in lvls) +
                  f"  {100*ratio:6.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         sys.argv[4] if len(sys.argv) > 4 else "results")
