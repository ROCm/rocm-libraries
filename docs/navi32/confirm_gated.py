#!/usr/bin/env python3
"""Compare two independent runs of the gated re-map arm.

Written BEFORE the second run finished, so the comparison is pre-committed rather than
chosen after seeing the numbers.

What this decides: run 1 gave +1.96% wall-clock against a 100.13% A/A floor. Per-stratum
resolution is ~±2.5%, so one run cannot separate a real 2% from drift. Three things settle it:

  1. Does the aggregate REPRODUCE? Two independent +2%s are worth far more than one.
  2. What is the A/A floor across TWO samples? One A/A draw is a point estimate of noise,
     not the noise itself.
  3. Do per-shape deltas CORRELATE between runs? If the same shapes win both times, the
     effect is structural. If not, it is noise that happened to average positive.

(3) is the strongest test and the one a single run cannot offer at all.
"""
import csv, collections, math, statistics, sys


def load(path):
    per = collections.defaultdict(dict); meta = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok": continue
        try: us, gf = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or gf <= 0: continue
        per[r["shape_id"]][r["arm"]] = (us, gf)
        meta[r["shape_id"]] = (int(r["M"]), int(r["N"]), int(r["K"]))
    return {s: v for s, v in per.items() if len(v) == 3}, meta


def stratum(M, N):
    lo, hi = min(M, N), max(M, N)
    if lo <= 8: return "gemv"
    if hi <= 128: return "tiny"
    if N * 4 <= M: return "skinny_N"
    if M * 4 <= N: return "skinny_M"
    if hi >= 4096: return "large"
    return "med"


def geo(v): return math.exp(sum(map(math.log, v)) / len(v))


A, metaA = load("/home/vmijovic/navi32/results/P3g_gated.csv")
B, metaB = load("/home/vmijovic/navi32/results/P3h_gated_rep2.csv")
common = sorted(set(A) & set(B))
print(f"  run1 {len(A)} shapes | run2 {len(B)} | common {len(common)}\n")
if len(common) < 30:
    print("  too few common shapes yet"); sys.exit()

print(f"  {'':<12}{'run 1':>10}{'run 2':>10}{'spread':>9}")
print("-" * 42)
for arm in ("gated", "lean_aa"):
    def wall(D):
        tb = sum(D[s]["lean"][0] for s in common); tc = sum(D[s][arm][0] for s in common)
        return 100 * tb / tc
    a, b = wall(A), wall(B)
    print(f"  {arm+' wall':<12}{a:>9.2f}%{b:>9.2f}%{abs(a-b):>8.2f}")
for arm in ("gated", "lean_aa"):
    a = 100 * geo([A[s][arm][1] / A[s]["lean"][1] for s in common])
    b = 100 * geo([B[s][arm][1] / B[s]["lean"][1] for s in common])
    print(f"  {arm+' geo':<12}{a:>9.2f}%{b:>9.2f}%{abs(a-b):>8.2f}")

# (3) the decisive test: do the same shapes win in both runs?
da = [A[s]["lean"][0] / A[s]["gated"][0] - 1 for s in common]
db = [B[s]["lean"][0] / B[s]["gated"][0] - 1 for s in common]
n = len(common)
ma, mb = statistics.mean(da), statistics.mean(db)
cov = sum((x - ma) * (y - mb) for x, y in zip(da, db)) / n
r = cov / (statistics.pstdev(da) * statistics.pstdev(db)) if statistics.pstdev(da) and statistics.pstdev(db) else float("nan")
agree = sum(1 for x, y in zip(da, db) if (x > 0) == (y > 0)) / n
print(f"\n  per-shape gain correlation between runs: r = {r:.3f}")
print(f"  same sign in both runs:                  {100*agree:.0f}% of shapes")
print("  (high r + high agreement = structural; near zero = noise averaging positive)")

# same test on the A/A arm: this is what pure noise looks like on this workload
aa = [A[s]["lean"][0] / A[s]["lean_aa"][0] - 1 for s in common]
bb = [B[s]["lean"][0] / B[s]["lean_aa"][0] - 1 for s in common]
ma2, mb2 = statistics.mean(aa), statistics.mean(bb)
cov2 = sum((x - ma2) * (y - mb2) for x, y in zip(aa, bb)) / n
r2 = cov2 / (statistics.pstdev(aa) * statistics.pstdev(bb)) if statistics.pstdev(aa) and statistics.pstdev(bb) else float("nan")
print(f"  same statistic on the A/A arm:           r = {r2:.3f}  <- the noise reference")

print(f"\n  {'stratum':<10}{'n':>4}{'run1':>9}{'run2':>9}{'spread':>9}")
print("-" * 42)
by = collections.defaultdict(list)
for s in common: by[stratum(*metaA[s][:2])].append(s)
for st in ("gemv", "tiny", "skinny_M", "skinny_N", "med", "large"):
    sh = by.get(st)
    if not sh: continue
    def w(D):
        tb = sum(D[s]["lean"][0] for s in sh); tc = sum(D[s]["gated"][0] for s in sh)
        return 100 * tb / tc
    a, b = w(A), w(B)
    print(f"  {st:<10}{len(sh):>4}{a:>8.2f}%{b:>8.2f}%{abs(a-b):>8.2f}")
