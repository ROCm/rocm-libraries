#!/usr/bin/env python3
"""Phase 0 gate: does cold measurement change kernel RANKINGS, or only absolute throughput?

All prior measurement in this work was warm-cache (hipblaslt-bench defaults --rotating 0, no
--flush; the TUNING environment defaults to rotating=512/flush=true). Warm vs cold measured 3.1x
on a 512^3 shape.

That gap is harmless for an A/B of two catalogs -- it is common-mode. It is NOT harmless for a
re-map, which is about picking WHICH kernel wins. So before spending a cold sweep:

  rankings stable  -> the existing warm matrix can still pick winners; cold only for validation
  rankings shift   -> full cold re-sweep needed, AND the already-shipped lean representatives
                      for small shapes were chosen warm and become suspect

Reports top-1 agreement, top-5 overlap and Spearman, broken out by size band, since cache
residency should bite hardest where the working set fits.
"""
import collections, json, math, statistics, sys


def load(path):
    d = {}
    for line in open(path):
        try: m = json.loads(line)
        except Exception: continue
        gf = {int(k): v for k, v in m["gf"].items() if v > 0}
        if gf: d[(m["M"], m["N"], m["B"], m["K"])] = (m["stratum"], gf)
    return d


def spearman(a, b, keys):
    ra = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -a[x]))}
    rb = {k: i for i, k in enumerate(sorted(keys, key=lambda x: -b[x]))}
    n = len(keys)
    if n < 3: return float("nan")
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


warm = load("/home/vmijovic/navi32/results/P0_matrix.jsonl")
cold = load("/home/vmijovic/navi32/results/P0c_cold_check.jsonl")
common = sorted(set(warm) & set(cold))
print(f"shapes measured both warm and cold: {len(common)}\n")

rows = collections.defaultdict(list)
for key in common:
    st, gw = warm[key]; _, gc = cold[key]
    ks = [k for k in gw if k in gc]
    if len(ks) < 10: continue
    tw = max(ks, key=lambda k: gw[k]); tc = max(ks, key=lambda k: gc[k])
    top5w = {k for k in sorted(ks, key=lambda x: -gw[x])[:5]}
    top5c = {k for k in sorted(ks, key=lambda x: -gc[x])[:5]}
    # what does picking the WARM winner cost, judged by COLD measurement?
    regret = gc[tc] / gc[tw] - 1.0
    rows[st].append(dict(top1=tw == tc, top5=len(top5w & top5c),
                         rho=spearman(gw, gc, ks), regret=regret,
                         ratio=statistics.median(gc[k] / gw[k] for k in ks)))

print(f"{'stratum':<10}{'n':>4}{'top-1 same':>12}{'top-5 overlap':>15}{'Spearman':>11}"
      f"{'cold/warm':>11}{'regret of warm pick':>21}")
print("-" * 84)
allr = []
for st in ("gemv", "tiny", "skinny_M", "skinny_N", "med", "large"):
    v = rows.get(st)
    if not v: continue
    allr += v
    print(f"{st:<10}{len(v):>4}{100*sum(x['top1'] for x in v)/len(v):>11.0f}%"
          f"{statistics.mean(x['top5'] for x in v):>14.1f}/5"
          f"{statistics.mean(x['rho'] for x in v):>11.3f}"
          f"{statistics.median(x['ratio'] for x in v):>11.3f}"
          f"{100*statistics.median(x['regret'] for x in v):>20.1f}%")
print("-" * 84)
print(f"{'ALL':<10}{len(allr):>4}{100*sum(x['top1'] for x in allr)/len(allr):>11.0f}%"
      f"{statistics.mean(x['top5'] for x in allr):>14.1f}/5"
      f"{statistics.mean(x['rho'] for x in allr):>11.3f}"
      f"{statistics.median(x['ratio'] for x in allr):>11.3f}"
      f"{100*statistics.median(x['regret'] for x in allr):>20.1f}%")

t1 = 100 * sum(x["top1"] for x in allr) / len(allr)
print(f"\nGATE: top-1 agreement {t1:.0f}%")
print("  >=90% and no size drift -> warm matrix usable for selection; cold for validation only")
print("  <90% or size-dependent  -> full cold re-sweep, and re-check lean's small-shape reps")
print("\n'regret of warm pick' = how much throughput the WARM-chosen kernel gives up, judged")
print("cold. That is the number that matters: it is the cost of selecting on warm data.")
