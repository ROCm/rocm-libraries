#!/usr/bin/env python3
"""Does the catalog win survive genuine 60-CU EXECUTION?

The campaign headline (+23.9% wall-clock) was measured with 60-CU *selection*
(--sm_count_target 60) but 96-CU *execution*, because the CU-masked stream was
believed to hang ~37% of runs. Re-measured it hangs ~4-8%, so the masked run is
affordable and the assumption is testable.

Compares the SAME shapes under both regimes:
  P6_main.csv    navi32ship vs gridcat   96-CU execution   (already measured)
  P12_masked60   ship       vs wide      60-CU execution   (this run)
If the ratios agree within the A/A floor, the emulation shortcut was sound.
"""
import csv, collections, math, statistics, sys

def load(path, arms):
    per = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok" or r["arm"] not in arms: continue
        try: us, g = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or g <= 0: continue
        d = per[r["shape_id"]].setdefault(r["arm"], [])
        d.append((us, g))
    return per

def geo(v): return math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")

def summarize(per, base, cand, label):
    """geomean of per-shape gflops ratio, and flops-weighted wall-clock ratio."""
    ratios, t_base, t_cand = [], 0.0, 0.0
    for sid, d in per.items():
        if base not in d or cand not in d: continue
        ub = min(x[0] for x in d[base]); uc = min(x[0] for x in d[cand])
        gb = max(x[1] for x in d[base]); gc = max(x[1] for x in d[cand])
        ratios.append(gc / gb); t_base += ub; t_cand += uc
    if not ratios: return None
    return dict(n=len(ratios), geomean=100*geo(ratios), wall=100*t_base/t_cand, label=label)

MASK = sys.argv[1] if len(sys.argv) > 1 else "/home/vmijovic/navi32/results/P12_masked60.csv"
UNMASK = "/home/vmijovic/navi32/results/P6_main.csv"

m = load(MASK, {"ship", "wide", "ship_aa"})
u = load(UNMASK, {"navi32ship", "gridcat", "navi32ship_aa"})
common = set(m) & set(u)
m = {k: v for k, v in m.items() if k in common}
u = {k: v for k, v in u.items() if k in common}
print(f"shapes common to both regimes: {len(common)}\n")

rows = [summarize(u, "navi32ship", "gridcat",   "96-CU exec (P6, shipped result)"),
        summarize(m, "ship",       "wide",      "60-CU exec (P12, this run)"),
        summarize(u, "navi32ship", "navi32ship_aa", "  A/A floor, 96-CU"),
        summarize(m, "ship",       "ship_aa",       "  A/A floor, 60-CU")]
print(f"{'regime':<34}{'n':>5}{'geomean':>10}{'wall-clock':>12}")
print("-" * 61)
for r in rows:
    if r: print(f"{r['label']:<34}{r['n']:>5}{r['geomean']:>9.1f}%{r['wall']:>11.1f}%")
if rows[0] and rows[1]:
    dg, dw = rows[1]["geomean"] - rows[0]["geomean"], rows[1]["wall"] - rows[0]["wall"]
    floor = max(abs(rows[2]["wall"] - 100) if rows[2] else 0,
                abs(rows[3]["wall"] - 100) if rows[3] else 0)
    print(f"\n60-CU minus 96-CU:  geomean {dg:+.1f} pt   wall-clock {dw:+.1f} pt")
    print(f"A/A floor (max |dev| from 100): {floor:.2f} pt")
    print("VERDICT:", "ratios agree within the floor -- the emulation shortcut was sound"
          if abs(dw) <= max(floor, 1.0) else
          f"wall-clock ratio MOVES by {dw:+.1f} pt -- 96-CU execution was NOT a safe proxy")
