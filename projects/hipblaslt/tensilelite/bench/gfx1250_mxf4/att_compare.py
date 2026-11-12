#!/usr/bin/env python3
"""Per-slice (100-WMMA) main-loop breakdown for ours vs gemm_wt, same method for both.

usage: att_compare.py ours.csv ours_wmma_per_iter gemm_wt.csv
Values in the stats CSVs are sums over hits. Ours: loop = backward s_cbranch_scc0 with most hits,
normalised by its hits. gemm_wt: loop = rows with Hitcount >= 2000 (16 waves x 128 iterations, body = 4 slices).
Everything is reported per slice of 100 WMMAs: cycles = Latency/Stall/Idle per instruction mnemonic.
"""
import csv, re, sys, collections

def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["Instruction"] and not r["Instruction"].startswith(";")]
    for r in rows:
        for k in ("Vaddr", "Hitcount", "Latency", "Stall", "Idle"):
            r[k] = int(r[k])
    return rows

def ours(path):
    rows = load(path)
    best = None
    for r in rows:
        m = re.match(r"s_cbranch_scc0 (\d+)$", r["Instruction"])
        if m and int(m.group(1)) > 32768 and (best is None or r["Hitcount"] > best["Hitcount"]):
            best = r
    end = best["Vaddr"]
    start = end + 4 + (int(re.search(r"\d+$", best["Instruction"]).group()) - 65536) * 4
    return [r for r in rows if start <= r["Vaddr"] <= end], best["Hitcount"]

def table(loop, n, wmma_per_body, slices_per_body):
    cat = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in loop:
        m = r["Instruction"].split()[0]
        if m.startswith("s_barrier_wait"): m = "s_barrier_wait"
        c = cat[m]; c[0] += 1; c[1] += r["Latency"] / n / slices_per_body
        c[2] += r["Stall"] / n / slices_per_body; c[3] += r["Idle"] / n / slices_per_body
    tot = [sum(c[i] for c in cat.values()) for i in (1, 2, 3)]
    return cat, tot

o_loop, o_hits = ours(sys.argv[1])
o_wmma = sum(1 for r in o_loop if r["Instruction"].startswith("v_wmma"))
o_slices = o_wmma / 100.0
g_rows = load(sys.argv[3]); g_loop = [r for r in g_rows if r["Hitcount"] >= 2000]
g_wmma = sum(1 for r in g_loop if r["Instruction"].startswith("v_wmma"))
oc, ot = table(o_loop, o_hits, o_wmma, o_slices)
gc, gt = table(g_loop, 2048, g_wmma, g_wmma / 100.0)
print(f"ours: {o_wmma} WMMA/body, hits {o_hits}; gemm_wt: {g_wmma} WMMA/body (per 100-WMMA slice below)\n")
print(f"{'mnemonic':32s} | {'ours n':>6s} {'lat':>6s} {'stall':>6s} {'idle':>6s} | {'wt n':>5s} {'lat':>6s} {'stall':>6s} {'idle':>6s}")
keys = sorted(set(oc) | set(gc), key=lambda k: -(oc.get(k, [0, 0])[1] + gc.get(k, [0, 0])[1]))
for k in keys[:16]:
    a = oc.get(k, [0, 0, 0, 0]); b = gc.get(k, [0, 0, 0, 0])
    print(f"{k:32s} | {a[0]/o_slices:6.0f} {a[1]:6.0f} {a[2]:6.0f} {a[3]:6.0f} | {b[0]/(g_wmma/100):5.0f} {b[1]:6.0f} {b[2]:6.0f} {b[3]:6.0f}")
print(f"{'TOTAL per slice':32s} | {'':6s} {ot[0]:6.0f} {ot[1]:6.0f} {ot[2]:6.0f} | {'':5s} {gt[0]:6.0f} {gt[1]:6.0f} {gt[2]:6.0f}")
print(f"\nlatency+idle per slice: ours {ot[0]+ot[2]:.0f}, gemm_wt {gt[0]+gt[2]:.0f}")
