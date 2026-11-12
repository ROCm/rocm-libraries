#!/usr/bin/env python3
"""Per-instruction (stall+idle) per iteration in our main loop, in program order, with the WMMA index.
usage: loop_hot.py ours.csv [threshold]   (loop selection as in att_compare.py)"""
import re, sys, csv
def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["Instruction"] and not r["Instruction"].startswith(";")]
    for r in rows:
        for k in ("Vaddr", "Hitcount", "Latency", "Stall", "Idle"):
            r[k] = int(r[k])
    return rows
rows = load(sys.argv[1]); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 25
best = None
for r in rows:
    m = re.match(r"s_cbranch_scc0 (\d+)$", r["Instruction"])
    if m and int(m.group(1)) > 32768 and (best is None or r["Hitcount"] > best["Hitcount"]):
        best = r
end = best["Vaddr"]; norm = best["Hitcount"]
start = end + 4 + (int(re.search(r"\d+$", best["Instruction"]).group()) - 65536) * 4
loop = sorted([r for r in rows if start <= r["Vaddr"] <= end], key=lambda r: r["Vaddr"])
nw = 0; tot = 0
for r in loop:
    mn = r["Instruction"].split()[0]
    if "wmma" in mn: nw += 1
    lat = (r["Stall"] + r["Idle"]) / norm; tot += r["Latency"] / norm + r["Idle"] / norm
    if lat >= thr:
        print(f"wmma#{nw:3d} {r['Instruction'][:60]:60s} lat {r['Latency']/norm:7.0f} stall {r['Stall']/norm:7.0f} idle {r['Idle']/norm:7.0f}")
print(f"hits {norm}  total lat+idle per iter {tot:.0f}")
