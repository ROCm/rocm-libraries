#!/usr/bin/env python3
"""Per-iteration main-loop breakdown from rocprofv3 ATT stats CSVs.

usage: att_loop.py name=stats.csv [name=stats.csv ...]

The loop is the Vaddr range of the backward s_cbranch_scc0 with the most hits. Cycles per
iteration per wave = (Latency + Idle) summed over the loop's instructions / back-branch hits
(idle is time the wave could not issue at all, so latency alone undercounts).
"""
import csv, re, sys
from collections import OrderedDict

def category(ins):
    rules = [("v_wmma", "WMMA"), ("ds_load", "ds_load"), ("s_wait_dscnt", "s_wait_dscnt"),
             ("s_wait_tensorcnt", "s_wait_tensorcnt"), ("tensor_load_to_lds", "tensor_load"),
             ("s_barrier_wait 0xfffd", "cluster wait"), ("s_barrier_signal -3", "cluster signal"),
             ("s_barrier_wait", "WG barrier wait"), ("s_barrier_signal", "WG barrier signal"),
             ("s_set_vgpr_msb", "s_set_vgpr_msb"), ("s_wait_alu", "s_wait_alu"), ("s_nop", "s_nop"),
             ("v_", "other VALU"), ("s_", "other SALU")]
    for p, c in rules:
        if ins.startswith(p):
            return c
    return "other"

def loop(path):
    rows = [r for r in csv.DictReader(open(path)) if r["Instruction"] and not r["Instruction"].startswith(";")]
    best = None
    for r in rows:
        m = re.match(r"s_cbranch_scc0 (\d+)$", r["Instruction"])
        if m and int(m.group(1)) > 32768 and (best is None or int(r["Hitcount"]) > int(best["Hitcount"])):
            best = r
    end = int(best["Vaddr"])
    start = end + 4 + (int(re.search(r"\d+$", best["Instruction"]).group()) - 65536) * 4
    iters = int(best["Hitcount"])
    body = [r for r in rows if start <= int(r["Vaddr"]) <= end]
    cats = OrderedDict()
    for r in body:
        c = category(r["Instruction"])
        cats[c] = cats.get(c, 0) + int(r["Latency"]) + int(r["Idle"])
    total = sum(cats.values())
    return iters, total / iters, {c: v / iters for c, v in cats.items()}, len(body)

res = {}
for a in sys.argv[1:]:
    n, p = a.split("=", 1)
    res[n] = loop(p)
names = list(res)
allCats = []
for n in names:
    for c in res[n][2]:
        if c not in allCats:
            allCats.append(c)
allCats.sort(key=lambda c: -max(res[n][2].get(c, 0) for n in names))
print(f"{'cycles/iter/wave':22s}" + "".join(f"{n:>14s}" for n in names))
print(f"{'loop instructions':22s}" + "".join(f"{res[n][3]:>14d}" for n in names))
print(f"{'back-branch hits':22s}" + "".join(f"{res[n][0]:>14d}" for n in names))
for c in allCats:
    print(f"{c:22s}" + "".join(f"{res[n][2].get(c, 0):>14.0f}" for n in names))
print(f"{'TOTAL':22s}" + "".join(f"{res[n][1]:>14.0f}" for n in names))
