#!/usr/bin/env python3
"""Prefetch distance of ds_loads in an ATT main loop: WMMAs between a ds_load's issue and the
s_wait_dscnt that retires it (LDS returns in order), plus the stall at that wait.

usage: ds_distance.py ours.csv gemm_wt.csv     (loop selection as in att_compare.py)"""
import re, sys, csv, collections

class ac:
    @staticmethod
    def load(path):
        rows = [r for r in csv.DictReader(open(path)) if r["Instruction"] and not r["Instruction"].startswith(";")]
        for r in rows:
            for k in ("Vaddr", "Hitcount", "Latency", "Stall", "Idle"):
                r[k] = int(r[k])
        return rows

    @staticmethod
    def ours(path):
        rows = ac.load(path); best = None
        for r in rows:
            m = re.match(r"s_cbranch_scc0 (\d+)$", r["Instruction"])
            if m and int(m.group(1)) > 32768 and (best is None or r["Hitcount"] > best["Hitcount"]):
                best = r
        end = best["Vaddr"]
        start = end + 4 + (int(re.search(r"\d+$", best["Instruction"]).group()) - 65536) * 4
        return [r for r in rows if start <= r["Vaddr"] <= end], best["Hitcount"]

def analyse(name, loop, norm, per100):
    loop = sorted(loop, key=lambda r: r["Vaddr"])
    q = []                      # outstanding ds_loads: [issue_wmma_count, mnemonic]
    nw = 0
    rows = []                   # (distance_in_wmma, stall_per_iter_at_retiring_wait)
    waits = []
    for r in loop:
        ins = r["Instruction"]; mn = ins.split()[0]; h = max(r["Hitcount"], 1)
        if "wmma" in mn:
            nw += 1
        elif mn.startswith("ds_load"):
            q.append(nw)
        elif mn == "s_wait_dscnt":
            n = int(ins.split()[1], 0)
            retired = q[:max(0, len(q) - n)]; q = q[len(retired):]
            if retired:
                dist = [nw - i for i in retired]
                stall = r["Stall"] / norm
                waits.append((min(dist), sum(dist) / len(dist), len(retired), stall, r["Latency"] / norm))
    tot = sum(w[3] for w in waits)
    print(f"== {name}: {nw} WMMA in loop, {len(waits)} waits that retire loads, total wait stall {tot*per100:.0f} cyc per 100 WMMA")
    print("   min-dist(WMMAs)  avg-dist  loads  stall/iter   <- per wait, in program order")
    for w in waits:
        print(f"   {w[0]:6d}  {w[1]:8.1f}  {w[2]:5d}  {w[3]:9.1f}")
    byd = collections.defaultdict(float)
    for w in waits:
        byd[min(w[0], 8)] += w[3]
    print("   stall by min distance (8 = 8+):", {k: round(v * per100) for k, v in sorted(byd.items())})
    return waits

ours, hits = ac.ours(sys.argv[1])
nw_o = sum(1 for r in ours if "wmma" in r["Instruction"].split()[0])
analyse("ours", ours, hits, 100 / nw_o)
wt = [r for r in ac.load(sys.argv[2]) if r["Hitcount"] >= 2000]
nw_w = sum(1 for r in wt if "wmma" in r["Instruction"].split()[0])
analyse("gemm_wt", wt, 2048, 100 / nw_w)
