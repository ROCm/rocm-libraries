#!/usr/bin/env python3
"""Static occupancy exposure of the shipped navi32 catalog on navi33 (gfx1102).

navi33 has a 1024-VGPR/SIMD file vs 1536 on navi31/navi32. RDNA3 wave32 occupancy
is waves/SIMD = min(MaxWavesPerSimd, vgprPerSimd // align(vgprs)), taken from
rocisa/hardware_caps.hpp (PhysicalMaxVgprCU = 2*vgprPerSimd*32; MaxWavesPerSimd=16
for gfx11). NOTE Tensile/OccupancyMeasure.py is gfx9/wave64-specific and returns
None for RDNA3 -- it must NOT be used here.

BOUNDS EXPOSURE, DOES NOT PREDICT PERFORMANCE: a kernel loses throughput from
lower occupancy only if it was latency-bound. This says how much of the catalog
is even *at risk*, which is the part that can be settled without navi33 silicon.
"""
import csv, re, subprocess, collections
from pathlib import Path

MAX_WAVES, GRAN = 16, 8
H = Path.home() / "navi32"

def waves(v, per_simd):
    return max(1, min(MAX_WAVES, per_simd // (-(-v // GRAN) * GRAN)))

def toks(s):
    out = []
    for p in s.split("_"):
        if out and re.fullmatch(r"[0-9]+", p): out[-1] += "_" + p
        else: out.append(p)
    return set(out)

def co_map(lib):
    co = next((H / f"libs/{lib}/library/gfx1100").glob("TensileLibrary*.co"))
    txt = subprocess.run(["llvm-readelf", "--notes", str(co)],
                         capture_output=True, text=True).stdout
    m, sym = [], None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith(".symbol:"): sym = s.split(":", 1)[1].strip().removesuffix(".kd")
        elif s.startswith(".vgpr_count:") and sym:
            m.append((toks(sym), int(s.split(":", 1)[1]))); sym = None
    return m

ARMS = [("HHS", "wgm8", "P6_main", "gridcat"), ("BBS", "bbs_wide", "P9_bbs", "bbs_wide"),
        ("AuxH", "aux_wide", "P10_aux", "aux_wide"), ("AuxB", "auxb_wide", "P11_auxb", "auxb_wide")]

print(f"{'PT':<6}{'kern':>6}{'shapes':>8}{'matched':>9}{'% TIME at risk':>16}{'% SHAPES at risk':>18}")
print("-" * 63)
tot_t = tot_lost = 0.0; agg = collections.Counter(); vdist = collections.Counter()
for pt, lib, csvf, arm in ARMS:
    cm = co_map(lib); cache = {}
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(H / f"results/{csvf}.csv")):
        if r["arm"] != arm or r.get("status") != "ok": continue
        try: per[r["shape_id"]].append((float(r["us"]), r["kernel"]))
        except ValueError: pass
    t_all = t_lost = 0.0; n = nlost = 0; miss = 0
    for sid, reps in per.items():
        us, k = min(reps, key=lambda x: x[0])
        if k not in cache:
            hit = [v for t, v in cm if t <= toks(k)]
            cache[k] = hit[0] if len(hit) == 1 else None
        v = cache[k]
        if v is None: miss += 1; continue
        w31, w33 = waves(v, 1536), waves(v, 1024)
        t_all += us; n += 1; vdist[v] += 1
        if w33 < w31:
            t_lost += us; nlost += 1; agg[(v, w31, w33)] += 1
    # GUARD: an empty join must never print as 0%
    assert n > 0, f"{pt}: ZERO kernels matched -- join is broken, not a 0% result"
    if miss: print(f"  ({pt}: {miss} shapes unmatched, excluded)")
    print(f"{pt:<6}{len(cm):>6}{len(per):>8}{n:>9}{100*t_lost/t_all:>15.1f}%{100*nlost/n:>17.1f}%")
    tot_t += t_all; tot_lost += t_lost
print("-" * 63)
print(f"{'ALL':<6}{'':>6}{'':>8}{'':>9}{100*tot_lost/tot_t:>15.1f}%{'':>18}")

print("\nWhere the loss lands (VGPRs -> waves/SIMD, navi32 -> navi33):")
for (v, a, b), c in sorted(agg.items(), key=lambda x: -x[1]):
    print(f"  {v:>4} VGPR: {a:>2} -> {b:>2} waves ({100*(a-b)/a:>3.0f}% fewer)  {c:>5} shapes")
safe = sum(c for v, c in vdist.items() if waves(v,1536) == waves(v,1024))
print(f"\nUnaffected (same occupancy on both): {safe} shapes"
      f" -- these are kernels at <=64 VGPR, already at the 16-wave cap.")
