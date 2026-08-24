"""RDNA3 wave32 occupancy from a .co, honouring BOTH the VGPR and LDS limits.

waves/SIMD from VGPR = min(MaxWavesPerSimd=16, vgprPerSimd // align8(vgpr))
WGs/CU from LDS      = DeviceLDS(65536) // group_segment_fixed_size
Effective WGs/CU     = max(1, min(both));  waves/CU = WGs/CU * waves_per_wg
"""
import subprocess, re
from pathlib import Path
DEVICE_LDS, MAX_W, GRAN = 65536, 16, 8

def parse(co):
    txt = subprocess.run(["llvm-readelf", "--notes", str(co)],
                         capture_output=True, text=True).stdout
    ks, cur = [], None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("- .args"):
            if cur and len(cur) == 4: ks.append(cur)
            cur = {}
        if cur is None: continue
        for key, f in ((".group_segment_fixed_size:", "lds"), (".vgpr_count:", "v"),
                       (".max_flat_workgroup_size:", "thr"), (".symbol:", "sym")):
            if s.startswith(key):
                val = s.split(":", 1)[1].strip()
                cur[f] = val.removesuffix(".kd") if f == "sym" else int(val)
    if cur and len(cur) == 4: ks.append(cur)
    return ks

def occupancy(k, vgpr_per_simd):
    """Return (wgs_per_cu, waves_per_cu, limiter)."""
    wpw = max(1, -(-k["thr"] // 32))
    vw_simd = min(MAX_W, vgpr_per_simd // (-(-max(k["v"],1) // GRAN) * GRAN))
    vgpr_wgs = max(1, (2 * vw_simd) // wpw)
    lds_r = ((k["lds"] + 255) // 256) * 256   # Tensile rounds LDS to 256 B
    lds_wgs = (DEVICE_LDS // lds_r) if lds_r > 0 else 10**6
    wgs = max(1, min(vgpr_wgs, lds_wgs))
    lim = "lds" if lds_wgs < vgpr_wgs else ("vgpr" if vgpr_wgs < lds_wgs else "tie")
    return wgs, wgs * wpw, lim

def toks(s):
    out = []
    for p in s.split("_"):
        if out and re.fullmatch(r"[0-9]+", p): out[-1] += "_" + p
        else: out.append(p)
    return set(out)
