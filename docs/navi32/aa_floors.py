import csv, collections, math, os
H = os.path.expanduser("~/navi32/results")
def load(path, keep=None):
    per = collections.defaultdict(dict)
    if not os.path.exists(path): return per
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok": continue
        if keep and r["shape_id"] not in keep: continue
        try: us, g = float(r["us"]), float(r["gflops"])
        except ValueError: continue
        if us <= 0 or g <= 0: continue
        per[r["shape_id"]].setdefault(r["arm"], []).append((us, g))
    return per
def wall(per, base, cand):
    tb = tc = 0.0; n = 0
    for sid, d in per.items():
        if base not in d or cand not in d: continue
        tb += min(x[0] for x in d[base]); tc += min(x[0] for x in d[cand]); n += 1
    return (n, 100*tb/tc) if n else (0, float("nan"))
ROWS = [("HHS","P6_main.csv","navi32ship","navi32ship_aa","P12_masked60.csv"),
        ("BBS","P9_bbs.csv","bbs_ship","bbs_ship_aa","P15_bbs_m60.csv"),
        ("AuxH","P10_aux.csv","aux_ship","aux_ship_aa","P16_auxh_m60.csv"),
        ("AuxB","P11_auxb.csv","auxb_ship","auxb_ship_aa","P17_auxb_m60.csv")]
print(f"{'PT':<6}{'96-CU A/A':>11}{'60-CU A/A':>11}{'observed shift':>16}{'  drift explains it?'}")
print("-" * 66)
for lab, c96, b96, a96, c60 in ROWS:
    p60 = load(f"{H}/{c60}"); keep = set(p60)
    p96 = load(f"{H}/{c96}", keep)
    _, w96 = wall(p96, b96, a96)
    _, w60 = wall(p60, "ship", "ship_aa")
    print(f"{lab:<6}{w96:>10.2f}%{w60:>10.2f}%")
