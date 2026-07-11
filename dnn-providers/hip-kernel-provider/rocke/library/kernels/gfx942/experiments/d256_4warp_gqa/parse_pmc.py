import csv, sys, glob
csvf = sorted(glob.glob(sys.argv[1]+"/**/*counter_collection.csv", recursive=True))
if not csvf:
    print("NO CSV"); sys.exit(0)
rows = list(csv.DictReader(open(csvf[-1])))
def side(kn):
    kn = kn or ""
    if "unified_attention" in kn: return "AITER"
    if ("rocke" in kn or "gqa" in kn or "uattn" in kn): return "OURS"
    return None
agg = {}
for r in rows:
    s = side(r.get("Kernel_Name") or r.get("Dispatch_Id") or "")
    if not s: continue
    c = r.get("Counter_Name"); v = r.get("Counter_Value")
    if c and v:
        try: agg.setdefault(s, {}).setdefault(c, 0.0); agg[s][c] += float(v)
        except: pass
o, a = agg.get("OURS", {}), agg.get("AITER", {})
ctrs = sorted(set(o) | set(a))
print(f"{'counter':28} {'OURS':>12} {'AITER':>12} {'ratio':>7}")
for c in ctrs:
    ov, av = o.get(c, 0.0), a.get(c, 0.0)
    rt = ov/av if av else 0.0
    print(f"{c:28} {ov:12.3e} {av:12.3e} {rt:7.2f}")
