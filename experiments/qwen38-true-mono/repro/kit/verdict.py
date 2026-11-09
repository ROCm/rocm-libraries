"""PASS / FAIL of a quick-harness run against the expected ratio bands (REPRO.md "Expected ratios").

    python verdict.py <out>/q/ratios.json

Reads summarize.py's ratios.json (per-pass ratios to the same-session `hybrid`; `med` = median decode tok/s per
request, `rnd_med` = median ms per verify round) and checks the means over passes. Ratios only.
"""
import json
import sys

BANDS = {  # (K, arm, metric): (lo, hi)
    ("K3", "c2s3b", "med"): (1.11, 1.15),
    ("K3", "c2s3b", "rnd_med"): (0.88, 0.895),
    ("K0", "c2s0b", "med"): (1.12, 1.15),
}
TWINS = {"K3": [("hybrid_b", None), ("c2s3b_b", "c2s3b")], "K0": [("hybrid_b", None), ("c2s0b_b", "c2s0b")]}
TOL = {"med": 0.015, "rnd_med": 0.005}

r = json.load(open(sys.argv[1]))
bad = 0


def mean(k, a, m):
    c = r.get(k, {}).get(a, {}).get(m)
    return c["mean"] if c else None


def say(ok, what):
    global bad
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {what}")


for (k, a, m), (lo, hi) in BANDS.items():
    if k not in r or a not in r[k]:
        continue
    v = mean(k, a, m)
    c = r[k][a][m]
    say(v is not None and lo <= v <= hi,
        f"{k} {a} / hybrid {m}: {v:.3f} [{c['min']:.3f}-{c['max']:.3f}] over {c['n']} passes, band {lo}-{hi}")
for k, pairs in TWINS.items():
    for twin, base in pairs:
        if k not in r or twin not in r[k]:
            continue
        for m in ("med", "rnd_med") if k == "K3" else ("med",):
            t = mean(k, twin, m)
            b = 1.0 if base is None else mean(k, base, m)
            if t is None or b is None:
                continue
            say(abs(t / b - 1) <= TOL[m], f"{k} twin {twin} / {base or 'hybrid'} {m}: {t / b:.3f} (within +-{TOL[m]:.1%})")
chk = r.get("_checks", {})
nbad = [n for n, c in chk.items() if not c.get("ok")]
say(not nbad, f"server-log checks of {len(chk)} arm-runs" + (f": failed {nbad}" if nbad else ""))
print("VERDICT", "PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
