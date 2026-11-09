"""truemono-quick summary: python summarize.py <out dir> "<Ks>" <passes> "<arms>"

Per K and arm, over the passes (k<K>_<pass>_<arm>.json, single_request_bench):
  req    per-request mean decode tok/s (the headline metric: "tok/s per request")
  med    per-request MEDIAN decode tok/s (robust to neighbour bursts: a request slowed by another job's burst pulls
         the mean down; slow = share of requests below 0.85 x the run's median, printed per run)
  tw     token-weighted decode tok/s
  e2e    end-to-end tok/s (TTFT included)
  round  ms per verify round (K > 0), all requests' decode time / all rounds
  rnd_med  the median over requests of each request's ms per round (robust to neighbour bursts; K > 0)
  tok    ms per output token (K > 0: round / acceptance; K = 0: 1000 / tw)
  acc    acceptance length (K > 0)
  right  gsm8k correct / checked
Ratios to the `hybrid` arm paired by pass (same job, same GPU, same prompts): req, tw, e2e (> 1 faster), round, tok
(< 1 faster), acc. Server-log checks come from runs.txt (quick.sh's check_log line per run).
Writes <out>/abs.json (absolute; private) and <out>/ratios.json (ratios + checks), prints both (absolute lines "ABS").
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys

o, ks, npass, arms = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), sys.argv[4].split()
REF = "hybrid"


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def measure(k, p, a, sfx=""):
    d = jload(f"{o}/k{k}_{p}_{a}{sfx}.json")
    if not d:
        return None
    s = d.get("summary", {})
    tw, rnd, acc = s.get("decode_tok_s_token_weighted"), s.get("ms_per_round"), s.get("acceptance")
    tok = (rnd / acc if rnd and acc else None) if k != "0" else (1000 / tw if tw else None)
    rates = [r.get("decode_tok_s") for r in d.get("rows", []) if r.get("decode_tok_s")]
    rnds = [r.get("ms_per_round") for r in d.get("rows", []) if r.get("ms_per_round")]
    med = statistics.median(rates) if rates else None
    slow = sum(x < 0.85 * med for x in rates) / len(rates) if rates else None
    return {"req": s.get("decode_tok_s_per_request_mean"), "med": med, "slow": slow, "tw": tw, "e2e": s.get("end_to_end_tok_s"),
            "round": rnd if k != "0" else None, "rnd_med": statistics.median(rnds) if rnds and k != "0" else None, "tok": tok, "acc": acc if k != "0" else None,
            "correct": s.get("correct"), "checked": s.get("checked"), "at_cap": s.get("at_cap"),
            "out_tokens": s.get("mean_output_tokens"), "ttft_ms": s.get("mean_ttft_ms")}


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"per_pass": [round(x, 4) for x in xs], "mean": round(statistics.mean(xs), 4),
            "min": round(min(xs), 4), "max": round(max(xs), 4), "n": len(xs)}


checks = {}
if os.path.exists(f"{o}/runs.txt"):
    for ln in open(f"{o}/runs.txt"):
        m = re.match(r"(k\d+_\d+_\w+): .*checks ok=(\d)(.*)", ln.strip())
        if m:
            checks[m.group(1)] = {"ok": m.group(2) == "1", "line": m.group(3).strip()}

M = ["req", "med", "tw", "e2e", "round", "rnd_med", "tok", "acc"]
absout, ratout, lines, alines = {}, {}, [], []
KARMS = {}
for k in ks:
    f = f"{o}/arms_k{k}.txt"
    KARMS[k] = open(f).read().split() if os.path.exists(f) else arms
CELLS = open(f"{o}/cells.txt").read().split() if os.path.exists(f"{o}/cells.txt") else []
SPEED = ([("c1", "")] if not os.path.exists(f"{o}/speed_c1_off") else []) + [
    (c, "__" + c.replace(":", "-").replace(",", "-").replace("+", "-")) for c in CELLS if c.startswith("speed:")]
ACC = [(c, "__" + c.replace(":", "-").replace(",", "-").replace("+", "-")) for c in CELLS if not c.startswith("speed:")]


def acc_scores(k, p, a, spec, sfx):
    kind = spec.split(":")[0]
    base = f"{o}/k{k}_{p}_{a}{sfx}"
    if kind == "lmeval":
        fs = sorted(glob.glob(f"{base}.lmeval/**/results_*.json", recursive=True))
        if not fs:
            return None
        r = json.load(open(fs[-1])).get("results", {}).get("gsm8k", {})
        return {"flexible": r.get("exact_match,flexible-extract"), "strict": r.get("exact_match,strict-match")}
    if kind == "acc":
        d = jload(base + ".json")
        return {"accuracy": d.get("summary", {}).get("accuracy")} if d else None
    if kind == "ruler":
        fs = sorted(glob.glob(f"{base}.ruler/**/results_*.json", recursive=True))
        if not fs:
            return None
        out = {}
        for task, r in ((t, r) for f in fs for t, r in json.load(open(f)).get("results", {}).items()):
            for key, v in r.items():
                if isinstance(v, (int, float)) and v >= 0 and "stderr" not in key and key.split(",")[0].isdigit():
                    out.setdefault(key.split(",")[0], []).append(v)
        return {f"ruler{L}": statistics.mean(v) for L, v in sorted(out.items(), key=lambda kv: int(kv[0]))}
    return None


for k in ks:
    arms = KARMS[k]
    for cname, sfx in SPEED:
        tagk = f"K{k}" if cname == "c1" else f"K{k} {cname}"
        runs = {a: [measure(k, p, a, sfx) for p in range(1, npass + 1)] for a in arms}
        if not any(x for v in runs.values() for x in v):
            continue
        absout[tagk] = runs
        for a in arms:
            rs = runs[a]
            alines.append(f"ABS {tagk} {a:<10} " + "  ".join(
                f"{m}=" + "/".join(f"{x[m]:.{3 if m in ('round', 'rnd_med', 'tok', 'acc') else 1}f}" if x and x.get(m) is not None else "-" for x in rs)
                for m in M if not (k == "0" and m in ("round", "rnd_med", "acc")))
                + "  right=" + "/".join(f"{x['correct']}of{x['checked']}" if x else "-" for x in rs)
                + "  slow=" + "/".join(f"{x['slow']:.2f}" if x and x.get("slow") is not None else "-" for x in rs))
            mean = {m: stats([x[m] if x else None for x in rs]) for m in M}
            alines.append(f"ABS {tagk} {a:<10} mean " + "  ".join(f"{m}={mean[m]['mean']:.{3 if m in ('round', 'rnd_med', 'tok', 'acc') else 1}f}"
                                                                for m in M if mean[m]))
        if REF in arms:
            cell = {}
            for a in arms:
                if a == REF:
                    continue
                c = {}
                for m in M:
                    r = []
                    for p in range(npass):
                        x, y = runs[a][p], runs[REF][p]
                        r.append(x[m] / y[m] if x and y and x.get(m) and y.get(m) else None)
                    c[m] = stats(r)
                c["right"] = [x["correct"] if x else None for x in runs[a]]
                cell[a] = c
                lines.append(f"{tagk} {a:<10} / {REF}: " + "  ".join(
                    f"{m}={c[m]['mean']:.3f}[{c[m]['min']:.3f}-{c[m]['max']:.3f}]" for m in M if c.get(m)))
            cell[REF] = {"right": [x["correct"] if x else None for x in runs[REF]]}
            ratout[tagk] = cell
    for spec, sfx in ACC:
        tagk = f"K{k} {spec}"
        runs = {a: [acc_scores(k, p, a, spec, sfx) for p in range(1, npass + 1)] for a in arms}
        if not any(x for v in runs.values() for x in v):
            continue
        absout[tagk] = runs
        cell = {}
        for a in arms:
            keys = sorted({m for x in runs[a] if x for m in x})
            cell[a] = {m: [x.get(m) if x else None for x in runs[a]] for m in keys}
            lines.append(f"ACC {tagk} {a:<10} " + "  ".join(
                f"{m}=" + "/".join(f"{v:.4f}" if v is not None else "-" for v in vs) for m, vs in cell[a].items()))
        if REF in arms:
            for m in cell.get(REF, {}):
                ref = [v for v in cell[REF][m] if v is not None]
                spread = (max(ref) - min(ref)) if len(ref) > 1 else None
                for a in arms:
                    if a == REF:
                        continue
                    vs = [v for v in cell[a].get(m, []) if v is not None]
                    if vs and ref:
                        d = statistics.mean(vs) - statistics.mean(ref)
                        inside = (min(ref) - (spread or 0) <= min(vs) and max(vs) <= max(ref) + (spread or 0)) if spread is not None else None
                        lines.append(f"ACC {tagk} {a} - {REF} {m}: mean delta {d:+.4f}; {REF} A-vs-A spread {spread if spread is None else round(spread, 4)}; "
                                     f"{a} range [{min(vs):.4f}, {max(vs):.4f}] vs {REF} [{min(ref):.4f}, {max(ref):.4f}]"
                                     + ("" if inside is None else f"; inside spread-widened {REF} range: {inside}"))
        ratout[tagk] = cell
ratout["_checks"] = checks
ratout["_orders"] = [ln.strip() for ln in open(f"{o}/order.txt")] if os.path.exists(f"{o}/order.txt") else []
json.dump(absout, open(f"{o}/abs.json", "w"), indent=1)
json.dump(ratout, open(f"{o}/ratios.json", "w"), indent=1)
print(f"== ratios to {REF} (mean [min-max] over {npass} passes)")
print("\n".join(lines))
print("\n".join(alines))
for n, c in sorted(checks.items()):
    print(f"LOG {n}: ok={c['ok']} {c['line']}")
timed = [f"k{k}_{p}_{a}" for k in ks for p in range(1, npass + 1) for a in KARMS[k]]
bad = [t for t in timed if not checks.get(t, {}).get("ok")]
print("LOG CHECK FAILURES:", bad or "none")
sys.exit(1 if bad else 0)
