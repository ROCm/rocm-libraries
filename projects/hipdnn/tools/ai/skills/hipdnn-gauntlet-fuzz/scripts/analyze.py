#!/usr/bin/env python3
"""Compare gauntlet fuzz work dirs of a BASE build and a PR build.

  analyze.py --root <dir holding the work dirs> --base <name> --pr <name> [--pr-only]

Reads <root>/<name>/{outcomes.jsonl,samples.jsonl,summary.json,failures/}. Prints, per space:
outcome counts, coverage flags, base->pr transitions and the WRONG/CRASH list with axes.

The one rule that matters: in PAIRWISE mode only the primary rows (case ids below the
space's `rows` count) are the same graph in both runs. Repair rows are built from the
outcomes of the run, so ids above that differ between base and pr. This script compares
primary rows only for pairwise runs and every row for random runs (same --seed, same graphs).
Run it where the work dirs are; it needs only python3.
"""
import argparse, collections, json, os, re, sys


def jl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load(root, name):
    d = os.path.join(root, name)
    if not os.path.exists(os.path.join(d, "outcomes.jsonl")):
        return None
    out = {(r["space"], r["case"]): r for r in jl(os.path.join(d, "outcomes.jsonl"))}
    axes = {(r["space"], r["case"]): r["axes"] for r in jl(os.path.join(d, "samples.jsonl"))}
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    return {"dir": d, "o": out, "a": axes, "s": summary}


def log_tail(run, space, case):
    p = os.path.join(run["dir"], "failures", f"{space.replace('/', '_')}_{case}", "log_tail.txt")
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def detail(text, case):
    """One line per failing tensor: name, mismatched count, max abs diff, tolerance, non-finite."""
    i = text.find("sweep.json#" + case)
    blk = text[i:] if i >= 0 else text
    j = blk.find("[  FAILED")
    blk = blk[:j] if j > 0 else blk
    out = []
    for t in blk.split("Tensor: ")[1:]:
        name = t.split("\n")[0].split("::")[-1].split(" ")[0]
        mm = re.search(r"Mismatched:\s+(\S+ \(\S+\))", t)
        mx = re.search(r"Max abs diff:\s+(\S+)", t)
        tol = re.search(r"Tolerance: (.*)", t)
        nf = re.search(r"Non-finite:\s+(\d+)", t)
        out.append(f"{name} mism={mm.group(1) if mm else '?'} max={mx.group(1) if mx else '?'}"
                   f" [{tol.group(1) if tol else '?'}]" + (f" NONFINITE={nf.group(1)}" if nf else ""))
    return " ; ".join(out)


def short(axes, keep=14):
    return {k: v for k, v in list(axes.items())[:keep]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--list", type=int, default=25, help="max WRONG/CRASH rows listed")
    a = ap.parse_args()
    pr = load(a.root, a.pr)
    base = load(a.root, a.base) if a.base else None
    if pr is None:
        sys.exit(f"no outcomes in {a.root}/{a.pr}")
    s = pr["s"]
    print(f"pr  : {s['mode']} stopped_because={s['stopped_because']} elapsed={round(s['elapsed_seconds'])}s"
          f" arch={s.get('arch')} outcomes={s['outcomes']}")
    if base:
        b = base["s"]
        print(f"base: {b['mode']} stopped_because={b['stopped_because']} elapsed={round(b['elapsed_seconds'])}s"
              f" arch={b.get('arch')} outcomes={b['outcomes']}")
    pairwise = s["mode"] == "pairwise"
    rows = {sp: c["rows"] for sp, c in s.get("pair_coverage", {}).items()}
    for sp, c in s.get("pair_coverage", {}).items():
        print(f"coverage {sp}: " + " ".join(f"{k}={c.get(k)}" for k in
              ("tuples", "illegal", "dead", "unproven", "budget_exceeded", "rows", "repair_rows",
               "generation_complete", "complete")))

    def comparable(k):
        return (not pairwise) or int(re.sub(r"\D", "", k[1])) < rows.get(k[0], 0)

    keys = [k for k in pr["o"] if comparable(k)]
    if base:
        same = sum(1 for k in keys if k in base["a"] and base["a"][k] == pr["a"][k])
        print(f"comparable rows: {len(keys)}; identical axes in base: {same} (must be equal; else the runs differ)")
    for sp in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == sp]
        line = f"[{sp}] pr {dict(collections.Counter(pr['o'][k]['outcome'] for k in ks))}"
        if base:
            line += f"  base {dict(collections.Counter(base['o'][k]['outcome'] for k in ks if k in base['o']))}"
        print(line)
        if base:
            tr = collections.defaultdict(list)
            for k in ks:
                if k in base["o"] and base["o"][k]["outcome"] != pr["o"][k]["outcome"]:
                    tr[(base["o"][k]["outcome"], pr["o"][k]["outcome"])].append(k[1])
            for t, v in sorted(tr.items()):
                tag = ""
                if t[0] == "PASS" and t[1] in ("WRONG", "CRASH", "DECLINED"):
                    tag = "  <== REGRESSION"
                elif t[0] == "DECLINED" and t[1] in ("WRONG", "CRASH"):
                    tag = "  <== new acceptance, wrong"
                elif t[0] == "DECLINED" and t[1] == "PASS":
                    tag = "  (new acceptance)"
                print(f"   {t[0]} -> {t[1]}: {len(v)} {v[:8]}{tag}")
    bad = [k for k in keys if pr["o"][k]["outcome"] in ("WRONG", "CRASH")]
    print(f"\nWRONG/CRASH on pr (comparable rows): {len(bad)}")
    nonfinite = []
    for k in sorted(bad)[: a.list]:
        txt = log_tail(pr, *k)
        d = detail(txt, k[1])
        if "NONFINITE" in d:
            nonfinite.append(k[1])
        bo = base["o"].get(k, {}).get("outcome") if base else None
        print(f" {k[0].split('/')[-1]} {k[1]} {pr['o'][k]['outcome']} base={bo} {short(pr['a'][k])}\n     {d or (pr['o'][k].get('reason') or '')[:200]}")
    if nonfinite:
        print("NON-FINITE (NaN/unwritten outputs; check batch dependence with --batch-size 1):", nonfinite)
    viol = collections.Counter()
    for k in keys:
        r = pr["o"][k]
        if r.get("expectation") not in (None, "met"):
            viol[(k[0].split("/")[-1], pr["a"][k].get("family"), r["outcome"], r.get("expect"))] += 1
    if viol:
        print("\nprobe violations on pr (space, family, outcome, expect):")
        for k, n in sorted(viol.items(), key=str):
            bn = ""
            if base:
                bn = sum(1 for kk in keys if kk in base["o"] and base["o"][kk].get("expectation") not in (None, "met")
                         and (kk[0].split("/")[-1], base["a"].get(kk, {}).get("family"), base["o"][kk]["outcome"],
                              base["o"][kk].get("expect")) == k)
                bn = f"  (same on base: {bn})"
            print(f"   {n} x {k}{bn}")


if __name__ == "__main__":
    main()
