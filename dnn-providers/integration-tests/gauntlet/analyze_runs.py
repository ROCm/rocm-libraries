#!/usr/bin/env python3
"""Compare gauntlet fuzz work dirs of a BASE build and a PR build.

  analyze_runs.py --root <dir holding the work dirs> --base <name> --pr <name> [--pr-only]

Reads <root>/<name>/{outcomes.jsonl,samples.jsonl,summary.json,failures/}. Prints, per space:
outcome counts, coverage flags, base->pr transitions and the WRONG/CRASH list with axes.

The one rule that matters: in PAIRWISE mode only the primary rows (case ids below the
space's `rows` count) are the same graph in both runs. Repair rows are built from the
outcomes of the run, so ids above that differ between base and pr. This script compares
primary rows only for pairwise runs and every row for random runs (same --seed, same graphs).
Run it where the work dirs are; it needs only python3.

Every WRONG/CRASH is printed with WHAT was wrong (per tensor: mismatched count and percent, max abs
diff against the tolerance in force, ratio, non-finite) and a first WHY (a kind: TAIL / BULK / LARGE /
NONFINITE, and the axis signature that usually explains it). `--findings OUT.md` writes the same as a
table the report can quote, with empty `verdict` and `evidence` columns that triage must fill (domain
A/B replay, baseline on a shipped engine, shrunk repro). WRONG is a measurement against a tolerance,
never a verdict.
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
    axes = {
        (r["space"], r["case"]): r["axes"] for r in jl(os.path.join(d, "samples.jsonl"))
    }
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    return {"dir": d, "o": out, "a": axes, "s": summary}


def log_tail(run, space, case):
    p = os.path.join(
        run["dir"], "failures", f"{space.replace('/', '_')}_{case}", "log_tail.txt"
    )
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
        out.append(
            f"{name} mism={mm.group(1) if mm else '?'} max={mx.group(1) if mx else '?'}"
            f" [{tol.group(1) if tol else '?'}]"
            + (f" NONFINITE={nf.group(1)}" if nf else "")
        )
    return " ; ".join(out)


def tensors(text, case):
    """Per failing tensor: dict(name, mism, pct, max, atol, rtol, nonfinite). Same parse as detail()."""
    i = text.find("sweep.json#" + case)
    blk = text[i:] if i >= 0 else text
    j = blk.find("[  FAILED")
    blk = blk[:j] if j > 0 else blk
    out = []
    for t in blk.split("Tensor: ")[1:]:
        name = t.split("\n")[0].split("::")[-1].split(" ")[0]
        mm = re.search(r"Mismatched:\s+(\d+) \(([\d.]+)%\)", t)
        mx = re.search(r"Max abs diff:\s+(\S+)", t)
        tol = re.search(r"atol=([\d.eE+-]+)\s+rtol=([\d.eE+-]+)", t)
        nf = re.search(r"Non-finite:\s+(\d+)", t)
        out.append(
            {
                "name": name,
                "mism": int(mm.group(1)) if mm else None,
                "pct": float(mm.group(2)) if mm else None,
                "max": float(mx.group(1)) if mx else None,
                "atol": float(tol.group(1)) if tol else None,
                "rtol": float(tol.group(2)) if tol else None,
                "nonfinite": int(nf.group(1)) if nf else 0,
            }
        )
    return out


def fmt_num(x):
    """0.6708 -> '0.671'; 1e-4 -> '1e-4'; 4463 stays whole."""
    if x is None:
        return "?"
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:.3g}"


def plain(ts, sep="<br>"):
    """The failing tensors in plain words, one per line:
    'dK: 43.6% of elements off (586); worst diff 0.671 = 6708x the 1e-4 limit'. The limit is atol (the part of the
    allowed band that does not scale with the reference value); an element passes if its diff is within atol + rtol*|ref|.
    """
    lines = []
    for t in ts:
        if t["max"] is None:
            continue
        parts = [
            f"**{t['name']}**: "
            + (
                f"{fmt_num(t['pct'])}% of elements off ({t['mism']})"
                if t["mism"] is not None
                else "elements off"
            )
        ]
        if t["atol"]:
            parts.append(
                f"worst diff {fmt_num(t['max'])} = {fmt_num(round(t['max'] / t['atol']))}x the {fmt_num(t['atol'])} limit"
            )
        else:
            parts.append(f"worst diff {fmt_num(t['max'])}")
        if t["nonfinite"]:
            parts.append(f"{t['nonfinite']} NaN/Inf/unwritten")
        lines.append("; ".join(parts))
    return sep.join(lines)


# domains that widen Q/K (large logits): the softmax is ill-conditioned there, so tolerance-class
# effects show first. A hint, never a verdict.
WIDE_DOMAINS = ("wide", "large", "positive", "nonzero")


def classify(ts, axes):
    """(kind, hints). kind: NONFINITE (NaN/unwritten output), BULK (>= 5 percent of an output
    differs), LARGE (max abs diff >= 20x the tolerance), TAIL (a few elements, near the tolerance),
    MIXED. hints: axis signatures that usually explain a TAIL, for the triager to confirm.
    """
    if any(t["nonfinite"] for t in ts):
        kind = "NONFINITE"
    elif any((t["pct"] or 0) >= 5 for t in ts):
        kind = "BULK"
    elif any(t["max"] and t["atol"] and t["max"] >= 20 * t["atol"] for t in ts):
        kind = "LARGE"
    elif ts and all((t["pct"] or 0) < 1 for t in ts):
        kind = "TAIL"
    else:
        kind = "MIXED"
    hints = []
    dom = axes.get("domain")
    if dom in WIDE_DOMAINS:
        hints.append(f"domain={dom} (large logits, peaked softmax)")
    sc = axes.get("scale")
    if isinstance(sc, (int, float)) and abs(sc) >= 1:
        hints.append(f"scale={sc} (>= 1, peaked softmax)")
    if isinstance(axes.get("skv"), int) and axes["skv"] <= 2:
        hints.append(f"skv={axes['skv']} (dS cancels to ~0; engine TOML class K1)")
    if kind in ("BULK", "NONFINITE", "LARGE") and dom not in WIDE_DOMAINS:
        hints.append("NOT explained by a wide domain: look for a real defect first")
    return kind, hints


OOM_RE = re.compile(
    r"out of memory|OutOfMemory|bad_alloc|cannot allocate|hipErrorMemoryAllocation|allocation (?:failed|of)",
    re.I,
)

# Report order: the worst problems first. rank 0 out of memory; 1 the biggest tolerance failures (non-finite,
# a bulk of an output off, error >= 20x the tolerance), worst first; 2 crashes/hangs and applicability
# (an engine that accepts a graph it must decline); 3 smaller tolerance failures; 4 failures that pass at the
# engine's own class tolerance; 5 accepted but not gradable.
CATEGORY_RANK = {
    "Out of memory": 0,
    "Non-finite output (NaN or unwritten)": 1,
    "Tolerance: bulk mismatch": 1,
    "Tolerance: large error": 1,
    "Crash or hang": 2,
    "Applicability: accepted a graph it should decline": 2,
    "Tolerance: few elements near the limit": 3,
    "Tolerance: only at the flat limit (passes at the engine's class tolerance)": 4,
    "Harness gap: accepted but could not be graded": 5,
}


def tolerance_badness(ts):
    """(ratio, pct): worst max abs diff over its atol, and the largest mismatch share, of a failing case."""
    ratio = max(
        (t["max"] / t["atol"] for t in ts if t["max"] and t["atol"]), default=0.0
    )
    pct = max((t["pct"] or 0 for t in ts), default=0.0)
    return ratio, pct


def categorize(outcome, reason, log, ts, kind, violation, expect, class_passes):
    """(category, rank, score): a plain-words category (a tolerance issue, an applicability issue, a crash...), its
    report rank (CATEGORY_RANK) and a score that orders rows within a rank, worst first.
    """
    text = (reason or "") + "\n" + (log or "")
    ratio, pct = tolerance_badness(ts)
    if outcome == "CRASH":
        cat = "Out of memory" if OOM_RE.search(text) else "Crash or hang"
        return cat, CATEGORY_RANK[cat], 0.0
    if (
        violation
        and expect == "declined"
        and outcome in ("PASS", "WRONG", "UNEXERCISED")
    ):
        cat = "Applicability: accepted a graph it should decline"
        return cat, CATEGORY_RANK[cat], ratio
    if outcome == "UNEXERCISED":
        cat = "Harness gap: accepted but could not be graded"
        return cat, CATEGORY_RANK[cat], 0.0
    if outcome == "WRONG":
        if kind == "NONFINITE":
            cat = "Non-finite output (NaN or unwritten)"
        elif class_passes:
            cat = "Tolerance: only at the flat limit (passes at the engine's class tolerance)"
        elif kind == "BULK":
            cat = "Tolerance: bulk mismatch"
        elif kind == "LARGE":
            cat = "Tolerance: large error"
        else:
            cat = "Tolerance: few elements near the limit"
        return cat, CATEGORY_RANK[cat], ratio + pct
    return "Other", 9, 0.0


def short(axes, keep=14):
    # the late axes (domain, alignment, placement) sit at the end: always show the ones that matter
    out = {k: v for k, v in list(axes.items())[:keep]}
    for k in ("domain", "scale", "alignment", "placement", "p", "seed", "offset"):
        if k in axes:
            out[k] = axes[k]
    return out


def class_entry(cls, replay, run, k):
    """'class U (atol 0.04 rtol 0.016): WITHIN | replay under it: PASS' for one WRONG row, or ''."""
    folder = f"{k[0].replace('/', '_')}_{k[1]}"
    c = cls.get(f"{run}/{folder}")
    if not c:
        return ""
    txt = f"class {c['class']} (atol {c['atol']} rtol {c['rtol']}): {c['verdict']}"
    if c["info"].get("escalated_from"):
        txt += f", escalated from {c['info']['escalated_from']}"
    rp = replay.get(folder, {}).get("class")
    if rp:
        txt += f" | replayed under the class tolerance: {rp}"
    return txt


def write_findings(path, name, findings, pr, cls=None, replay=None):
    """One row per WRONG/CRASH: what was wrong, a first why, and empty verdict/evidence cells."""
    with open(path, "w") as f:
        f.write(f"# WRONG/CRASH findings of {name}\n\n")
        f.write(
            "`kind` and `hints` are automatic first guesses. `verdict` (defect | precision class | "
            "harness gap | pre-existing | unknown) and `evidence` (domain A/B replay, baseline on a "
            "shipped engine or arch, shrunk repro, failure folder) are the triager's to fill.\n\n"
        )
        f.write(
            "| space | case | outcome | kind | what (tensor: mismatched, max abs vs tolerance) | hints | engine class | axes | verdict | evidence |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for k, kind, hints, ts, bo in findings:
            what = (
                "; ".join(
                    f"{t['name']}: {t['mism']} ({t['pct']}%), max {t['max']:.3g} vs atol {t['atol']} rtol {t['rtol']}"
                    + (f", NON-FINITE {t['nonfinite']}" if t["nonfinite"] else "")
                    for t in ts
                    if t["max"] is not None
                )
                or (pr["o"][k].get("reason") or "")[:160]
            )
            axes = json.dumps(pr["a"][k], separators=(",", ":"))
            f.write(
                f"| {k[0].split('/')[-1]} | {k[1]} | {pr['o'][k]['outcome']} | {kind} | {what} | "
                f"{'; '.join(hints)} | {class_entry(cls or {}, replay or {}, name, k)} | `{axes}` | | |\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base")
    ap.add_argument("--pr", required=True)
    ap.add_argument("--list", type=int, default=25, help="max WRONG/CRASH rows listed")
    ap.add_argument(
        "--findings",
        help="write every WRONG/CRASH as a markdown table (what, why, verdict)",
    )
    ap.add_argument(
        "--class-check",
        help="class_check.json: add the engine tolerance class and its verdict to every WRONG row",
    )
    ap.add_argument(
        "--class-replay",
        action="append",
        default=[],
        help="triage_wrong.py summary.json (dab_<tag>/summary.json) holding the class-tolerance replay outcomes; repeatable",
    )
    a = ap.parse_args()
    pr = load(a.root, a.pr)
    cls = json.load(open(a.class_check)) if a.class_check else {}
    replay = {}
    for path in a.class_replay:
        replay.update(json.load(open(path)))
    base = load(a.root, a.base) if a.base else None
    if pr is None:
        sys.exit(f"no outcomes in {a.root}/{a.pr}")
    s = pr["s"]
    print(
        f"pr  : {s['mode']} stopped_because={s['stopped_because']} elapsed={round(s['elapsed_seconds'])}s"
        f" arch={s.get('arch')} outcomes={s['outcomes']}"
    )
    if base:
        b = base["s"]
        print(
            f"base: {b['mode']} stopped_because={b['stopped_because']} elapsed={round(b['elapsed_seconds'])}s"
            f" arch={b.get('arch')} outcomes={b['outcomes']}"
        )
    pairwise = s["mode"] == "pairwise"
    rows = {sp: c["rows"] for sp, c in s.get("pair_coverage", {}).items()}
    for sp, c in s.get("pair_coverage", {}).items():
        print(
            f"coverage {sp}: "
            + " ".join(
                f"{k}={c.get(k)}"
                for k in (
                    "tuples",
                    "illegal",
                    "dead",
                    "unproven",
                    "budget_exceeded",
                    "rows",
                    "repair_rows",
                    "generation_complete",
                    "complete",
                )
            )
        )

    def comparable(k):
        return (not pairwise) or int(re.sub(r"\D", "", k[1])) < rows.get(k[0], 0)

    keys = [k for k in pr["o"] if comparable(k)]
    if base:
        same = sum(1 for k in keys if k in base["a"] and base["a"][k] == pr["a"][k])
        print(
            f"comparable rows: {len(keys)}; identical axes in base: {same} (must be equal; else the runs differ)"
        )
    for sp in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == sp]
        line = (
            f"[{sp}] pr {dict(collections.Counter(pr['o'][k]['outcome'] for k in ks))}"
        )
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
    findings = []
    for k in sorted(bad):
        txt = log_tail(pr, *k)
        d = detail(txt, k[1])
        ts = tensors(txt, k[1])
        kind, hints = classify(ts, pr["a"][k])
        if pr["o"][k]["outcome"] == "CRASH":
            kind = "CRASH"
        if "NONFINITE" in d:
            nonfinite.append(k[1])
        bo = base["o"].get(k, {}).get("outcome") if base else None
        findings.append((k, kind, hints, ts, bo))
        if len(findings) <= a.list:
            why = f"{kind}" + (": " + "; ".join(hints) if hints else "")
            print(
                f" {k[0].split('/')[-1]} {k[1]} {pr['o'][k]['outcome']} base={bo} {short(pr['a'][k])}\n"
                f"     what: {d or (pr['o'][k].get('reason') or '')[:200]}\n     why? {why}"
            )
            ce = class_entry(cls, replay, a.pr, k)
            if ce:
                print(f"     engine class: {ce}")
    if bad:
        print(
            "\nWRONG/CRASH by kind:", dict(collections.Counter(f[1] for f in findings))
        )
        if cls:
            ents = [
                cls.get(f"{a.pr}/{f[0][0].replace('/', '_')}_{f[0][1]}")
                for f in findings
            ]
            print(
                "engine class verdicts:",
                dict(collections.Counter(e["verdict"] for e in ents if e)),
            )
            if replay:
                rp = [
                    replay.get(f"{f[0][0].replace('/', '_')}_{f[0][1]}", {}).get(
                        "class"
                    )
                    for f in findings
                ]
                print(
                    "replayed under the engine class tolerance:",
                    dict(collections.Counter(x for x in rp if x)),
                    "(PASS = not wrong by the engine's own contract)",
                )
        for axis in ("dtype", "domain", "scale", "mask", "pad_on", "p"):
            tab = collections.defaultdict(lambda: [0, 0])
            for k in keys:
                if axis not in pr["a"][k]:
                    continue
                oc = pr["o"][k]["outcome"]
                if oc in ("PASS", "WRONG"):
                    tab[str(pr["a"][k][axis])][oc == "WRONG"] += 1
            if len(tab) > 1:
                print(
                    f"  WRONG rate by {axis}: "
                    + "  ".join(f"{v}={w}/{p + w}" for v, (p, w) in sorted(tab.items()))
                )
        if a.findings:
            write_findings(a.findings, a.pr, findings, pr, cls, replay)
            print(f"findings table: {a.findings}  (fill verdict + evidence per row)")
    if nonfinite:
        print(
            "NON-FINITE (NaN/unwritten outputs; check batch dependence with --batch-size 1):",
            nonfinite,
        )
    viol = collections.Counter()
    for k in keys:
        r = pr["o"][k]
        if r.get("expectation") not in (None, "met"):
            viol[
                (
                    k[0].split("/")[-1],
                    pr["a"][k].get("family"),
                    r["outcome"],
                    r.get("expect"),
                )
            ] += 1
    if viol:
        print("\nprobe violations on pr (space, family, outcome, expect):")
        for k, n in sorted(viol.items(), key=str):
            bn = ""
            if base:
                bn = sum(
                    1
                    for kk in keys
                    if kk in base["o"]
                    and base["o"][kk].get("expectation") not in (None, "met")
                    and (
                        kk[0].split("/")[-1],
                        base["a"].get(kk, {}).get("family"),
                        base["o"][kk]["outcome"],
                        base["o"][kk].get("expect"),
                    )
                    == k
                )
                bn = f"  (same on base: {bn})"
            print(f"   {n} x {k}{bn}")


if __name__ == "__main__":
    main()
