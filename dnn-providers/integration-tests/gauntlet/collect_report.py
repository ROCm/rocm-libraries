#!/usr/bin/env python3
"""Build a findings folder from gauntlet work dirs: the report, the issues file and EVERY graph that caused a problem.

  collect_report.py --root <work dirs> --run <dir> [--run <dir> ...] --out <report dir> --title "PR <N> ..."
                    [--aot-rule <dir of aot_*.json>] [--class-check cc.json] [--class-replay <summary.json> ...] [--base <dir>]

Run it where the work dirs are (python3 only).

Layout of <out>:
  REPORT.md        what was run (engine, arch, build, mode, seeds), outcome counts per run and space, links
  ISSUES.md        a summary by category, then one table row per problem, WORST FIRST: category in plain words (out of
                   memory; a tolerance issue: bulk / large / few elements / only at the flat limit; applicability: accepted a
                   graph it should decline; crash or hang; harness gap), what was wrong, engine class, verdict and evidence
                   (the last two empty: the triager fills them)
  analysis_<run>.txt   analyze_runs.py's full output for each run
  graphs/<run>/<Space>/<case>/   for every WRONG / CRASH: the concrete graph (<case>.json), its fill metadata
                   (<case>.meta.json), row.json or repro.json (replay with `--row`), outcome.json, replay.sh,
                   shrunk/ (with --shrink), and case.md (what, kind, class, outcome, replay command)
                   for every UNEXERCISED row and every probe violation: case.md and row.json built from the
                   sample's axes (no graph is kept for those: `run_gauntlet_fuzz.py --row row.json
                   --generate-only` or the replay command regenerates it)
  class_check.json, triage/*.json   the engine-class check and the replay summaries, when given
  index.json       every problem: run, space, case, outcome, kind, folder
Anything that cannot be copied is listed in REPORT.md under "Missing", never skipped silently.
"""
import argparse, collections, glob, json, os, re, shutil, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import analyze_runs as analyze  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--run", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Gauntlet fuzz findings")
ap.add_argument(
    "--base", help="a base work dir to compare each run with (passed to analyze.py)"
)
ap.add_argument(
    "--aot-rule",
    help="dir with aot_*.json from tools/sdpa_bwd_aotriton_check.py: adds AOTriton's tolerance verdict per row",
)
ap.add_argument("--class-check")
ap.add_argument("--class-replay", action="append", default=[])
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
cls = json.load(open(a.class_check)) if a.class_check else {}
replay = {}
for p in a.class_replay:
    replay.update(json.load(open(p)))
aot = {}
if a.aot_rule:
    for p in glob.glob(os.path.join(a.aot_rule, "aot_*.json")):
        aot.update(json.load(open(p)))


def aot_group(v, axes):
    """A short name for what a failing row has in common with others (first match wins). The names are this report's
    reading of the data: edit them with the findings."""
    bad = [t for t in v["tensors"].values() if not t["pass"]]
    dt, dom, sc = v["dtype"], (axes or {}).get("domain"), (axes or {}).get("scale")
    if sc is not None and abs(sc) >= 2**20:
        return "scale at the 2^20 limit"
    if dt == "float16" and dom == "subnormal":
        return "fp16 flush: subnormal inputs"
    if dt == "float16" and (axes or {}).get("skv", 0) >= 16385:
        return "fp16 flush: P below 2^-14 at long Skv"
    if any(t["ref_error"] == 0 for t in bad):
        return "exact-zero gradient, engine noise"
    if dt == "float32":
        return "fp32 error above the rule's limit"
    return "fp16/bf16 error above the rule's limit"


def aot_verdict(run, case, axes=None):
    """(sort group, text): FAIL first. A PASS whose limit is >= half of the tensor's largest value is near-vacuous."""
    v = aot.get(f"{run}/{case}")
    if not a.aot_rule:
        return 1, ""
    if v is None or "error" in v:
        return 2, "not graded" + (f" ({v['error']})" if v else "")
    ts = v["tensors"]
    if not v["pass"]:
        bad = [(n, t) for n, t in ts.items() if not t["pass"]]
        n, t = max(bad, key=lambda x: x[1]["ratio"])
        why = (
            "true gradient exactly 0, engine nonzero"
            if t["ref_error"] == 0
            else f"{t['ratio']:.3g}x the limit"
        )
        if "test_error_truth" in t:
            why += f"; vs truth {t['test_error_truth']:.3g}, vs its inputs {t['test_error_given']:.3g}"
        return (
            0,
            f"**FAIL** [{aot_group(v, axes)}] {n}: error {t['test_error']:.3g} vs limit {t['threshold']:.3g} ({why})",
        )
    vac = any(
        t["ref_max"] > 0 and t["threshold"] >= 0.5 * t["ref_max"] for t in ts.values()
    )
    return 1, "pass (limit near-vacuous: >= half the largest value)" if vac else "pass"


missing, index, rows_md = [], [], []
report = [f"# {a.title}\n"]


def slug(space):
    return space.replace("/", "_")


for run in a.run:
    d = os.path.join(a.root, run)
    run_data = analyze.load(a.root, run)
    if run_data is None:
        missing.append(f"{run}: no outcomes.jsonl")
        continue
    s = run_data["s"]
    report.append(f"## {run}\n")
    report.append(
        f"- engine `{s.get('engine') or s.get('test_engine')}`, arch `{s.get('arch')}`, mode `{s['mode']}`, "
        f"seed {s['seed']}, data seed {s.get('data_seed')}, stopped because `{s['stopped_because']}`, "
        f"{round(s['elapsed_seconds'])} s"
    )
    report.append(
        f"- binary `{s.get('binary')}`, plugin `{s.get('test_article')}`, config `{s.get('test_config')}`"
    )
    report.append(f"- outcomes: {s['outcomes']}\n")
    per = collections.defaultdict(collections.Counter)
    for (sp, case), r in run_data["o"].items():
        per[sp][r["outcome"]] += 1
    report.append(
        "| space | "
        + " | ".join(
            analyze_o
            for analyze_o in ("PASS", "DECLINED", "WRONG", "CRASH", "UNEXERCISED")
        )
        + " |"
    )
    report.append("|---|---|---|---|---|---|")
    for sp in sorted(per):
        report.append(
            f"| {sp} | "
            + " | ".join(
                str(per[sp].get(o, 0))
                for o in ("PASS", "DECLINED", "WRONG", "CRASH", "UNEXERCISED")
            )
            + " |"
        )
    report.append("")

    # analysis text
    cmd = [
        sys.executable,
        os.path.join(here, "analyze_runs.py"),
        "--root",
        a.root,
        "--pr",
        run,
        "--list",
        "1000",
    ]
    if a.base:
        cmd += ["--base", a.base]
    if a.class_check:
        cmd += ["--class-check", a.class_check]
    for p in a.class_replay:
        cmd += ["--class-replay", p]
    cmd += ["--findings", os.path.join(a.out, f"_findings_{run}.md")]
    text = subprocess.run(cmd, capture_output=True, text=True).stdout
    open(os.path.join(a.out, f"analysis_{run}.txt"), "w").write(text)

    # problems: WRONG / CRASH (failure folders), UNEXERCISED and probe violations (axes only)
    problems = []
    for key, r in run_data["o"].items():
        viol = r.get("expectation") not in (None, "met")
        aot_fail = (
            bool(aot.get(f"{run}/{key[1]}"))
            and "tensors" in aot[f"{run}/{key[1]}"]
            and not aot[f"{run}/{key[1]}"]["pass"]
        )
        if r["outcome"] in ("WRONG", "CRASH", "UNEXERCISED") or viol or aot_fail:
            problems.append((key, r, viol))
    for (sp, case), r, viol in sorted(problems):
        folder = os.path.join(
            a.out,
            "graphs",
            run,
            slug(sp).split("_", 1)[1] if "_" in slug(sp) else slug(sp),
            case,
        )
        os.makedirs(folder, exist_ok=True)
        src = os.path.join(d, "failures", f"{slug(sp)}_{case}")
        axes = run_data["a"].get((sp, case), {})
        log = analyze.log_tail(run_data, sp, case)
        ts = analyze.tensors(log, case) if log else []
        kind, hints = analyze.classify(ts, axes) if ts else ("-", [])
        if r["outcome"] == "CRASH":
            kind = "CRASH"
        copied = []
        if os.path.isdir(src):
            for f in os.listdir(src):
                if f == "log_tail.txt":
                    continue  # the batch log: this case's slice is written below
                p = os.path.join(src, f)
                if os.path.isdir(p):
                    shutil.copytree(p, os.path.join(folder, f), dirs_exist_ok=True)
                else:
                    shutil.copy2(p, os.path.join(folder, f))
                copied.append(f)
            i = log.find("sweep.json#" + case)
            blk = log[i:] if i >= 0 else log
            j = blk.find("[  FAILED")
            open(os.path.join(folder, "log_case.txt"), "w").write(
                blk[: j if j > 0 else len(blk)]
            )
        else:
            row = {
                "space": sp,
                "axes": axes,
                "data_seed": s.get("data_seed"),
                "case_id": case,
            }
            json.dump(row, open(os.path.join(folder, "row.json"), "w"), indent=1)
            if r["outcome"] in ("WRONG", "CRASH"):
                missing.append(
                    f"{run}/{sp}/{case}: no failure folder (the graph was not kept); row.json written from the sample's axes"
                )
        ce = analyze.class_entry(cls, replay, run, (sp, case)) if cls else ""
        class_pass = "replayed under the class tolerance: PASS" in ce
        category, rank, score = analyze.categorize(
            r["outcome"],
            r.get("reason"),
            log,
            ts,
            kind,
            viol,
            r.get("expect"),
            class_pass,
        )
        what = analyze.detail(log, case) if log else ""
        av = aot.get(f"{run}/{case}")
        aot_failed = bool(av) and "tensors" in av and not av["pass"]
        if aot_failed and r["outcome"] == "PASS":
            # the flat tolerance passes but AOTriton's rule fails: there is no failure log, so the numbers come from the rule's own record
            category, rank, score = (
                "AOTriton rule: fails while the flat tolerance passes",
                1,
                max(t["ratio"] for t in av["tensors"].values()),
            )
        reason = (r.get("reason") or "")[:400]
        what_md = analyze.plain(ts) or reason.replace("|", "/").replace("\n", " ")[:300]
        if aot_failed and not ts:
            what_md = "<br>".join(
                f"**{n}**: error {t['test_error']:.3g} vs limit {t['threshold']:.3g} ({'true gradient exactly 0' if t['ref_error'] == 0 else format(t['ratio'], '.3g') + 'x the limit'}; "
                f"fp32-vs-low-precision reference distance {t['ref_error']:.3g}, fudge {t['fudge']:g})"
                for n, t in av["tensors"].items()
                if not t["pass"]
            )
        worst_ratio = analyze.tolerance_badness(ts)[0]
        with open(os.path.join(folder, "case.md"), "w") as f:
            f.write(f"# {sp} {case}: {category}\n\n")
            f.write(
                f"- **category:** {category} (outcome `{r['outcome']}`"
                + (", probe expectation violated" if viol else "")
                + ")\n"
            )
            f.write(
                f"- run `{run}`, engine `{s.get('engine') or s.get('test_engine')}`, arch `{s.get('arch')}`\n"
            )
            f.write(f"- axes: `{json.dumps(axes, separators=(',', ':'))}`\n")
            if what:
                f.write(
                    "- **what was wrong:**\n"
                    + "".join(
                        f"  - {x}\n"
                        for x in analyze.plain(ts, sep="\n").split("\n")
                        if x
                    )
                )
                f.write(f"  - (raw: {what})\n")
            if reason:
                f.write(f"- reason: {reason}\n")
            if viol:
                f.write(
                    f"- expectation: expected `{r.get('expect')}`, got `{r['outcome']}`\n"
                )
            f.write(
                f"- first guess (kind and hints): {kind}"
                + (f": {'; '.join(hints)}" if hints else "")
                + "\n"
            )
            if ce:
                f.write(f"- engine class: {ce}\n")
            f.write(
                "- **verdict:** _to fill_ (defect | precision class | harness gap | pre-existing | unknown)\n"
            )
            f.write(
                "- **evidence:** _to fill_ (domain A/B, class-tolerance replay, baseline on a shipped engine, shrunk repro)\n"
            )
            if os.path.exists(os.path.join(folder, "replay.sh")):
                f.write(
                    "- replay: `replay.sh` in this folder (solo), or `run_gauntlet_fuzz.py --row row.json ...`\n"
                )
            if os.path.isdir(os.path.join(folder, "shrunk")):
                f.write("- shrunk reproducer: `shrunk/` (own `replay.sh`)\n")
        rel = os.path.relpath(folder, a.out).replace("\\", "/")
        grp, rule_txt = aot_verdict(run, case, axes)
        if rule_txt:
            with open(os.path.join(folder, "case.md"), "a") as f:
                f.write(f"- **AOTriton tolerance rule:** {rule_txt}\n")
        index.append(
            {
                "aot_rule": rule_txt,
                "run": run,
                "space": sp,
                "case": case,
                "outcome": r["outcome"],
                "violation": viol,
                "kind": kind,
                "category": category,
                "rank": rank,
                "score": score,
                "folder": rel,
                "files": sorted(os.listdir(folder)),
            }
        )
        rows_md.append(
            (
                grp,
                rank,
                -score,
                run,
                sp,
                case,
                f"| {category} | {('non-finite' if kind == 'NONFINITE' or worst_ratio in (float('inf'), float('nan')) else format(worst_ratio, '.3g') + 'x') if worst_ratio else ''} | "
                f"{rule_txt + ' | ' if a.aot_rule else ''}{run} | {sp.split('/')[-1]} | [{case}]({rel}/case.md) | "
                f"{what_md} | {ce} | |",
            )
        )

# ISSUES.md: worst first
rows_md.sort()
by_cat = collections.Counter(i["category"] for i in index)


def write_table(f, rows):
    if a.aot_rule:
        f.write(
            "| # | category | worst (x atol) | AOTriton rule | run | space | case | what was wrong (per tensor) | engine class | verdict / evidence |\n|---|---|---|---|---|---|---|---|---|---|\n"
        )
    else:
        f.write(
            "| # | category | worst (x atol) | run | space | case | what was wrong (per tensor) | engine class | verdict / evidence |\n|---|---|---|---|---|---|---|---|---|\n"
        )
    for n, row in enumerate(rows, 1):
        f.write(f"| {n} " + row[6] + "\n")


# with --aot-rule the issues are the rows that fail it (and any it could not grade); the rows that only fail the flat
# fuzz tolerance go to OTHER_FLAT_TOLERANCE_ONLY.md
issue_rows = [r for r in rows_md if r[0] != 1] if a.aot_rule else rows_md
other_rows = [r for r in rows_md if r[0] == 1] if a.aot_rule else []
by_cat = collections.Counter(i["category"] for i in index)
with open(os.path.join(a.out, "ISSUES.md"), "w") as f:
    f.write(f"# Issues: {a.title}\n\n")
    if a.aot_rule:
        by_rule = collections.Counter(
            (
                "FAIL"
                if i["aot_rule"].startswith("**FAIL")
                else (
                    "pass, limit near-vacuous"
                    if "vacuous" in i["aot_rule"]
                    else (
                        "pass, limit meaningful"
                        if i["aot_rule"] == "pass"
                        else "not graded"
                    )
                )
            )
            for i in index
        )
        f.write(
            f"**{len(issue_rows)} issue(s)** among {len(index)} rows that failed the fuzz's flat tolerance or AOTriton's rule. A row is an issue when it "
            "fails AOTriton's tolerance rule (sdpa_conformance: an output passes if its max error is within "
            "`max(1e-5, fudge x |fp32 reference - same math in the test dtype|)`) against BOTH references: the truth recomputed "
            "from Q, K and V, and the fp64 backward of the O and LSE the engine was given (an engine that matches the latter is "
            "faithful to its inputs); details and caveats in `aot_rule/README.md`). "
            "The other rows pass that rule and are in `OTHER_FLAT_TOLERANCE_ONLY.md`; every row still has its folder under `graphs/`.\n\n"
            "| verdict | rows |\n|---|---|\n"
        )
        for k2 in (
            "FAIL",
            "pass, limit meaningful",
            "pass, limit near-vacuous",
            "not graded",
        ):
            f.write(f"| {k2} | {by_rule.get(k2, 0)} |\n")
        groups = collections.Counter(
            i["aot_rule"].split("[", 1)[1].split("]", 1)[0]
            for i in index
            if i["aot_rule"].startswith("**FAIL")
        )
        f.write(
            "\n### The failing rows by what they have in common\n\n| group | rows |\n|---|---|\n"
        )
        for k2, n2 in groups.most_common():
            f.write(f"| {k2} | {n2} |\n")
        f.write(
            "\nA `pass, limit near-vacuous` row passes only because the low-precision reference is itself far from the fp32 one, "
            "so the limit is at least half of the tensor's largest value: the rule cannot tell good from bad there.\n\n"
        )
    else:
        f.write(
            "Every row is a graph that caused a problem; its folder under `graphs/` holds the graph and everything needed to "
            "replay it. **Worst first**: out of memory; then the biggest tolerance failures (non-finite output, a bulk of an "
            "output off, error far over the tolerance), worst ratio first; then crashes/hangs and applicability problems "
            "(an engine that accepts a graph it must decline); then smaller tolerance failures; then failures that pass at the "
            "engine's own class tolerance; last, accepted graphs the harness could not grade.\n\n"
        )
    f.write(
        "Per failing tensor, `what` gives the share of elements off, the count, and the worst diff as a multiple of the atol limit "
        "(an element passes if its diff is within atol + rtol x |reference|; the multiple uses atol only); the engine class column is "
        "the class check and its replay; **verdict** and **evidence** are for the triager (defect | precision class | harness gap | "
        "pre-existing | unknown, with the domain A/B, class-tolerance replay and shrunk-repro evidence).\n\n"
    )
    f.write("## By category (issues only)\n\n| category | rows |\n|---|---|\n")
    cat_issue = collections.Counter(
        r[6].split(" | ")[0].lstrip("| ").strip() for r in issue_rows
    )
    for cat, n in sorted(
        cat_issue.items(), key=lambda kv: (analyze.CATEGORY_RANK.get(kv[0], 9), -kv[1])
    ):
        f.write(f"| {cat} | {n} |\n")
    f.write("\n## Issues, worst first\n\n")
    write_table(f, issue_rows)
if other_rows:
    with open(os.path.join(a.out, "OTHER_FLAT_TOLERANCE_ONLY.md"), "w") as f:
        f.write(
            f"# Rows that fail only the fuzz's flat tolerance: {a.title}\n\n{len(other_rows)} rows. They pass AOTriton's tolerance rule "
            "(see `ISSUES.md`), so they are not listed as issues; most are fp16/bf16 rows in wide input domains where that rule's "
            "limit is near-vacuous. Same columns as `ISSUES.md`, worst flat-tolerance ratio first.\n\n"
        )
        write_table(f, other_rows)
index.sort(
    key=lambda i: (
        0 if i["aot_rule"].startswith("**FAIL") else 1,
        i["rank"],
        -i["score"],
        i["run"],
        i["space"],
        i["case"],
    )
)
for tmp in glob.glob(os.path.join(a.out, "_findings_*.md")):
    os.remove(tmp)

# extras
if a.class_check:
    shutil.copy2(a.class_check, os.path.join(a.out, "class_check.json"))
if a.class_replay:
    os.makedirs(os.path.join(a.out, "triage"), exist_ok=True)
    for p in a.class_replay:
        shutil.copy2(
            p,
            os.path.join(
                a.out, "triage", os.path.basename(os.path.dirname(p)) + "_summary.json"
            ),
        )
json.dump(index, open(os.path.join(a.out, "index.json"), "w"), indent=1)

report.append("## Problems\n")
report.append(
    f"{len(index)} graphs/rows in `graphs/`, worst first in `ISSUES.md` (`index.json` is in the same order):\n"
)
for cat, n in sorted(
    by_cat.items(), key=lambda kv: (analyze.CATEGORY_RANK.get(kv[0], 9), -kv[1])
):
    report.append(f"- {cat}: {n}")
report.append("")
report.append(
    "Files: `ISSUES.md`"
    + (", `OTHER_FLAT_TOLERANCE_ONLY.md`" if a.aot_rule else "")
    + ", `index.json`, `analysis_<run>.txt`"
    + (", `class_check.json`, `triage/`" if (a.class_check or a.class_replay) else "")
    + ".\n"
)
if missing:
    report.append("## Missing\n")
    report += [f"- {m}" for m in missing]
open(os.path.join(a.out, "REPORT.md"), "w").write("\n".join(report) + "\n")
print(f"wrote {a.out}: {len(index)} problem folders; missing: {len(missing)}")
