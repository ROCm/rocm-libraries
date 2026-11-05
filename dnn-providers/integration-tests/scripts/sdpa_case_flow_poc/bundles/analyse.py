#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Summarise GPU validation runs of the SdpaFwdPlan bundles.

Part A reads <results>/validate/<arch>-job<id>/: <tier>.xml and <tier>.log from
hipdnn_integration_tests (--gtest_output=xml:<tier>.xml), optional <tier>-hostfill-<n>.xml
re-runs with HIPDNN_TEST_HOST_INPUT_FILL=1, and an optional steps.log with lines
"asm <tier> rc=<rc> wall_s=<s>".
Part B reads <results>/runner/<arch>-job<id>/<engine>/results.jsonl from
hipdnn_sdpa_case_flow_runner, and compares the accepted rows with --predicted when given.

--skips TOML applies the engine's [[test_skips]] to the runs and reports every failure that
is not skipped, skipped under a reason of another tag, and every passing case that is skipped.

Usage: analyse.py --results DIR --levels levels.csv --placements placements.csv
                  [--part a|b|both] [--out DIR] [--skips TOML [--validate DIR ...]]
"""
import argparse
import collections
import csv
import fnmatch
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.dont_write_bytecode = True
BUNDLES = Path(__file__).resolve().parents[3] / "integration-test-bundles"

TIERS = ["quick", "standard", "comprehensive", "full"]
LANE_TIMEOUT = {"quick": 1200, "standard": 3600, "comprehensive": 7200, "full": 28800}
TICKET = {
    "T2": "ALMIOPEN-3042",
    "T3": "ALMIOPEN-3042",
    "T4": "ALMIOPEN-3124",
    "T5": "ALMIOPEN-3125",
    "T6": "ALMIOPEN-3126",
    "T7": "ALMIOPEN-3043",
    "T9": "ALMIOPEN-3041",
    "T1": "ALMIOPEN-3123",
    "T8": "ALMIOPEN-3127",
    "T11": "ALMIOPEN-3133",
}
REASON = {
    "T11": "Q/K/V descales on non-fp8 inputs are accepted and ignored (unverifiable until ALMIOPEN-3132)",
    "T5": "FP8_E4M3_FNUZ inputs give a wrong or non-finite O",
    "T7": "a negative or zero attention scale gives NaN in O",
    "T6": "the dropout RNG dump is accepted but never written",
    "T4": "the Max and Sum-exp outputs are accepted but never written",
    "T3": "Stats is written with only the head stride, so other Stats layouts are wrong",
    "T2": "overlapping or zero Q/K/V sequence strides give a wrong O",
}


def skip_reason(tag):
    return f"{TICKET[tag]}: {REASON[tag]}"


def reason_tag(reason):
    """Return the tag whose skip reason is exactly `reason`, or None."""
    return next((t for t in REASON if skip_reason(t) == reason), None)


AXES = [
    "dtype_q",
    "dtype_o",
    "batch_size",
    "seq_q",
    "seq_kv",
    "q_heads",
    "head_dim_qk",
    "head_dim_v",
    "dim_order_q",
    "strides_q",
    "strides_k",
    "strides_v",
    "strides_o",
    "alignment_q",
    "causal_mask",
    "sliding_window",
    "bias",
    "alibi",
    "attention_sinks",
    "scale_source",
    "scale_value",
    "dropout",
    "dropout_probability",
    "dropout_pattern_output",
    "stats_output",
    "dim_order_stats",
    "strides_stats",
    "dtype_stats",
    "alignment_stats",
    "max_sumexp_output",
    "sizes",
]


_STATS_LAYOUT = None


def stats_layouts():
    """case_key -> (dims, strides) of the Stats tensor, read from the SdpaFwdPlan bundles.

    A case whose absent_tensors names the Stats tensor has no entry.
    """
    global _STATS_LAYOUT
    if _STATS_LAYOUT is None:
        _STATS_LAYOUT = {}
        for tpl in sorted(BUNDLES.glob("*/SdpaFwdPlan/*/graph.template.json")):
            graph = json.loads(tpl.read_text())
            uid = next(
                n["outputs"]["stats_tensor_uid"]
                for n in graph["nodes"]
                if n["type"] == "SdpaAttributes"
            )
            if uid is None:
                continue
            sweep = json.loads((tpl.parent / "sweep.json").read_text())
            for case in sweep["cases"]:
                if uid in case.get("absent_tensors", ()):
                    continue
                t = next(
                    (t for t in case["values"]["tensors"] if t["uid"] == uid), None
                )
                if t is None:
                    raise ValueError(
                        f"{tpl.parent / 'sweep.json'}: case {case.get('id')!r} has no"
                        f" values for Stats tensor {uid} and does not list it absent"
                    )
                _STATS_LAYOUT[case["metadata"]["case_key"]] = (t["dims"], t["strides"])
    return _STATS_LAYOUT


def stats_unaddressable(case_key):
    """True when the kernel's Stats offset ((b*Hq + h)*head_stride + s) misses the layout."""
    layout = stats_layouts().get(case_key)
    if layout is None:
        return False
    (b, hq, sq, _), (sb, sh, ss, _) = layout
    return (sq > 1 and ss != 1) or (b > 1 and sb != hq * sh)


def tags_for(row, arch):
    """Candidate defect tags for one CSV row. A tag is a hypothesis until a run fails."""
    tags = []
    strides = [row[c] for c in ("strides_q", "strides_k", "strides_v")]
    if "self_overlapping" in strides:
        tags.append("T2")
    elif "zero_broadcast" in strides:
        tags.append("T2?")
    if row["stats_output"] == "requested" and stats_unaddressable(row["case_key"]):
        tags.append("T3")
    if row["max_sumexp_output"] != "not_requested":
        tags.append("T4")
    if row["dtype_q"] == "fp8_e4m3_fnuz" and arch == "gfx942":
        tags.append("T5")
    if (
        row["dropout"] == "random"
        and row["dropout_probability"] == "zero"
        and row["dropout_pattern_output"] == "requested"
    ):
        tags.append("T6")
    if row["scale_value"] in ("zero", "negative"):
        tags.append("T7")
    if not row["dtype_q"].startswith("fp8") and "present" in (
        row["descale_q"],
        row["descale_k"],
        row["descale_v"],
    ):
        tags.append("T11")
    return tags


def load_rows(levels, placements):
    by_key = {}
    with open(levels, newline="") as f:
        for r in csv.DictReader(f):
            by_key[r["case_key"]] = r
    by_name = {}
    with open(placements, newline="") as f:
        for p in csv.DictReader(f):
            by_name[p["gtest_name"]] = by_key[p["case_key"]]
    return by_key, by_name


ABS_PATH = re.compile(r"(?<![\w.])/[^\s/]+/\S+")


def clean(text):
    text = ABS_PATH.sub("<path>", text or "")
    return re.sub(r"[0-9a-f]{16}", "<key>", text)


def classify(tc):
    fail = tc.find("failure")
    skip = tc.find("skipped")
    if fail is not None:
        return "failed", fail.text or fail.get("message") or ""
    if skip is not None:
        text = skip.text or ""
        if "Engine could not execute bundle" in text:
            return "declined", text
        if "Unverifiable:" in text:
            return "unverifiable", text
        if "[arch " in text:
            return "toml_skipped", text
        return "other_skip", text
    return "passed", ""


def failure_kind(text):
    tensors = sorted(set(re.findall(r"Tensor: (\S+)", text)))
    if "Comparison FAILED" in text:
        return "comparison:" + ",".join(tensors)
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and ln.strip() != "Failed" and not ln.strip().startswith("/")
    ]
    return clean(lines[0] if lines else "?")[:120]


def parse_xml(path):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, FileNotFoundError):
        return None
    out = {}
    for tc in root.iter("testcase"):
        name = tc.get("classname") + "." + tc.get("name")
        out[name] = classify(tc)
    return out


def ref_sections(log):
    sections = {"REFERENCE EXECUTOR ERRORS": [], "UNVERIFIABLE BUNDLES": []}
    current = None
    try:
        lines = open(log, errors="replace").read().splitlines()
    except FileNotFoundError:
        return sections
    for ln in lines:
        m = re.match(r"^==== (.+) \((\d+)\) ====$", ln)
        if m:
            current = m.group(1) if m.group(1) in sections else None
            continue
        if current and ln.startswith("  - "):
            sections[current].append(clean(ln[4:])[:200])
        elif current and ln.strip() == "":
            continue
        elif current:
            current = None
    return sections


def walls(steps):
    w = {}
    if not steps.exists():
        return w
    for ln in open(steps).read().splitlines():
        m = re.search(r" asm (\w+) rc=(\d+) wall_s=(\d+)", ln)
        if m:
            w[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        m = re.search(r" (\w+) hostfill wall_s=(\d+)", ln)
        if m:
            w[m.group(1) + "-hostfill"] = (0, int(m.group(2)))
    return w


def part_a(results, out, by_name):
    report = []
    detail = []
    for d in sorted(glob.glob(str(results / "validate" / "*-job*"))):
        arch, job = os.path.basename(d).split("-job")
        steps = Path(d) / "steps.log"
        w = walls(steps)
        for tier in TIERS:
            res = parse_xml(Path(d) / f"{tier}.xml")
            if res is None:
                continue
            host = {}
            for hx in sorted(glob.glob(str(Path(d) / f"{tier}-hostfill-*.xml"))):
                host.update(parse_xml(hx) or {})
            counts = collections.Counter(v[0] for v in res.values())
            refs = ref_sections(Path(d) / f"{tier}.log")
            rc, wall = w.get(tier, (None, None))
            report.append(
                f"{arch} job{job} {tier}: total={len(res)} passed={counts['passed']} "
                f"failed={counts['failed']} declined={counts['declined']} "
                f"toml_skipped={counts['toml_skipped']} unverifiable={counts['unverifiable']} "
                f"other_skip={counts['other_skip']} ref_errors={len(refs['REFERENCE EXECUTOR ERRORS'])} "
                f"unverifiable_listed={len(refs['UNVERIFIABLE BUNDLES'])} rc={rc} wall_s={wall} "
                f"lane_timeout_s={LANE_TIMEOUT[tier]} hostfill_wall_s={w.get(tier + '-hostfill', (0, None))[1]}"
            )
            by_tag = collections.Counter()
            by_kind = collections.Counter()
            unmapped = []
            for name, (status, text) in sorted(res.items()):
                if status == "unverifiable":
                    detail.append(
                        f"UNVERIFIABLE {arch} {name}: {clean(text).strip()[:200]}"
                    )
                if status == "other_skip":
                    detail.append(
                        f"OTHER_SKIP {arch} {name}: {clean(text).strip()[:200]}"
                    )
                if status != "failed":
                    continue
                row = by_name.get(name)
                tags = tags_for(row, arch) if row else ["no-row"]
                kind = failure_kind(text)
                hstat = host.get(name, ("not-rerun", ""))[0]
                by_kind[kind] += 1
                for t in tags or ["UNMAPPED"]:
                    by_tag[t] += 1
                axes = " ".join(f"{a}={row[a]}" for a in AXES) if row else ""
                line = (
                    f"FAIL {arch} {name} case_id={row['case_id'] if row else '?'} "
                    f"tags={'+'.join(tags) or 'UNMAPPED'} kind={kind} hostfill={hstat} | {axes}"
                )
                detail.append(line)
                if not tags:
                    unmapped.append(line)
            report.append(
                "    failures by tag: " + json.dumps(dict(sorted(by_tag.items())))
            )
            report.append(
                "    failures by kind: " + json.dumps(dict(by_kind.most_common()))
            )
            hcount = collections.Counter(
                host.get(n, ("not-rerun",))[0]
                for n, v in res.items()
                if v[0] == "failed"
            )
            report.append(
                "    hostfill re-run of failures: " + json.dumps(dict(hcount))
            )
            for sec, items in refs.items():
                for it in items:
                    detail.append(f"{sec} {arch} {tier}: {it}")
            for u in unmapped:
                report.append("    " + u)
        if (Path(d) / "mlops.xml").exists():
            res = parse_xml(Path(d) / "mlops.xml") or {}
            counts = collections.Counter(v[0] for v in res.values())
            ml = (
                [ln for ln in steps.read_text().splitlines() if " mlops " in ln]
                if steps.exists()
                else []
            )
            report.append(
                f"{arch} job{job} HIP_MLOPS quick+standard: total={len(res)} "
                + json.dumps(dict(counts))
            )
            report += ["    " + re.sub(r"\[  SKIPPED \].*", "", ln)[:200] for ln in ml]
    (out / "part-a.txt").write_text("\n".join(report) + "\n")
    (out / "part-a-detail.txt").write_text("\n".join(detail) + "\n")
    print("\n".join(report))


def part_b(results, out, by_key, predicted_csv=None):
    predicted = collections.defaultdict(set)
    if predicted_csv:
        with open(predicted_csv, newline="") as f:
            for p in csv.DictReader(f):
                predicted[(p["engine"], p["arch"])].add(p["case_id"])
    rows_by_id = {r["case_id"]: r for r in by_key.values()}
    report = []
    detail = []
    for d in sorted(
        glob.glob(str(results / "runner" / "*-job*" / "*" / "results.jsonl"))
    ):
        engine = Path(d).parent.name
        arch = Path(d).parent.parent.name.split("-job")[0]
        if "-" in arch:
            continue
        cases = {}
        crashes = []
        runs = []
        for ln in open(d):
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("record") == "case":
                cases[rec["case_id"]] = rec
            elif rec.get("record") == "crash":
                crashes.append(rec)
            elif rec.get("record") in ("run", "summary"):
                runs.append(rec)
        outc = collections.Counter(r.get("outcome") for r in cases.values())
        acc = {cid for cid, r in cases.items() if r.get("graph_outcome") == "accepted"}
        verd = collections.Counter(
            r.get("verdict", "-") for cid, r in cases.items() if cid in acc
        )
        pred = predicted.get((engine, arch), set())
        report.append(
            f"{arch} {engine}: rows={len(cases)} accepted={len(acc)} predicted={len(pred)} "
            f"accepted&predicted={len(acc & pred)} accepted-not-predicted={len(acc - pred)} "
            f"predicted-not-accepted={len(pred - acc)} crashes={len(crashes)} "
            f"verdicts={json.dumps(dict(verd))} outcomes={json.dumps(dict(outcomes_clean(outc)))}"
        )
        fatal = [r.get("fatal") for r in runs if r.get("fatal")]
        if fatal:
            report.append(f"    FATAL: {fatal}")
        for cid in sorted(acc):
            r = cases[cid]
            v = r.get("verdict", "-")
            if v in ("pass",):
                continue
            row = rows_by_id.get(cid, {})
            axes = " ".join(f"{a}={row.get(a, '?')}" for a in AXES)
            comp = [
                (
                    c.get("against"),
                    c.get("pass"),
                    c.get("max_abs_err", c.get("summary")),
                )
                for c in r.get("comparisons", [])
                if isinstance(c, dict)
            ]
            detail.append(
                f"{arch} {engine} {cid} level={r.get('level')} verdict={v} "
                f"engine_status={r.get('engine_status')} gpu_ref={r.get('gpu_ref_status')} "
                f"cpu_ref={r.get('cpu_ref_status')} unwritten={r.get('engine_unwritten_outputs')} "
                f"skip={r.get('skip_reason')} engine_error={clean(r.get('engine_error', ''))[:160]} "
                f"comps={json.dumps(comp)[:300]} | {axes}"
            )
        for cid in sorted(acc - pred):
            row = rows_by_id.get(cid, {})
            detail.append(
                f"{arch} {engine} {cid} ACCEPTED-NOT-PREDICTED | "
                + " ".join(f"{a}={row.get(a, '?')}" for a in AXES)
            )
        for cid in sorted(pred - acc):
            r = cases.get(cid, {})
            detail.append(
                f"{arch} {engine} {cid} PREDICTED-NOT-ACCEPTED outcome={r.get('outcome')} "
                f"msg={clean(r.get('engine_message', r.get('message', '')))[:160]}"
            )
    (out / "part-b.txt").write_text("\n".join(report) + "\n")
    (out / "part-b-detail.txt").write_text("\n".join(detail) + "\n")
    print("\n".join(report))


def outcomes_clean(c):
    return {str(k): v for k, v in c.items()}


TAG_ORDER = ["T11", "T5", "T7", "T6", "T4", "T3", "T2", "T9"]
TAG_TENSORS = {
    "T3": ("sdpa_fwd::STATS",),
    "T4": ("logit_max", "score_sum_exp"),
    "T6": ("rng_dump",),
    "T2": ("sdpa_fwd::O",),
    "T5": ("sdpa_fwd::O",),
    "T7": ("sdpa_fwd::O",),
    "T11": ("Unverifiable",),
}


def collect_outcomes(validate_dirs):
    """arch -> gtest name -> list of (run file, status, text), over default and host-fill runs."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for vdir in validate_dirs:
        for d in sorted(glob.glob(str(Path(vdir) / "*-job*"))):
            arch = os.path.basename(d).split("-job")[0]
            for x in sorted(glob.glob(str(Path(d) / "*.xml"))):
                if os.path.basename(x) == "mlops.xml":
                    continue
                for name, (status, text) in (parse_xml(x) or {}).items():
                    out[arch][name].append((x, status, text))
    return out


def verdict(runs):
    """failed if any run failed, passed if any run passed, else the first status seen."""
    statuses = [s for _, s, _ in runs]
    if "failed" in statuses:
        return "failed"
    if "passed" in statuses:
        return "passed"
    return statuses[0]


def primary_tag(row, arch, runs):
    tags = [t for t in tags_for(row, arch) if t in TICKET]
    if not tags:
        return None
    if "T5" in tags:
        return "T5"
    text = " ".join(t for _, s, t in runs if s == "failed")
    hit = [
        t
        for t in TAG_ORDER
        if t in tags and any(k in text for k in TAG_TENSORS.get(t, ()))
    ]
    return (hit or [t for t in TAG_ORDER if t in tags])[0]


def find_skip(entries, name, arch):
    for e in entries:
        if e.get("archs") and not any(a in arch for a in e["archs"]):
            continue
        if any(fnmatch.fnmatchcase(name, f) for f in e["filters"]):
            return e
    return None


def skip_check(toml_path, validate_dirs, out, levels, placements):
    import tomllib

    entries = tomllib.loads(Path(toml_path).read_text()).get("test_skips", [])
    _, by_name = load_rows(levels, placements)
    outcomes = collect_outcomes(validate_dirs)
    report = [f"skip check: {toml_path} ({len(entries)} test_skips entries)"]
    problems = 0
    for arch in sorted(outcomes):
        counts = collections.Counter()
        per_entry = collections.Counter()
        for name, runs in sorted(outcomes[arch].items()):
            v = verdict(runs)
            e = find_skip(entries, name, arch)
            counts[(v, "skipped" if e else "run")] += 1
            if e:
                per_entry[str(reason_tag(e["reason"]))] += 1 if v == "failed" else 0
            row = by_name.get(name)
            if v == "failed" and not e:
                problems += 1
                tag = primary_tag(row, arch, runs) if row else None
                report.append(f"    NOT SKIPPED {arch} {name} tag={tag}")
            elif v == "failed":
                tags = tags_for(row, arch) if row else []
                if reason_tag(e["reason"]) not in tags:
                    problems += 1
                    report.append(
                        f"    WRONG REASON {arch} {name} tags={tags} reason={e['reason']}"
                    )
            elif v == "passed" and e:
                problems += 1
                report.append(f"    PASSING SKIPPED {arch} {name} reason={e['reason']}")
        report.append(
            f"{arch}: "
            + json.dumps({f"{a}/{b}": n for (a, b), n in sorted(counts.items())})
        )
        report.append(
            f"    failures skipped by tag: "
            + json.dumps(dict(sorted(per_entry.items())))
        )
    report.append(f"problems={problems}")
    (out / "skip-check.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return problems


def main():
    global BUNDLES
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results",
        type=Path,
        required=True,
        help="folder that holds validate/ and runner/ run folders",
    )
    ap.add_argument(
        "--levels", type=Path, required=True, help="levels.csv the bundles come from"
    )
    ap.add_argument(
        "--placements",
        type=Path,
        required=True,
        help="placements.csv from make_bundles.py",
    )
    ap.add_argument(
        "--bundle-dir",
        type=Path,
        default=BUNDLES,
        help="bundle tree to read Stats layouts from (default: the repository tree)",
    )
    ap.add_argument(
        "--predicted",
        type=Path,
        help="part B: CSV of engine,arch,case_id expected to be accepted",
    )
    ap.add_argument("--part", default="both", choices=["a", "b", "both"])
    ap.add_argument(
        "--out", type=Path, help="report folder (default: <results>/analysis)"
    )
    ap.add_argument(
        "--skips", default=None, help="engine TOML whose [[test_skips]] to apply"
    )
    ap.add_argument(
        "--validate",
        action="append",
        type=Path,
        default=None,
        help="validate folders for --skips (default: <results>/validate); repeat for more",
    )
    args = ap.parse_args()
    BUNDLES = args.bundle_dir
    results = args.results
    out = args.out or results / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if args.skips:
        vdirs = args.validate or [results / "validate"]
        sys.exit(
            1 if skip_check(args.skips, vdirs, out, args.levels, args.placements) else 0
        )
    by_key, by_name = load_rows(args.levels, args.placements)
    if args.part in ("a", "both"):
        part_a(results, out, by_name)
    if args.part in ("b", "both"):
        part_b(results, out, by_key, args.predicted)


if __name__ == "__main__":
    main()
