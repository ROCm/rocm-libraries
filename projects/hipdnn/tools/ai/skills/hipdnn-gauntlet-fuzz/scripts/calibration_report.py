#!/usr/bin/env python3
"""Read an oracle calibration: per axis value, how often the oracle cannot be trusted, before the full fuzz.

    python calibration_report.py --rows <dir of cal*.json> --graded <grader output .json> [--run <key prefix>]
Needs the SDPA backward grader's output (gauntlet/tools/sdpa_bwd_aotriton_check.py); for other ops compare the CPU and GPU references.
Flags per case:
  ORACLE   the harness reference differs from the independent truth by > 1% of the value range: if it concentrates on an axis value,
           that value makes graphs no fixed tolerance can judge (bound it, or move it to a probe); if it covers a whole feature,
           suspect the grader's rebuild of the graph first
  VACUOUS  the rule's limit is at least half of the largest value: the rule cannot tell good from bad there
  FAIL     the engine misses both references (an early signal; the full run measures it)
"""
import argparse, collections, glob, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--graded", required=True)
ap.add_argument("--run", default=None)
a = ap.parse_args()
rows = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(a.rows, "cal*.json"))}
graded = json.load(open(a.graded))
per = collections.defaultdict(collections.Counter)
cases = collections.Counter()
for key, v in graded.items():
    case = key.split("/")[-1]
    if case not in rows or "tensors" not in v:
        cases["not graded"] += 1
        continue
    ts = v["tensors"].values()
    flags = set()
    if any(t["ref_max"] > 0 and t["harness_ref_vs_torch"] > 0.01 * t["ref_max"] for t in ts):
        flags.add("ORACLE")
    if any(t["ref_max"] > 0 and t["threshold"] >= 0.5 * t["ref_max"] for t in ts):
        flags.add("VACUOUS")
    if not v["pass"]:
        flags.add("FAIL")
    cases["graded"] += 1
    for f in flags:
        cases[f] += 1
    for ax, val in rows[case]["axes"].items():
        per[(ax, json.dumps(val))]["n"] += 1
        for f in flags:
            per[(ax, json.dumps(val))][f] += 1
print("cases:", dict(cases))
print("axis values with ORACLE flags (the ones to bound or move to a probe):")
for (ax, val), c in sorted(per.items(), key=lambda kv: -kv[1]["ORACLE"]):
    if c["ORACLE"]:
        print(f"  {ax}={val}: {c['ORACLE']} of {c['n']} ORACLE, {c['VACUOUS']} VACUOUS, {c['FAIL']} FAIL")
