#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Summarize a kernel corpus run: coverage, outcomes, and every failing kernel with a repro.

Merges <out>/shard-*/results.jsonl (the latest row per case wins, so a rerun replaces earlier
results) and writes <out>/summary.json, <out>/summary.md and one single-case YAML per failing
case under <out>/repro/, runnable with hipblaslt-test --yaml.

    report.py --out corpus-gfx90a
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_index  # noqa: E402
import plan  # noqa: E402

FAILING = ("fail", "crash", "timeout")
STATUS_ORDER = (
    "pass",
    "fail",
    "crash",
    "timeout",
    "refused",
    "skip",
    "unsupported",
    "deferred",
    "unmapped",
    "unreachable",
    "harness_error",
)


# hipblaslt-test fails, rather than skips, a fast_check case whose configuration fast_check
# cannot check. Those say nothing about the kernel.
_ORACLE_REFUSAL = re.compile(
    r"fast_check (does not support|requires|supports|checks only|checks an? |refuses|"
    r"with MX scales requires)"
)


def classify_failure(message: str, fault: bool = False) -> str:
    """The outcome of a gtest FAILED case from its output."""
    if "NO solution found!" in message:
        return "refused"
    if _ORACLE_REFUSAL.search(message) and not fault:
        return "unsupported"
    return "fail"


def load_results(out: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for path in sorted(out.glob("shard-*/results.jsonl")):
        for row in corpus_index.read_jsonl(path):
            if row["status"] == "fail":
                # Rows written before a classification rule existed are classified again.
                row["status"] = classify_failure(row.get("message", ""), row.get("fault", False))
            latest[row["case_id"]] = row
    # A device-specific solution is unreachable on some devices of an architecture and runs on
    # others. Once any device ran it, its unreachable rows from the rest are not a result.
    ran = {r["index"] for r in latest.values() if r.get("shape") is not None}
    return {
        k: r for k, r in latest.items() if not (r["status"] == "unreachable" and r["index"] in ran)
    }


def case_from_row(row: dict) -> Optional[plan.Case]:
    if row.get("shape") is None or "types" not in row:
        return None
    shape = plan.Shape(row["shape"], row["m"], row["n"], row["k"], row.get("batch", 1))
    trans = row.get("trans", "NN")
    return plan.Case(row["index"], row.get("name", ""), shape, row["types"], trans[0], trans[1])


def first_line(text: str, needle: str = "") -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line and (not needle or needle in line):
            return line[:200]
    return ""


def failure_reason(row: dict) -> str:
    message = row.get("message") or ""
    if row.get("fault"):
        return "GPU memory fault"
    if row["status"] == "timeout":
        return "hung (no output before the case timeout)"
    if row["status"] == "crash":
        return "process crashed"
    for needle in ("fast_check", "mismatch", "expected", "Failure"):
        line = first_line(message, needle)
        if line:
            return line
    return first_line(message) or row["status"]


def summarize(results: dict[str, dict], corpus: list[dict]) -> dict:
    status_counts = collections.Counter(r["status"] for r in results.values())
    by_kernel: dict[int, dict] = {}
    for row in results.values():
        k = by_kernel.setdefault(row["index"], {"statuses": collections.Counter(), "rows": []})
        k["statuses"][row["status"]] += 1
        k["rows"].append(row)

    def kernel_outcome(statuses: collections.Counter) -> str:
        if any(statuses[s] for s in FAILING):
            return "failing"
        if statuses["pass"]:
            return "passing" if statuses["pass"] == sum(statuses.values()) else "partly passing"
        if statuses["refused"] and not statuses["skip"]:
            return "refused every case"
        return "not tested"

    kernel_outcomes = collections.Counter(kernel_outcome(k["statuses"]) for k in by_kernel.values())
    failing_rows = sorted(
        (r for r in results.values() if r["status"] in FAILING),
        key=lambda r: (r["index"], r.get("shape") or ""),
    )
    by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in results.values():
        by_type[row.get("problem_type", "?")][row["status"]] += 1
    skip_reasons = collections.Counter(
        first_line(r.get("message", ""), "fast_check") or first_line(r.get("message", ""))
        for r in results.values()
        if r["status"] == "skip"
    )
    not_run_reasons = collections.Counter(
        first_line(r.get("message", ""), "fast_check") or first_line(r.get("message", ""))
        for r in results.values()
        if r["status"] in ("deferred", "unmapped", "unreachable", "unsupported")
    )
    meta = {}
    for row in results.values():
        for key in ("arch", "device", "hipblaslt_git_version"):
            if row.get(key):
                meta.setdefault(key, row[key])
    return {
        **meta,
        "solutions_in_corpus": len(corpus),
        "solutions_with_results": len(by_kernel),
        "case_status": {s: status_counts[s] for s in STATUS_ORDER if status_counts[s]},
        "kernel_outcomes": dict(kernel_outcomes),
        "failing_cases": len(failing_rows),
        "failing_solutions": len({r["index"] for r in failing_rows}),
        "failures": [
            {
                "case_id": r["case_id"],
                "index": r["index"],
                "status": r["status"],
                "reason": failure_reason(r),
                "problem_type": r.get("problem_type"),
                "shape": [r.get("m"), r.get("n"), r.get("k"), r.get("batch")],
                "name": r.get("name"),
                "bench_cmd": r.get("bench_cmd"),
                "host": r.get("host"),
            }
            for r in failing_rows
        ],
        "by_problem_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "skip_reasons": dict(skip_reasons.most_common(20)),
        "not_run_reasons": dict(not_run_reasons.most_common(20)),
    }


def markdown(summary: dict) -> str:
    lines = [
        f"# Kernel corpus results: {summary.get('arch', '?')}",
        "",
        f"Device {summary.get('device', '?')}, hipBLASLt {summary.get('hipblaslt_git_version', '?')}. "
        f"{summary['solutions_with_results']} of {summary['solutions_in_corpus']} solutions have "
        f"results; {summary['failing_solutions']} solutions fail {summary['failing_cases']} cases.",
        "",
        "| Case outcome | Count |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {v} |" for k, v in summary["case_status"].items()]
    lines += ["", "| Solution outcome | Count |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in sorted(summary["kernel_outcomes"].items())]
    if summary["failures"]:
        lines += [
            "",
            "## Failing cases",
            "",
            "| Solution | Shape (M, N, K, batch) | Outcome | Problem type | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
        for f in summary["failures"]:
            shape = ", ".join(str(x) for x in f["shape"])
            reason = f["reason"].replace("|", "\\|")
            lines.append(
                f"| {f['index']} | {shape} | {f['status']} | {f['problem_type']} | " f"{reason} |"
            )
        lines += [
            "",
            "Each case has a one-case YAML in `repro/` for `hipblaslt-test --yaml`. "
            "The `hipblaslt-bench` command for each is in summary.json; add `-v` to check it "
            "against the host reference.",
        ]
    lines += ["", "## By problem type", "", "| Problem type | Outcomes |", "| --- | --- |"]
    for key, counts in summary["by_problem_type"].items():
        lines.append(f"| {key} | " + ", ".join(f"{s} {n}" for s, n in counts.items()) + " |")
    if summary["skip_reasons"]:
        lines += ["", "## Skip reasons", ""]
        lines += [f"- {n} cases: {k}" for k, n in summary["skip_reasons"].items()]
    if summary["not_run_reasons"]:
        lines += ["", "## Solutions not planned", ""]
        lines += [f"- {n} solutions: {k}" for k, n in summary["not_run_reasons"].items()]
    return "\n".join(lines) + "\n"


def write_repros(results: dict[str, dict], out: Path, common_yaml: Optional[Path]) -> None:
    repro = out / "repro"
    for row in results.values():
        if row["status"] not in FAILING:
            continue
        case = case_from_row(row)
        if case is None:
            continue
        path = repro / f"{row['index']}_{row['shape']}.yaml"
        plan.write_cases_yaml([case], path, common_yaml or Path("hipblaslt_common.yaml"))


def write_atomically(path: Path, text: str) -> None:
    """Shards finishing together each rewrite the merged summary; readers never see half."""
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_reports(out: Path, common_yaml: Optional[Path] = None) -> dict:
    results = load_results(out)
    corpus_path = out / "corpus.jsonl"
    corpus = corpus_index.read_jsonl(corpus_path) if corpus_path.exists() else []
    summary = summarize(results, corpus)
    write_atomically(out / "summary.json", json.dumps(summary, indent=2) + "\n")
    write_atomically(out / "summary.md", markdown(summary))
    if common_yaml is None:
        for path in sorted(out.glob("shard-*/run.json")):
            rocm_dir = json.loads(path.read_text()).get("rocm_dir")
            if rocm_dir:
                common_yaml = Path(rocm_dir) / "bin" / "hipblaslt_common.yaml"
                break
    write_repros(results, out, common_yaml)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Results directory of run.py")
    args = parser.parse_args(argv)
    summary = write_reports(args.out)
    print((args.out / "summary.md").read_text())
    return 1 if summary["failing_cases"] else 0


if __name__ == "__main__":
    sys.exit(main())
