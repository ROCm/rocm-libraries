# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Output formats. Any CI can read at least one of these.

text      for people at a terminal
json      every number (`report`) or every finding (`check`)
junit     one test case per check id; CTest, Jenkins, GitLab and GitHub dashboards read it
github    `::warning file=...,line=...::` lines, shown on the PR diff
markdown  before/after table for a PR description
"""
from __future__ import annotations

import json
from xml.sax.saxutils import escape, quoteattr

import slo_config as C
from slo_check import CHECK_TITLE

_WARN_KINDS = ("worse", "over_target", "error")


def warnings_of(findings):
    return [f for f in findings if f["kind"] in _WARN_KINDS]


def failing(findings, fail_on: str):
    if fail_on == "never":
        return []
    return [
        f
        for f in findings
        if f["kind"] in _WARN_KINDS and f["enforce"] and f["kind"] != "error"
    ]


def text(findings, current, fail_on: str) -> str:
    warn = warnings_of(findings)
    better = [f for f in findings if f["kind"] == "better"]
    label = "FAIL" if fail_on != "never" else "WARN"
    lines = []
    for f in sorted(warn, key=lambda f: (f["check"], f["path"])):
        tag = "ERROR" if f["kind"] == "error" else (label if f["enforce"] else "WARN")
        lines.append(f"{tag} {f['check']} {f['path']}:{f['line']}  {f['message']}")
    if better:
        lines.append("")
        lines.append(f"Better than baseline ({len(better)}):")
        for f in sorted(better, key=lambda f: (f["check"], f["path"])):
            lines.append(f"  {f['check']} {f['path']}  {f['message']}")
    lines.append("")
    lines.append(
        f"{len(current['files'])} files checked: {len(warn)} warnings, {len(better)} better."
    )
    return "\n".join(lines)


def junit(findings, current, fail_on: str) -> str:
    checks = sorted(CHECK_TITLE)
    fail_set = {id(f) for f in failing(findings, fail_on)}
    cases = []
    n_fail = n_err = 0
    for cid in checks + (
        ["SLO000"] if any(f["check"] == "SLO000" for f in findings) else []
    ):
        mine = [f for f in findings if f["check"] == cid and f["kind"] in _WARN_KINDS]
        body = "\n".join(f"{f['path']}:{f['line']} {f['message']}" for f in mine)
        name = quoteattr(f"{cid} {CHECK_TITLE.get(cid, 'Parse errors')}")
        inner = ""
        errs = [f for f in mine if f["kind"] == "error"]
        fails = [f for f in mine if id(f) in fail_set]
        if errs:
            n_err += 1
            inner += f'<error message={quoteattr(f"{len(errs)} files could not be parsed")}/>'
        elif fails:
            n_fail += 1
            inner += f'<failure message={quoteattr(f"{len(fails)} findings")}>{escape(body)}</failure>'
        if body:
            inner += f"<system-out>{escape(body)}</system-out>"
        cases.append(
            f'  <testcase classname="rocke.slo" name={name}>{inner}</testcase>'
        )
    head = (
        f'<testsuite name="rocke-slo" tests="{len(cases)}" failures="{n_fail}" '
        f'errors="{n_err}" skipped="0">'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + head
        + "\n"
        + "\n".join(cases)
        + "\n</testsuite>\n"
    )


def github(findings, fail_on: str, prefix: str = "") -> str:
    fail_set = {id(f) for f in failing(findings, fail_on)}
    out = []
    for f in warnings_of(findings):
        level = "error" if (id(f) in fail_set or f["kind"] == "error") else "warning"
        msg = f["message"].replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        out.append(
            f"::{level} file={prefix}{f['path']},line={f['line']},title={f['check']}::{msg}"
        )
    return "\n".join(out)


def markdown(findings, current) -> str:
    warn = warnings_of(findings)
    better = [f for f in findings if f["kind"] == "better"]
    lines = ["### rocKE SLO check", ""]
    lines.append(
        f"{len(current['files'])} files checked, {len(warn)} warnings, "
        f"{len(better)} better than baseline. Warn only."
    )
    for title, rows in (
        ("Worse or over target", warn),
        ("Better than baseline", better),
    ):
        if not rows:
            continue
        lines += [
            "",
            f"**{title}**",
            "",
            "| Check | File | Metric | Before | After | Target |",
            "|---|---|---|---|---|---|",
        ]
        for f in sorted(rows, key=lambda f: (f["check"], f["path"])):
            tgt = "" if f["target"] is None else f["target"]
            old = "" if f["old"] is None else f["old"]
            lines.append(
                f"| {f['check']} | `{f['path']}` | {f['metric']} | {old} | {f['new']} | {tgt} |"
            )
    return "\n".join(lines) + "\n"


def report_text(current, summary: dict) -> str:
    """Human summary for `report`: worst files against each target."""
    files = current["files"]
    lines = [f"{len(files)} files measured.", ""]
    lines.append("Files over each target today:")
    for metric, (limit, direction) in C.TARGETS.items():
        if metric not in summary:
            continue
        op = "<=" if direction == "max" else ">="
        lines.append(f"  {metric:15} target {op} {limit:<6} over: {summary[metric]}")
    for metric, label in (
        ("sloc", "Biggest files (SLOC)"),
        ("raw_per_100", "Most raw IR ops per 100 SLOC"),
        ("longest_fn", "Longest functions"),
        ("overlap_max", "Most copy-paste overlap"),
    ):
        lines += ["", label + ":"]
        for r in sorted(files.values(), key=lambda r: r[metric] or 0, reverse=True)[:8]:
            extra = (
                f"  {r['longest_fn_name']}"
                if metric == "longest_fn"
                else f"  with {r['overlap_with']}" if metric == "overlap_max" else ""
            )
            lines.append(f"  {r[metric]:>8}  {r['path']}{extra}")
    return "\n".join(lines)


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=sorted)
