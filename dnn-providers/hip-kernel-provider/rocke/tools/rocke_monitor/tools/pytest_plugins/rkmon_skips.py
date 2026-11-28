# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Skipped tests, never silent: the line the review checks' pytest plugin (rkmon_env.py) prints for each test a run
skips, and the reports the checks build from those lines (tools/pytest_vs_base.py, tools/installed_tests.py).

A skip for a tool CI's test lanes have and this host may lack (TOOLS: cmake with ctest, PyYAML) reads
"NOT RUN: <tool name> missing", never a pass: a skip counts so when its reason names the tool and the tool is missing
from the run's own environment (PATH, the importable modules). Every other skip is listed with its reason.

The line: "rkmon-skip:", then the missing tool (or -), the test id and the skip's reason, each after a tab.

As a script (python -I rkmon_skips.py result|report [--where PATH] FILE...), it reads such lines from FILEs:
`result` prints three lines: the skipped count, the NOT RUN text and the summary (either may be empty), and `report`
the section's lines (the NOT RUN lines, then "SKIPPED k:" with each test and its reason, grouped by reason over
LISTED)."""

import argparse
import importlib.util
import re
import shutil
import sys

PREFIX = "rkmon-skip:"
# (tool, what its skip reasons say, whether the run's environment has it)
TOOLS = (
    (
        "cmake",
        re.compile(r"(?i)\b(cmake|ctest)\b"),
        lambda: bool(shutil.which("cmake") and shutil.which("ctest")),
    ),
    (
        "PyYAML",
        re.compile(r"(?i)\b(py)?yaml\b"),
        lambda: importlib.util.find_spec("yaml") is not None,
    ),
)
LISTED = 20  # up to this many skips each is listed on its own line; over it, grouped by reason
GROUPS = 12  # reasons shown when grouped
REASON_SHOWN = 70  # a reason's characters in the summary


def missing_tools() -> set[str]:
    """The TOOLS this process's environment lacks."""
    return {name for name, _pat, have in TOOLS if not have()}


def _one_line(s) -> str:
    return " ".join(str(s).split())


def line(test_id: str, reason: str, missing: set[str]) -> str:
    """The plugin's line for one skipped test; `missing` is missing_tools() of the run."""
    reason = _one_line(reason)
    tool = next(
        (name for name, pat, _have in TOOLS if name in missing and pat.search(reason)),
        "-",
    )
    return f"{PREFIX}\t{tool}\t{_one_line(test_id)}\t{reason}"


def parse(text: str) -> list[tuple[str, str, str]]:
    """(missing tool or "", test id, reason) for each skip line in `text`, once each, in order."""
    out, seen = [], set()
    for ln in text.splitlines():
        if not ln.startswith(PREFIX + "\t"):
            continue
        parts = ln.split("\t", 3)
        if len(parts) < 4:
            continue
        item = ("" if parts[1] == "-" else parts[1], parts[2], parts[3])
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def not_run(skips) -> dict[str, list[str]]:
    """Missing tool -> the ids of the tests it kept from running."""
    out: dict[str, list[str]] = {}
    for tool, tid, _reason in skips:
        if tool:
            out.setdefault(tool, []).append(tid)
    return out


def not_run_text(skips) -> str:
    """ "NOT RUN: cmake missing (3 tests), PyYAML missing (1 test)", or "" when no tool was missing."""
    miss = not_run(skips)
    return (
        (
            "NOT RUN: "
            + ", ".join(
                f"{t} missing ({len(ids)} test{'s' if len(ids) != 1 else ''})"
                for t, ids in sorted(miss.items())
            )
        )
        if miss
        else ""
    )


def _short(reason: str) -> str:
    return reason if len(reason) <= REASON_SHOWN else reason[: REASON_SHOWN - 1] + "…"


def by_reason(skips) -> list[tuple[str, list[str]]]:
    """(reason, ids) for the skips, the most frequent reason first."""
    groups: dict[str, list[str]] = {}
    for _tool, tid, reason in skips:
        groups.setdefault(reason, []).append(tid)
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def summary(skips) -> str:
    """ "k skipped: <n> <reason>; <n> <reason>; …" (the two most frequent reasons), or "" without skips."""
    if not skips:
        return ""
    groups = by_reason(skips)
    shown = "; ".join(f"{len(ids)} {_short(r)}" for r, ids in groups[:2])
    return f"{len(skips)} skipped: {shown}{'; …' if len(groups) > 2 else ''}"


def report(skips, where: str = "") -> list[str]:
    """The section's lines: a NOT RUN line per missing tool, then "SKIPPED k:" with each skipped test and its reason
    (grouped by reason over LISTED, every id then in `where`)."""
    out = []
    for tool, ids in sorted(not_run(skips).items()):
        out.append(
            f"NOT RUN: {tool} missing ({len(ids)} test{'s' if len(ids) != 1 else ''}; CI's lanes have it): "
            + " ".join(ids[:6])
            + (f" (+{len(ids) - 6} more)" if len(ids) > 6 else "")
        )
    if not skips:
        return out
    tag = lambda tool: f" [NOT RUN: {tool} missing]" if tool else ""  # noqa: E731
    if len(skips) <= LISTED:
        out.append(f"SKIPPED {len(skips)}:")
        out += [f"   {tid} - {reason}{tag(tool)}" for tool, tid, reason in skips]
        return out
    groups = by_reason(skips)
    out.append(
        f"SKIPPED {len(skips)}, by reason{f' (every test and its reason: {where})' if where else ''}:"
    )
    for reason, ids in groups[:GROUPS]:
        more = f" (+{len(ids) - 3} more)" if len(ids) > 3 else ""
        out.append(f"   {len(ids)} x {reason}: {' '.join(ids[:3])}{more}")
    if len(groups) > GROUPS:
        rest = sum(len(ids) for _r, ids in groups[GROUPS:])
        out.append(f"   ({len(groups) - GROUPS} more reasons, {rest} tests)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("what", choices=("result", "report"))
    ap.add_argument(
        "--where",
        default="",
        help="the file that lists every skip, named when the report groups them",
    )
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    text = ""
    for f in a.files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text += fh.read() + "\n"
        except OSError:
            pass
    skips = parse(text)
    if a.what == "result":
        print(f"{len(skips)}\n{not_run_text(skips)}\n{summary(skips)}")
    else:
        for ln in report(skips, a.where):
            print(ln)
    return 0


if __name__ == "__main__":
    sys.exit(main())
