# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scan a PR's added lines for repo-rule violations (tracker IDs, PR numbers in code,
absolute perf numbers, kernel-selecting env reads). Usage: python -I rule_scan.py <diff file>
Every added line of every file counts, `experiments/` and data files included; the PR body's diff (PR_BODY.md) is
scanned the same way but for the committed-files sections. After the four first sections:
- "perf (data files)": a measured figure in a .json, .jsonl or .md file under a key or column that names it (tok/s,
  TFLOP/s, GFLOPS, GB/s, bandwidth, latency, ms, us, MFU), where the plain perf pattern sees no unit;
- "PR refs (committed files)": `#NNNN` in a Markdown file, `PR NNNN` and `/pull/NNNN` in any committed file (a
  committed file never points to a PR: the number names its author);
- "dated narrative": an added line, comment or docstring that starts with a date and a colon (`11-05:`,
  `2026-11-05 14:00Z:`), or that tells a run by its date (`measured on 2026-11-05`).
The tracker keys come from RKMON_TRACKER_KEYS (comma-separated; config tracker_keys and jira.project); without them,
any Jira-style ID (KEY-NNN) counts, except a few technical names (SHA-256, UTF-8, ...).
"""
import collections
import os
import re
import sys

_KEYS = [
    k.strip() for k in os.environ.get("RKMON_TRACKER_KEYS", "").split(",") if k.strip()
]
NOT_TRACKERS = {
    "SHA",
    "UTF",
    "ISO",
    "RFC",
    "LLVM",
    "GFX",
    "FP",
    "BF",
    "MI",
    "CVE",
    "X86",
    "AVX",
    "ARM",
    "POSIX",
}
TRACKER = re.compile(
    r"\b(" + "|".join(map(re.escape, _KEYS)) + r")-\d+\b"
    if _KEYS
    else r"\b(?!(?:"
    + "|".join(sorted(NOT_TRACKERS))
    + r")-)([A-Z][A-Z0-9]{1,9})-\d{2,}\b"
)
# A PR or issue number such as "#1234" in a code comment (Rule 1, like tracker IDs). Not "owner/repo#N", "&#N;" or hex
# colours; Markdown docs may cite PRs, so only non-.md files count. Data files are not code either: a hipTensor manifest
# writes sizes as `K=#1234`, hundreds of false hits.
PRREF = re.compile(r"(?<![\w/&#])#\d{4,6}\b")
PRREF_SKIP = (".md", ".manifest", ".json", ".csv", ".tsv")
PERF = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|TFLOPS|TFLOP/s|GFLOPS|GB/s|TB/s|tok/s|tokens/s)(?![\w/])"
    r"|\b\d+(?:\.\d+)?\s*x\s+(?:faster|slower)\b|\bspeedup\s*[=:]?\s*\d"
    # A unit before its figures, "TFLOPS (...): median **N.N**, p10 ..." (a plan file's table, for example).
    r"|\b(?:TFLOPS|TFLOP/s|GFLOPS|GB/s|TB/s)\b[^\n]{0,80}?\b\d+\.\d+\b"
)
# Measured times and sizes (Compliance: comments, docstrings and commit messages stay qualitative; a PR body, which a
# squash commits to develop, may give ratios only): "about 1-5 MB", "Measured, one process:" over build counts and
# seconds, a body that tables MB and seconds. In code, a figure counts on a line with a measuring word or within the
# three lines after one; in the body, every figure counts.
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
FIGURE = re.compile(
    rf"(?<![\w.])(?:{NUM})(?:\s*[-–]\s*(?:{NUM}))?\s*(?:s|secs?|seconds|mins?|minutes|[KMG]i?B)\b"
)
MEASURE_WORD = re.compile(
    r"(?i)\b(?:measured|took|takes|median|average|mean|about|roughly|approx(?:imately)?)\b|~\s*\d"
)
BODY = "PR_BODY.md"
DATA_FILES = (".json", ".jsonl", ".md")
# A figure under a key or a column that names a measure: `"tflops": 512.3`, `tok_per_s: 1800`, `latency_ms = 0.21`.
DATA_PERF = re.compile(
    r"(?i)(?<![\w-])(?:\w+_)?(?:tok_?s|toks?_?per_?s(?:ec)?|tokens?_?per_?s(?:ec)?|tflop_?s|tflops?|gflops|gb_?s|gbps"
    r"|tb_?s|mfu|ms|us|(?:bandwidth|latency)\w*)[\"'`]?\s*[:=]\s*\[?\s*-?\d"
)
PRREF_ANY = re.compile(r"(?<![\w/])PR\s*#?\s*\d{3,6}\b|/pull/\d+")
DATED = re.compile(
    r"^\s*(?:[#*>/-]+\s*|[\"']{3}\s*)?(?:\d{4}-)?(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2})?Z?)?\s*(?::|#\d)"
    r"|(?i:\b(?:measured|ran|run|validated|benchmarked)\s+(?:on\s+)?)(?:\d{4}-)?(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
)
ENV = re.compile(
    r"os\.environ(?:\.get)?\s*[\[(]\s*[\"']([A-Z0-9_]+)[\"']\s*(\]\s*=(?!=))?"
)
# Text I/O with no encoding: Windows reads and writes cp1252, so a UTF-8 file (an arrow, a µ) fails the Windows lanes
# on Windows. Python files only, tests included; a one-line call only (a call split over lines is not seen), binary
# modes skipped. The arguments are read to the call's own closing parenthesis: `write_text(fmt(x), encoding="utf-8")`
# has one.
TEXT_CALL = re.compile(r"\.(?:read_text|write_text)\(|(?<![\w.])open\(")
ENCODING_KW = re.compile(r"\bencoding\s*=")
BINARY_MODE = re.compile(r"[\"'][rwax+]*b[rwax+]*[\"']")
SHOW = 8  # lines per section; every section keeps its header and count, so a long list never hides the next one
ROCKE_PREFIX = (os.environ.get("RKMON_ROCKE_PATH", "").strip("/") + "/").lstrip(
    "/"
)  # shown relative to rocKE


def call_args(line, start):
    """The text between a call's `(` (ending at `start`) and its matching `)`; None if the call goes on past the line."""
    depth, quote, i = 1, None, start
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return line[start:i]
        i += 1
    return None


def text_io_without_encoding(line):
    for m in TEXT_CALL.finditer(line):
        args = call_args(line, m.end())
        if args is None or ENCODING_KW.search(args):
            continue
        if m.group(0).endswith("open(") and BINARY_MODE.search(args):
            continue
        return True
    return False


hits = collections.defaultdict(list)
path = None
measured_left = 0  # added lines still inside a measuring word's reach
for raw in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if raw.startswith("+++ b/"):
        path = raw[6:].strip()
        measured_left = 0
        continue
    if not raw.startswith("+") or raw.startswith("+++"):
        continue
    line = raw[1:].rstrip("\n")
    is_test = "/tests/" in path or "/benchmarks/" in path
    for m in TRACKER.finditer(line):
        hits["tracker"].append(f"{path}: {m.group(0)}")
    if not path.endswith(PRREF_SKIP):
        for m in PRREF.finditer(line):
            hits["tracker"].append(f"{path}: {m.group(0)}")
    if MEASURE_WORD.search(line):
        measured_left = 4
    figure = FIGURE.search(line) and (path == BODY or measured_left > 0)
    measured_left -= 1
    if PERF.search(line) or figure:
        hits["perf"].append(f"{path}: {line.strip()[:110]}")
    elif path.endswith(DATA_FILES) and DATA_PERF.search(line):
        hits["perf (data files)"].append(f"{path}: {line.strip()[:110]}")
    if path != BODY:
        if PRREF_ANY.search(line) or (path.endswith(".md") and PRREF.search(line)):
            hits["PR refs (committed files)"].append(f"{path}: {line.strip()[:110]}")
        if DATED.search(line):
            hits["dated narrative"].append(f"{path}: {line.strip()[:110]}")
    if not is_test:
        for m in ENV.finditer(line):
            if not m.group(
                2
            ):  # `os.environ["X"] = ...` sets the variable for a child; Rule 3 is about reads
                hits["env (non-test)"].append(f"{path}: {m.group(1)}")
    if (
        path.endswith(".py")
        and not line.lstrip().startswith("#")
        and text_io_without_encoding(line)
    ):
        hits["text I/O without encoding"].append(f"{path}: {line.strip()[:110]}")
for kind in (
    "tracker",
    "perf",
    "env (non-test)",
    "text I/O without encoding",
    "perf (data files)",
    "PR refs (committed files)",
    "dated narrative",
):
    rows = sorted(set(hits.get(kind, [])))
    print(f"== {kind}: {len(hits.get(kind, []))}")
    for r in rows[:SHOW]:
        print("  " + (r.replace(ROCKE_PREFIX, "") if ROCKE_PREFIX else r))
    if len(rows) > SHOW:
        print(f"  … {len(rows) - SHOW} more")
