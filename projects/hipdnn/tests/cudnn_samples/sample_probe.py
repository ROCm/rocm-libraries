#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launcher for one cuDNN-sample ctest case; also writes that sample's report sidecar.

Two modes:

  compile  run a build command for one translation unit
  run      run the built sample, with a no-crash pass bar

The run bar is deliberately *not* the exit code. A sample whose REQUIRE fails because no
provider plan exists on this GPU is a capability gap, not a shim defect, and must not red
the job; a sample that dies on a signal must. Both are non-zero exits, so the distinction
has to be drawn here rather than by ctest.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Catch2 v3 end-of-run summary, in its three shapes:
#   "All tests passed (7 assertions in 1 test case)"
#   "assertions: 4 | 2 passed | 2 failed"
#   "assertions: - none -"          (paired with "test cases: 1 | 1 skipped")
_ALL_PASSED_RE = re.compile(r"All tests passed \((\d+) assertion", re.IGNORECASE)
_ASSERT_NONE_RE = re.compile(r"assertions:\s*-\s*none\s*-", re.IGNORECASE)
_ASSERT_TOTAL_RE = re.compile(r"assertions:\s+(\d+)\s*\|([^\n]*)", re.IGNORECASE)
_CASES_RE = re.compile(r"test cases:\s+(\d+)\s*\|([^\n]*)", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)
# Windows surfaces crashes as a 0xC0000005-style exception code rather than a signal.
_WINDOWS_EXCEPTION_MIN = 0xC0000000


def parse_catch2(output: str) -> tuple:
    """(assertions_total, assertions_failed, cases_skipped) from a Catch2 run."""
    total = failed = skipped = 0
    m = _ALL_PASSED_RE.search(output)
    if m:
        total = int(m.group(1))
    elif _ASSERT_NONE_RE.search(output):
        total = 0
    else:
        for m in _ASSERT_TOTAL_RE.finditer(output):
            total += int(m.group(1))
            f = _FAILED_RE.search(m.group(2))
            failed += int(f.group(1)) if f else 0
    for m in _CASES_RE.finditer(output):
        s = _SKIPPED_RE.search(m.group(2))
        skipped += int(s.group(1)) if s else 0
    return total, failed, skipped


def sidecar_path(report_dir: Path, tu: str) -> Path:
    return report_dir / (tu.replace("/", "__").replace(".cpp", "") + ".json")


def write_sidecar(report_dir: Path, tu: str, **fields) -> None:
    """Merge fields into this TU's sidecar.

    The compile case runs before the run case for a given TU (ctest fixtures enforce it),
    so read-modify-write is safe and lets the run case keep the compile case's fields.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_path(report_dir, tu)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    data.update(fields)
    data["tu"] = tu
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def crashed(returncode: int) -> bool:
    if returncode < 0:  # POSIX: killed by signal -N
        return True
    return os.name == "nt" and (returncode & 0xFFFFFFFF) >= _WINDOWS_EXCEPTION_MIN


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["compile", "run"])
    parser.add_argument(
        "--tu", required=True, help="corpus-relative path, e.g. sdpa/fp16_fwd.cpp"
    )
    parser.add_argument(
        "--tier", required=True, choices=["RUN", "XFAIL_COMPILE", "EXCLUDED"]
    )
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = (
        args.command[1:] if args.command and args.command[0] == "--" else args.command
    )
    if not command:
        print("sample_probe: no command given", file=sys.stderr)
        return 2

    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace"
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    # ctest matches PASS_REGULAR_EXPRESSION against this, so the compiler diagnostic has
    # to reach our stdout rather than stay in the child's pipe.
    sys.stdout.write(output)
    sys.stdout.flush()

    if args.mode == "compile":
        ok = completed.returncode == 0
        if args.tier == "XFAIL_COMPILE":
            outcome = "xfail-now-compiles" if ok else "xfail-still-failing"
        else:
            outcome = "compiled" if ok else "compile-failed"
        write_sidecar(args.report_dir, args.tu, tier=args.tier, outcome=outcome)
        # For XFAIL_COMPILE, ctest's verdict comes from PASS_REGULAR_EXPRESSION alone and
        # ignores this status; returning the real one keeps the RUN tier honest.
        return completed.returncode

    total, failures, skipped = parse_catch2(output)

    if crashed(completed.returncode):
        write_sidecar(
            args.report_dir,
            args.tu,
            tier=args.tier,
            outcome="crashed",
            assertions=total,
            assertion_failures=failures,
            cases_skipped=skipped,
            exit_code=completed.returncode,
        )
        print(
            f"::error title=cuDNN sample crashed::{args.tu} died with code {completed.returncode}"
        )
        return 1

    if failures:
        outcome = "ran-with-assertion-failures"
    elif total == 0:
        # Every case skipped, so nothing was actually exercised. Kept distinct from
        # ran-clean: the arch predicates are pinned to a non-NVIDIA answer, which closes
        # a lot of the corpus, and folding that into "clean" would report a green run
        # over samples that asserted nothing.
        outcome = "ran-no-assertions"
    else:
        outcome = "ran-clean"

    write_sidecar(
        args.report_dir,
        args.tu,
        tier=args.tier,
        outcome=outcome,
        assertions=total,
        assertion_failures=failures,
        cases_skipped=skipped,
        exit_code=completed.returncode,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
