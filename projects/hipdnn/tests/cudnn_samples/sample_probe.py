#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launcher for one cuDNN-sample ctest case; also writes that sample's report sidecar.

Two modes:

  compile  run a build command for one translation unit
  run      run the built sample, with a no-crash pass bar

The run bar is deliberately *not* the exit code. A sample whose REQUIRE fails because no
provider plan exists on this GPU is a capability gap, not a shim defect, and must not red
the job; a sample that dies on a signal, or exits before Catch2 reports a result (a
missing shared library, say), must. All of these are non-zero exits, so the distinction
has to be drawn here rather than by ctest.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Catch2 v3 end-of-run summary, in its four shapes:
#   "All tests passed (7 assertions in 1 test case)"
#   "assertions: 4 | 2 passed | 2 failed"
#   "assertions: - none -"          (paired with "test cases: 1 | 1 skipped")
#   "No tests ran"                  (no test case registered or selected)
_ALL_PASSED_RE = re.compile(r"All tests passed \((\d+) assertion", re.IGNORECASE)
_ASSERT_NONE_RE = re.compile(r"assertions:\s*-\s*none\s*-", re.IGNORECASE)
_NO_TESTS_RE = re.compile(r"No tests ran", re.IGNORECASE)
_ASSERT_TOTAL_RE = re.compile(r"assertions:\s+(\d+)\s*\|([^\n]*)", re.IGNORECASE)
_CASES_RE = re.compile(r"test cases:\s+(\d+)\s*\|([^\n]*)", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)
_SKIPPED_RE = re.compile(r"(\d+)\s+skipped", re.IGNORECASE)
# Windows surfaces crashes as a 0xC0000005-style exception code rather than a signal.
_WINDOWS_EXCEPTION_MIN = 0xC0000000


def parse_catch2(output: str):
    """(assertions_total, assertions_failed, cases_skipped) from a Catch2 run.

    None when the output carries no Catch2 summary at all, meaning Catch2 never reached
    the end of its run. That is not the same as "nothing asserted", and the caller must
    not treat it as such.
    """
    total = failed = skipped = 0
    m = _ALL_PASSED_RE.search(output)
    if m:
        total = int(m.group(1))
    elif _ASSERT_NONE_RE.search(output) or _NO_TESTS_RE.search(output):
        total = 0
    else:
        matches = list(_ASSERT_TOTAL_RE.finditer(output))
        if not matches:
            return None
        for m in matches:
            total += int(m.group(1))
            f = _FAILED_RE.search(m.group(2))
            failed += int(f.group(1)) if f else 0
    for m in _CASES_RE.finditer(output):
        s = _SKIPPED_RE.search(m.group(2))
        skipped += int(s.group(1)) if s else 0
    return total, failed, skipped


def write_sidecar(path: Path, tu: str, **fields) -> None:
    """Replace this TU's sidecar with exactly these fields.

    Never merged into: each write is the whole current verdict for the TU, so nothing an
    earlier compile or run left behind (assertion counts, an exit code, a reason) can
    outlive a later case that no longer produces it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"tu": tu, **fields}
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
    parser.add_argument(
        "--sidecar",
        required=True,
        type=Path,
        help="this TU's report sidecar; the harness decides its name",
    )
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
        write_sidecar(args.sidecar, args.tu, tier=args.tier, outcome=outcome)
        # For XFAIL_COMPILE, ctest's verdict comes from PASS_REGULAR_EXPRESSION alone and
        # ignores this status; returning the real one keeps the RUN tier honest.
        return completed.returncode

    summary = parse_catch2(output)
    total, failures, skipped = summary or (0, 0, 0)

    if crashed(completed.returncode):
        write_sidecar(
            args.sidecar,
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

    # A non-zero exit is accepted below only because Catch2's summary accounts for it: a
    # failed REQUIRE exits non-zero. Without a summary nothing accounts for it, and the
    # exit means the sample never got as far as running its cases -- the loader could not
    # resolve a library (127), or the process exited on its own.
    if summary is None and completed.returncode != 0:
        write_sidecar(
            args.sidecar,
            args.tu,
            tier=args.tier,
            outcome="run-failed",
            exit_code=completed.returncode,
            reason=f"exited {completed.returncode} before Catch2 reported a result",
        )
        print(
            f"::error title=cuDNN sample did not run::{args.tu} exited with code "
            f"{completed.returncode} and no Catch2 summary"
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
        args.sidecar,
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
