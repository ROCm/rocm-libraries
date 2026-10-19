#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launcher for one cuDNN-sample ctest case; also writes that sample's report sidecar.

Three modes:

  compile  run a build command for one translation unit
  run      run the built sample, with a no-crash pass bar
  case     run one Catch2 case of the legacy corpus against its manifest expectation

The run bar is deliberately *not* the exit code. A sample whose REQUIRE fails because no
provider plan exists on this GPU is a capability gap, not a shim defect, and must not red
the job; a sample that dies on a signal, or exits before Catch2 reports a result (a
missing shared library, say), must. All of these are non-zero exits, so the distinction
has to be drawn here rather than by ctest.

A case passes only when its outcome is the one the manifest predicts for its tier:
clean and free of the forbidden text for RUN_CLEAN, anything but crashed or run-failed
for RUN, and for XFAIL_RUN one `expect` entry's outcome with its anchor present and its
forbid absent. Every other result, including an XFAIL_RUN case that now runs clean,
fails it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

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
_CASE_COUNT_RE = re.compile(r"in (\d+) test case|test cases:\s+(\d+)", re.IGNORECASE)
_NO_MATCH_TEXT = "No test cases matched"
# Catch2's exit code under --warn UnmatchedTestSpec when the test spec matched nothing.
_UNMATCHED_SPEC_EXIT = 3
# Windows surfaces crashes as a 0xC0000005-style exception code rather than a signal.
_WINDOWS_EXCEPTION_MIN = 0xC0000000

_LEGACY_TIERS = ("RUN_CLEAN", "RUN", "XFAIL_RUN")


class RunResult(NamedTuple):
    outcome: str
    assertions: int
    assertion_failures: int
    cases_skipped: int
    reason: Optional[str]  # run-failed only


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


def classify_run(returncode: int, output: str) -> RunResult:
    summary = parse_catch2(output)
    total, failures, skipped = summary or (0, 0, 0)
    reason = None
    if crashed(returncode):
        outcome = "crashed"
    # A non-zero exit is accepted below only because Catch2's summary accounts for it: a
    # failed REQUIRE exits non-zero. Without a summary nothing accounts for it, and the
    # exit means the sample never got as far as running its cases -- the loader could not
    # resolve a library (127), or the process exited on its own.
    elif summary is None and returncode != 0:
        outcome = "run-failed"
        reason = f"exited {returncode} before Catch2 reported a result"
    elif failures:
        outcome = "ran-with-assertion-failures"
    elif total == 0:
        # Every case skipped, so nothing was actually exercised. Kept distinct from
        # ran-clean: the arch predicates are pinned to a non-NVIDIA answer, which closes
        # a lot of the corpus, and folding that into "clean" would report a green run
        # over samples that asserted nothing.
        outcome = "ran-no-assertions"
    else:
        outcome = "ran-clean"
    return RunResult(outcome, total, failures, skipped, reason)


def run_command(command):
    completed = subprocess.run(
        command, capture_output=True, text=True, errors="replace"
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    # ctest matches PASS_REGULAR_EXPRESSION against this, so the compiler diagnostic has
    # to reach our stdout rather than stay in the child's pipe.
    sys.stdout.write(output)
    sys.stdout.flush()
    return completed.returncode, output


def compile_mode(args, command) -> int:
    returncode, _ = run_command(command)
    ok = returncode == 0
    if args.tier == "XFAIL_COMPILE":
        outcome = "xfail-now-compiles" if ok else "xfail-still-failing"
    else:
        outcome = "compiled" if ok else "compile-failed"
    write_sidecar(args.sidecar, args.tu, tier=args.tier, outcome=outcome)
    # For XFAIL_COMPILE, ctest's verdict comes from PASS_REGULAR_EXPRESSION alone and
    # ignores this status; returning the real one keeps the RUN tier honest.
    return returncode


def run_mode(args, command) -> int:
    returncode, output = run_command(command)
    result = classify_run(returncode, output)

    if result.outcome == "crashed":
        write_sidecar(
            args.sidecar,
            args.tu,
            tier=args.tier,
            outcome="crashed",
            assertions=result.assertions,
            assertion_failures=result.assertion_failures,
            cases_skipped=result.cases_skipped,
            exit_code=returncode,
        )
        print(
            f"::error title=cuDNN sample crashed::{args.tu} died with code {returncode}"
        )
        return 1

    if result.outcome == "run-failed":
        write_sidecar(
            args.sidecar,
            args.tu,
            tier=args.tier,
            outcome="run-failed",
            exit_code=returncode,
            reason=result.reason,
        )
        print(
            f"::error title=cuDNN sample did not run::{args.tu} exited with code "
            f"{returncode} and no Catch2 summary"
        )
        return 1

    write_sidecar(
        args.sidecar,
        args.tu,
        tier=args.tier,
        outcome=result.outcome,
        assertions=result.assertions,
        assertion_failures=result.assertion_failures,
        cases_skipped=result.cases_skipped,
        exit_code=returncode,
    )
    return 0


def load_case(manifest: Path, name: str):
    """The manifest entry for `name`, its regexes compiled; ValueError if unusable."""
    try:
        cases = json.loads(manifest.read_text(encoding="utf-8"))["legacy"]["cases"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"cannot read legacy.cases from {manifest}: {e}") from e
    entry = cases.get(name)
    if entry is None:
        raise ValueError(f"case {name!r} is not in {manifest}")
    tier = entry.get("tier")
    if tier not in _LEGACY_TIERS:
        raise ValueError(f"case {name!r} has tier {tier!r}; only {_LEGACY_TIERS} run")

    def compile_re(pattern):
        if pattern is None:
            return None
        try:
            return re.compile(pattern, re.MULTILINE)
        except re.error as e:
            raise ValueError(f"case {name!r}: bad regex {pattern!r}: {e}") from e

    forbid = compile_re(entry.get("forbid"))
    expect = [
        (e["outcome"], compile_re(e["anchor"]), compile_re(e.get("forbid")))
        for e in entry.get("expect", [])
    ]
    if tier == "XFAIL_RUN" and not expect:
        raise ValueError(f"XFAIL_RUN case {name!r} has no expect entries")
    return tier, entry.get("reason", ""), forbid, expect


def case_count(output: str):
    m = _CASE_COUNT_RE.search(output)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def describe(result: RunResult, returncode: int) -> str:
    if result.outcome == "crashed":
        if os.name == "nt":
            return f"crashed with code {returncode & 0xFFFFFFFF:#x}"
        return f"crashed with code {returncode}"
    return result.outcome


def case_verdict(tier, forbid, expect, result, returncode, output):
    """(expected, matched expect index or None, mismatch reason or None)."""
    outcome = result.outcome
    # No tier lists run-failed, and its own reason says more than any tier-specific one.
    if outcome == "run-failed":
        return False, None, result.reason
    if tier == "RUN_CLEAN":
        if outcome != "ran-clean":
            got = describe(result, returncode)
            return False, None, f"expected ran-clean, got {got}"
        hit = forbid.search(output) if forbid else None
        if hit:
            return False, None, f"forbidden output present: {hit.group(0)!r}"
        return True, None, None

    if tier == "RUN":
        if outcome == "crashed":
            return False, None, describe(result, returncode)
        return True, None, None

    misses = []
    for i, (want, anchor, entry_forbid) in enumerate(expect):
        if want != outcome:
            continue
        if not anchor.search(output):
            misses.append(f"expect[{i}] anchor {anchor.pattern!r} not found")
            continue
        hit = entry_forbid.search(output) if entry_forbid else None
        if hit:
            misses.append(f"expect[{i}] forbidden output present: {hit.group(0)!r}")
            continue
        return True, i, None
    if misses:
        return False, None, f"{outcome}, but " + "; ".join(misses)
    if outcome == "ran-clean":
        return False, None, "xfail now passes"
    return False, None, f"no expect entry for {describe(result, returncode)}"


def case_mode(args, command) -> int:
    try:
        tier, manifest_reason, forbid, expect = load_case(args.manifest, args.case)
    except ValueError as e:
        print(f"sample_probe: {e}", file=sys.stderr)
        return 2

    returncode, output = run_command(command)
    result = classify_run(returncode, output)

    # A run that selected anything but exactly this one case says nothing about it.
    # Where Catch2 never reported, the case count is unknown rather than wrong, so the
    # earlier run-failed reason is the more useful one.
    if result.outcome != "crashed":
        unmatched = _NO_MATCH_TEXT in output or returncode == _UNMATCHED_SPEC_EXIT
        if unmatched or (result.outcome != "run-failed" and case_count(output) != 1):
            result = result._replace(outcome="run-failed", reason="case not registered")

    expected, matched, mismatch = case_verdict(
        tier, forbid, expect, result, returncode, output
    )
    reason = manifest_reason if expected else mismatch
    write_sidecar(
        args.sidecar,
        f"legacy_samples::{args.case}",
        **{
            "case": args.case,
            "tier": tier,
            "outcome": result.outcome,
            "verdict": "expected" if expected else "unexpected",
            "matched": matched,
            "assertions": result.assertions,
            "assertion_failures": result.assertion_failures,
            "cases_skipped": result.cases_skipped,
            "exit_code": returncode,
            "reason": reason,
        },
    )

    if not expected:
        print(f"::error title=cuDNN legacy case unexpected::{args.case}: {reason}")
        return 1
    if result.outcome == "crashed":
        print(
            f"::notice title=cuDNN legacy case crashed as expected::{args.case}: "
            f"{describe(result, returncode)}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["compile", "run", "case"])
    parser.add_argument(
        "--tu", help="compile|run: corpus-relative path, e.g. sdpa/fp16_fwd.cpp"
    )
    parser.add_argument(
        "--tier", choices=["RUN", "XFAIL_COMPILE", "EXCLUDED"], help="compile|run only"
    )
    parser.add_argument(
        "--manifest", type=Path, help="case: manifest.json holding legacy.cases"
    )
    parser.add_argument("--case", help="case: exact Catch2 TEST_CASE name")
    parser.add_argument(
        "--sidecar",
        required=True,
        type=Path,
        help="this TU's report sidecar; the harness decides its name",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.mode == "case":
        if args.tu is not None or args.tier is not None:
            parser.error("--tu and --tier are not accepted with --mode case")
        if args.manifest is None or args.case is None:
            parser.error("--mode case requires --manifest and --case")
    else:
        if args.manifest is not None or args.case is not None:
            parser.error(
                f"--manifest and --case are not accepted with --mode {args.mode}"
            )
        if args.tu is None or args.tier is None:
            parser.error(f"--mode {args.mode} requires --tu and --tier")

    command = (
        args.command[1:] if args.command and args.command[0] == "--" else args.command
    )
    if not command:
        print("sample_probe: no command given", file=sys.stderr)
        return 2

    if args.mode == "compile":
        return compile_mode(args, command)
    if args.mode == "run":
        return run_mode(args, command)
    return case_mode(args, command)


if __name__ == "__main__":
    sys.exit(main())
