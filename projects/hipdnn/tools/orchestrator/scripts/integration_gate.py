#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Decide whether stage 2's integration loop is actually done.

The orchestrator's condition grammar can compare two numbers but cannot count a set
or divide, so a loop whose `until` reads a single suite number is trivially satisfied
by a run that verified nothing: a gtest filter that matches zero tests prints
`PASSED 0 tests` and exits 0, an engine that declines every graph it is offered
reports zero failures right alongside zero passes, and a positive run's passes mean
nothing unless the same suite is shown to fail when the engine is not loaded at all.
None of those are visible from `suite_failed == 0` alone. So this collects five
independent measurements of the same integration -- the suite run, a control run
naming an absent engine, `hipdnn_validate_descriptors`, a census run, and the CTest
registry -- into one report with a single 0/1 the loop's `until` can read, and a
feedback string that names exactly which of the underlying checks did not hold.

    integration_gate.py --out gate.json --contract integration.json \\
        --suite-passed 12 --suite-failed 0 --suite-selected 12 --suite-skipped 0 \\
        --absent-exit 1 --absent-passed 0 \\
        --validator validator.json --census-ran 12 --census-failed 0 \\
        --ctest-listed 1

The nine checks, in the order they are reported:

  cases_declared    the contract's `bundle_case_ids` is non-empty. An integration
                     that added no graphs has not been verified by anything.
  cases_passed      suite_passed >= len(bundle_case_ids). Passing fewer cases than
                     were declared means the pack declined its own graphs, which the
                     suite records as a skip and the exit code as success.
  no_failures       suite_failed == 0.
  selection_nonempty  suite_selected > 0. A gtest filter matching nothing prints
                     `PASSED 0 tests` and exits 0.
  served_not_skipped  suite_skipped < suite_selected. An engine that declines every
                     graph it is offered reports zero failures right alongside it.
  absent_control    naming the engine while it is not loaded must exit 1 and pass
                     nothing. This is the only thing that proves the positive run's
                     passes were conditional on the engine existing at all -- if this
                     is the one check that fails, the positive run proved nothing.
  descriptors_valid  the validator succeeded, named no expected engine as missing,
                     and lists this engine among the ones it found.
  census_ran        the census actually ran at least one case and none of them
                     failed. Zero cases run is absence of evidence, not success.
  ctest_registered  the engine has exactly one CTest entry -- not zero (nobody else
                     can run it) and not several (an ambiguous registration).

Exit codes: 0 the report was written (a failing check is a verdict, not an error --
the flow asserts on `meets_target` and `failed_checks`), 1 `--contract` or
`--validator` could not be read/parsed as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Names, in report order, of the nine checks this gate computes. Kept as an ordered
#: tuple (not just dict insertion order) so the "nine names" the flow depends on are
#: one visible list, not an implicit side effect of the code below.
CHECK_NAMES = (
    "cases_declared",
    "cases_passed",
    "no_failures",
    "selection_nonempty",
    "served_not_skipped",
    "absent_control",
    "descriptors_valid",
    "census_ran",
    "ctest_registered",
)

_ERROR_SEVERITIES = {"ERROR", "FATAL"}


def _load_json(path: str, role: str) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: could not read {role} {path}: {error}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"error: {role} {path}'s top level is not a JSON object", file=sys.stderr)
        return None
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument("--contract", required=True, help="integration.json to read")
    parser.add_argument("--suite-passed", type=int, required=True)
    parser.add_argument("--suite-failed", type=int, required=True)
    parser.add_argument("--suite-selected", type=int, required=True)
    parser.add_argument("--suite-skipped", type=int, required=True)
    parser.add_argument(
        "--absent-exit", type=int, required=True, help="exit code of the control run"
    )
    parser.add_argument(
        "--absent-passed", type=int, required=True, help="pass count of the control run"
    )
    parser.add_argument(
        "--validator", required=True, help="hipdnn_validate_descriptors --json output"
    )
    parser.add_argument("--census-ran", type=int, required=True)
    parser.add_argument("--census-failed", type=int, required=True)
    parser.add_argument(
        "--ctest-listed",
        type=int,
        required=True,
        help="number of CTest entries registered for this engine",
    )
    args = parser.parse_args()

    contract = _load_json(args.contract, "--contract")
    validator = _load_json(args.validator, "--validator")
    if contract is None or validator is None:
        return 1

    bundle_case_ids = contract.get("bundle_case_ids")
    if not isinstance(bundle_case_ids, list):
        bundle_case_ids = []
    engine_name = contract.get("engine_name")
    engine_name = engine_name if isinstance(engine_name, str) else ""
    declared_cases = len(bundle_case_ids)

    # A missing or malformed validator field must fail the check it feeds, not
    # silently pass it -- defaulting "expected_engines_missing" to [] when the key
    # is simply absent would read a broken validator run as a clean one.
    validator_success = bool(validator.get("success", False))
    validator_engines = validator.get("engines")
    if not isinstance(validator_engines, list):
        validator_engines = []
    missing_engines = validator.get("expected_engines_missing")
    if not isinstance(missing_engines, list):
        missing_engines = [
            "<validator.json malformed: 'expected_engines_missing' absent>"
        ]
    diagnostics = validator.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    validator_errors = sum(
        1
        for entry in diagnostics
        if isinstance(entry, dict)
        and str(entry.get("severity", "")).upper() in _ERROR_SEVERITIES
    )

    checks = {
        "cases_declared": 1 if declared_cases > 0 else 0,
        "cases_passed": 1 if args.suite_passed >= declared_cases else 0,
        "no_failures": 1 if args.suite_failed == 0 else 0,
        "selection_nonempty": 1 if args.suite_selected > 0 else 0,
        "served_not_skipped": 1 if args.suite_skipped < args.suite_selected else 0,
        "absent_control": (
            1 if (args.absent_exit == 1 and args.absent_passed == 0) else 0
        ),
        "descriptors_valid": (
            1
            if (
                validator_success
                and not missing_engines
                and engine_name in validator_engines
            )
            else 0
        ),
        "census_ran": 1 if (args.census_ran > 0 and args.census_failed == 0) else 0,
        "ctest_registered": 1 if args.ctest_listed == 1 else 0,
    }
    failed_checks = [name for name in CHECK_NAMES if checks[name] == 0]

    feedback_lines: list[str] = []
    if checks["cases_declared"] == 0:
        feedback_lines.append(
            "cases_declared: the contract's bundle_case_ids is empty. An integration "
            "that added no graphs has not been verified by anything."
        )
    if checks["cases_passed"] == 0:
        feedback_lines.append(
            f"cases_passed: suite_passed ({args.suite_passed}) is fewer than the "
            f"{declared_cases} case(s) declared in the contract. Passing fewer cases "
            f"than were declared means the pack declined its own graphs, which the "
            f"suite records as a skip and the exit code as success."
        )
    if checks["no_failures"] == 0:
        feedback_lines.append(
            f"no_failures: the suite reported {args.suite_failed} failure(s)."
        )
    if checks["selection_nonempty"] == 0:
        feedback_lines.append(
            "selection_nonempty: suite_selected is 0. A gtest filter that matches "
            "nothing prints `PASSED 0 tests` and exits 0; this run tested no case at all."
        )
    if checks["served_not_skipped"] == 0:
        feedback_lines.append(
            f"served_not_skipped: {args.suite_skipped} of {args.suite_selected} "
            f"selected case(s) were skipped. An engine that declines every graph "
            f"reports zero failures."
        )
    if checks["absent_control"] == 0:
        feedback_lines.append(
            f"absent_control: naming {engine_name or 'the engine'!s} while it is not "
            f"loaded exited {args.absent_exit} and passed {args.absent_passed} case(s) "
            f"(expected exit 1 and 0 passes). Naming an engine that is not loaded is "
            f"supposed to be a hard failure, not a skip; because this is the check "
            f"that proves the positive run's passes were conditional on the engine "
            f"existing at all, the positive run proved nothing."
        )
    if checks["descriptors_valid"] == 0:
        reasons = []
        if not validator_success:
            reasons.append("validator reported success=false")
        if missing_engines:
            reasons.append(f"expected_engines_missing={missing_engines}")
        if engine_name not in validator_engines:
            reasons.append(
                f"{engine_name or '<empty engine_name>'!s} is not in the validator's "
                f"engines list {validator_engines}"
            )
        feedback_lines.append("descriptors_valid: " + "; ".join(reasons) + ".")
    if checks["census_ran"] == 0:
        feedback_lines.append(
            f"census_ran: census_ran={args.census_ran}, census_failed={args.census_failed}. "
            f"Zero cases run is absence of evidence, not success."
        )
    if checks["ctest_registered"] == 0:
        feedback_lines.append(
            f"ctest_registered: {args.ctest_listed} CTest entry/entries found for "
            f"{engine_name or 'the engine'!s} (expected exactly 1). Registration is "
            f"what makes anyone else able to run the engine; zero means nobody can, "
            f"and several is an ambiguous registration."
        )

    report = {
        "meets_target": 1 if not failed_checks else 0,
        "checks": checks,
        "failed_checks": failed_checks,
        "failed_check_count": len(failed_checks),
        "declared_cases": declared_cases,
        "suite_passed": args.suite_passed,
        "suite_failed": args.suite_failed,
        "suite_selected": args.suite_selected,
        "suite_skipped": args.suite_skipped,
        "engine_name": engine_name,
        "validator_errors": validator_errors,
        "feedback": "\n\n".join(feedback_lines),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"integration gate for {engine_name or '<unnamed>'}: "
        f"{len(CHECK_NAMES) - len(failed_checks)}/{len(CHECK_NAMES)} checks passed"
        + (f"; failed: {', '.join(failed_checks)}" if failed_checks else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
