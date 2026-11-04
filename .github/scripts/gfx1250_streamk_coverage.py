# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Record StreamK outcomes and require every available test to pass."""

import argparse
import collections
import json
import pathlib
import re
import xml.etree.ElementTree as ET


def gtest_results(log, unavailable=None, expected=()):
    unavailable = unavailable or {}
    results = {}
    starts = collections.Counter()
    reasons = {}
    context = []
    for line in log.read_text().splitlines():
        match = re.match(r"\[\s*(RUN|OK|FAILED|SKIPPED)\s*\]\s+(\S+)", line)
        if not match or "." not in match[2]:
            context.append(line)
            continue
        if match[1] == "RUN":
            starts[match[2]] += 1
            context = []
        elif match[1] == "SKIPPED" and match[2] not in reasons:
            reasons[match[2]] = "\n".join(context).strip()
        status = {
            "RUN": "incomplete",
            "OK": "passed",
            "FAILED": "failed",
            "SKIPPED": "skipped",
        }
        results[match[2]] = status[match[1]]
    cases = [
        {"case": name, "status": status} for name, status in sorted(results.items())
    ]
    for case in cases:
        if case["status"] == "skipped":
            case["reason"] = reasons.get(case["case"], "")
            pattern = unavailable.get(case["case"])
            case["unavailable"] = bool(pattern and re.search(pattern, case["reason"]))
    fast = [case for case in cases if "fast_check_streamk" in case["case"]]
    invalid_execution = sorted(name for name in results if starts[name] != 1)
    complete = (
        bool(fast)
        and not invalid_execution
        and set(expected).issubset(results)
        and all(case["status"] == "passed" for case in cases)
    )
    available_complete = (
        bool(fast)
        and not invalid_execution
        and any(case["status"] == "passed" for case in fast)
        and set(expected).issubset(results)
        and all(
            case["status"] == "passed"
            or (case["status"] == "skipped" and case.get("unavailable", False))
            for case in cases
        )
    )
    return {
        "cases": cases,
        "fast_check_streamk": fast,
        "missing_cases": sorted(set(expected) - results.keys()),
        "invalid_execution_cases": invalid_execution,
        "coverage_complete": complete,
        "available_coverage_complete": available_complete,
    }


def tensilelite_results(xml, mirrors, unavailable=None):
    unavailable = unavailable or {}
    cases = []
    for case in ET.parse(xml).iter("testcase"):
        status = "passed"
        for tag in ("skipped", "failure", "error"):
            if case.find(tag) is not None:
                status = {"skipped": "skipped", "failure": "failed", "error": "error"}[
                    tag
                ]
        result = {
            "case": f"{case.get('classname', '')}::{case.get('name', '')}",
            "status": status,
        }
        skipped = case.find("skipped")
        if skipped is not None:
            result["reason"] = skipped.get("message", "")
        cases.append(result)
    selected = []
    for config in sorted(mirrors.rglob("*.yaml")):
        name = "common/streamk/gfx1250-strict/" + config.relative_to(mirrors).as_posix()
        matched = [case for case in cases if name in case["case"]]
        outcomes = [case["status"] for case in matched]
        entry = {"config": name, "outcomes": outcomes}
        if outcomes == ["skipped"]:
            entry["reason"] = matched[0].get("reason", "")
            pattern = unavailable.get(name)
            entry["unavailable"] = bool(pattern and re.search(pattern, entry["reason"]))
        selected.append(entry)
    complete = bool(selected) and all(
        entry["outcomes"] and all(status == "passed" for status in entry["outcomes"])
        for entry in selected
    )
    return {
        "cases": cases,
        "strict_streamk_configs": selected,
        "coverage_complete": complete,
        "available_coverage_complete": bool(selected)
        and any(entry["outcomes"] == ["passed"] for entry in selected)
        and all(
            entry["outcomes"] == ["passed"]
            or (entry["outcomes"] == ["skipped"] and entry.get("unavailable", False))
            for entry in selected
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("gtest", "tensilelite"))
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--mirrors", type=pathlib.Path)
    parser.add_argument("--availability", type=pathlib.Path)
    args = parser.parse_args()
    if args.suite == "tensilelite" and args.mirrors is None:
        parser.error("tensilelite requires --mirrors")
    availability = (
        json.loads(args.availability.read_text()) if args.availability else {}
    )
    data = (
        gtest_results(
            args.results,
            availability.get("gtest_unavailable", {}),
            availability.get("expected_gtest_cases", []),
        )
        if args.suite == "gtest"
        else tensilelite_results(
            args.results, args.mirrors, availability.get("tensilelite_unavailable", {})
        )
    )
    data["counts"] = dict(collections.Counter(case["status"] for case in data["cases"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "suite": args.suite,
                "counts": data["counts"],
                "coverage_complete": data["coverage_complete"],
                "available_coverage_complete": data["available_coverage_complete"],
            }
        )
    )
    if not data["available_coverage_complete"]:
        raise SystemExit(
            "Available StreamK coverage is incomplete; inspect the saved report"
        )


if __name__ == "__main__":
    main()
