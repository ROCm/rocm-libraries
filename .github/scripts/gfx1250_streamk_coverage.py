# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Record test outcomes without hardware details and reject missing StreamK coverage."""

import argparse
import collections
import json
import pathlib
import re
import xml.etree.ElementTree as ET


def gtest_results(log):
    results = {}
    for line in log.read_text().splitlines():
        match = re.match(r"\[\s*(RUN|OK|FAILED|SKIPPED)\s*\]\s+(\S+)", line)
        if not match or "." not in match[2]:
            continue
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
    fast = [case for case in cases if "fast_check_streamk" in case["case"]]
    complete = bool(fast) and all(case["status"] == "passed" for case in fast)
    complete &= all(case["status"] in ("passed", "skipped") for case in cases)
    return {"cases": cases, "fast_check_streamk": fast, "coverage_complete": complete}


def tensilelite_results(xml, mirrors):
    cases = []
    for case in ET.parse(xml).iter("testcase"):
        status = "passed"
        for tag in ("skipped", "failure", "error"):
            if case.find(tag) is not None:
                status = {"skipped": "skipped", "failure": "failed", "error": "error"}[
                    tag
                ]
        cases.append(
            {
                "case": f"{case.get('classname', '')}::{case.get('name', '')}",
                "status": status,
            }
        )
    selected = []
    for config in sorted(mirrors.rglob("*.yaml")):
        name = "common/streamk/gfx1250-strict/" + config.relative_to(mirrors).as_posix()
        outcomes = [case["status"] for case in cases if name in case["case"]]
        selected.append({"config": name, "outcomes": outcomes})
    complete = bool(selected) and all(
        entry["outcomes"] and all(status == "passed" for status in entry["outcomes"])
        for entry in selected
    )
    return {
        "cases": cases,
        "strict_streamk_configs": selected,
        "coverage_complete": complete,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("gtest", "tensilelite"))
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("report", type=pathlib.Path)
    parser.add_argument("--mirrors", type=pathlib.Path)
    args = parser.parse_args()
    if args.suite == "tensilelite" and args.mirrors is None:
        parser.error("tensilelite requires --mirrors")
    data = (
        gtest_results(args.results)
        if args.suite == "gtest"
        else tensilelite_results(args.results, args.mirrors)
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
            }
        )
    )
    if not data["coverage_complete"]:
        raise SystemExit("StreamK coverage is incomplete; inspect the saved report")


if __name__ == "__main__":
    main()
