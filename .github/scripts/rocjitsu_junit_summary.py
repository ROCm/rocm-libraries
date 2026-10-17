# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Print per-test timing and outcome counts from the suite's JUnit XML.

Also writes durations.json next to the report; copying it over
rocjitsu_tensilelite_durations.json refreshes the order the suite runs in.
"""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def outcome(testcase):
    if testcase.find("failure") is not None or testcase.find("error") is not None:
        return "FAILED"
    if testcase.find("skipped") is not None:
        return "SKIPPED"
    return "PASSED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    junit = args.report_dir / "junit" / "tensilelite.xml"
    if not junit.is_file():
        print(f"No JUnit report at {junit}")
        return 0

    tests = [
        (float(tc.get("time", "0")), tc.get("name", ""), outcome(tc))
        for tc in ET.parse(junit).iter("testcase")
    ]
    tests.sort(key=lambda test: test[0], reverse=True)
    durations = {name: round(seconds) for seconds, name, _ in tests}
    (args.report_dir / "durations.json").write_text(
        json.dumps(durations, indent=2, sort_keys=True) + "\n"
    )
    total = sum(seconds for seconds, _, _ in tests)
    passed = sum(1 for _, _, status in tests if status == "PASSED")
    failed = sum(1 for _, _, status in tests if status == "FAILED")
    skipped = sum(1 for _, _, status in tests if status == "SKIPPED")

    print()
    print("=" * 80)
    print("Per-test timing from JUnit XML (rocjitsu emulation)")
    print("=" * 80)
    print("%-55s %10s %8s" % ("Test", "Time", "Status"))
    print("-" * 80)
    for seconds, name, status in tests:
        if seconds >= 60:
            time_str = "%dm%02ds" % (int(seconds // 60), int(seconds % 60))
        else:
            time_str = "%.1fs" % seconds
        if "[" in name:
            short = name.split("[")[-1].rstrip("]").split("/")[-1].replace(".yaml", "")
        else:
            short = name
        print("%-55s %10s %8s" % (short, time_str, status))
    print("-" * 80)
    print(
        "Total: %d tests, %d passed, %d failed, %d skipped, "
        "%.0fs aggregate test time (%.0f min)"
        % (len(tests), passed, failed, skipped, total, total / 60)
    )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
