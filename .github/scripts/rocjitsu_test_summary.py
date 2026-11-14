# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""List the suite's tests by outcome in the GitHub step summary.

Reads progress.json from rocjitsu_pytest, which keeps pytest's xfail detail
that JUnit flattens into skips. Skipped tests are listed only when there are
at most MAX_LISTED_SKIPS of them; otherwise only their number is given.
"""

import argparse
import json
import os
from pathlib import Path
import sys

MAX_LISTED_SKIPS = 10
SECTIONS = ("failed", "incomplete", "xpassed", "xfailed", "passed", "skipped")
TITLES = dict(
    failed="Failed",
    incomplete="Incomplete",
    xpassed="XPASS",
    xfailed="XFAIL",
    passed="Passed",
    skipped="Skipped",
)


def classify(phases):
    """Return the outcome of one test from its setup/call/teardown reports."""
    call = phases.get("call", {})
    details = " ".join(phase.get("detail") or "" for phase in phases.values())
    if any(phase["outcome"] == "failed" for phase in phases.values()):
        # A strict xfail that passes is reported as a failure.
        return "xpassed" if "[XPASS(strict)]" in details else "failed"
    if any(phase.get("wasxfail") is not None for phase in phases.values()):
        return "xpassed" if call.get("outcome") == "passed" else "xfailed"
    if any(phase["outcome"] == "skipped" for phase in phases.values()):
        return "skipped"
    if "teardown" not in phases or call.get("outcome") != "passed":
        return "incomplete"
    return "passed"


def short_name(nodeid):
    test = nodeid.rpartition("::")[2]
    if "[" in test:
        test = test.partition("[")[2].rstrip("]")
        test = test.split("Tests/common/", 1)[-1].removesuffix(".yaml")
    return test


def duration(phases):
    seconds = sum(phase.get("seconds") or 0 for phase in phases.values())
    return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def summarize(progress, title):
    groups = {section: [] for section in SECTIONS}
    for nodeid, test in progress.get("tests", {}).items():
        phases = test.get("phases", {})
        note = ""
        if "Timeout (>" in " ".join(p.get("detail") or "" for p in phases.values()):
            note = ", timeout"
        groups[classify(phases)].append(
            f"{short_name(nodeid)} ({duration(phases)}{note})"
        )
    counts = ", ".join(
        f"{len(groups[section])} {section}" for section in SECTIONS if groups[section]
    )
    lines = [f"### {title}: {counts or 'no tests'}", ""]
    for section in SECTIONS:
        tests = sorted(groups[section])
        if not tests:
            continue
        if section == "skipped" and len(tests) > MAX_LISTED_SKIPS:
            lines += [f"**{TITLES[section]}:** {len(tests)} tests", ""]
            continue
        body = [f"- {test}" for test in tests]
        if section == "passed":
            lines += [
                f"<details><summary><b>{TITLES[section]} ({len(tests)})</b></summary>",
                "",
                *body,
                "",
                "</details>",
                "",
            ]
        else:
            lines += [f"**{TITLES[section]} ({len(tests)})**", "", *body, ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--title", default="rocjitsu TensileLite results")
    args = parser.parse_args()

    progress_path = args.report_dir / "progress.json"
    if not progress_path.is_file():
        print(f"No progress report at {progress_path}")
        return 0
    text = summarize(json.loads(progress_path.read_text()), args.title)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
