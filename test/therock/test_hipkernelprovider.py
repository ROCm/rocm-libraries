# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = Path(
    os.environ.get("THEROCK_DIR") or SCRIPT_DIR.parent.parent.parent
).resolve()

logging.basicConfig(level=logging.INFO)


class TestResult(NamedTuple):
    name: str
    status: str  # "Passed" | "Failed" | "Skipped"
    seconds: float


def parse_junit(path: Path) -> list[TestResult]:
    """Parse a ``ctest --output-junit`` file into per-test results.

    Status comes from the child element ctest writes for each ``<testcase>``:
    a ``<failure>``/``<error>`` means the test failed, ``<skipped>`` means it
    was skipped, and otherwise it passed.
    """
    root = ET.parse(path).getroot()
    results: list[TestResult] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            status = "Failed"
        elif case.find("skipped") is not None:
            status = "Skipped"
        else:
            status = "Passed"
        try:
            seconds = float(case.get("time") or 0)
        except ValueError:
            seconds = 0.0
        results.append(TestResult(case.get("name", "?"), status, seconds))
    return results


_MARK = {"Passed": "✅", "Failed": "❌", "Skipped": "⚪"}


def render_markdown(results: list[TestResult], arch: str) -> str:
    passed = sum(1 for r in results if r.status == "Passed")
    lines = [
        f"### rocKE tests — {arch or 'unknown arch'}",
        "",
        f"{passed}/{len(results)} ctest entries passed — each row is one ctest "
        "entry (often a whole gtest binary, not a single case).",
        "",
        "| | test | time |",
        "| --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda t: -t.seconds):
        lines.append(
            f"| {_MARK.get(r.status, r.status)} | `{r.name}` | {r.seconds:.2f}s |"
        )
    return "\n".join(lines) + "\n"


def write_step_summary(junit_path: Path) -> None:
    """Best-effort: render a per-test table to ``$GITHUB_STEP_SUMMARY``.

    Never raises — a summary problem must not fail the test job — but never
    fails silently either: on error it logs and writes a visible marker so a
    missing table is diagnosable rather than mistaken for a clean pass.
    """
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return  # local run: no GitHub summary to write
    try:
        results = parse_junit(junit_path)
        markdown = render_markdown(results, os.getenv("AMDGPU_FAMILIES", ""))
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown)
    except Exception as exc:  # best-effort, but loud (see docstring)
        logging.warning("rocKE test-summary generation failed: %s", exc)
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(
                    "### rocKE tests\n\n"
                    f"⚠️ test-result summary generation failed: `{exc}` "
                    "— see job log.\n"
                )
        except Exception:
            logging.exception("could not write test-summary failure marker")


def main() -> int:
    junit_path = THEROCK_DIR / "test_logs" / "ctest-junit-hipkernelprovider.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop any prior run's file so a ctest that dies before rewriting it surfaces
    # as the loud "missing XML" marker rather than a stale (possibly all-green) table.
    junit_path.unlink(missing_ok=True)

    environ_vars = os.environ.copy()
    # Some of our runtime kernel compilations have been relying on either ROCM_PATH being set, or ROCm being installed at
    # /opt/rocm. Neither of these is true in TheRock so we need to supply ROCM_PATH to our tests.
    environ_vars["ROCM_PATH"] = str(Path(THEROCK_BIN_DIR).resolve().parent)

    cmd = [
        "ctest",
        "--test-dir",
        f"{THEROCK_BIN_DIR}/hip_kernel_provider",
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "600",
        # JUnit XML is a machine-readable side channel for the step summary
        # below; ctest still streams to stdout, which the workflow tees.
        "--output-junit",
        str(junit_path),
    ]

    if os.getenv("TEST_TYPE", "full") == "smoke":
        environ_vars["GTEST_FILTER"] = "-Full*"

    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

    # check=False so the summary still runs on failure and the ctest exit code
    # is propagated exactly. The summary is decorative; ctest's exit code is the
    # sole pass/fail authority.
    result = subprocess.run(cmd, cwd=THEROCK_DIR, env=environ_vars, check=False)

    write_step_summary(junit_path)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
