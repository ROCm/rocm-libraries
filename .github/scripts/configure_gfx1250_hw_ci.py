#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decide what the gfx1250 hardware (HW) test job runs.

The gfx1250 HW job builds hipBLASLt and TensileLite and tests them on a real gfx1250 GPU
(gfx1250_hipblaslt_tensilelite_test.yml). This script is its gate. It writes three
outputs to GITHUB_OUTPUT, or prints them when run locally:

  run_hipblaslt    "true" to run the hipBLASLt gtest suite
  run_tensilelite  "true" to run the TensileLite tests
  run              "true" if either runs (the job is skipped otherwise)

For a pull request it reads the PR's changed files and applies the path rules in
.github/gfx1250-hw-ci-paths.toml. In CI the files come from the GitHub API (the gate job
checks out no PR code). Locally they come from git:

  python3 .github/scripts/configure_gfx1250_hw_ci.py                 # vs origin/develop
  python3 .github/scripts/configure_gfx1250_hw_ci.py --base my-base   # vs another ref

Environment (the workflow sets these from the event; all optional locally):
  EVENT            pull_request (default), schedule or workflow_dispatch
  ACTION           the pull_request action, e.g. labeled
  LABEL            the label that triggered a labeled event
  DRAFT            "true" for a draft PR
  LABELS           the PR's labels as a JSON list; unset skips the org-label check
  REPO, PR         the repository and PR number; with both set, files come from the API
  CHANGED_FILES    the PR's changed_files count; if the API lists fewer (it stops at
                   3000 files), both suites run
  RUN_HIPBLASLT, RUN_TENSILELITE
                   workflow_dispatch inputs ("true"/"false"); ignored for other events
"""

import argparse
import fnmatch
import json
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_utils import set_github_output  # noqa: E402

PATHS = Path(__file__).resolve().parents[1] / "gfx1250-hw-ci-paths.toml"
# pr-org-label.yml labels a ROCm org member's PR "organization: ROCm". Outside
# contributors don't get it, so their PR code never reaches the self-hosted runner.
# "organization: streamhpc" (the streamhpc team's label) is left out: that team doesn't
# work on hipBLASLt or TensileLite, and a manual dispatch still covers a one-off PR.
ORG_LABELS = {"organization: ROCm"}

log = logging.getLogger("configure_gfx1250_hw_ci")


class Decision:
    """What the job runs, and why (the reason is shown as a workflow notice)."""

    def __init__(self, run_hipblaslt, run_tensilelite, reason=None):
        self.run_hipblaslt = run_hipblaslt
        self.run_tensilelite = run_tensilelite
        self.reason = reason

    @property
    def run(self):
        return self.run_hipblaslt or self.run_tensilelite

    def outputs(self):
        return {
            "run": str(self.run).lower(),
            "run_hipblaslt": str(self.run_hipblaslt).lower(),
            "run_tensilelite": str(self.run_tensilelite).lower(),
        }


def matches_any(path, patterns):
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def is_true(value, default=True):
    if value is None or value == "":
        return default
    return value.lower() == "true"


def decide(env, changed_files, rules):
    """Decide the suites for this event. changed_files is None if they couldn't be listed."""
    event = env.get("EVENT") or "pull_request"
    if event == "schedule":
        return Decision(True, True, None)
    if event == "workflow_dispatch":
        hipblaslt = is_true(env.get("RUN_HIPBLASLT"))
        tensilelite = is_true(env.get("RUN_TENSILELITE"))
        reason = (
            None if hipblaslt or tensilelite else "Skipping: both suites deselected"
        )
        return Decision(hipblaslt, tensilelite, reason)

    pr = env.get("PR", "")
    label = env.get("LABEL", "")
    if env.get("ACTION") == "labeled" and label not in ORG_LABELS:
        return Decision(
            False, False, f"Skipping: label '{label}' does not affect this workflow"
        )
    if env.get("DRAFT") == "true":
        return Decision(False, False, f"Skipping: PR #{pr} is a draft")
    labels = env.get("LABELS")
    if labels is not None and not ORG_LABELS & set(json.loads(labels) or []):
        return Decision(False, False, f"Skipping: PR #{pr} has no ROCm org label")

    if changed_files is None:
        return Decision(
            True, True, f"Running both suites: could not list PR #{pr}'s changed files"
        )
    expected = env.get("CHANGED_FILES")
    if expected and len(changed_files) < int(expected):
        return Decision(
            True,
            True,
            f"Running both suites: the files API listed {len(changed_files)} of PR #{pr}'s "
            f"{expected} changed files",
        )

    relevant = [
        path
        for path in changed_files
        if matches_any(path, rules["run"]) and not matches_any(path, rules["ignore"])
    ]
    log.info(
        "%d of %d changed files are relevant to this job",
        len(relevant),
        len(changed_files),
    )
    for path in relevant[:20]:
        log.info("  relevant: %s", path)
    if len(relevant) > 20:
        log.info("  ... and %d more", len(relevant) - 20)

    if not relevant:
        return Decision(
            False,
            False,
            "Skipping: no change outside docs to hipBLASLt, TensileLite or the TheRock pin",
        )
    tensilelite_relevant = [
        p for p in relevant if not matches_any(p, rules["hipblaslt_only"])
    ]
    if not tensilelite_relevant:
        return Decision(
            True,
            False,
            "Skipping the TensileLite tests: only hipBLASLt's library and clients changed",
        )
    log.info("TensileLite runs because of: %s", ", ".join(tensilelite_relevant[:5]))
    return Decision(True, True, None)


def list_files_from_api(repo, pr):
    """The PR's changed files from the GitHub API, or None if the call fails."""
    command = [
        "gh",
        "api",
        "--paginate",
        f"repos/{repo}/pulls/{pr}/files",
        "--jq",
        ".[].filename",
    ]
    log.info("Listing PR #%s's changed files: %s", pr, " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"::warning::could not list PR #{pr}'s changed files: {result.stderr.strip()}"
        )
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def list_files_from_git(base):
    """Files changed on this branch since it left `base` (git diff base...HEAD)."""
    command = ["git", "diff", "--name-only", f"{base}...HEAD"]
    log.info("Listing changed files: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base",
        default="origin/develop",
        help="local runs: compare HEAD against this ref (default: origin/develop)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    env = os.environ
    rules = tomllib.loads(PATHS.read_text())
    event = env.get("EVENT") or "pull_request"
    changed_files = []
    if event == "pull_request":
        if env.get("REPO") and env.get("PR"):
            changed_files = list_files_from_api(env["REPO"], env["PR"])
        else:
            changed_files = list_files_from_git(args.base)
        if changed_files is not None:
            log.info("%d changed files", len(changed_files))

    decision = decide(env, changed_files, rules)
    if decision.reason:
        print(f"::notice::{decision.reason}")
    log.info(
        "Decision: run_hipblaslt=%s run_tensilelite=%s",
        decision.run_hipblaslt,
        decision.run_tensilelite,
    )
    set_github_output(decision.outputs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
