#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decide whether the gfx1250 HW job runs, and whether it runs the TensileLite tests.

The workflow's gate job runs this with the event in the environment and, for a pull
request, the PR's changed files on stdin. The path rules live in
.github/gfx1250-hw-ci-paths.toml. Writes run= and skip_tensilelite= to GITHUB_OUTPUT,
or prints them when run locally:

  git diff --name-only origin/develop... | python3 .github/scripts/configure_gfx1250_hw_ci.py

Environment (all optional locally; the workflow sets them from the event):
  EVENT    pull_request (default), schedule or workflow_dispatch
  ACTION   the pull_request action, e.g. labeled
  LABEL    the label that triggered a labeled event
  DRAFT    "true" for a draft PR
  LABELS   the PR's labels as a JSON list; unset skips the org-label check
  PR       the PR number, for messages
  CHANGED_FILES  the PR's changed_files count; a shorter list on stdin (the files API lists
                 at most 3000 files and rejects very large diffs) runs both suites
"""

import fnmatch
import json
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_utils import set_github_output  # noqa: E402

PATHS = Path(__file__).resolve().parents[1] / "gfx1250-hw-ci-paths.toml"
# pr-org-label.yml sets these from org membership; non-members never get them.
ORG_LABELS = {"organization: ROCm", "organization: streamhpc"}


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def decide(env, files, rules):
    """(run, skip_tensilelite, notice) for this event and changed-file list."""
    event = env.get("EVENT") or "pull_request"
    if event in ("schedule", "workflow_dispatch"):
        # Scheduled runs test everything; dispatch inputs pick the suites.
        return True, False, None
    pr = env.get("PR", "")
    label = env.get("LABEL", "")
    if env.get("ACTION") == "labeled" and label not in ORG_LABELS:
        return False, False, f"Skipping: label '{label}' does not affect this workflow"
    if env.get("DRAFT") == "true":
        return False, False, f"Skipping: PR #{pr} is a draft"
    labels = env.get("LABELS")
    if labels is not None and not ORG_LABELS & set(json.loads(labels) or []):
        return False, False, f"Skipping: PR #{pr} has no ROCm org label"
    expected = env.get("CHANGED_FILES")
    if expected and len(files) < int(expected):
        return (
            True,
            False,
            f"Running both suites: the files API listed {len(files)} of PR #{pr}'s "
            f"{expected} changed files",
        )
    relevant = [
        f for f in files if matches(f, rules["run"]) and not matches(f, rules["ignore"])
    ]
    if not relevant:
        return (
            False,
            False,
            "Skipping: no change outside docs to hipBLASLt, TensileLite or the TheRock pin",
        )
    if all(matches(f, rules["hipblaslt_only"]) for f in relevant):
        return (
            True,
            True,
            "Skipping the TensileLite tests: only hipBLASLt's library and clients changed",
        )
    return True, False, None


def main():
    rules = tomllib.loads(PATHS.read_text())
    files = (
        [line.strip() for line in sys.stdin if line.strip()]
        if not sys.stdin.isatty()
        else []
    )
    run, skip_tensilelite, notice = decide(os.environ, files, rules)
    if notice:
        print(f"::notice::{notice}")
    set_github_output(
        {"run": str(run).lower(), "skip_tensilelite": str(skip_tensilelite).lower()}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
