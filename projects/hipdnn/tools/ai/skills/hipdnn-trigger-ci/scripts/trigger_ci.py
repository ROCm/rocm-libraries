#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Trigger CI workflows on a rocm-libraries branch.

Defaults to the current git branch (its upstream name when it tracks one).
Override with --branch.

dispatch needs --dry-run (print the gh command only) or --yes (dispatch for
real, after the --dry-run command was approved).

Examples:
    # Dry-run TheRock CI for integration-tests on gfx94X (current branch)
    python3 <skill>/scripts/trigger_ci.py dispatch -w therock-ci --gfx gfx94X \\
        --projects "dnn-providers/integration-tests" --dry-run

    # Same, dispatched for real after approval
    python3 <skill>/scripts/trigger_ci.py dispatch -w therock-ci --gfx gfx94X \\
        --projects "dnn-providers/integration-tests" --yes

    # Dry-run on a specific branch
    python3 <skill>/scripts/trigger_ci.py --branch users/someone/feature dispatch \\
        -w hipdnn-superbuild --dry-run

    # Dry-run multi-arch CI with test labels
    python3 <skill>/scripts/trigger_ci.py dispatch -w multi-arch --gfx gfx94X,gfx950 \\
        --test-labels test:hipdnn,test:miopenprovider --dry-run

    # Check CI status for a PR
    python3 <skill>/scripts/trigger_ci.py --pr 10770 status

    # Check CI status for the current branch (no --pr needed)
    python3 <skill>/scripts/trigger_ci.py status

    # Watch the most recent active run on the current branch
    python3 <skill>/scripts/trigger_ci.py watch

    # Watch a specific run by ID
    python3 <skill>/scripts/trigger_ci.py watch --run-id 33443403983
"""

import json
import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = "ROCm/rocm-libraries"
CANONICAL_REMOTE_URL = re.compile(
    r"github\.com[:/]ROCm/rocm-libraries(\.git)?/?$", re.IGNORECASE
)

# TheRock commit the --help family and label lists were taken from. The Sources
# links in SKILL.md use the same SHA; update both together.
THEROCK_SNAPSHOT_REF = "7440cb8578f4daae0d85a428fadd6645dc5464a0"
CI_ENV_ACTION = ".github/actions/ci-env/action.yml"

WORKFLOWS = {
    "therock-ci": {
        "file": "therock-ci.yml",
        "fields": ["projects", "gfx", "windows_gfx"],
    },
    "multi-arch": {
        "file": "therock-multi-arch-ci.yml",
        "fields": ["gfx", "windows_gfx", "test_labels", "windows_test_labels"],
    },
    "hipdnn-superbuild": {
        "file": "hipdnn-superbuild-ci.yml",
        "fields": [],
    },
}

INPUT_MAP = {
    "gfx": "linux_amdgpu_families",
    "windows_gfx": "windows_amdgpu_families",
    "projects": "projects",
    "test_labels": "linux_test_labels",
    "windows_test_labels": "windows_test_labels",
}

OPTION_NAMES = {
    "gfx": "--gfx",
    "windows_gfx": "--windows-gfx",
    "projects": "--projects",
    "test_labels": "--test-labels",
    "windows_test_labels": "--windows-test-labels",
}

INSTALL_HINTS = {
    "gh": "install the GitHub CLI (https://cli.github.com)",
    "git": "install git",
}


def exit_not_found(cmd):
    hint = INSTALL_HINTS.get(cmd[0], "install it")
    print(f"error: '{cmd[0]}' not found on PATH; {hint}", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd, check=True, capture=True):
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
        )
    except FileNotFoundError:
        exit_not_found(cmd)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if stderr:
            print(f"error: {stderr}", file=sys.stderr)
        else:
            print(f"error: command failed: {shlex.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip() if capture else ""


def check_gh_auth():
    cmd = ["gh", "auth", "status"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        exit_not_found(cmd)
    if result.returncode != 0:
        print(
            "error: gh is not authenticated. Run 'gh auth login' first.",
            file=sys.stderr,
        )
        sys.exit(1)


def current_git_branch():
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        print("error: detached HEAD — use --branch to specify a ref", file=sys.stderr)
        sys.exit(1)
    # Dispatch needs the name on the remote, which differs from the local name
    # when the branch tracks an upstream under another name.
    merge_ref = run_cmd(
        ["git", "config", "--get", f"branch.{branch}.merge"], check=False
    )
    if not merge_ref.startswith("refs/heads/"):
        return branch
    # The upstream name is only valid on REPO; a branch tracking a fork would
    # otherwise dispatch on whatever REPO branch has the same name.
    remote = run_cmd(["git", "config", "--get", f"branch.{branch}.remote"], check=False)
    url = run_cmd(["git", "remote", "get-url", remote], check=False) if remote else ""
    if not CANONICAL_REMOTE_URL.search(url):
        print(
            f"error: branch '{branch}' tracks remote '{remote}' ({url or 'no URL'}), "
            f"not {REPO}; push it to {REPO} and pass --branch",
            file=sys.stderr,
        )
        sys.exit(1)
    return merge_ref[len("refs/heads/") :]


def checkout_therock_ref():
    root = run_cmd(["git", "rev-parse", "--show-toplevel"], check=False)
    if not root:
        return ""
    try:
        text = Path(root, CI_ENV_ACTION).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    match = re.search(r'therock-ref:\n(?:.*\n)*?\s*value:\s*"([0-9a-f]+)"', text)
    return match.group(1) if match else ""


def pr_branch(pr_number):
    head = json.loads(
        run_cmd(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--repo",
                REPO,
                "--json",
                "headRefName,isCrossRepository",
            ]
        )
    )
    # A fork PR's head branch lives in the fork; on REPO the same name would
    # select an unrelated branch.
    if head["isCrossRepository"]:
        print(
            f"error: PR #{pr_number} is from a fork, so its head branch "
            f"'{head['headRefName']}' is not on {REPO}; push it to {REPO} and "
            "pass --branch, or use watch --run-id",
            file=sys.stderr,
        )
        sys.exit(1)
    return head["headRefName"]


def resolve_branch(args):
    if args.branch:
        return args.branch
    if args.pr:
        return pr_branch(args.pr)
    return current_git_branch()


def latest_run_id(workflow_file, ref):
    # Filter on the event so a push-triggered run of the same workflow on the
    # same branch is not mistaken for the dispatched one.
    out = run_cmd(
        [
            "gh",
            "run",
            "list",
            "--repo",
            REPO,
            "--workflow",
            workflow_file,
            "--branch",
            ref,
            "--event",
            "workflow_dispatch",
            "--limit",
            "1",
            "--json",
            "databaseId",
        ]
    )
    runs = json.loads(out) if out else []
    return runs[0]["databaseId"] if runs else 0


def find_new_run(workflow_file, ref, before_id, timeout=15, interval=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(interval)
        current = latest_run_id(workflow_file, ref)
        if current > before_id:
            return current
    return None


def dispatch_workflow(workflow_file, ref, inputs, dry_run=False):
    cmd = [
        "gh",
        "workflow",
        "run",
        workflow_file,
        "--repo",
        REPO,
        "--ref",
        ref,
    ]
    for key, value in inputs.items():
        if value:
            cmd.extend(["-f", f"{key}={value}"])

    print(f"  {shlex.join(cmd)}")

    if dry_run:
        print("  (dry-run — not dispatched; after approval, re-run with --yes)")
        return None

    before_id = latest_run_id(workflow_file, ref)
    run_cmd(cmd, capture=False)
    print(f"  -> dispatched {workflow_file} on ref '{ref}'")

    print("  waiting for run to appear...", end="", flush=True)
    run_id = find_new_run(workflow_file, ref, before_id)
    if run_id:
        print(f" run {run_id}")
        print(f"\n  Watch: gh run watch {run_id} --repo {REPO}")
        print(f"  Logs:  gh run view {run_id} --repo {REPO} --log")
    else:
        print(" timed out")
        print(
            f"\n  Check manually: gh run list --repo {REPO} "
            f"--workflow {workflow_file} --branch {ref} --limit 5"
        )
    return run_id


def find_active_run(ref):
    active = []
    for status in ("in_progress", "queued"):
        out = run_cmd(
            [
                "gh",
                "run",
                "list",
                "--repo",
                REPO,
                "--branch",
                ref,
                "--status",
                status,
                "--limit",
                "1",
                "--json",
                "databaseId,workflowName",
            ]
        )
        active.extend(json.loads(out) if out else [])
    return max(active, key=lambda run: run["databaseId"], default=None)


def cmd_dispatch(args):
    wf = WORKFLOWS[args.workflow]
    for field, option in OPTION_NAMES.items():
        if getattr(args, field, "") and field not in wf["fields"]:
            valid_for = [name for name, w in WORKFLOWS.items() if field in w["fields"]]
            print(
                f"error: {option} is only valid for -w {' or -w '.join(valid_for)}",
                file=sys.stderr,
            )
            sys.exit(1)
    # therock-ci has no default for projects; an empty value selects no subtrees,
    # so the run skips every build job and still completes.
    if args.workflow == "therock-ci" and not args.projects:
        print(
            "error: -w therock-ci needs --projects (subtree paths or 'all')",
            file=sys.stderr,
        )
        sys.exit(1)
    ref = resolve_branch(args)
    inputs = {}
    for field in wf["fields"]:
        value = getattr(args, field, "") or ""
        if value:
            inputs[INPUT_MAP[field]] = value
    pinned = checkout_therock_ref()
    if pinned and pinned != THEROCK_SNAPSHOT_REF:
        print(
            f"warning: {CI_ENV_ACTION} pins TheRock {pinned}, but the --help family "
            f"and label lists are from {THEROCK_SNAPSHOT_REF}; re-read the TheRock "
            f"files at {pinned} before relying on them",
            file=sys.stderr,
        )
    print(f"Dispatching '{args.workflow}' on '{ref}':")
    dispatch_workflow(wf["file"], ref, inputs, dry_run=args.dry_run)


def cmd_status(args):
    if args.pr and not args.branch:
        cmd = ["gh", "pr", "checks", str(args.pr), "--repo", REPO]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        stderr = result.stderr.strip()
        # 0 = all passed, 8 = some pending. 1 covers both failing checks and
        # errors such as "no checks reported"; only errors write to stderr.
        if result.returncode not in (0, 1, 8) or (result.returncode == 1 and stderr):
            print(
                f"error: {stderr or 'command failed: ' + shlex.join(cmd)}",
                file=sys.stderr,
            )
            sys.exit(1)
        if stderr:
            print(stderr, file=sys.stderr)
    else:
        ref = resolve_branch(args)
        print(f"Recent runs on '{ref}':\n")
        run_cmd(
            [
                "gh",
                "run",
                "list",
                "--repo",
                REPO,
                "--branch",
                ref,
                "--limit",
                "10",
            ],
            capture=False,
        )


def cmd_watch(args):
    if args.run_id:
        run_id = args.run_id
    else:
        ref = resolve_branch(args)
        active = find_active_run(ref)
        if not active:
            # Exit non-zero so "nothing to watch" is not read as a passing run.
            print(f"No active runs on '{ref}'.", file=sys.stderr)
            print(
                f"Check: gh run list --repo {REPO} --branch {ref} --limit 5",
                file=sys.stderr,
            )
            sys.exit(1)
        run_id = active["databaseId"]
        print(f"Watching '{active['workflowName']}' (run {run_id}):\n")

    sys.exit(
        subprocess.run(
            [
                "gh",
                "run",
                "watch",
                str(run_id),
                "--repo",
                REPO,
                "--exit-status",
            ]
        ).returncode
    )


COMMANDS = {
    "dispatch": cmd_dispatch,
    "status": cmd_status,
    "watch": cmd_watch,
}


def main():
    parser = argparse.ArgumentParser(
        description="Trigger CI on a rocm-libraries branch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
The lists below are a snapshot of TheRock at {THEROCK_SNAPSHOT_REF}.
If therock-ref in {CI_ENV_ACTION} differs, re-read the TheRock files
named below at that ref (dispatch warns when it does).

GPU families (case-insensitive; from TheRock build_tools/github_actions/amdgpu_family_matrix.py):
  presubmit:   gfx94X, gfx110X, gfx1151, gfx120X, gfx125X
  postsubmit:  gfx90a, gfx950
  nightly:     gfx900, gfx90c, gfx906, gfx908, gfx101X, gfx103X, gfx1150, gfx1152, gfx1153
  multi-arch only: gfx1250-strict, all, none
  Families without an entry for the target platform are dropped.

hipDNN test labels (multi-arch; from TheRock fetch_test_configurations.py test_matrix):
  test:hipdnn, test:hipdnn_install, test:hipdnn-integration-tests, test:hipdnn-samples,
  test:miopenprovider, test:hipblasltprovider, test:hipkernelprovider
  The test: prefix is optional. Any test label selects the full test tier; add
  test_filter:<quick|standard|comprehensive|full> to the same list to pick the tier.
        """,
    )
    parser.add_argument(
        "--branch",
        "-b",
        default="",
        help="Branch to dispatch on (default: current git branch, or PR branch if --pr given)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number; its head branch is used when --branch is not given",
    )

    sub = parser.add_subparsers(dest="command")

    dispatch = sub.add_parser("dispatch", help="Dispatch a workflow_dispatch run")
    dispatch.add_argument(
        "--workflow",
        "-w",
        choices=list(WORKFLOWS.keys()),
        required=True,
        help="Which workflow to trigger",
    )
    dispatch.add_argument(
        "--gfx", default="", help="Linux GPU families (comma-separated)"
    )
    dispatch.add_argument(
        "--windows-gfx", dest="windows_gfx", default="", help="Windows GPU families"
    )
    dispatch.add_argument(
        "--projects",
        default="",
        help="Space-separated subtree paths from .github/scripts/therock_matrix.py, "
        "or 'all' (therock-ci only, required there)",
    )
    dispatch.add_argument(
        "--test-labels",
        dest="test_labels",
        default="",
        help="Linux test labels, comma-separated (multi-arch only)",
    )
    dispatch.add_argument(
        "--windows-test-labels",
        dest="windows_test_labels",
        default="",
        help="Windows test labels, comma-separated (multi-arch only)",
    )
    # A real dispatch needs --yes, so it cannot happen by leaving a flag out.
    mode = dispatch.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Print the gh command without executing"
    )
    mode.add_argument(
        "--yes",
        action="store_true",
        help="Dispatch for real; pass only after the --dry-run command was approved",
    )

    sub.add_parser("status", help="Show CI check status for a PR or branch")

    watch = sub.add_parser("watch", help="Watch an active CI run until it completes")
    watch.add_argument(
        "--run-id", type=int, default=None, help="Specific run ID to watch"
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    check_gh_auth()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
