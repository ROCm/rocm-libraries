#!/usr/bin/env python3
"""Trigger CI workflows on a rocm-libraries branch.

Defaults to the current git branch. Override with --branch.

Examples:
    # Dispatch TheRock CI for integration-tests on gfx94X (current branch)
    python3 .github/scripts/trigger_ci.py dispatch -w therock-ci --gfx gfx94X \\
        --projects "dnn-providers/integration-tests"

    # Dry-run: print the gh command without executing
    python3 .github/scripts/trigger_ci.py dispatch -w therock-ci --gfx gfx94X --dry-run

    # Dispatch on a specific branch
    python3 .github/scripts/trigger_ci.py --branch users/someone/feature dispatch \\
        -w hipdnn-superbuild

    # Multi-arch CI with test labels
    python3 .github/scripts/trigger_ci.py dispatch -w multi-arch --gfx gfx94X,gfx950

    # Add PR labels to trigger CI via the normal PR path
    python3 .github/scripts/trigger_ci.py --pr 10770 label \\
        --add test:integration-tests --add test_type:comprehensive

    # Check CI status for a PR
    python3 .github/scripts/trigger_ci.py --pr 10770 status

    # Check CI status for the current branch (no --pr needed)
    python3 .github/scripts/trigger_ci.py status

    # Watch the most recent active run on the current branch
    python3 .github/scripts/trigger_ci.py watch

    # Watch a specific run by ID
    python3 .github/scripts/trigger_ci.py watch --run-id 33443403983

    # Clean up labels
    python3 .github/scripts/trigger_ci.py --pr 10770 label \\
        --remove test:integration-tests
"""

import json
import argparse
import subprocess
import sys
import time

REPO = "ROCm/rocm-libraries"

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


def run_cmd(cmd, check=True, capture=True):
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            check=check,
        )
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        if stderr:
            print(f"error: {stderr}", file=sys.stderr)
        else:
            print(f"error: command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip() if capture else ""


def check_gh_auth():
    result = subprocess.run(
        ["gh", "auth", "status"],
        capture_output=True,
        text=True,
    )
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
    return branch


def pr_branch(pr_number):
    return run_cmd(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            REPO,
            "--json",
            "headRefName",
            "--jq",
            ".headRefName",
        ]
    )


def resolve_branch(args):
    if args.branch:
        return args.branch
    if args.pr:
        return pr_branch(args.pr)
    return current_git_branch()


def latest_run_id(workflow_file, ref):
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

    print(f"  {' '.join(cmd)}")

    if dry_run:
        print("  (dry-run — not dispatched)")
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
        runs = json.loads(out) if out else []
        if runs:
            return runs[0]
    return None


def add_labels(pr_number, labels):
    for label in labels:
        run_cmd(
            [
                "gh",
                "pr",
                "edit",
                str(pr_number),
                "--repo",
                REPO,
                "--add-label",
                label,
            ],
            capture=False,
        )
        print(f"  added label: {label}")


def remove_labels(pr_number, labels):
    for label in labels:
        run_cmd(
            [
                "gh",
                "pr",
                "edit",
                str(pr_number),
                "--repo",
                REPO,
                "--remove-label",
                label,
            ],
            capture=False,
        )
        print(f"  removed label: {label}")


def cmd_dispatch(args):
    ref = resolve_branch(args)
    wf = WORKFLOWS[args.workflow]
    inputs = {}
    for field in wf["fields"]:
        value = getattr(args, field, "") or ""
        if value:
            inputs[INPUT_MAP[field]] = value
    print(f"Dispatching '{args.workflow}' on '{ref}':")
    dispatch_workflow(wf["file"], ref, inputs, dry_run=args.dry_run)


def cmd_label(args):
    if not args.pr:
        print("error: --pr is required for label operations", file=sys.stderr)
        sys.exit(1)
    if args.add:
        add_labels(args.pr, args.add)
    if args.remove:
        remove_labels(args.pr, args.remove)
    if not args.add and not args.remove:
        print("error: specify --add or --remove", file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    if args.pr:
        run_cmd(
            [
                "gh",
                "pr",
                "checks",
                str(args.pr),
                "--repo",
                REPO,
            ],
            check=False,
            capture=False,
        )
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
            check=False,
            capture=False,
        )


def cmd_watch(args):
    if args.run_id:
        run_id = args.run_id
    else:
        ref = resolve_branch(args)
        active = find_active_run(ref)
        if not active:
            print(f"No active runs on '{ref}'.")
            print(f"Check: gh run list --repo {REPO} --branch {ref} --limit 5")
            return
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


def main():
    parser = argparse.ArgumentParser(
        description="Trigger CI on a rocm-libraries branch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
available GFX families: gfx94X, gfx950, gfx125X, gfx1151, gfx110X, gfx120X
available test labels:  test:hipdnn, test:integration-tests, test:hipblaslt, test:miopen, ...
available test_type:    test_type:quick, test_type:standard, test_type:comprehensive, test_type:full
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
        help="PR number (required for label, optional for dispatch/status)",
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
        "--projects", default="", help="Projects to build (therock-ci only)"
    )
    dispatch.add_argument(
        "--test-labels",
        dest="test_labels",
        default="",
        help="Test labels (multi-arch only)",
    )
    dispatch.add_argument(
        "--windows-test-labels",
        dest="windows_test_labels",
        default="",
        help="Windows test labels",
    )
    dispatch.add_argument(
        "--dry-run", action="store_true", help="Print the gh command without executing"
    )

    label = sub.add_parser("label", help="Add/remove PR labels to trigger CI")
    label.add_argument(
        "--add", action="append", default=[], help="Label to add (repeatable)"
    )
    label.add_argument(
        "--remove", action="append", default=[], help="Label to remove (repeatable)"
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

    {
        "dispatch": cmd_dispatch,
        "label": cmd_label,
        "status": cmd_status,
        "watch": cmd_watch,
    }[args.command](args)


if __name__ == "__main__":
    main()
