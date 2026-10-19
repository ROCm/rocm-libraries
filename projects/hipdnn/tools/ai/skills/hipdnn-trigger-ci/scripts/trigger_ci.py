#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Trigger CI workflows on a rocm-libraries branch.

Defaults to the current git branch (its upstream name when it tracks one).
Override with --branch.

dispatch needs --dry-run (print the gh command only) or --yes (dispatch for
real, after the --dry-run command was approved).

Workflow inputs are passed as -f name=value, as with gh workflow run. They are
checked against the workflow_dispatch inputs of the workflow file on the
target branch, so a renamed or removed input fails before dispatching.

Examples:
    # Dry-run TheRock CI for integration-tests on gfx94X (current branch)
    python3 <skill>/scripts/trigger_ci.py dispatch -w therock-ci \\
        -f linux_amdgpu_families=gfx94X \\
        -f projects=dnn-providers/integration-tests --dry-run

    # Same, dispatched for real after approval
    python3 <skill>/scripts/trigger_ci.py dispatch -w therock-ci \\
        -f linux_amdgpu_families=gfx94X \\
        -f projects=dnn-providers/integration-tests --yes

    # Dry-run on a specific branch
    python3 <skill>/scripts/trigger_ci.py --branch users/someone/feature dispatch \\
        -w hipdnn-superbuild --dry-run

    # Dry-run multi-arch CI with test labels
    python3 <skill>/scripts/trigger_ci.py dispatch -w multi-arch \\
        -f linux_amdgpu_families=gfx94X,gfx950 \\
        -f linux_test_labels=test:hipdnn,test:miopenprovider --dry-run

    # Re-run multi-arch tests at another tier, reusing run 36880722738's build
    python3 <skill>/scripts/trigger_ci.py --pr 12751 dispatch -w multi-arch \\
        -f linux_amdgpu_families=gfx94X \\
        -f linux_test_labels=test:hipdnn,test_filter:comprehensive \\
        --reuse-build 36880722738 --dry-run

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
from urllib.parse import quote

REPO = "ROCm/rocm-libraries"
# gh follows GH_HOST and a GitHub Enterprise login otherwise, so name the host.
HOST = "github.com"
REPO_ARG = f"{HOST}/{REPO}"
CANONICAL_REMOTE_URL = re.compile(
    r"github\.com[:/]ROCm/rocm-libraries(\.git)?/?$", re.IGNORECASE
)

WORKFLOWS = {
    "therock-ci": "therock-ci.yml",
    "multi-arch": "therock-multi-arch-ci.yml",
    "hipdnn-superbuild": "hipdnn-superbuild-ci.yml",
}

# Inputs --reuse-build sets; the workflow must declare both.
REUSE_INPUTS = ("baseline_run_id", "prebuilt_stages")

YAML_KEY = re.compile(r"""^(\s*)["']?([\w-]+)["']?:(?:\s+(.*))?$""")

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


def gh_api(*args):
    return ["gh", "api", "--hostname", HOST, *args]


def check_gh_auth():
    # Without --hostname, a problem with any other logged-in host fails this.
    cmd = ["gh", "auth", "status", "--hostname", HOST]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        exit_not_found(cmd)
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        if details:
            print(details, file=sys.stderr)
        print(
            f"error: gh is not authenticated to {HOST}. "
            f"Run 'gh auth login --hostname {HOST}' first.",
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


def workflow_text(workflow_file, ref):
    path = f".github/workflows/{workflow_file}"
    cmd = gh_api(
        "-H",
        "Accept: application/vnd.github.raw",
        f"repos/{REPO}/contents/{path}?ref={quote(ref, safe='')}",
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        exit_not_found(cmd)
    if result.returncode != 0:
        print(
            f"error: could not read {path} at '{ref}': {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    return result.stdout


def _yaml_scalar(value):
    """Plain or quoted one-line scalar; None for an empty (null) value."""
    value = (value or "").strip()
    if value[:1] in ("'", '"'):
        return value[1 : value.index(value[0], 1)]
    value = "" if value.startswith("#") else value.split(" #")[0].strip()
    return value or None


def dispatch_inputs(text):
    """Return {name: default, or None when unset} for the workflow_dispatch
    inputs in a workflow file, or None when it has no workflow_dispatch trigger.

    Reads only the block-style mappings GitHub workflows use for on: and its
    inputs, which keeps the script stdlib-only.
    """
    stack = []  # (indent, key) of the mappings enclosing the current line
    found = False
    inputs = {}
    for line in text.splitlines():
        match = YAML_KEY.match(line)
        if not match:
            continue
        indent, key, value = len(match.group(1)), match.group(2), match.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = [k for _, k in stack]
        stack.append((indent, key))
        if path == ["on"] and key == "workflow_dispatch":
            found = True
        elif path == [] and key == "on" and "workflow_dispatch" in (value or ""):
            found = True  # flow style, e.g. on: [push, workflow_dispatch]
        elif path == ["on", "workflow_dispatch", "inputs"]:
            inputs[key] = None
        elif path[:3] == ["on", "workflow_dispatch", "inputs"] and len(path) == 4:
            if key == "default":
                inputs[path[3]] = _yaml_scalar(value)
    return inputs if found else None


def pr_branch(pr_number):
    head = json.loads(
        run_cmd(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--repo",
                REPO_ARG,
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
            REPO_ARG,
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
        REPO_ARG,
        "--ref",
        ref,
    ]
    for key, value in inputs.items():
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
        print(f"\n  Watch: gh run watch {run_id} --repo {REPO_ARG}")
        print(f"  Logs:  gh run view {run_id} --repo {REPO_ARG} --log")
    else:
        print(" timed out")
        print(
            f"\n  Check manually: gh run list --repo {REPO_ARG} "
            f"--workflow {workflow_file} --branch {ref} --limit 5"
        )
    return run_id


def ref_head_sha(ref):
    return run_cmd(gh_api(f"repos/{REPO}/commits/{ref}", "--jq", ".sha"))


def runs_cancelled_by_dispatch(workflow_file, sha):
    # The workflows' concurrency group is "<workflow>-<PR number or SHA>" with
    # cancel-in-progress. A dispatch has no PR number, so it joins the group of
    # every non-pull_request run of the workflow at that SHA, on any branch
    # (push postsubmit, schedule, other dispatches), and cancels them.
    out = run_cmd(
        [
            "gh",
            "run",
            "list",
            "--repo",
            REPO_ARG,
            "--workflow",
            workflow_file,
            "--commit",
            sha,
            "--limit",
            "50",
            "--json",
            "databaseId,status,event,headBranch",
        ]
    )
    runs = json.loads(out) if out else []
    return [
        run
        for run in runs
        if run["status"] != "completed"
        and run["event"] not in ("pull_request", "pull_request_target")
    ]


def check_local_head(ref, head_sha, args):
    # A ref taken from the checkout reads as "test my code", but the run tests
    # the remote head, which misses unpushed commits and is newer than a stale
    # checkout. --reuse-build is checked against the remote head too.
    local_sha = run_cmd(["git", "rev-parse", "HEAD"])
    if local_sha == head_sha:
        return
    print(
        f"warning: local HEAD is {local_sha[:11]}, but '{ref}' on {REPO} is at "
        f"{head_sha[:11]}; the run tests {head_sha[:11]}",
        file=sys.stderr,
    )
    if args.yes and not args.allow_head_mismatch:
        print(
            "error: push or pull so they match, or pass --allow-head-mismatch "
            "after the user approved testing the remote head",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.dry_run:
        print(
            "  after approval, the --yes run also needs --allow-head-mismatch",
            file=sys.stderr,
        )


def check_reuse_baseline(run_id, workflow_file, ref, head_sha):
    run = json.loads(
        run_cmd(
            gh_api(
                f"repos/{REPO}/actions/runs/{run_id}",
                "--jq",
                "{path, head_sha, status, repo: .head_repository.full_name}",
            )
        )
    )
    problems = []
    # The path can carry a "@ref" suffix, so compare only the file part.
    if run["path"].split("@")[0] != f".github/workflows/{workflow_file}":
        problems.append(f"is a run of {run['path']}, not {workflow_file}")
    if run["repo"] != REPO:
        problems.append(f"built code from {run['repo']}, not {REPO}")
    # Artifacts built from another commit would be tested as if they were ref's.
    if run["head_sha"] != head_sha:
        problems.append(
            f"built {run['head_sha'][:11]}, but '{ref}' is at {head_sha[:11]}"
        )
    # An unfinished run may not have uploaded every artifact yet, and the new
    # dispatch cancels it through the workflow's concurrency group.
    if run["status"] != "completed":
        problems.append(f"is still {run['status']}")
    if problems:
        for problem in problems:
            print(f"error: --reuse-build run {run_id} {problem}", file=sys.stderr)
        sys.exit(1)


def find_active_run(ref):
    active = []
    for status in ("in_progress", "queued"):
        out = run_cmd(
            [
                "gh",
                "run",
                "list",
                "--repo",
                REPO_ARG,
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


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def parse_fields(fields):
    inputs = {}
    for field in fields:
        name, sep, value = field.partition("=")
        if not sep or not name:
            fail(f"-f '{field}' is not name=value")
        if name in inputs:
            fail(f"-f {name} given twice")
        inputs[name] = value
    return inputs


def describe_inputs(declared):
    if not declared:
        return "it declares no inputs"
    return "its inputs are: " + ", ".join(
        name if default is None else f"{name} (default '{default}')"
        for name, default in declared.items()
    )


def cmd_dispatch(args):
    workflow_file = WORKFLOWS[args.workflow]
    ref = resolve_branch(args)
    # A dispatch on a postsubmit branch shares the concurrency group of that
    # commit's push run and cancels it on shared CI.
    if (
        ref == "develop" or ref.startswith("release/")
    ) and not args.allow_shared_branch:
        print(
            f"error: dispatching on '{ref}' can cancel its postsubmit run; use a "
            "user branch, or pass --allow-shared-branch",
            file=sys.stderr,
        )
        sys.exit(1)
    inputs = parse_fields(args.field)
    # Check the inputs against the workflow as it is on the target branch, so a
    # renamed or removed input fails here instead of being ignored by GitHub.
    declared = dispatch_inputs(workflow_text(workflow_file, ref))
    if declared is None:
        fail(f"{workflow_file} at '{ref}' has no workflow_dispatch trigger")
    unknown = [name for name in inputs if name not in declared]
    if unknown:
        fail(
            f"{workflow_file} at '{ref}' has no input {', '.join(unknown)}; "
            f"{describe_inputs(declared)}"
        )
    if args.reuse_build:
        missing = [name for name in REUSE_INPUTS if name not in declared]
        if missing:
            fail(
                f"--reuse-build needs {', '.join(missing)}, which {workflow_file} "
                f"at '{ref}' does not declare"
            )
        given = [name for name in REUSE_INPUTS if name in inputs]
        if given:
            fail(f"--reuse-build sets {', '.join(given)}; do not pass it with -f")
    # An input without a default is one the caller must choose. An empty
    # therock-ci projects, for example, selects no subtrees, so the run skips
    # every build job and still passes.
    unset = [
        name
        for name, default in declared.items()
        if default is None and not inputs.get(name)
    ]
    if unset:
        fail(
            f"{workflow_file} input {', '.join(unset)} has no default; "
            "pass -f name=value"
        )
    head_sha = ref_head_sha(ref)
    if not args.branch and not args.pr:
        check_local_head(ref, head_sha, args)
    if args.reuse_build:
        check_reuse_baseline(args.reuse_build, workflow_file, ref, head_sha)
        inputs["baseline_run_id"] = str(args.reuse_build)
        # Copy every build stage from the baseline run; only tests run.
        inputs["prebuilt_stages"] = "all"
    doomed = runs_cancelled_by_dispatch(workflow_file, head_sha)
    if doomed:
        print(
            f"warning: this dispatch cancels {len(doomed)} active run(s) of "
            f"{workflow_file} at {head_sha[:11]} (shared concurrency group):",
            file=sys.stderr,
        )
        for run in doomed:
            print(
                f"  {run['databaseId']}  {run['status']}  {run['event']}  "
                f"{run['headBranch']}",
                file=sys.stderr,
            )
        if args.yes and not args.cancel_active:
            print(
                "error: pass --cancel-active to dispatch anyway, after the user "
                "approved cancelling the runs above",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.dry_run:
            print(
                "  after approval, the --yes run also needs --cancel-active",
                file=sys.stderr,
            )
    print(f"Dispatching '{args.workflow}' on '{ref}':")
    dispatch_workflow(workflow_file, ref, inputs, dry_run=args.dry_run)


def cmd_status(args):
    if args.pr and not args.branch:
        cmd = ["gh", "pr", "checks", str(args.pr), "--repo", REPO_ARG]
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
                REPO_ARG,
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
                f"Check: gh run list --repo {REPO_ARG} --branch {ref} --limit 5",
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
                REPO_ARG,
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
        epilog="""
Workflow aliases: """
        + ", ".join(f"{alias} = {file}" for alias, file in WORKFLOWS.items())
        + """
Valid -f inputs are the workflow_dispatch inputs of that file on the target
branch; an unknown name is rejected with the current list. Valid values
(GPU families, test labels, projects) are defined by the files SKILL.md
points to.
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
        "--field",
        "-f",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Workflow input, as with gh workflow run; repeat for each input",
    )
    dispatch.add_argument(
        "--reuse-build",
        dest="reuse_build",
        type=int,
        default=None,
        metavar="RUN_ID",
        help="Skip the build and copy its artifacts from this completed run of "
        "the same workflow and commit (sets baseline_run_id and prebuilt_stages)",
    )
    dispatch.add_argument(
        "--cancel-active",
        dest="cancel_active",
        action="store_true",
        help="Allow --yes to cancel the active runs of the workflow at the same "
        "commit that the dry-run lists",
    )
    dispatch.add_argument(
        "--allow-head-mismatch",
        dest="allow_head_mismatch",
        action="store_true",
        help="Allow --yes when the branch comes from the checkout and its head "
        "on GitHub is not the local HEAD; the run tests the GitHub head",
    )
    dispatch.add_argument(
        "--allow-shared-branch",
        dest="allow_shared_branch",
        action="store_true",
        help="Allow dispatching on develop or release/*, which can cancel "
        "their postsubmit runs",
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
