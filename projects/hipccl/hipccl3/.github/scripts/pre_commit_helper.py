#!/usr/bin/env python3

"""
Pre-commit checkout preparation script
--------------------------------------
Prepares a blobless, sparse checkout for the pre-commit workflow so that only
the project directories touched by a change are materialized instead of the
full tree.

Steps:
    1. Select the diff base, falling back to the default branch when the event
       carries no source ref (e.g. a branch-creation push with an all-zero SHA).
    2. Fetch that base ref (unshallowing a shallow clone when needed) so a
       three-dot diff has a valid merge base.
    3. Compute the three-dot diff (base...HEAD) of changed files.
    4. Map the changed files to sparse-checkout cones and materialize them.

Arguments:
    --from-ref            : Event-derived diff reference (PR base ref or push
                            before-SHA). An all-zero SHA triggers the fallback.
    --default-ref         : Default branch reference used as the fallback base.
    --pushed-branch-name  : Name of the branch being pushed, used to build the
                            fetch refspec when --from-ref is not an origin/ ref.

Outputs (written to GITHUB_OUTPUT):
    diff_ref      : The resolved diff base, consumed by pre-commit --from-ref.
    changed_files : New line -separated, sorted list of changed files.
"""

from ci_utils import set_github_output, get_modified_paths
from pathlib import Path
import subprocess
import argparse
import sys

NULL_REF = "0" * 40


def select_diff_base(from_ref: str, default_ref: str) -> str:
    if from_ref == NULL_REF:
        return default_ref
    return from_ref


def is_shallow_repo() -> bool:
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-dir"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return Path(git_dir, "shallow").is_file()


def resolve_fetch_target(from_ref: str, pushed_branch_name: str) -> tuple[str, str]:
    if from_ref.startswith("origin/"):
        base_branch = from_ref.removeprefix("origin/")
        target_ref = "refs/remotes/" + from_ref
    else:
        base_branch = pushed_branch_name
        target_ref = "refs/remotes/origin/" + pushed_branch_name
    return base_branch, target_ref


def fetch_diff_base(from_ref: str, pushed_branch_name: str) -> None:
    args = [
        "git",
        "fetch",
        "--no-tags",
        "--prune",
        "--no-recurse-submodules",
        "--filter=blob:none",
    ]

    if is_shallow_repo():
        args.append("--unshallow")

    base_branch, target_ref = resolve_fetch_target(from_ref, pushed_branch_name)

    args.extend(["origin", f"+refs/heads/{base_branch}:{target_ref}"])

    subprocess.run(args, check=True)


def compute_three_dot_diff(from_ref: str) -> list[str]:
    changed_files = get_modified_paths(f"{from_ref}...HEAD")
    return sorted(changed_files)


def resolve_sparse_checkout_cones(changed_files: list[str]) -> list[str]:
    DIRS = set([])

    for f in changed_files:
        directory = f.split("/")
        if len(directory) == 2:
            DIRS.add(directory[0])
        elif len(directory) >= 3:
            DIRS.add(directory[0] + "/" + directory[1])

    return sorted(DIRS)


def sparse_checkout_changed_projects(from_ref: str) -> list[str]:
    changed_files = compute_three_dot_diff(from_ref)

    cones = resolve_sparse_checkout_cones(changed_files)
    subprocess.run(
        ["git", "sparse-checkout", "set", "--cone", *cones],
        check=True,
    )

    return changed_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from-ref", required=True, help="event-derived diff reference"
    )
    parser.add_argument("--default-ref", required=True, help="default reference branch")
    parser.add_argument(
        "--pushed-branch-name",
        required=True,
        help="name of branch to be pushed, if event is a push instance",
    )

    args = parser.parse_args()
    try:
        from_ref = select_diff_base(args.from_ref, args.default_ref)
        set_github_output({"diff_ref": f"{from_ref}"})
        fetch_diff_base(from_ref, args.pushed_branch_name)
        changed_files = sparse_checkout_changed_projects(from_ref)
        set_github_output({"changed_files": "\n".join(changed_files)})
    except subprocess.CalledProcessError as e:
        if e.stderr:
            sys.stderr.write(e.stderr)
        sys.stderr.write(f"{e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
