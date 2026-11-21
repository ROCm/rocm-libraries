# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fetch the dnn-benchmarking workload archives when a pull request changes them.

The hipDNN superbuild lanes have no GPU, so they cannot run the workloads. They
check only that the pinned DVC pointers fetch and that configure extracts the
archives. That check matters only when a pull request changes the pointers, so
this script:

1. Compares the pull request base with HEAD under the workloads directory. If
   nothing changed, it fetches nothing and exits 0.
2. Otherwise runs ``dvc pull`` on the directory and fails if no ``*.tar.gz``
   archive results, because ``dvc pull`` can exit 0 having fetched nothing.

Run it before configure, from the repository root:

    python .github/scripts/fetch_dnn_workloads.py --base-sha <pull request base sha>
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

WORKLOADS_DIR = "dnn-providers/integration-tests/workloads"


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure_commit(sha: str) -> None:
    """Fetch sha unless it is already present."""
    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True
    )
    if has_commit.returncode == 0:
        return
    fetch = ["git", "fetch", "--no-tags"]
    # CI checkouts are shallow: fetch only the commit's tree. Never pass
    # --depth to a full clone, because that makes the whole clone shallow.
    if git_output("rev-parse", "--is-shallow-repository") == "true":
        fetch += ["--depth=1", "--filter=blob:none"]
    subprocess.run([*fetch, "origin", sha], check=True)


def workloads_changed(base_sha: str, workloads_dir: str) -> bool:
    """Return True if any file under workloads_dir differs between base_sha and HEAD."""
    ensure_commit(base_sha)
    return bool(
        git_output("diff", "--name-only", base_sha, "HEAD", "--", workloads_dir)
    )


def fetch_archives(workloads_dir: str) -> int:
    """Pull the workload archives and return how many are present afterwards."""
    # Resolve the full path so a Windows .cmd shim also starts without a shell.
    dvc = shutil.which("dvc")
    if dvc is None:
        raise SystemExit("dvc is not on PATH")
    subprocess.run([dvc, "pull", "--recursive", workloads_dir], check=True)
    return sum(1 for _ in Path(workloads_dir).rglob("*.tar.gz"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-sha", required=True, help="Pull request base commit")
    parser.add_argument("--workloads-dir", default=WORKLOADS_DIR)
    args = parser.parse_args(argv)

    if not workloads_changed(args.base_sha, args.workloads_dir):
        print(f"No change under {args.workloads_dir}; not fetching workload archives.")
        return 0

    count = fetch_archives(args.workloads_dir)
    if count == 0:
        print("dvc pull produced no workload archives", file=sys.stderr)
        return 1
    print(f"Fetched {count} workload archive(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
