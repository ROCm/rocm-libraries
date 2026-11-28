#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Publish the package into its own repository (docs/maintaining.md, "Publishing the package"): the committed HEAD
of the package checkout this script lives in, as one commit in TARGET_REPO, never a merge:

  tools/publish_package.py TARGET_REPO -- --deny FILE [PRIVACY_ARGS...]

Every change is made in the package: what is published is the package checkout's own HEAD commit, never a branch of a
live toolkit (a monitor's own checkout of the toolkit). It refuses a package work tree with changes (only a commit is
published), and a HEAD whose config.json is not the blank configuration (tools/make_package.py BLANK_CONFIG): a live
toolkit's tree carries a machine's settings there, the package's never does. Then the steps of tools/sync_from_live.py
run in TARGET_REPO: its tracked files replaced by that commit's tree with the blank config.json, the privacy check over
the files and over the staged change (the arguments after `--` go to `python3 -m rkmon_lib.privacy`, which requires
--deny FILE, the deny list), the tests, and one commit "rocKE monitor package: publish <commit>" as rkmon
<rkmon@localhost>; a failed step commits nothing.

Exit: 0 published (or nothing to publish), 1 a refusal or a failed step, 2 a usage error (no deny list named among
them).
"""

import os
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent
sys.path.insert(0, str(HERE))

import sync_from_live  # noqa: E402

USAGE = """\
# Publish the package into its own repository: the committed HEAD of the package checkout this script lives in.
#   tools/publish_package.py TARGET_REPO -- --deny FILE [PRIVACY_ARGS...]
"""


def text(argv: list[str], cwd) -> tuple[int, str]:
    p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, text=True)
    return p.returncode, p.stdout


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        sys.stdout.write(USAGE)
        return 2
    target, args = argv[0], argv[1:]
    if not os.path.isdir(target):
        print(f"no folder {target}")
        return 2
    rc, sha = text(["git", "rev-parse", "-q", "--verify", "HEAD^{commit}"], PACKAGE)
    if rc != 0:
        print(f"{PACKAGE} has no commit to publish")
        return 1
    sha = sha.strip()
    _, dirty = text(
        ["git", "status", "--porcelain", "--untracked-files=normal"], PACKAGE
    )
    if dirty.strip():
        print(
            f"the package checkout {PACKAGE} has changes: commit them (only a commit is published)"
        )
        return 1
    rc, cfg = text(["git", "show", f"{sha}:config.json"], PACKAGE)
    _, blank = text([sys.executable, "-B", "-c", sync_from_live.BLANK], PACKAGE)
    if rc != 0 or cfg != blank:
        print(
            f"{sha[:10]}'s config.json is not the blank configuration: not the package's tree (a live toolkit's "
            "carries a machine's settings); nothing published"
        )
        return 1
    os.chdir(target)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return sync_from_live.sync(
                [str(PACKAGE), sha, *args], tmp, label=f"publish {sha[:10]}"
            )
        except sync_from_live.Stop as s:
            return s.code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
