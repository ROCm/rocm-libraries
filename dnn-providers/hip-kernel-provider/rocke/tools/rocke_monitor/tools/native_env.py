# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Print `export VAR=path` lines for run_all.py's NATIVE_PARITY_TESTS, so a checks job's pytest runs the native C++
comparisons (for example ROCKE_OPTIMIZATION_BARRIER_EMIT) instead of skipping them.

Usage: python3 -I native_env.py <run_all.py> <build root> [<owner's user name>]
The tuple is read with ast (the PR's file is parsed, never imported). Only binaries that exist in the build are
exported; the others keep skipping, as they would in CI without them. Only rocKE's own switches are (NAME: ROCKE_,
then capitals, digits or underscores): the names are the PR's, and the job script that evaluates these lines is the
monitor's, so no other variable (PATH, LD_PRELOAD, BASH_ENV) and nothing that is no variable name is ever
set.

The build is what PR code wrote (a slot's build folder, as its runner), and the monitor reads it, so it trusts none of
it. The build root is opened as the result reader opens a slot's folder (worker/pr_results.py open_root: O_DIRECTORY
and O_NOFOLLOW, never while a process of its user runs) and refused unless `owner` owns it: a slot's runner, or without
`owner` the user running this (a host's build is the monitor's own). The walk (os.fwalk) follows no link and opens no
folder more than the reader's DEPTH_MAX folders below the root, and a binary is a regular file with an x bit, by lstat:
a link is never one. A refused build exports nothing, with why on stderr (exit 1).
"""
import ast
import os
import pwd
import re
import shlex
import stat
import sys

sys.dont_write_bytecode = True  # the package folder is read-only at run time
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "worker"
    ),
)
import pr_results  # noqa: E402  (the result reader: how the monitor reads a slot's files)

NAME = re.compile(r"ROCKE_[A-Z0-9_]+")


def switches(src: str) -> list[tuple[str, str]]:
    """run_all.py's NATIVE_PARITY_TESTS as (switch, binary) pairs, rocKE's own switches only."""
    try:
        with open(src, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError, ValueError):
        return []
    pairs = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "NATIVE_PARITY_TESTS" for t in node.targets
        ):
            try:
                pairs = [(v, n) for v, n, _label in ast.literal_eval(node.value)]
            except (ValueError, TypeError):
                pairs = []
    return [(v, n) for v, n in pairs if isinstance(v, str) and NAME.fullmatch(v)]


def build_root(build: str, owner: str) -> int:
    """The build root, opened: Refused (an OSError) when it is a link or no folder, a process of its user runs, or
    `owner` (without it, this user) does not own it."""
    try:
        uid = pwd.getpwnam(owner).pw_uid if owner else os.geteuid()
    except KeyError:
        raise pr_results.Refused(f"{build}: no user {owner}") from None
    fd, got = pr_results.open_root(build)
    if got != uid:
        os.close(fd)
        raise pr_results.Refused(
            f"{build}: not {owner or 'this user'}'s build (its owner's id is {got})"
        )
    return fd


def programs(fd: int, build: str, pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Each switch's binary in the build root `fd` (`build`): the first regular file of that name with an x bit, by
    lstat, in a walk that follows no link and opens no folder more than DEPTH_MAX folders below the root.
    """
    found = {}
    for top, dirs, files, dfd in os.fwalk(".", dir_fd=fd, follow_symlinks=False):
        if top.count("/") >= pr_results.DEPTH_MAX:
            dirs[:] = []  # the folders in it are deeper than the walk goes
        for var, name in pairs:
            if var in found or name not in files:
                continue
            try:
                s = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(s.st_mode) and s.st_mode & 0o111:
                found[var] = os.path.normpath(os.path.join(build, top, name))
    return found


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "usage: native_env.py <run_all.py> <build root> [<owner's user name>]",
            file=sys.stderr,
        )
        return 2
    pairs = switches(argv[0])
    if not pairs:
        return 0
    try:
        fd = build_root(argv[1], argv[2] if len(argv) == 3 else "")
    except OSError as exc:
        print(f"native_env: {exc}", file=sys.stderr)
        return 1
    try:
        found = programs(fd, argv[1], pairs)
    finally:
        os.close(fd)
    for var, path in found.items():
        print(f"export {var}={shlex.quote(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
