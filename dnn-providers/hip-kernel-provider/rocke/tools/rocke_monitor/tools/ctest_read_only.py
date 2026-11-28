#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ctest on a native build this process may only read: a kept slot's, the review's shared build (rkmon_lib/slots.py),
whose tests the verification and code quality passes run as PR code, in a slot of their own.

Usage: ctest_read_only.py BUILD [CTEST ARGS...]
  rk pr-run [--out DIR] -- python3 -B <package>/tools/ctest_read_only.py <build dir>/core -R ...

ctest writes its logs and its scratch (Testing/) in the folder it tests, and a kept slot's build is read-only to every
other runner. Here ctest runs in RKMON_OUT/ctest instead, the results folder rk pr-run gives the command, whose
CTestTestfile.cmake names BUILD as its one subfolder: ctest reads BUILD's CTest files from there, runs each test as they
say, in BUILD's folders, and writes only in RKMON_OUT/ctest, which `rk pr-run --out DIR` keeps. A test that writes into
its own folder fails here, as every write into BUILD does: it needs a build of its own.
The CTEST ARGS go to ctest as they are, but for --test-dir, which this sets. Exit: ctest's; 2 when no ctest ran.
"""

import os
import shutil
import sys

FOLDER = "ctest"  # the folder in RKMON_OUT that ctest runs and writes in
USAGE = "usage: ctest_read_only.py BUILD [CTEST ARGS...]"


def cmake_quoted(path: str) -> str:
    """`path` as one CMake quoted argument: its quotes and dollars escaped, so CMake reads it as it is. A backslash it
    cannot hold: CMake reads one in a path as a folder separator."""
    for c in ('"', "$"):
        path = path.replace(c, "\\" + c)
    return f'"{path}"'


def refused(why: str) -> int:
    print(f"ctest_read_only.py: {why}", file=sys.stderr)
    return 2


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        return refused(USAGE)
    build, args = os.path.abspath(argv[0]), argv[1:]
    if any(a == "--test-dir" or a.startswith("--test-dir=") for a in args):
        return refused("--test-dir is this tool's: name the build first")
    out = os.environ.get("RKMON_OUT", "")
    if not out:
        return refused(
            "RKMON_OUT is not set: run it through rk pr-run, which gives the command a results folder"
        )
    if "\\" in build:
        return refused(
            f"{build}: CMake reads a backslash in a path as a folder separator"
        )
    if not os.path.isfile(os.path.join(build, "CTestTestfile.cmake")):
        return refused(f"{build} has no CTestTestfile.cmake: no CTest build")
    ctest = shutil.which("ctest")
    if not ctest:
        return refused("no ctest on PATH")
    here = os.path.join(out, FOLDER)
    os.makedirs(here, exist_ok=True)
    with open(
        os.path.join(here, "CTestTestfile.cmake"),
        "w",
        encoding="utf-8",
        errors="surrogateescape",
    ) as f:
        f.write(
            f"# ctest_read_only.py: the tests of the build it names\nsubdirs({cmake_quoted(build)})\n"
        )
    os.execv(ctest, [ctest, "--test-dir", here, *args])
    return 2  # not reached: execv replaces this process or raises


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
