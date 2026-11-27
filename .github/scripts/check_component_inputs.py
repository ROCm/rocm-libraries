#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fail when a Component CI build reads a tracked file its path filter doesn't cover.

component_ci.py decides from changed paths whether a lane runs. If a lane's build reads
a file outside its patterns, a PR that changes only that file skips the lane. Run after
a Ninja build: the inputs are the CMake files the configure read (build.ninja's
RERUN_CMAKE rule) and every source and header the compiler read (`ninja -t deps`).
Not seen: files read only at test run time or by execute_process()/file(COPY) at
configure time; keep those inside the component's own directories.

  check_component_inputs.py <component> <build dir>
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ci_utils import matches_paths  # noqa: E402
from component_ci import COMPONENTS, INFRA_PATHS  # noqa: E402


def cmake_inputs(build_ninja):
    """Files listed as dependencies of the rule that re-runs CMake."""
    lines = iter(build_ninja.splitlines())
    for line in lines:
        if line.startswith("build build.ninja") and "RERUN_CMAKE" in line:
            while line.endswith("$"):
                line = line[:-1] + next(lines, "")
            deps = line.split("RERUN_CMAKE", 1)[1].partition("|")[2]
            # Ninja escapes a space in a path as "$ ".
            return [p.replace("\0", " ") for p in deps.replace("$ ", "\0").split()]
    return []


def ninja_program(build_dir):
    """The ninja CMake configured with (CMAKE_MAKE_PROGRAM), else the one on PATH."""
    try:
        with open(os.path.join(build_dir, "CMakeCache.txt")) as f:
            for line in f:
                if line.startswith("CMAKE_MAKE_PROGRAM:"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return "ninja"


def depfile_inputs(ninja_deps):
    """Paths `ninja -t deps` prints: each object's indented dependency lines."""
    return [line.strip() for line in ninja_deps.splitlines() if line.startswith("    ")]


def uncovered(paths, build_dir, repo_root, tracked, patterns):
    """Tracked repo files among `paths` that `patterns` don't match, repo-relative."""
    root = os.path.realpath(repo_root) + os.sep
    out = set()
    for p in paths:
        full = os.path.realpath(os.path.join(build_dir, p))
        if full.startswith(root):
            rel = full[len(root) :]
            if rel in tracked and not matches_paths({rel}, patterns):
                out.add(rel)
    return sorted(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("component", choices=sorted(COMPONENTS))
    parser.add_argument("build_dir")
    args = parser.parse_args(argv)

    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # The full tree, even in a sparse checkout.
    tracked = set(
        subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")
    )
    with open(os.path.join(args.build_dir, "build.ninja")) as f:
        paths = cmake_inputs(f.read())
    paths += depfile_inputs(
        subprocess.run(
            [ninja_program(args.build_dir), "-C", args.build_dir, "-t", "deps"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if not paths:
        print(
            f"::error::no build inputs found in {args.build_dir}; run after a Ninja build"
        )
        return 1

    missing = uncovered(
        paths,
        args.build_dir,
        repo_root,
        tracked,
        COMPONENTS[args.component] + INFRA_PATHS,
    )
    for rel in missing:
        print(
            f"::error file={rel}::{args.component}'s build reads {rel}, which its "
            f"path filter in .github/scripts/component_ci.py doesn't cover"
        )
    print(f"{len(set(paths))} build inputs checked, {len(missing)} not covered")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
