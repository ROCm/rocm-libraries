#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_build_tree_ctest.py -- build targets of the platform and run selected
# CTest entries of its build tree.
#
# Some CTest entries exist in the build tree only: they run a program the build
# makes against files the installed test tree does not carry. This runner lets a
# job that has no build tree of its own (a pre-commit hook) run such entries:
# it configures the platform, builds the targets the entries need (or, when it
# names none, every program of the tree), and runs the entries through CTest,
# so the command each entry runs stays defined in one place, its CMake
# registration. It selects entries by exact name, by label, or both. It needs
# cmake, ctest and a C++ compiler on PATH, and fails with one line naming cmake
# or ctest when either is missing; a new build tree uses Ninja when it is on
# PATH.
#
# The configure leaves out the sibling library and the install rules, so it
# reads no CMake file outside the platform. The build tree is kept between runs,
# one per user and source tree, in the temporary directory, so a repeated run
# rebuilds only what changed. Before building, the runner checks that every
# named entry is registered and that every label selects an entry, so a
# renamed or removed entry, or a label no entry carries, fails here instead of
# leaving nothing to run.
#
# Usage:
#   python rocke/platform/tools/run_build_tree_ctest.py \
#       [--target <cmake target> ...] [--test <ctest entry> ...] \
#       [--label <label regex> ...] [--build-root DIR]
# At least one --test or --label is required.

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM / "python") not in sys.path:
    sys.path.insert(0, str(PLATFORM / "python"))
from rocke.tests.ctest_listing import list_ctest_entries
from rocke.tests.host_cpus import available_cpus
from rocke.tests.platform_build import build_platform_targets, configure_platform

# Single- and multi-config generators both build and test this configuration.
CONFIG = "Release"

# The programs the runner starts itself; CMake finds the C++ compiler.
REQUIRED_PROGRAMS = ("cmake", "ctest")

# The platform alone: its tests, without the sibling library or install rules.
PLATFORM_ONLY = (
    "-DBUILD_TESTING=ON",
    "-DROCKE_BUILD_LIBRARY_CPP=OFF",
    "-DROCKE_INSTALL_TESTS=OFF",
    "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
)


def _user() -> str:
    """The current user's name, or an empty string where none is known."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return ""


def default_build_root() -> Path:
    """The kept build tree: one per user and source tree, so two users or two
    checkouts do not share one."""
    key = f"{_user()}\0{PLATFORM}".encode()
    digest = hashlib.sha256(key).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"rocke_build_tree_ctest_{digest}"


def missing_programs() -> list[str]:
    """The programs of ``REQUIRED_PROGRAMS`` that are not on PATH."""
    return [program for program in REQUIRED_PROGRAMS if shutil.which(program) is None]


def _exact_names(tests: Sequence[str]) -> str:
    """A CTest ``-R`` regex that matches exactly the entries ``tests``."""
    return "^(" + "|".join(re.escape(test) for test in tests) + ")$"


def require_registered(build_root: Path, tests: Sequence[str]) -> None:
    """Raise ``ValueError`` naming each of ``tests`` that ``build_root`` does
    not register."""
    if not tests:
        return
    registered = {
        entry["name"]
        for entry in list_ctest_entries(
            build_root, config=CONFIG, name_pattern=_exact_names(tests)
        )
    }
    unregistered = sorted(set(tests) - registered)
    if unregistered:
        raise ValueError(
            f"build-tree CTest run failed - unregistered={unregistered!r}, "
            f"expected every named test registered in {build_root}"
        )


def labelled_entries(build_root: Path, labels: Sequence[str]) -> set[str]:
    """The names of the entries of ``build_root`` with a label matching one of
    the ``labels`` regexes. Raises ``ValueError`` naming each regex that
    selects no entry."""
    selected: set[str] = set()
    unmatched = []
    for label in labels:
        names = {
            entry["name"]
            for entry in list_ctest_entries(
                build_root, config=CONFIG, label_pattern=label
            )
        }
        if not names:
            unmatched.append(label)
        selected |= names
    if unmatched:
        raise ValueError(
            f"build-tree CTest run failed - labels={unmatched!r} select no entry, "
            f"expected each label to select an entry registered in {build_root}"
        )
    return selected


def run_tests(build_root: Path, tests: Sequence[str]) -> int:
    """Run exactly ``tests``, side by side, and return CTest's exit status."""
    return subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_root),
            "-C",
            CONFIG,
            "-R",
            _exact_names(tests),
            "--output-on-failure",
            "--no-tests=error",
            "--parallel",
            str(available_cpus()),
        ],
        check=False,
        # The entries run Python scripts of the source tree; keep their
        # bytecode out of it.
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    ).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run build-tree CTest entries.")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="a CMake target the entries need (repeatable); with none, the "
        "build builds every program of the tree",
    )
    parser.add_argument(
        "--test",
        action="append",
        default=[],
        help="a CTest entry to run, by exact name (repeatable)",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="a regex of CTest labels; runs every entry with a matching label "
        "(repeatable)",
    )
    parser.add_argument("--build-root", type=Path, default=default_build_root())
    args = parser.parse_args(argv)
    if not (args.test or args.label):
        parser.error("name at least one --test or --label")

    if missing := missing_programs():
        print(
            f"error: build-tree CTest run failed - missing={missing!r}, expected "
            f"{' and '.join(REQUIRED_PROGRAMS)} on PATH",
            file=sys.stderr,
        )
        return 1
    try:
        configure_platform(args.build_root, *PLATFORM_ONLY, config=CONFIG)
        require_registered(args.build_root, args.test)
        tests = sorted(set(args.test) | labelled_entries(args.build_root, args.label))
        build_platform_targets(args.build_root, args.target, config=CONFIG)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(
            f"error: {error}; the build tree is {args.build_root}, delete it to "
            "start over",
            file=sys.stderr,
        )
        return 1
    return run_tests(args.build_root, tests)


if __name__ == "__main__":
    raise SystemExit(main())
