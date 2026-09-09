#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the hipDNN forwarding parity harness.

This is the command behind a single ctest entry. It replays the shim surface
twice -- once with forwarding disabled, once with it enabled -- compares the two
runs test for test, and then checks the wrapper's exported ABI.

Three steps in one process rather than three ctest entries tied together by a
fixture: a sharded ctest run distributes whole entries, so the members of a
fixture can land in different shards and fail as unsatisfied. Sequencing them
here also keeps the fixture's best property, that a replay which dies never
reaches the comparison, without depending on ctest to enforce it.

Registered only where MIOPEN_ENABLE_HIPDNN_WRAPPER is on, so it never has to
work out which kind of tree it is running in. It runs from both the build tree
and an installed one, which differ only in where its inputs sit; the defaults
below suit the installed layout, where everything lands in one directory.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_library(lib_dirs, stem):
    """Return the real (non-symlink) shared object for `stem`, or None.

    Skips the symlinks so the versioned file is what gets inspected.
    """
    for lib_dir in lib_dirs:
        matches = [
            p
            for p in sorted(lib_dir.glob(f"{stem}.so.*"))
            if p.is_file() and not p.is_symlink()
        ]
        if matches:
            return matches[0]
    return None


def run(argv, what, env=None):
    print(f"+ {' '.join(str(a) for a in argv)}", flush=True)
    code = subprocess.run([str(a) for a in argv], env=env).returncode
    if code != 0:
        print(f"FAIL: {what} exited {code}", flush=True)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtest", required=True, help="the test binary to replay")
    parser.add_argument(
        "--filter", required=True, help="gtest filter selecting the shim surface"
    )
    parser.add_argument(
        "--output-dir", default=".", help="where the two replay XMLs are written"
    )
    parser.add_argument(
        "--lib-dir",
        help="directory holding libMIOpen.so*; defaults to searching lib*/ beside "
        "this script's parent directory",
    )
    parser.add_argument("--compare", default=SCRIPT_DIR / "compare_forwarding_runs.py")
    parser.add_argument("--abi-check", default=SCRIPT_DIR / "check_public_abi.py")
    parser.add_argument("--baseline", default=SCRIPT_DIR / "public_symbols.baseline")
    parser.add_argument(
        "--excluded", default=SCRIPT_DIR / "wrapper_excluded_symbols.txt"
    )
    args = parser.parse_args()

    gtest = Path(args.gtest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for mode in ("disabled", "enabled"):
        report = output_dir / f"{gtest.name}_forwarding_{mode}.xml"
        # Removed rather than overwritten: a replay that dies before writing would
        # otherwise leave the previous run's file for the comparison to read.
        report.unlink(missing_ok=True)
        reports.append(report)
        env = dict(os.environ, MIOPEN_HIPDNN_FORWARDING=mode)
        ok = run(
            [
                gtest,
                f"--gtest_filter={args.filter}",
                f"--gtest_output=xml:{report}",
            ],
            f"forwarding={mode} replay",
            env=env,
        )
        if not ok:
            # Short-circuit: the comparison of a run that failed reports whatever
            # partial XML it left behind, which buries the failure that caused it.
            return 1

    # --newer-than: two XML files left over from an earlier build compare just as
    # cleanly as two fresh ones, so it holds the replays to this run's binary.
    ok = run(
        [args.compare, *reports, "--newer-than", gtest],
        "forwarding parity comparison",
    )

    lib_dirs = (
        [Path(args.lib_dir)] if args.lib_dir else sorted(SCRIPT_DIR.parent.glob("lib*"))
    )
    wrapper_lib = find_library(lib_dirs, "libMIOpen")
    private_lib = find_library(lib_dirs, "libMIOpen_private")
    if wrapper_lib is None or private_lib is None:
        # Both are built whenever this entry is registered, so a missing one is a
        # packaging or layout regression rather than a configuration to skip over.
        print(
            "FAIL: expected libMIOpen.so and libMIOpen_private.so under "
            f"{', '.join(str(d) for d in lib_dirs) or '<no lib directory>'}",
            flush=True,
        )
        return 1

    # --public-header is deliberately absent: the check it enables compares the
    # exclusion list against miopen.h, both source files, so it belongs to the
    # build and an installed tree has no include directory to point it at.
    ok &= run(
        [
            sys.executable,
            args.abi_check,
            "check-wrapper",
            wrapper_lib,
            "--baseline",
            args.baseline,
            "--excluded",
            args.excluded,
            "--private-lib",
            private_lib,
        ],
        "wrapper ABI check",
    )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
