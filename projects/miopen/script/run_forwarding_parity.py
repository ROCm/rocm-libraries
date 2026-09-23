#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the hipDNN forwarding parity harness.

This is the command behind a single ctest entry. It replays the shim surface
twice -- once with forwarding disabled, once with it enabled -- and compares the
two runs test for test. The wrapper's exported ABI is a separate concern with its
own entry, check_wrapper_abi.py: it needs only the two built libraries, not a
GPU, so tying it to this harness would make it wait on a GPU it does not need.

The two replays and the comparison run in one process rather than as three
ctest entries tied together by a fixture: a sharded ctest run distributes whole
entries, so the members of a fixture can land in different shards and fail as
unsatisfied. Sequencing them here also keeps the fixture's best property, that a
replay which dies never reaches the comparison, without depending on ctest to
enforce it.

The two replays can only diverge for entry points named in kForwardingEntries, in
src/private/routing.cpp. That array is empty today, so both replays run the same
code and this entry cannot fail; adding the first name there is what gives it
detection power. Until then it is a harness kept warm, not a check.

Registered only where MIOPEN_ENABLE_HIPDNN_WRAPPER is on, so it never has to
work out which kind of tree it is running in. It runs from both the build tree
and an installed one, which differ only in where its inputs sit; the defaults
below suit the installed layout, where everything lands in one directory.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from miopen_wrapper_libs import resolve_pair

SCRIPT_DIR = Path(__file__).resolve().parent


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
        "--output-dir",
        help="where the two replay XMLs are written; defaults to a temporary directory",
    )
    parser.add_argument(
        "--lib-dir",
        help="directory holding libMIOpen.so*; defaults to searching lib*/ beside "
        "this script's parent directory",
    )
    parser.add_argument("--compare", default=SCRIPT_DIR / "compare_forwarding_runs.py")
    args = parser.parse_args()

    gtest = Path(args.gtest).resolve()

    lib_dirs = (
        [Path(args.lib_dir)] if args.lib_dir else sorted(SCRIPT_DIR.parent.glob("lib*"))
    )
    wrapper_lib, private_lib, problems = resolve_pair(lib_dirs)
    if problems:
        # Both are built whenever this entry is registered, so a missing one is a
        # packaging or layout regression rather than a configuration to skip over.
        # Resolved before the replays so that regression is reported directly
        # instead of as a loader error inside the first replay.
        for problem in problems:
            print(f"FAIL: {problem}", flush=True)
        return 1
    # Named, so a load failure below can be tied back to the files it is about.
    print(f"libraries under test: {wrapper_lib}, {private_lib}", flush=True)

    # The replays have to load the pair. An installed test binary's RUNPATH names
    # the ROCm library directory, not the tree it was installed into, so without
    # this an install to any other prefix silently replays some other MIOpen -- or,
    # with no private library beside it, fails to start at all.
    ld_path = os.pathsep.join(
        p for p in (str(private_lib.parent), os.environ.get("LD_LIBRARY_PATH")) if p
    )

    # A temporary directory rather than the working directory, because ctest runs the installed
    # entry from inside the install tree, which on a shipping prefix is root-owned and read-only
    # to whoever runs the tests. The XMLs only feed the comparison below, so the directory is
    # removed after a pass. After a failure it is kept, and its printed path is how to find it.
    temporary = not args.output_dir
    if temporary:
        output_dir = Path(tempfile.mkdtemp(prefix="miopen_forwarding_parity_"))
    else:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    print(f"replay reports: {output_dir}", flush=True)

    reports = []
    for mode in ("disabled", "enabled"):
        report = output_dir / f"{gtest.name}_forwarding_{mode}.xml"
        # Removed rather than overwritten: a replay that dies before writing would
        # otherwise leave the previous run's file for the comparison to read.
        report.unlink(missing_ok=True)
        reports.append(report)
        env = dict(os.environ, MIOPEN_HIPDNN_FORWARDING=mode, LD_LIBRARY_PATH=ld_path)
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
    # Both helpers are launched through this interpreter rather than their shebangs,
    # so the harness and the scripts it drives cannot end up on different Pythons,
    # and a lost exec bit becomes a FAIL line instead of a PermissionError traceback.
    ok = run(
        [sys.executable, args.compare, *reports, "--newer-than", gtest],
        "forwarding parity comparison",
    )

    if ok and temporary:
        shutil.rmtree(output_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
