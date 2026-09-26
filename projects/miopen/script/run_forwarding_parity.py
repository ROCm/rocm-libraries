#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run the hipDNN forwarding parity harness.

Replays the shim surface with forwarding disabled and then enabled, and compares
the two runs test for test. See test/gtest/README.md for how it is registered.
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


# Printed by the library when forwarding is enabled; the only outside sign of
# which mode a replay actually ran in.
ENABLED_BANNER = "[MIOpen] MIOPEN_HIPDNN_FORWARDING="


def run(argv, what, env=None):
    print(f"+ {' '.join(str(a) for a in argv)}", flush=True)
    code = subprocess.run([str(a) for a in argv], env=env).returncode
    if code != 0:
        print(f"FAIL: {what} exited {code}", flush=True)
    return code == 0


def replay(argv, mode, env):
    what = f"forwarding={mode} replay"
    print(f"+ {' '.join(str(a) for a in argv)}", flush=True)
    result = subprocess.run(
        [str(a) for a in argv],
        env=env,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    sys.stderr.write(result.stderr)
    sys.stderr.flush()
    if result.returncode != 0:
        print(f"FAIL: {what} exited {result.returncode}", flush=True)
        return False

    announced = ENABLED_BANNER in result.stderr
    if mode == "enabled" and not announced:
        print(
            f"FAIL: {what} never printed '{ENABLED_BANNER}...', so it ran with "
            "forwarding disabled.",
            flush=True,
        )
        return False
    if mode == "disabled" and announced:
        print(
            f"FAIL: {what} printed '{ENABLED_BANNER}...', so it ran with "
            "forwarding enabled.",
            flush=True,
        )
        return False
    return True


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
        # packaging regression, not something to skip.
        for problem in problems:
            print(f"FAIL: {problem}", flush=True)
        return 1
    print(f"libraries under test: {wrapper_lib}, {private_lib}", flush=True)

    # An installed test binary's RUNPATH names the ROCm library directory, so an
    # install to any other prefix would otherwise load some other MIOpen.
    ld_path = os.pathsep.join(
        p for p in (str(private_lib.parent), os.environ.get("LD_LIBRARY_PATH")) if p
    )

    # Not the working directory: ctest runs the installed entry inside the install
    # tree, which may be read-only. Kept after a failure for inspection.
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
        # A replay that dies before writing must not leave an old report behind.
        report.unlink(missing_ok=True)
        reports.append(report)
        env = dict(os.environ, MIOPEN_HIPDNN_FORWARDING=mode, LD_LIBRARY_PATH=ld_path)
        ok = replay(
            [
                gtest,
                f"--gtest_filter={args.filter}",
                f"--gtest_output=xml:{report}",
            ],
            mode,
            env,
        )
        if not ok:
            # Comparing a partial report would bury the failure that caused it.
            return 1

    # --newer-than rejects stale reports from an earlier build. sys.executable rather
    # than the shebang, so a lost exec bit fails cleanly.
    ok = run(
        [sys.executable, args.compare, *reports, "--newer-than", gtest],
        "forwarding parity comparison",
    )

    if ok and temporary:
        shutil.rmtree(output_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
