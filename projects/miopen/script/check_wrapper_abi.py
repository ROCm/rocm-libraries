#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Gate the exported ABI of a hipDNN wrapper build.

This is the command behind its own ctest entry, separate from the forwarding
parity harness in run_forwarding_parity.py. It only reads the two built shared
objects -- it never loads or executes them -- so unlike the parity harness it
needs no GPU, and is registered wherever MIOPEN_ENABLE_HIPDNN_WRAPPER is on
regardless of whether one is available.

Registered from both the build tree and an installed one, which differ only in
where the two libraries and --public-header sit; the defaults below suit the
installed layout, where everything but --public-header lands beside this script.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from miopen_wrapper_libs import resolve_pair

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lib-dir",
        help="directory holding libMIOpen.so* and libMIOpen_private.so*; defaults "
        "to searching lib*/ beside this script's parent directory",
    )
    parser.add_argument("--abi-check", default=SCRIPT_DIR / "check_public_abi.py")
    parser.add_argument("--baseline", default=SCRIPT_DIR / "public_symbols.baseline")
    parser.add_argument(
        "--excluded", default=SCRIPT_DIR / "wrapper_excluded_symbols.txt"
    )
    parser.add_argument(
        "--public-header",
        help="path to include/miopen/miopen.h; only meaningful from a build tree, "
        "since an installed tree has no include directory. Enables the check that "
        "no excluded symbol is declared there",
    )
    args = parser.parse_args()

    lib_dirs = (
        [Path(args.lib_dir)] if args.lib_dir else sorted(SCRIPT_DIR.parent.glob("lib*"))
    )
    wrapper_lib, private_lib, problems = resolve_pair(lib_dirs)
    if problems:
        # Both are built whenever this entry is registered, so a missing one is a
        # packaging or layout regression rather than a configuration to skip over.
        for problem in problems:
            print(f"FAIL: {problem}", flush=True)
        return 1
    # Named, so a failure below can be tied back to the files it is about.
    print(f"libraries under test: {wrapper_lib}, {private_lib}", flush=True)

    command = [
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
    ]
    if args.public_header:
        command += ["--public-header", args.public_header]
    print(f"+ {' '.join(str(a) for a in command)}", flush=True)
    return subprocess.run([str(a) for a in command]).returncode


if __name__ == "__main__":
    sys.exit(main())
