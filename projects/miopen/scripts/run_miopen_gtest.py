#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dapper-aware gtest runner (the yaml `gtest_runner`) for MIOpen.

Invoked by the generated CTestTestfile for a dapper-enabled category:

    run_miopen_gtest.py --dapper-json <json> --category <name> \
        --category-filter "<patterns>" -- <gtest_exe> [exe args...]

It computes the effective --gtest_filter from the dapper JSON (impact filter
intersected with the category, honoring fallback_mode) and execs the gtest binary
with that filter appended, propagating the exit code. This is what makes Dapper
"active": the reduced (subtractive) set actually runs, transparently to CTest.

Self-contained on the runner: this script and dapper_union.py are co-installed
next to the CTestTestfile (bin/<PROJECT>/), so `import dapper_union` resolves from
this script's own directory.
"""

import argparse
import os
import subprocess
import sys

# Make the co-located dapper_union importable (installed layout), with a
# source-tree fallback to shared/ctest for local testing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dapper_union
except ImportError:
    _shared = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "ctest")
    )
    sys.path.insert(0, _shared)
    import dapper_union


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit(
            "run_miopen_gtest.py: expected '--' separating options from the gtest "
            "command, e.g. --dapper-json J --category C --category-filter F -- ./gtest"
        )
    split = argv.index("--")
    our_args, gtest_cmd = argv[:split], argv[split + 1 :]
    if not gtest_cmd:
        sys.exit("run_miopen_gtest.py: no gtest command given after '--'")

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dapper-json", required=True)
    parser.add_argument("--category", default="none")
    parser.add_argument("--category-filter", default="*")
    args = parser.parse_args(our_args)

    gtest_filter = dapper_union.compute_filter(
        args.dapper_json, args.category, args.category_filter
    )

    full_cmd = gtest_cmd + [f"--gtest_filter={gtest_filter}"]
    print("run_miopen_gtest: " + " ".join(full_cmd), flush=True)
    result = subprocess.run(full_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
