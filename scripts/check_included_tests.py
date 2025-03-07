#!/usr/bin/env python3

################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
Prints every `.cpp` file in `tests_dir` that is not mentioned in `cmakelist_path`.
Then returns the number of missing file paths.

Usage:
scripts/check_included_tests.py -t test/ -c CMakeLists.txt

"""

import sys
import argparse
from pathlib import Path


def check_included_tests(tests_dir: str, cmakelist_path: str) -> int:
    tests_dir = Path(tests_dir)
    assert tests_dir.is_dir()
    cmakelist_path = Path(cmakelist_path)
    assert cmakelist_path.is_file()

    tests_in_dir = set(tests_dir.glob("**/*.cpp"))
    cmakelist_str = cmakelist_path.read_text()

    missingTestCount = 0
    for subpath in tests_in_dir:
        relative_subpath = subpath.resolve().relative_to(
            cmakelist_path.joinpath("..").resolve()
        )
        if str(relative_subpath) not in cmakelist_str:
            missingTestCount += 1
            print("Paths not found:", relative_subpath)

    print(
        "Checked %i paths, found %i missing paths"
        % (len(tests_in_dir), missingTestCount)
    )
    return missingTestCount


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prints every `.cpp` file in `tests_dir` that is not mentioned in `cmakelist_path`"
    )
    parser.add_argument(
        "-t",
        "--test-dir",
        default="../test/",
        help="Directory to tests",
    )
    parser.add_argument(
        "-c",
        "--cmakelist-path",
        default="../test/CMakeLists.txt",
        help="Path to CMakeLists file",
    )

    args = parser.parse_args()
    sys.exit(check_included_tests(args.test_dir, args.cmakelist_path))
