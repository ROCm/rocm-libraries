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
Prints every `.cpp` file in `tests_dir` that is not mentioned in CMakeLists.txt.
Then returns the number of missing file paths.

Usage:
scripts/check_included_tests.py -t test/
"""

import sys
import argparse
from pathlib import Path
from typing import List


def check_included_tests(
    tests_dir: str,
    exclude_paths: List[str],
) -> int:
    tests_dir = Path(tests_dir)
    assert tests_dir.is_dir()

    def should_exclude(test_path: Path, exclude_paths: List[str]):
        for exclude_path in exclude_paths:
            if test_path.is_relative_to(Path(exclude_path)):
                return True
        return False

    def missing_tests(cmakelist_path):
        assert cmakelist_path.is_file()
        missingTestCount = 0
        for subpath in tests_in_dir:
            relative_subpath = subpath.resolve().relative_to(
                cmakelist_path.joinpath("..").resolve()
            )

            if str(relative_subpath) not in cmakelist_path.read_text():
                missingTestCount += 1
                print("Paths not found:", relative_subpath)
        return missingTestCount

    test_dirs = [
        Path(d) for d in tests_dir.iterdir() if d.is_dir() and not should_exclude(d, exclude_paths)
    ]
    missingTestCount = 0
    test_count = 0
    for test_dir in test_dirs:
        tests_in_dir = {x for x in test_dir.glob("**/*.cpp")}
        test_count += len(tests_in_dir)
        missingTestCount += missing_tests(test_dir / "CMakeLists.txt")

    print(
        "Checked %i paths, found %i missing paths"
        % (test_count, missingTestCount)
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
        "-x",
        "--exclude-path",
        action="append",
        default=["../test/common"],
        help="Paths to exclude",
    )

    args = parser.parse_args()
    sys.exit(
        check_included_tests(
            args.test_dir,
            args.exclude_path,
        )
    )
