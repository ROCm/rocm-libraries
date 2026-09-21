#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Prints every `.cpp` file in `tests_dir` that is not mentioned in CMakeLists.txt.
Then returns the number of missing file paths.

Usage:
scripts/check_included_tests.py -t test/
"""

import argparse
import sys
from pathlib import Path


def check_included_tests(
    tests_dir: str,
    exclude_paths: list[str],
) -> int:
    tests_dir = Path(tests_dir)
    assert tests_dir.is_dir()

    def should_exclude(test_path: Path, exclude_paths: list[str]):
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
        Path(d)
        for d in tests_dir.iterdir()
        if d.is_dir() and not should_exclude(d, exclude_paths)
    ]
    missingTestCount = 0
    test_count = 0
    for test_dir in test_dirs:
        tests_in_dir = {x for x in test_dir.glob("**/*.cpp")}
        test_count += len(tests_in_dir)
        missingTestCount += missing_tests(test_dir / "CMakeLists.txt")

    print("Checked %i paths, found %i missing paths" % (test_count, missingTestCount))
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
