#!/usr/bin/env python3
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
        default="../CMakeLists.txt",
        help="Path to CMakeLists file",
    )

    args = parser.parse_args()
    sys.exit(check_included_tests(args.test_dir, args.cmakelist_path))
