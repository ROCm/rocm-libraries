# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for check_component_inputs.py."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from check_component_inputs import (  # noqa: E402
    cmake_inputs,
    depfile_inputs,
    ninja_program,
    uncovered,
)


class Parse(unittest.TestCase):
    def test_cmake_inputs_follow_continuations_and_escaped_spaces(self):
        ninja = (
            "rule RERUN_CMAKE\n"
            "build build.ninja /b/cmake_install.cmake: RERUN_CMAKE | /r/CMakeLists.txt $\n"
            "    /r/cmake/a.cmake /r/my$ dir/b.cmake\n"
            "  pool = console\n"
        )
        self.assertEqual(
            cmake_inputs(ninja),
            ["/r/CMakeLists.txt", "/r/cmake/a.cmake", "/r/my dir/b.cmake"],
        )

    def test_cmake_inputs_without_the_rule(self):
        self.assertEqual(cmake_inputs("rule CXX\n"), [])

    def test_depfile_inputs_are_the_indented_lines(self):
        deps = (
            "obj/a.o: #deps 2, deps mtime 1 (VALID)\n"
            "    ../src/a.cpp\n"
            "    /usr/include/stdio.h\n"
            "\n"
        )
        self.assertEqual(depfile_inputs(deps), ["../src/a.cpp", "/usr/include/stdio.h"])


class NinjaProgram(unittest.TestCase):
    def test_reads_cmake_make_program_else_path(self):
        with tempfile.TemporaryDirectory() as build:
            self.assertEqual(ninja_program(build), "ninja")
            with open(os.path.join(build, "CMakeCache.txt"), "w") as f:
                f.write("CMAKE_MAKE_PROGRAM:FILEPATH=/opt/venv/bin/ninja\n")
            self.assertEqual(ninja_program(build), "/opt/venv/bin/ninja")


class Uncovered(unittest.TestCase):
    def test_reports_tracked_repo_files_the_patterns_miss(self):
        with tempfile.TemporaryDirectory() as root:
            build = os.path.join(root, "build")
            os.makedirs(build)
            paths = [
                "../lib/x.cpp",  # covered
                os.path.join(root, "other", "y.h"),  # missed
                os.path.join(root, "build", "gen.h"),  # generated, not tracked
                "/usr/include/stdio.h",  # outside the repo
            ]
            tracked = {"lib/x.cpp", "other/y.h"}
            self.assertEqual(
                uncovered(paths, build, root, tracked, ["lib/**"]), ["other/y.h"]
            )


if __name__ == "__main__":
    unittest.main()
