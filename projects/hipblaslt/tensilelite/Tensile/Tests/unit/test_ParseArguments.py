################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
import sys
from pathlib import Path

from Tensile.TensileLogic.ParseArguments import parseArguments


@pytest.mark.unit
class TestParseArguments:
    """Tests for ParseArguments function"""

    def test_parse_arguments_basic(self):
        """Test parsing basic arguments"""
        test_args = ["test_script.py", "/path/to/logic"]
        sys.argv = test_args

        args = parseArguments()

        assert args.LogicPath == "/path/to/logic"
        assert args.Verbose == 1  # default
        # Note: Default is int, but argument parser doesn't specify type=int
        assert args.Jobs == 48  # default is int

    def test_parse_arguments_with_verbose(self):
        """Test parsing with verbose flag"""
        test_args = ["test_script.py", "/path/to/logic", "-v", "2"]
        sys.argv = test_args

        args = parseArguments()

        assert args.Verbose == 2

    def test_parse_arguments_with_jobs(self):
        """Test parsing with jobs argument"""
        test_args = ["test_script.py", "/path/to/logic", "--jobs", "16"]
        sys.argv = test_args

        args = parseArguments()

        # Note: Jobs becomes a string when passed as argument because
        # the argument parser doesn't specify type=int (inconsistent with default)
        assert args.Jobs == "16"

    def test_parse_arguments_with_check_all(self):
        """Test parsing with check-all flag"""
        test_args = ["test_script.py", "/path/to/logic", "--check-all"]
        sys.argv = test_args

        args = parseArguments()

        assert args.CheckAll is True
        assert args.CheckOnlyCustomKernels is False

    def test_parse_arguments_with_check_only_custom_kernels(self):
        """Test parsing with check-only-custom-kernels flag"""
        test_args = ["test_script.py", "/path/to/logic", "--check-only-custom-kernels"]
        sys.argv = test_args

        args = parseArguments()

        assert args.CheckOnlyCustomKernels is True
        assert args.CheckAll is False

    def test_parse_arguments_with_cxx_compiler(self):
        """Test parsing with cxx-compiler argument"""
        test_args = ["test_script.py", "/path/to/logic", "--cxx-compiler", "/usr/bin/g++"]
        sys.argv = test_args

        args = parseArguments()

        assert args.CxxCompiler == "/usr/bin/g++"

    def test_parse_arguments_with_known_bugs(self):
        """Test parsing with known-bugs argument"""
        test_args = ["test_script.py", "/path/to/logic", "--known-bugs", "bugs.yaml"]
        sys.argv = test_args

        args = parseArguments()

        assert args.KnownBugs == Path("bugs.yaml")

    def test_parse_arguments_all_options(self):
        """Test parsing with all options"""
        test_args = [
            "test_script.py",
            "/path/to/logic",
            "-v", "3",
            "--jobs", "32",
            "--check-all",
            "--cxx-compiler", "/opt/rocm/bin/amdclang++",
            "--known-bugs", "known_issues.yaml"
        ]
        sys.argv = test_args

        args = parseArguments()

        assert args.LogicPath == "/path/to/logic"
        assert args.Verbose == 3
        assert args.Jobs == "32"
        assert args.CheckAll is True
        assert args.CxxCompiler == "/opt/rocm/bin/amdclang++"
        assert args.KnownBugs == Path("known_issues.yaml")
