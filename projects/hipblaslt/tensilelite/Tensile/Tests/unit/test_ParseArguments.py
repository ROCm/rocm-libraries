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
from argparse import ArgumentTypeError
from pathlib import Path

from Tensile.TensileLogic.ParseArguments import parseArguments, positive_int


@pytest.mark.unit
class TestPositiveInt:
    """Tests for positive_int validation function"""

    def test_positive_int_valid(self):
        """Test positive_int with valid positive integers"""
        assert positive_int("1") == 1
        assert positive_int("10") == 10
        assert positive_int("100") == 100
        assert positive_int("48") == 48

    def test_positive_int_zero_raises(self):
        """Test positive_int raises for zero"""
        with pytest.raises(ArgumentTypeError, match="is not a positive integer"):
            positive_int("0")

    def test_positive_int_negative_raises(self):
        """Test positive_int raises for negative integers"""
        with pytest.raises(ArgumentTypeError, match="is not a positive integer"):
            positive_int("-1")
        with pytest.raises(ArgumentTypeError, match="is not a positive integer"):
            positive_int("-10")

    def test_positive_int_invalid_string_raises(self):
        """Test positive_int raises for non-integer strings"""
        with pytest.raises(ArgumentTypeError, match="is not a valid integer"):
            positive_int("abc")
        with pytest.raises(ArgumentTypeError, match="is not a valid integer"):
            positive_int("12.5")
        with pytest.raises(ArgumentTypeError, match="is not a valid integer"):
            positive_int("")


@pytest.mark.unit
class TestParseArguments:
    """Tests for ParseArguments function"""

    def test_parse_arguments_basic(self, monkeypatch):
        """Test parsing basic arguments"""
        test_args = ["test_script.py", "/path/to/logic"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.LogicPath == "/path/to/logic"
        assert args.Verbose == 1  # default
        assert args.Jobs == 48  # default

    def test_parse_arguments_with_verbose(self, monkeypatch):
        """Test parsing with verbose flag"""
        test_args = ["test_script.py", "/path/to/logic", "-v", "2"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.Verbose == 2

    def test_parse_arguments_with_jobs(self, monkeypatch):
        """Test parsing with jobs argument"""
        test_args = ["test_script.py", "/path/to/logic", "--jobs", "16"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.Jobs == 16

    def test_parse_arguments_with_check_all(self, monkeypatch):
        """Test parsing with check-all flag"""
        test_args = ["test_script.py", "/path/to/logic", "--check-all"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.CheckAll is True
        assert args.CheckOnlyCustomKernels is False

    def test_parse_arguments_with_check_only_custom_kernels(self, monkeypatch):
        """Test parsing with check-only-custom-kernels flag"""
        test_args = ["test_script.py", "/path/to/logic", "--check-only-custom-kernels"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.CheckOnlyCustomKernels is True
        assert args.CheckAll is False

    def test_parse_arguments_with_cxx_compiler(self, monkeypatch):
        """Test parsing with cxx-compiler argument"""
        test_args = ["test_script.py", "/path/to/logic", "--cxx-compiler", "/usr/bin/g++"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.CxxCompiler == "/usr/bin/g++"

    def test_parse_arguments_with_known_bugs(self, monkeypatch):
        """Test parsing with known-bugs argument"""
        test_args = ["test_script.py", "/path/to/logic", "--known-bugs", "bugs.yaml"]
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.KnownBugs == Path("bugs.yaml")

    def test_parse_arguments_all_options(self, monkeypatch):
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
        monkeypatch.setattr(sys, "argv", test_args)

        args = parseArguments()

        assert args.LogicPath == "/path/to/logic"
        assert args.Verbose == 3
        assert args.Jobs == 32
        assert args.CheckAll is True
        assert args.CxxCompiler == "/opt/rocm/bin/amdclang++"
        assert args.KnownBugs == Path("known_issues.yaml")

    def test_parse_arguments_jobs_zero_raises(self, monkeypatch, capsys):
        """Test that jobs=0 raises error"""
        test_args = ["test_script.py", "/path/to/logic", "--jobs", "0"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit):
            parseArguments()

        captured = capsys.readouterr()
        assert "is not a positive integer" in captured.err

    def test_parse_arguments_jobs_negative_raises(self, monkeypatch, capsys):
        """Test that negative jobs value raises error"""
        test_args = ["test_script.py", "/path/to/logic", "--jobs", "-5"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit):
            parseArguments()

        captured = capsys.readouterr()
        assert "is not a positive integer" in captured.err

    def test_parse_arguments_jobs_invalid_raises(self, monkeypatch, capsys):
        """Test that non-integer jobs value raises error"""
        test_args = ["test_script.py", "/path/to/logic", "--jobs", "abc"]
        monkeypatch.setattr(sys, "argv", test_args)

        with pytest.raises(SystemExit):
            parseArguments()

        captured = capsys.readouterr()
        assert "is not a valid integer" in captured.err
