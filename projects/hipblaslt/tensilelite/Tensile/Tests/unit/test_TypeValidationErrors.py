################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################
"""Tests for Tensile.Common.TypeValidationErrors."""

import textwrap

import pytest

from Tensile.Common.TypeValidationErrors import (
    ConfigTypeError,
    formatMismatch,
)


class TestConfigTypeError:
    def test_is_exception_subclass(self):
        assert issubclass(ConfigTypeError, Exception)

    def test_distinct_from_value_error(self):
        # Exists to keep tests catching ConfigTypeError specifically.
        assert ConfigTypeError is not ValueError

    def test_raisable(self):
        with pytest.raises(ConfigTypeError):
            raise ConfigTypeError("boom")


class TestFormatMismatch:
    def test_no_srcfile_no_lineno(self):
        msg = formatMismatch("", "GlobalParameters.BoundsCheck", False, {int})
        assert "GlobalParameters.BoundsCheck = False (bool)" in msg
        assert "expected int" in msg
        assert "/" not in msg.split(":")[0] if ":" in msg else True

    def test_includes_repr(self):
        # repr keeps strings quoted so 1 vs "1" is obvious.
        msg = formatMismatch("", "X", "1", {int})
        assert "'1'" in msg

    def test_includes_actual_type(self):
        msg = formatMismatch("", "X", True, {int})
        assert "(bool)" in msg

    def test_expected_union_sorted(self):
        msg = formatMismatch("", "X", 0, {int, str, type(None)})
        # sorted -> NoneType, int, str
        assert "expected NoneType or int or str" in msg

    def test_explicit_lineno(self):
        msg = formatMismatch("foo.yaml", "X", 0, {str}, lineNo=42)
        assert "foo.yaml:42:" in msg

    def test_lineno_recovered_from_file(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(textwrap.dedent("""\
            GlobalParameters:
              BoundsCheck: false
              PinClocks: 0
        """))
        msg = formatMismatch(str(path), "GlobalParameters.BoundsCheck", False, {int})
        # The key "BoundsCheck" is on line 2.
        assert f"{path}:2:" in msg

    def test_missing_file_omits_lineno(self):
        msg = formatMismatch("/no/such/file.yaml", "X", 0, {str})
        # Path is kept, no lineno appended (no trailing ":N:").
        assert "/no/such/file.yaml:" in msg
        # The path prefix ends with ": " (just file, no lineno).
        assert "/no/such/file.yaml: " in msg

    def test_bracketed_keypath_leaf_extraction(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(textwrap.dedent("""\
            BenchmarkProblems:
              - - InternalSupportParams:
                    KernArgsVersion: two
        """))
        # Even with a bracketed keypath, the leaf name "KernArgsVersion" is
        # used for the line-number lookup.
        msg = formatMismatch(
            str(path),
            "BenchmarkProblems[0][0].InternalSupportParams.KernArgsVersion",
            "two",
            {int},
        )
        assert str(path) in msg
        # Line 3 holds "KernArgsVersion: two".
        assert ":3:" in msg
