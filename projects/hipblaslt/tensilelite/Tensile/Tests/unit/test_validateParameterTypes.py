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

import pytest

from Tensile.SolutionStructs.Solution import (
    validateParameterTypes,
    _getExpectedTypes,
    _expectedParamTypes,
    _skipTypeCheck,
    _typeMismatchCollector,
    resetTypeMismatchCollector,
    printTypeMismatchSummary,
)
from Tensile.Common.ValidParameters import validParameters


class TestGetExpectedTypes:
    """Tests for the _getExpectedTypes helper."""

    def test_returns_dict(self):
        result = _getExpectedTypes(validParameters)
        assert isinstance(result, dict)

    def test_int_param_has_int_type(self):
        """UseCustomMainLoopSchedule: [-1, 0, 1] should expect {int}."""
        result = _getExpectedTypes(validParameters)
        assert "UseCustomMainLoopSchedule" in result
        assert result["UseCustomMainLoopSchedule"] == {int}

    def test_bool_param_has_bool_type(self):
        """BufferLoad: [False, True] should expect {bool}."""
        result = _getExpectedTypes(validParameters)
        assert "BufferLoad" in result
        assert result["BufferLoad"] == {bool}

    def test_sentinel_minus_one_skipped(self):
        """Parameters with validParams[name] == -1 should be skipped."""
        result = _getExpectedTypes({"Foo": -1, "Bar": [0, 1]})
        assert "Foo" not in result
        assert "Bar" in result

    def test_precomputed_matches_fresh(self):
        """The module-level _expectedParamTypes should match a fresh computation."""
        fresh = _getExpectedTypes(validParameters)
        assert _expectedParamTypes == fresh


class TestValidateParameterTypes:
    """Tests for the validateParameterTypes function (warning-collection mode)."""

    def setup_method(self):
        """Reset the collector before each test."""
        resetTypeMismatchCollector()

    # --- Passing cases (no mismatches collected) ---

    def test_int_param_with_int_value_passes(self):
        """int value for an int param should not collect a mismatch."""
        state = {"UseCustomMainLoopSchedule": 0}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    def test_int_param_with_negative_int_passes(self):
        state = {"UseCustomMainLoopSchedule": -1}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    def test_bool_param_with_bool_value_passes(self):
        """bool value for a bool param should not collect a mismatch."""
        state = {"BufferLoad": True}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    def test_bool_param_with_false_passes(self):
        state = {"BufferLoad": False}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    def test_unknown_param_ignored(self):
        """Parameters not in validParameters should be silently skipped."""
        state = {"SomeUnknownParameter": "whatever"}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    def test_empty_state_passes(self):
        validateParameterTypes({})
        assert len(_typeMismatchCollector) == 0

    def test_multiple_valid_params_pass(self):
        state = {
            "UseCustomMainLoopSchedule": 0,
            "BufferLoad": True,
            "BufferStore": False,
            "PrefetchGlobalRead": 1,
        }
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0

    # --- Mismatch collection cases ---

    def test_bool_where_int_expected_collects(self):
        """bool(False) for an int param should be collected, not raise."""
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state)  # should NOT raise
        assert len(_typeMismatchCollector) == 1
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert key in _typeMismatchCollector
        assert _typeMismatchCollector[key]["count"] == 1

    def test_bool_true_where_int_expected_collects(self):
        """bool(True) for an int param should be collected."""
        state = {"UseCustomMainLoopSchedule": True}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert key in _typeMismatchCollector
        assert _typeMismatchCollector[key]["count"] == 1

    def test_int_where_bool_expected_collects(self):
        """int(0) for a bool param like BufferLoad should be collected."""
        state = {"BufferLoad": 0}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("BufferLoad", "int", "bool")
        assert key in _typeMismatchCollector

    def test_int_one_where_bool_expected_collects(self):
        """int(1) for a bool param like BufferLoad should be collected."""
        state = {"BufferLoad": 1}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("BufferLoad", "int", "bool")
        assert key in _typeMismatchCollector

    def test_string_where_int_expected_collects(self):
        state = {"UseCustomMainLoopSchedule": "0"}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("UseCustomMainLoopSchedule", "str", "int")
        assert key in _typeMismatchCollector

    def test_float_where_int_expected_collects(self):
        state = {"UseCustomMainLoopSchedule": 0.0}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("UseCustomMainLoopSchedule", "float", "int")
        assert key in _typeMismatchCollector

    # --- Collected entry details ---

    def test_collector_records_param_name(self):
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state)
        keys = list(_typeMismatchCollector.keys())
        assert keys[0][0] == "UseCustomMainLoopSchedule"

    def test_collector_records_value(self):
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state)
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert "False" in _typeMismatchCollector[key]["values"]

    def test_collector_records_actual_type(self):
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state)
        keys = list(_typeMismatchCollector.keys())
        assert keys[0][1] == "bool"

    def test_collector_records_expected_type(self):
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state)
        keys = list(_typeMismatchCollector.keys())
        assert keys[0][2] == "int"

    def test_collector_records_source_file(self):
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state, srcFile="my_file.yaml")
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert "my_file.yaml" in _typeMismatchCollector[key]["files"]

    def test_collector_empty_srcfile_not_recorded(self):
        """When srcFile is empty string, files set should remain empty."""
        state = {"UseCustomMainLoopSchedule": False}
        validateParameterTypes(state, srcFile="")
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert len(_typeMismatchCollector[key]["files"]) == 0

    # --- Accumulation across multiple calls ---

    def test_multiple_mismatches_all_collected(self):
        """All mismatches in a single state dict should be collected."""
        state = {
            "PrefetchGlobalRead": 1,              # valid int
            "UseCustomMainLoopSchedule": False,    # BAD: bool for int
            "BufferLoad": True,                    # valid bool
        }
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 1
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert key in _typeMismatchCollector

    def test_accumulates_across_calls(self):
        """Multiple validateParameterTypes calls accumulate into the collector."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml")
        validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml")
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert _typeMismatchCollector[key]["count"] == 2
        assert _typeMismatchCollector[key]["values"] == {"False", "True"}
        assert _typeMismatchCollector[key]["files"] == {"a.yaml", "b.yaml"}

    def test_different_params_get_separate_entries(self):
        """Mismatches on different params create separate collector entries."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        validateParameterTypes({"BufferLoad": 0})
        assert len(_typeMismatchCollector) == 2

    # --- Skip list ---

    def test_isa_in_skip_list(self):
        """ISA should be in _skipTypeCheck (YAML deserializes as list, not tuple)."""
        assert "ISA" in _skipTypeCheck

    def test_skipped_param_does_not_collect(self):
        """A param in _skipTypeCheck should not trigger collection."""
        # ISA expects tuple/SemanticVersion but YAML gives list
        state = {"ISA": [9, 0, 10]}
        validateParameterTypes(state)
        assert len(_typeMismatchCollector) == 0


class TestResetTypeMismatchCollector:
    """Tests for resetTypeMismatchCollector()."""

    def test_reset_clears_collector(self):
        """After reset, the collector should be empty."""
        resetTypeMismatchCollector()
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        assert len(_typeMismatchCollector) > 0
        resetTypeMismatchCollector()
        assert len(_typeMismatchCollector) == 0

    def test_reset_on_empty_is_safe(self):
        """Calling reset when already empty should not raise."""
        resetTypeMismatchCollector()
        resetTypeMismatchCollector()
        assert len(_typeMismatchCollector) == 0

    def test_reset_allows_fresh_collection(self):
        """After reset, new mismatches start from count=0."""
        resetTypeMismatchCollector()
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        resetTypeMismatchCollector()
        validateParameterTypes({"UseCustomMainLoopSchedule": True})
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert _typeMismatchCollector[key]["count"] == 1
        assert _typeMismatchCollector[key]["values"] == {"True"}


class TestPrintTypeMismatchSummary:
    """Tests for printTypeMismatchSummary()."""

    def setup_method(self):
        resetTypeMismatchCollector()

    def test_returns_zero_when_clean(self):
        """No mismatches -> returns 0 and prints nothing."""
        result = printTypeMismatchSummary()
        assert result == 0

    def test_returns_total_count(self):
        """Should return the total number of individual mismatches."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml")
        validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml")
        validateParameterTypes({"BufferLoad": 0}, srcFile="c.yaml")
        result = printTypeMismatchSummary()
        assert result == 3

    def test_outputs_warning_to_stdout(self, capsys):
        """Summary should be printed to stdout so it appears in build logs."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="test.yaml")
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert captured.err == ""
        assert "WARNING" in captured.out

    def test_output_contains_param_name(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "UseCustomMainLoopSchedule" in captured.out

    def test_output_contains_actual_type(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "bool" in captured.out

    def test_output_contains_expected_type(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "int" in captured.out

    def test_output_contains_solution_count(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml")
        validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml")
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "2 solutions" in captured.out

    def test_output_contains_file_count(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml")
        validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml")
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "2 files" in captured.out

    def test_output_contains_fix_message(self, capsys):
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "Fix these to prevent future build failures" in captured.out

    def test_no_output_when_clean_no_files(self, capsys):
        """When there are no mismatches and no files, nothing should be printed."""
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_clean_message_when_no_mismatches(self, capsys):
        """When files were checked but no mismatches found, print confirmation."""
        printTypeMismatchSummary(numFiles=42)
        captured = capsys.readouterr()
        assert "42" in captured.out
        assert "no type mismatches found" in captured.out

    def test_multiple_param_types_in_output(self, capsys):
        """Multiple different mismatched params should all appear in output."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        validateParameterTypes({"BufferLoad": 0})
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "UseCustomMainLoopSchedule" in captured.out
        assert "BufferLoad" in captured.out
