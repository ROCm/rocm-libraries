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
    mergeMismatchRecords,
    _getExpectedTypes,
    _expectedParamTypes,
    _skipTypeCheck,
    _typeMismatchCollector,
    resetTypeMismatchCollector,
    printTypeMismatchSummary,
)
from Tensile.SolutionStructs.Problem import (
    validateProblemTypeParameterTypes,
    _expectedProblemTypeParamTypes,
    _defaultProblemType,
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
    """Tests for the validateParameterTypes function (pure: returns records, no side effects)."""

    # --- Passing cases (empty return) ---

    def test_int_param_with_int_value_passes(self):
        """int value for an int param should return no records."""
        assert validateParameterTypes({"UseCustomMainLoopSchedule": 0}) == []

    def test_int_param_with_negative_int_passes(self):
        assert validateParameterTypes({"UseCustomMainLoopSchedule": -1}) == []

    def test_bool_param_with_bool_value_passes(self):
        """bool value for a bool param should return no records."""
        assert validateParameterTypes({"BufferLoad": True}) == []

    def test_bool_param_with_false_passes(self):
        assert validateParameterTypes({"BufferLoad": False}) == []

    def test_unknown_param_ignored(self):
        """Parameters not in validParameters should be silently skipped."""
        assert validateParameterTypes({"SomeUnknownParameter": "whatever"}) == []

    def test_empty_state_passes(self):
        assert validateParameterTypes({}) == []

    def test_multiple_valid_params_pass(self):
        state = {
            "UseCustomMainLoopSchedule": 0,
            "BufferLoad": True,
            "BufferStore": False,
            "PrefetchGlobalRead": 1,
        }
        assert validateParameterTypes(state) == []

    # --- Mismatch cases: return value content ---

    def test_bool_where_int_expected_returns_record(self):
        """bool(False) for an int param should return one record, not raise."""
        records = validateParameterTypes({"UseCustomMainLoopSchedule": False})
        assert len(records) == 1
        collectorKey, valueRepr, _ = records[0]
        assert collectorKey == ("UseCustomMainLoopSchedule", "bool", "int")
        assert valueRepr == "False"

    def test_bool_true_where_int_expected_returns_record(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": True})
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey == ("UseCustomMainLoopSchedule", "bool", "int")

    def test_int_where_bool_expected_returns_record(self):
        """int(0) for a bool param like BufferLoad should return a record."""
        records = validateParameterTypes({"BufferLoad": 0})
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey == ("BufferLoad", "int", "bool")

    def test_int_one_where_bool_expected_returns_record(self):
        records = validateParameterTypes({"BufferLoad": 1})
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey == ("BufferLoad", "int", "bool")

    def test_string_where_int_expected_returns_record(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": "0"})
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey == ("UseCustomMainLoopSchedule", "str", "int")

    def test_float_where_int_expected_returns_record(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": 0.0})
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey == ("UseCustomMainLoopSchedule", "float", "int")

    # --- Record field content ---

    def test_record_carries_param_name(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": False})
        collectorKey, _, _ = records[0]
        assert collectorKey[0] == "UseCustomMainLoopSchedule"

    def test_record_carries_value_repr(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": False})
        _, valueRepr, _ = records[0]
        assert valueRepr == "False"

    def test_record_carries_actual_type(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": False})
        collectorKey, _, _ = records[0]
        assert collectorKey[1] == "bool"

    def test_record_carries_expected_type(self):
        records = validateParameterTypes({"UseCustomMainLoopSchedule": False})
        collectorKey, _, _ = records[0]
        assert collectorKey[2] == "int"

    def test_record_carries_source_file(self):
        records = validateParameterTypes(
            {"UseCustomMainLoopSchedule": False}, srcFile="my_file.yaml"
        )
        _, _, srcFile = records[0]
        assert srcFile == "my_file.yaml"

    def test_empty_srcfile_recorded_as_empty_string(self):
        records = validateParameterTypes(
            {"UseCustomMainLoopSchedule": False}, srcFile=""
        )
        _, _, srcFile = records[0]
        assert srcFile == ""

    # --- Multiple mismatches ---

    def test_multiple_mismatches_all_returned(self):
        """All mismatches in a single state dict should be in the returned list."""
        state = {
            "PrefetchGlobalRead": 1,           # valid int
            "UseCustomMainLoopSchedule": False, # BAD: bool for int
            "BufferLoad": True,                 # valid bool
        }
        records = validateParameterTypes(state)
        assert len(records) == 1
        collectorKey, _, _ = records[0]
        assert collectorKey[0] == "UseCustomMainLoopSchedule"

    def test_different_params_return_separate_records(self):
        """Mismatches on different params each appear as a record."""
        state = {"UseCustomMainLoopSchedule": False, "BufferLoad": 0}
        records = validateParameterTypes(state)
        collectorKeys = {rec[0] for rec in records}
        assert ("UseCustomMainLoopSchedule", "bool", "int") in collectorKeys
        assert ("BufferLoad", "int", "bool") in collectorKeys
        assert len(records) == 2

    def test_does_not_mutate_collector(self):
        """Pure function: the module-level collector must not be touched."""
        resetTypeMismatchCollector()
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        assert len(_typeMismatchCollector) == 0

    # --- Skip list ---

    def test_isa_in_skip_list(self):
        """ISA should be in _skipTypeCheck (YAML deserializes as list, not tuple)."""
        assert "ISA" in _skipTypeCheck

    def test_skipped_param_returns_no_record(self):
        """A param in _skipTypeCheck should not appear in the returned records."""
        assert validateParameterTypes({"ISA": [9, 0, 10]}) == []


class TestValidateParameterTypesReturnContract:
    """Behavioural tests for the return-records contract (M1, folds in n2).

    validateParameterTypes returns a list of mismatch records and never
    mutates module state; Solution.__init__ folds them via
    mergeMismatchRecords. These tests pin that contract directly so they
    survive renames and exercise the real code (not source-text matching).
    """

    def setup_method(self):
        resetTypeMismatchCollector()

    def test_clean_state_returns_empty_list(self):
        """A well-typed state returns [] and touches no module state."""
        records = validateParameterTypes({"UseCustomMainLoopSchedule": 0})
        assert records == []
        assert len(_typeMismatchCollector) == 0

    def test_empty_state_returns_empty_list(self):
        assert validateParameterTypes({}) == []

    def test_single_mismatch_returned_as_record(self):
        """A bool where int is expected returns exactly that one record."""
        records = validateParameterTypes(
            {"UseCustomMainLoopSchedule": False}, srcFile="x.yaml"
        )
        assert records == [
            (("UseCustomMainLoopSchedule", "bool", "int"), "False", "x.yaml")
        ]

    def test_returns_without_mutating_collector(self):
        """The pure function must not populate the module-level collector."""
        validateParameterTypes({"UseCustomMainLoopSchedule": False})
        assert len(_typeMismatchCollector) == 0

    def test_multiple_mismatches_each_returned(self):
        """Every bad-typed key yields its own record in the returned list."""
        state = {"UseCustomMainLoopSchedule": False, "BufferLoad": 0}
        records = validateParameterTypes(state)
        collectorKeys = {rec[0] for rec in records}
        assert ("UseCustomMainLoopSchedule", "bool", "int") in collectorKeys
        assert ("BufferLoad", "int", "bool") in collectorKeys
        assert len(records) == 2

    def test_merge_records_populates_collector(self):
        """mergeMismatchRecords folds returned records into the collector."""
        records = validateParameterTypes(
            {"UseCustomMainLoopSchedule": False}, srcFile="x.yaml"
        )
        mergeMismatchRecords(records)
        key = ("UseCustomMainLoopSchedule", "bool", "int")
        assert _typeMismatchCollector[key]["count"] == 1
        assert "False" in _typeMismatchCollector[key]["values"]
        assert "x.yaml" in _typeMismatchCollector[key]["files"]

    def test_merge_empty_records_is_noop(self):
        """Folding an empty record list leaves the collector untouched."""
        mergeMismatchRecords(validateParameterTypes({"BufferLoad": True}))
        assert len(_typeMismatchCollector) == 0


class TestResetTypeMismatchCollector:
    """Tests for resetTypeMismatchCollector()."""

    def test_reset_clears_collector(self):
        """After reset, the collector should be empty."""
        resetTypeMismatchCollector()
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
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
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        resetTypeMismatchCollector()
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": True}))
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
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml"))
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml"))
        mergeMismatchRecords(validateParameterTypes({"BufferLoad": 0}, srcFile="c.yaml"))
        result = printTypeMismatchSummary()
        assert result == 3

    def test_outputs_warning_to_stdout(self, capsys):
        """Summary should be printed to stdout so it appears in build logs."""
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="test.yaml"))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert captured.err == ""
        assert "WARNING" in captured.out

    def test_output_contains_param_name(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "UseCustomMainLoopSchedule" in captured.out

    def test_output_contains_actual_type(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "bool" in captured.out

    def test_output_contains_expected_type(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "int" in captured.out

    def test_output_contains_solution_count(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml"))
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml"))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "2 solutions" in captured.out

    def test_output_contains_file_count(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}, srcFile="a.yaml"))
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": True}, srcFile="b.yaml"))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "2 files" in captured.out

    def test_output_contains_fix_message(self, capsys):
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
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
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        mergeMismatchRecords(validateParameterTypes({"BufferLoad": 0}))
        printTypeMismatchSummary()
        captured = capsys.readouterr()
        assert "UseCustomMainLoopSchedule" in captured.out
        assert "BufferLoad" in captured.out


class TestValidateProblemTypeParameterTypes:
    """Tests for the validateProblemTypeParameterTypes function.

    Step 4 promoted the default behaviour to raise on mismatch. These
    tests exercise the library-logic / collector mode by passing
    ``raiseOnMismatch=False``. See TestValidateProblemTypeRaiseMode for
    the strict-mode coverage.
    """

    def setup_method(self):
        """Reset the collector before each test."""
        resetTypeMismatchCollector()

    # --- Passing cases (no mismatches collected) ---

    def test_bool_param_with_bool_value_passes(self):
        """TransposeA: False (bool) should not collect a mismatch."""
        state = {"TransposeA": False}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_bool_param_with_true_passes(self):
        """TransposeA: True (bool) should not collect a mismatch."""
        state = {"TransposeA": True}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_usebeta_bool_param_with_bool_passes(self):
        """UseBeta: True (bool) should not collect a mismatch."""
        state = {"UseBeta": True}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_int_param_with_int_value_passes(self):
        """UseBias: 1 (int) should not collect a mismatch."""
        state = {"UseBias": 1}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_mxblock_int_param_passes(self):
        """MXBlockA: 16 (int) should not collect a mismatch."""
        state = {"MXBlockA": 16}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_unknown_param_ignored(self):
        """Parameters not in _defaultProblemType should be silently skipped."""
        state = {"UnknownProblemTypeParam": "anything"}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    def test_empty_state_passes(self):
        """Empty state should not collect any mismatches."""
        validateProblemTypeParameterTypes({}, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 0

    # --- Failing cases (mismatches collected; raise disabled) ---

    def test_bool_param_with_int_value_collects_mismatch(self):
        """TransposeA: 0 (int) should collect a bool-vs-int mismatch."""
        state = {"TransposeA": 0}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 1
        key = ("TransposeA", "int", "bool")
        assert key in _typeMismatchCollector

    def test_bool_param_with_int_one_collects_mismatch(self):
        """UseBeta: 1 (int) should collect a bool-vs-int mismatch."""
        state = {"UseBeta": 1}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 1
        key = ("UseBeta", "int", "bool")
        assert key in _typeMismatchCollector

    def test_int_param_with_bool_collects_mismatch(self):
        """UseBias: True (bool) should collect an int-vs-bool mismatch."""
        state = {"UseBias": True}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 1
        key = ("UseBias", "bool", "int")
        assert key in _typeMismatchCollector

    def test_mxblock_with_bool_collects_mismatch(self):
        """MXBlockA: False (bool) should collect an int-vs-bool mismatch."""
        state = {"MXBlockA": False}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 1
        key = ("MXBlockA", "bool", "int")
        assert key in _typeMismatchCollector

    def test_collector_tracks_srcfile(self):
        """srcFile should be recorded in the collector entry."""
        state = {"TransposeA": 0}
        validateProblemTypeParameterTypes(state, srcFile="problem_config.yaml",
                                          raiseOnMismatch=False)
        key = ("TransposeA", "int", "bool")
        assert "problem_config.yaml" in _typeMismatchCollector[key]["files"]

    def test_collector_tracks_values(self):
        """Different values should be collected in the values set."""
        validateProblemTypeParameterTypes({"TransposeA": 0}, raiseOnMismatch=False)
        validateProblemTypeParameterTypes({"TransposeA": 1}, raiseOnMismatch=False)
        key = ("TransposeA", "int", "bool")
        assert "0" in _typeMismatchCollector[key]["values"]
        assert "1" in _typeMismatchCollector[key]["values"]

    def test_multiple_params_collect_separately(self):
        """Different parameters should create separate collector entries."""
        state = {"TransposeA": 0, "UseBeta": 1, "UseBias": True}
        validateProblemTypeParameterTypes(state, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 3

    def test_accumulates_with_solution_params(self):
        """ProblemType and Solution mismatches should accumulate together."""
        mergeMismatchRecords(validateParameterTypes({"UseCustomMainLoopSchedule": False}))
        validateProblemTypeParameterTypes({"TransposeA": 0}, raiseOnMismatch=False)
        assert len(_typeMismatchCollector) == 2

    def test_precomputed_problemtype_types(self):
        """_expectedProblemTypeParamTypes should be precomputed correctly."""
        assert "TransposeA" in _expectedProblemTypeParamTypes
        assert _expectedProblemTypeParamTypes["TransposeA"] == {bool}
        assert "UseBeta" in _expectedProblemTypeParamTypes
        assert _expectedProblemTypeParamTypes["UseBeta"] == {bool}
        assert "UseBias" in _expectedProblemTypeParamTypes
        assert _expectedProblemTypeParamTypes["UseBias"] == {int}
        assert "MXBlockA" in _expectedProblemTypeParamTypes
        assert _expectedProblemTypeParamTypes["MXBlockA"] == {int}


class TestValidateProblemTypeRaiseMode:
    """Step 4: default raiseOnMismatch=True path."""

    pytestmark = pytest.mark.usefixtures("_force_strict_gate_on")

    def setup_method(self):
        resetTypeMismatchCollector()

    def test_usebeta_int_raises(self):
        """UseBeta: 1 (int where bool expected) raises ConfigTypeError."""
        from Tensile.Common.TypeValidationErrors import ConfigTypeError
        with pytest.raises(ConfigTypeError) as exc:
            validateProblemTypeParameterTypes({"UseBeta": 1})
        assert "UseBeta" in str(exc.value)
        assert "expected bool" in str(exc.value)

    def test_usebeta_bool_passes(self):
        """UseBeta: True passes."""
        validateProblemTypeParameterTypes({"UseBeta": True})

    def test_raises_on_first_bad_key(self):
        """Raises on the first mistyped key encountered."""
        from Tensile.Common.TypeValidationErrors import ConfigTypeError
        with pytest.raises(ConfigTypeError) as exc:
            validateProblemTypeParameterTypes({"TransposeA": 0, "UseBeta": 1})
        assert "TransposeA" in str(exc.value)

    def test_datatype_skip_set_not_touched(self):
        """DataType* string values are skipped (post-processed to DataType objects)."""
        # Passing a string for a DataType* key would be rejected if not in
        # the skip set, but DataType* aren't in _expectedProblemTypeParamTypes
        # because they aren't in _defaultProblemType either; in either case
        # this must not raise.
        validateProblemTypeParameterTypes({"DataType": "single"})

    def test_keypath_prefix_in_message(self):
        from Tensile.Common.TypeValidationErrors import ConfigTypeError
        with pytest.raises(ConfigTypeError) as exc:
            validateProblemTypeParameterTypes(
                {"UseBeta": 1}, keyPathPrefix="BenchmarkProblems[0][0].ProblemType",
            )
        assert "BenchmarkProblems[0][0].ProblemType.UseBeta" in str(exc.value)


class TestWorkerPassthroughBackstop:
    """Step 4 (B1): worker re-raises ConfigTypeError past the broad except."""

    def _make_minimal_fixtures(self):
        """Build the minimal arguments needed to enter _generate_single_solution.

        The worker accesses MatrixInstruction/WorkGroup/WavefrontSize on
        the merged dict and calls validateMIParameters before Solution.
        Patching validateMIParameters to return True is enough to drive
        execution to the Solution() call site.
        """
        class _Arch:
            archCaps = {"HasWave32": True}
        class _Cfg:
            splitGSU = False
            printSolutionRejectionReason = False
            printIndexAssignmentInfo = False
        class _Type:
            state = {"DataType": "single"}
        perm = {
            "MatrixInstruction": [],
            "WorkGroup": [16, 16, 1],
            "WavefrontSize": 32,
        }
        isaInfoMap = {(9, 0, 10): _Arch()}
        return perm, _Type(), _Cfg(), isaInfoMap

    def test_worker_re_raises_config_type_error(self):
        """A ConfigTypeError raised inside the worker propagates to the caller.

        Patches validateMIParameters and Solution so we don't need to
        stand up a full Solution-construction stack. The point is the
        typed-except in _generate_single_solution.
        """
        from Tensile.Common.TypeValidationErrors import ConfigTypeError
        from Tensile import BenchmarkProblems

        perm, ptype, cfg, isaInfoMap = self._make_minimal_fixtures()

        orig_solution = BenchmarkProblems.Solution
        orig_vmi = BenchmarkProblems.validateMIParameters
        BenchmarkProblems.validateMIParameters = lambda *a, **kw: True
        BenchmarkProblems.Solution = lambda *a, **kw: (_ for _ in ()).throw(
            ConfigTypeError("typed boom"))
        try:
            with pytest.raises(ConfigTypeError) as exc:
                BenchmarkProblems._generate_single_solution(
                    perm=perm,
                    problemType=ptype,
                    constantParams={},
                    assembler=None,
                    debugConfig=cfg,
                    isaInfoMap=isaInfoMap,
                )
            assert "typed boom" in str(exc.value)
        finally:
            BenchmarkProblems.Solution = orig_solution
            BenchmarkProblems.validateMIParameters = orig_vmi

    def test_worker_swallows_generic_exception(self, capsys):
        """A non-ConfigTypeError still gets swallowed (legacy behaviour)."""
        from Tensile import BenchmarkProblems

        perm, ptype, cfg, isaInfoMap = self._make_minimal_fixtures()

        orig_solution = BenchmarkProblems.Solution
        orig_vmi = BenchmarkProblems.validateMIParameters
        BenchmarkProblems.validateMIParameters = lambda *a, **kw: True
        BenchmarkProblems.Solution = lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("generic boom"))
        try:
            result = BenchmarkProblems._generate_single_solution(
                perm=perm,
                problemType=ptype,
                constantParams={},
                assembler=None,
                debugConfig=cfg,
                isaInfoMap=isaInfoMap,
            )
            assert result is None
            captured = capsys.readouterr()
            assert "Error processing permutation" in captured.out
        finally:
            BenchmarkProblems.Solution = orig_solution
            BenchmarkProblems.validateMIParameters = orig_vmi
