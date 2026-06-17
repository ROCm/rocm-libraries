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
"""Strict type-check coverage for Tensile.Common.ValidParameters.checkParametersAreValid."""

import pytest

from Tensile.Common.ValidParameters import (
    checkParametersAreValid,
    validParameters,
)
from Tensile.Common.TypeValidationErrors import ConfigTypeError

pytestmark = pytest.mark.usefixtures("_force_strict_gate_on")


class TestBoolIntTrap:
    """Bool vs int must be distinguished both directions (§5)."""

    def test_bool_where_int_expected_raises(self):
        # BoundsCheck/UseCustomMainLoopSchedule etc. expect int.
        # validParameters["UseCustomMainLoopSchedule"] = [-1, 0, 1] -> {int}.
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("UseCustomMainLoopSchedule", [False]), validParameters,
            )
        assert "UseCustomMainLoopSchedule" in str(exc.value)
        assert "bool" in str(exc.value)
        assert "expected int" in str(exc.value)

    def test_int_where_bool_expected_raises(self):
        # BufferLoad -> [False, True] -> {bool}
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("BufferLoad", [1]), validParameters,
            )
        assert "BufferLoad" in str(exc.value)
        assert "int" in str(exc.value)
        assert "expected bool" in str(exc.value)

    def test_bool_true_where_int_expected_raises(self):
        with pytest.raises(ConfigTypeError):
            checkParametersAreValid(
                ("UseCustomMainLoopSchedule", [True]), validParameters,
            )

    def test_valid_int_passes(self):
        # Should not raise.
        checkParametersAreValid(
            ("UseCustomMainLoopSchedule", [0]), validParameters,
        )

    def test_valid_bool_passes(self):
        checkParametersAreValid(
            ("BufferLoad", [False, True]), validParameters,
        )


class TestSentinelInFloatList:
    """validParameters entries with mixed numeric+sentinel are accepted."""

    def test_local_write_per_mfma_minus_one_accepted(self):
        # validParameters["LocalWritePerMfma"] = [i/100 ...] + [-1] -> {float, int}
        checkParametersAreValid(
            ("LocalWritePerMfma", [-1]), validParameters,
        )

    def test_local_write_per_mfma_float_accepted(self):
        checkParametersAreValid(
            ("LocalWritePerMfma", [0.5]), validParameters,
        )


class TestFreeFormListParams:
    """-1 sentinel parameters skip the type check entirely."""

    def test_matrix_instruction_skipped(self):
        # validParameters["MatrixInstruction"] == -1 means free-form. The
        # type map omits it, so any payload type is accepted.
        assert validParameters["MatrixInstruction"] == -1
        checkParametersAreValid(("MatrixInstruction", [[16, 16, 32, 1, 1, 16, 3, 1, 4]]),
                                validParameters)

    @pytest.mark.parametrize("name", [
        "WorkGroup", "ThreadTile",
    ])
    def test_value_list_params_accept_list_values(self, name):
        # These have value-list registrations of list-of-list payloads, so
        # the derived expected-type set is {list}. A list value passes.
        # (The legacy value-membership check requires the value to be in
        # the allowed list; pick the first allowed value to satisfy it.)
        allowed = validParameters[name]
        assert isinstance(allowed, list) and allowed
        checkParametersAreValid((name, [allowed[0]]), validParameters)

    def test_space_filling_algo_accepted(self):
        # SpaceFillingAlgo has its own shape checker; -1 sentinel here
        # means the type check is skipped and the shape checker validates
        # the [orderId] list.
        assert validParameters["SpaceFillingAlgo"] == -1
        checkParametersAreValid(("SpaceFillingAlgo", [[0, 1, 2]]), validParameters)

    def test_sfcwgm_accepted(self):
        assert validParameters["SFCWGM"] == -1
        checkParametersAreValid(("SFCWGM", [[[0, 1], [2, 3]]]), validParameters)


class TestSkipSet:
    """DataType*, MacDataType*, F32XdlMathOp, ISA are excluded from the type check.

    In the BenchmarkCommonParameters / ForkParameters scope these names
    aren't in validParameters at all (they're ProblemType keys), so the
    legacy unknown-name guard would fire before the type check ever
    runs. The skip set still applies on the Solution validator path
    (validateParameterTypes) where post-processed DataType objects are
    present. Here we only verify ISA, which IS registered in
    validParameters (as a value list of IsaVersion tuples).
    """

    def test_isa_skip_set_accepts_list(self):
        # validParameters["ISA"] is a list of IsaVersion tuples. The
        # _skipTypeCheck set lists "ISA" because the YAML form is a list
        # of ints, converted downstream. So any type passes -- but the
        # legacy value-membership check still runs against the allowed
        # list. Use the sentinel (0, 0, 0) which is always present.
        from Tensile.Common.Types import IsaVersion
        checkParametersAreValid(("ISA", [IsaVersion(0, 0, 0)]), validParameters)


class TestKeyPathBuilding:
    """Error messages include the supplied keypath prefix."""

    def test_keypath_in_error_no_prefix(self):
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("BufferLoad", [1]), validParameters,
            )
        # No prefix: keypath is just the name.
        assert "BufferLoad" in str(exc.value)

    def test_keypath_with_prefix(self):
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("BufferLoad", [1]), validParameters,
                keyPathPrefix="BenchmarkProblems[0][1].ForkParameters",
            )
        assert "BenchmarkProblems[0][1].ForkParameters.BufferLoad" in str(exc.value)

    def test_keypath_multi_value_has_index(self):
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("BufferLoad", [False, 1]), validParameters,
                keyPathPrefix="ForkParameters",
            )
        assert "ForkParameters.BufferLoad[1]" in str(exc.value)

    def test_keypath_groups(self):
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid(
                ("BufferLoad", [1]), validParameters,
                keyPathPrefix="ForkParameters.Groups[0][1]",
            )
        assert "ForkParameters.Groups[0][1].BufferLoad" in str(exc.value)


class TestLegacyChecksUnchanged:
    """The legacy unknown-name and invalid-value paths still raise generic Exception."""

    def test_unknown_name_raises(self):
        with pytest.raises(Exception) as exc:
            checkParametersAreValid(("NoSuchParam", [0]), validParameters)
        # Not a ConfigTypeError (legacy path).
        assert not isinstance(exc.value, ConfigTypeError)
        assert "Invalid parameter name" in str(exc.value)

    def test_problem_sizes_skipped(self):
        # ProblemSizes returns early -- any payload is accepted.
        checkParametersAreValid(("ProblemSizes", [[1, 2, 3]]), validParameters)

    def test_internal_support_params_rejected_by_name(self):
        # InternalSupportParams is validated by its own sibling validator
        # (validateInternalSupportParams). Passing it through this
        # function falls through to the unknown-name check.
        with pytest.raises(Exception) as exc:
            checkParametersAreValid(("InternalSupportParams", [{"x": 1}]), validParameters)
        assert "Invalid parameter name" in str(exc.value)
