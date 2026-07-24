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
"""Error-message and boundary contract for Tensile.Common.ValidParameters.

Locks the observable behaviour that mutation testing showed was otherwise
unpinned: the type-mismatch keypath (prefix, single-value bracketing), the
src-file prefix propagation, the mismatched-value rendering, the invalid-name
listing, the >32-combos boundary, the empty-list handling in the derived
expected-types map, and the SpaceFillingAlgo orderId upper bound.
"""

import pytest

from Tensile.Common.ValidParameters import (
    _expectedParamTypes,
    _getExpectedTypes,
    _skipTypeCheck,
    checkParametersAreValid,
    checkSpaceFillAlgoIsValid,
    validParameters,
)
from Tensile.Common.GlobalParameters import defaultInternalSupportParams
from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.Common.ValidParameters import validateInternalSupportParams


def _intParam():
    """A real registry param whose only expected type is int and that is not skipped."""
    for name, types in _expectedParamTypes.items():
        if types == {int} and name not in _skipTypeCheck:
            return name
    pytest.skip("no int-typed unskipped parameter in registry")


def _intISPKey():
    """An InternalSupportParams key whose default is an int."""
    for key, default in defaultInternalSupportParams.items():
        if type(default) is int:
            return key
    pytest.skip("no int-typed default InternalSupportParams key")


class TestGetExpectedTypesEmptyList:
    def test_empty_list_value_is_not_registered(self):
        result = _getExpectedTypes({"Foo": [], "Bar": [1, 2]})
        assert "Foo" not in result
        assert result["Bar"] == {int}


class TestSpaceFillAlgoUpperBound:
    def test_orderid_five_is_accepted(self):
        checkSpaceFillAlgoIsValid("SpaceFillingAlgo", [5])

    def test_orderid_above_bound_rejected(self):
        with pytest.raises(Exception, match="OrderID out of range"):
            checkSpaceFillAlgoIsValid("SpaceFillingAlgo", [6])


class TestCheckParametersKeyPath:
    def test_default_prefix_has_no_injected_marker(self):
        name = _intParam()
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid((name, ["x"]), {name: -1})
        msg = str(exc.value)
        assert "XXXX" not in msg
        assert name in msg

    def test_single_value_keypath_has_no_bracket(self):
        name = _intParam()
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid((name, ["x"]), {name: -1})
        assert "[0]" not in str(exc.value)

    def test_srcfile_prefix_is_propagated(self):
        name = _intParam()
        with pytest.raises(ConfigTypeError) as exc:
            checkParametersAreValid((name, ["x"]), {name: -1}, srcFile="probe.yaml")
        assert "probe.yaml" in str(exc.value)


class TestCheckParametersInvalidName:
    def test_invalid_name_lists_valid_parameters(self):
        with pytest.raises(Exception) as exc:
            checkParametersAreValid(("NotARealParameterName", [1]), validParameters)
        msg = str(exc.value)
        assert "Valid parameters are None" not in msg
        assert "PrefetchGlobalRead" in msg


class TestCheckParametersCombosBoundary:
    def test_thirtytwo_values_omits_truncation_note(self):
        vp = {"X": list(range(32))}
        with pytest.raises(Exception) as exc:
            checkParametersAreValid(("X", [999]), vp)
        assert "only first 32 combos" not in str(exc.value)

    def test_thirtythree_values_includes_truncation_note(self):
        vp = {"X": list(range(33))}
        with pytest.raises(Exception) as exc:
            checkParametersAreValid(("X", [999]), vp)
        assert "only first 32 combos" in str(exc.value)


class TestGetExpectedTypesSentinelInvariant:
    """Guards the equivalence of retargeting the ``-1`` skip sentinel.

    ``_getExpectedTypes`` only registers list-valued entries (the
    ``isinstance(list) and len>0`` guard). The ``if allowedValues == -1:
    continue`` fast-path is therefore observationally inert *as long as*
    the only non-list scalar in the registry is ``-1`` itself: no list is
    ever ``== -1`` (or ``+1`` / ``-2``), and any scalar the sentinel could
    match is dropped by the isinstance guard anyway. This test pins that
    invariant so that if a future registry entry introduces a confounding
    scalar the standing equivalence proof for those mutants is revisited.
    """

    def test_only_non_list_scalar_in_registry_is_minus_one(self):
        scalars = {v for v in validParameters.values() if not isinstance(v, list)}
        assert scalars == {-1}


class TestValidateInternalSupportParamsMessage:
    def test_default_srcfile_has_no_injected_marker(self):
        key = _intISPKey()
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({key: "notanint"})
        assert "XXXX" not in str(exc.value)

    def test_srcfile_prefix_is_propagated(self):
        key = _intISPKey()
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({key: "notanint"}, srcFile="probe.yaml")
        assert "probe.yaml" in str(exc.value)

    def test_mismatched_value_is_rendered(self):
        key = _intISPKey()
        with pytest.raises(ConfigTypeError) as exc:
            validateInternalSupportParams({key: "SENTINELVALUE"})
        assert "SENTINELVALUE" in str(exc.value)
