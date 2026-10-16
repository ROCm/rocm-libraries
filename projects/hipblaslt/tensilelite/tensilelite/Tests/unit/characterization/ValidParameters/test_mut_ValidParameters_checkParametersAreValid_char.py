# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted characterization tests for
``Tensile.Common.ValidParameters.checkParametersAreValid``.

These pin the validator's current boundary behavior and type-mismatch reporting:
unknown-name diagnostics include the global valid-parameter roster, value-list
messages switch to the truncated form only above 32 entries, and exact type
mismatches report the current source/key/value/expected-type fields.
"""

import pytest

import Tensile.Common.ValidParameters as VP
from Tensile.Common.TypeValidationErrors import ConfigTypeError

pytestmark = pytest.mark.unit


def test_check_params_unknown_name_lists_valid_parameter_roster():
    """Unknown-parameter errors include the real valid-parameter roster."""
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("NotARealParameter", [1]), {"P": [1, 2, 3]})

    msg = str(excinfo.value)
    assert msg.startswith("Invalid parameter name: NotARealParameter\n")
    assert "Valid parameters are [" in msg
    assert "'BufferLoad'" in msg
    assert msg.endswith(".")


def test_check_params_value_list_len_32_uses_full_message():
    """A 32-entry valid-values list is not reported as truncated."""
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("P", [999]), {"P": list(range(32))})

    assert str(excinfo.value) == (
        "Invalid parameter value: P = 999\n"
        "Valid values for P are "
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
        "16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, "
        "30, 31]."
    )


def test_check_params_value_list_len_33_uses_truncated_message():
    """A 33-entry valid-values list reports only the first 32 entries."""
    with pytest.raises(Exception) as excinfo:
        VP.checkParametersAreValid(("P", [999]), {"P": list(range(33))})

    assert str(excinfo.value) == (
        "Invalid parameter value: P = 999\n"
        "Valid values for P are "
        "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, "
        "16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, "
        "30, 31] (only first 32 combos printed)\n"
        "Refer to Common.py for more info."
    )


def test_check_params_type_mismatch_default_location_message():
    """Default error-location arguments produce an unprefixed key path."""
    with pytest.raises(ConfigTypeError) as excinfo:
        VP.checkParametersAreValid(("BufferLoad", [1]), VP.validParameters)

    assert str(excinfo.value) == "BufferLoad = 1 (int); expected bool"


def test_check_params_type_mismatch_multi_value_location_message():
    """Multi-value type mismatches include source, prefix, and value index."""
    with pytest.raises(ConfigTypeError) as excinfo:
        VP.checkParametersAreValid(
            ("BufferLoad", [True, 1]),
            VP.validParameters,
            keyPathPrefix="Root",
            srcFile="cfg.yaml",
        )

    assert str(excinfo.value) == "cfg.yaml: Root.BufferLoad[1] = 1 (int); expected bool"
