# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted tests for ``validateInternalSupportParams``."""

import pytest

from Tensile.Common.GlobalParameters import defaultInternalSupportParams
from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.Common.ValidParameters import validateInternalSupportParams

pytestmark = pytest.mark.unit


def test_validate_internal_support_params_empty_dict_returns_none():
    assert validateInternalSupportParams({}) is None


def test_validate_internal_support_params_accepts_full_default_table():
    assert validateInternalSupportParams(dict(defaultInternalSupportParams)) is None


def test_validate_internal_support_params_unknown_key_reports_roster():
    with pytest.raises(ConfigTypeError) as excinfo:
        validateInternalSupportParams({"NoSuchKey": 0})

    assert str(excinfo.value) == (
        "InternalSupportParams.NoSuchKey: unknown key. "
        "Valid keys are ['KernArgsVersion', 'SupportCustomStaggerU', "
        "'SupportCustomWGM', 'SupportUserGSU', 'UseSFC', 'UseUniversalArgs']."
    )


def test_validate_internal_support_params_default_type_mismatch_message():
    with pytest.raises(ConfigTypeError) as excinfo:
        validateInternalSupportParams({"KernArgsVersion": "two"})

    assert (
        str(excinfo.value)
        == "InternalSupportParams.KernArgsVersion = 'two' (str); expected int"
    )


def test_validate_internal_support_params_bool_does_not_accept_int():
    with pytest.raises(ConfigTypeError) as excinfo:
        validateInternalSupportParams({"SupportUserGSU": 1})

    assert str(excinfo.value) == (
        "InternalSupportParams.SupportUserGSU = 1 (int); expected bool"
    )


def test_validate_internal_support_params_custom_source_and_prefix_message():
    with pytest.raises(ConfigTypeError) as excinfo:
        validateInternalSupportParams(
            {"KernArgsVersion": "two"},
            srcFile="cfg.yaml",
            keyPathPrefix="BenchmarkProblems[0][1].InternalSupportParams",
        )

    assert str(excinfo.value) == (
        "cfg.yaml: BenchmarkProblems[0][1].InternalSupportParams.KernArgsVersion "
        "= 'two' (str); expected int"
    )
