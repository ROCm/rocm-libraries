# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for detecting a custom-kernel solution config."""

import pytest

from Tensile.Common.ValidParameters import validParameters
from Tensile.CustomKernels import isCustomKernelConfig
from Tensile.SolutionStructs.Naming import getKernelFileBase

pytestmark = pytest.mark.unit


# "CustomKernel" is seeded with the -1 placeholder for an unset parameter, and a
# partially populated solution can carry None, a stray scalar, or a dict with no
# name. None of those are subscriptable, so detection has to type-check first.
MALFORMED = [-1, None, "a_kernel_name", {}]


def test_unset_custom_kernel_placeholder_is_not_a_dict():
    assert not isinstance(validParameters["CustomKernel"], dict)


@pytest.mark.parametrize("value", MALFORMED)
def test_malformed_custom_kernel_is_not_a_custom_kernel_config(value):
    assert isCustomKernelConfig({"CustomKernel": value}) is False


def test_malformed_custom_kernel_still_honors_the_legacy_name():
    assert isCustomKernelConfig({"CustomKernel": -1, "CustomKernelName": "legacy"}) is True


def test_well_formed_custom_kernel_is_detected():
    assert isCustomKernelConfig({"CustomKernel": {"name": "a_kernel_name"}}) is True


def test_generated_custom_kernel_is_not_treated_as_a_custom_kernel():
    assert (
        isCustomKernelConfig({"CustomKernel": {"name": "a_kernel_name", "generated": True}})
        is False
    )


def test_kernel_file_base_falls_back_when_custom_kernel_is_malformed():
    assert getKernelFileBase(False, {"CustomKernel": -1, "CustomKernelName": "legacy"}) == "legacy"


def test_generated_custom_kernel_does_not_name_the_output_file():
    kernel = {
        "CustomKernel": {"name": "a_kernel_name", "generated": True},
        "CustomKernelName": "legacy",
    }

    assert getKernelFileBase(False, kernel) == "legacy"
