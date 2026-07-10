# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt


def test_core_version_is_string():
    assert isinstance(hipblaslt._core.__version__, str)
    assert hipblaslt._core.__version__


def test_hip_available_is_bool():
    assert isinstance(hipblaslt._core.hip_available(), bool)
