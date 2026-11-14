# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from tensilelite import _rocm


pytestmark = pytest.mark.unit


def test_python_sdk_runtime_is_not_gateable():
    assert not hasattr(_rocm, "_ENABLE_PYTHON_ROCM_RUNTIME")
