# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import hipblaslt


def test_error_type_exists():
    assert issubclass(hipblaslt._core.HipblasLtError, Exception)


def test_raise_status_helper():
    # _raise_test_status(int) is a debug hook that maps a status code to a raise.
    import pytest
    with pytest.raises(hipblaslt._core.HipblasLtError) as ei:
        hipblaslt._core._raise_test_status(3)  # 3 = HIPBLAS_STATUS_INVALID_VALUE
    assert "INVALID_VALUE" in str(ei.value)
