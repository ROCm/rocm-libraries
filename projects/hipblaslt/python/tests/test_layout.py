# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")


@requires_gpu
def test_layout_create():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    assert layout.ptr != 0


@requires_gpu
def test_layout_set_batch_count():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    layout.set_attribute(c.MatrixLayoutAttr.BATCH_COUNT, 2)  # no raise == pass
