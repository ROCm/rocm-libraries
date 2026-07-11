# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import pytest
import hipblaslt
c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")
pytestmark = pytest.mark.gpu


@requires_gpu
def test_layout_create():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    assert layout.ptr != 0


@requires_gpu
def test_layout_set_batch_count():
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    layout.set_attribute(c.MatrixLayoutAttr.BATCH_COUNT, 2)  # no raise == pass


@requires_gpu
def test_layout_set_batch_stride():
    # STRIDED_BATCH_OFFSET requires int64_t; use set_attribute_i64 to avoid
    # HIPBLAS_STATUS_INVALID_VALUE from a 4-byte size mismatch.
    layout = c.MatrixLayout(c.DataType.R_32F, 4, 8, 4)
    layout.set_attribute(c.MatrixLayoutAttr.BATCH_COUNT, 2)
    layout.set_attribute_i64(c.MatrixLayoutAttr.STRIDED_BATCH_OFFSET, 4 * 8)  # no raise == pass
