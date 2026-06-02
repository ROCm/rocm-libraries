# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for DeviceBuffer edge cases and the get_dtype_size utility."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


class TestGetDtypeSize:
    """get_dtype_size() over supported and unsupported numpy dtypes (no GPU)."""

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (np.float32, 4),
            (np.float16, 2),
            (np.float64, 8),
            (np.int32, 4),
            (np.uint8, 1),
            (np.int8, 1),
        ],
    )
    def test_supported_dtypes(self, dtype, expected):
        """Supported dtypes return their element size in bytes."""
        assert hipdnn.get_dtype_size(np.dtype(dtype)) == expected

    def test_unsupported_dtype_raises(self):
        """An unsupported dtype raises RuntimeError."""
        with pytest.raises(RuntimeError):
            hipdnn.get_dtype_size(np.dtype(np.complex64))


@pytest.mark.gpu
class TestDeviceBufferEdges:
    """DeviceBuffer boundary and mismatch behavior (require GPU)."""

    def test_zero_size_buffer(self):
        """A zero-byte DeviceBuffer reports size 0."""
        buf = hipdnn.DeviceBuffer(0)
        assert buf.size() == 0

    def test_copy_from_host_too_large_raises(self):
        """copy_from_host() with too many bytes raises RuntimeError."""
        buf = hipdnn.DeviceBuffer(16)
        data = np.zeros(8, dtype=np.float32)  # 32 bytes
        with pytest.raises(RuntimeError):
            buf.copy_from_host(data.tobytes())

    def test_copy_from_host_too_small_raises(self):
        """copy_from_host() with too few bytes raises RuntimeError."""
        buf = hipdnn.DeviceBuffer(16)
        data = np.zeros(2, dtype=np.float32)  # 8 bytes
        with pytest.raises(RuntimeError):
            buf.copy_from_host(data.tobytes())
