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


@pytest.mark.gpu
class TestDeviceBufferRoundTrip:
    """Host->device->host data round-trip across supported dtypes (require GPU).

    hipdnn.Tensor is a graph descriptor and holds no element data; the only way
    to fill a buffer with values and read them back is through DeviceBuffer. The
    reconstructed numpy array is what supports element indexing (e.g. out[0]).
    """

    @pytest.mark.parametrize(
        "dtype",
        [np.float32, np.float16, np.float64, np.int32, np.uint8, np.int8],
    )
    def test_fill_index_and_verify(self, dtype):
        """Filling a buffer and reading it back preserves every element."""
        data = np.arange(1, 9).astype(dtype)
        buf = hipdnn.DeviceBuffer(data.nbytes)
        buf.copy_from_host(data.tobytes())

        out = np.frombuffer(buf.copy_to_host(), dtype=dtype)

        assert out[0] == data[0]
        assert out[-1] == data[-1]
        np.testing.assert_array_equal(out, data)

    def test_zeros_clears_buffer(self):
        """zeros() overwrites prior contents with zero bytes."""
        data = np.arange(1, 5, dtype=np.float32)
        buf = hipdnn.DeviceBuffer(data.nbytes)
        buf.copy_from_host(data.tobytes())

        buf.zeros()
        out = np.frombuffer(buf.copy_to_host(), dtype=np.float32)

        assert np.all(out == 0)
