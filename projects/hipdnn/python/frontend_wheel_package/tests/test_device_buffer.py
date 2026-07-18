# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""GPU tests for DeviceBuffer allocation and data transfer."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


@pytest.mark.gpu
class TestDeviceBuffer:
    """Tests for DeviceBuffer creation and host-device data transfer."""

    def test_buffer_creation(self):
        """DeviceBuffer can be created with a given byte size."""
        buf = hipdnn.DeviceBuffer(1024)
        assert buf is not None

    def test_buffer_ptr_nonzero(self):
        """DeviceBuffer.ptr() returns a non-zero device pointer."""
        buf = hipdnn.DeviceBuffer(256)
        assert buf.ptr() != 0

    def test_buffer_size(self):
        """DeviceBuffer.size() returns the requested byte count."""
        buf = hipdnn.DeviceBuffer(512)
        assert buf.size() == 512

    def test_buffer_host_roundtrip(self):
        """Data copied to device and back matches the original."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = hipdnn.DeviceBuffer(data.nbytes)

        buf.copy_from_host(data.tobytes())
        result_bytes = buf.copy_to_host()
        result = np.frombuffer(result_bytes, dtype=np.float32)

        np.testing.assert_array_equal(result, data)

    def test_buffer_zeros(self):
        """DeviceBuffer.zeros() fills the buffer with zeros."""
        size = 64 * 4  # 64 float32 values
        buf = hipdnn.DeviceBuffer(size)

        # Fill with non-zero first to confirm zeros() works
        data = np.ones(64, dtype=np.float32)
        buf.copy_from_host(data.tobytes())

        buf.zeros()

        result_bytes = buf.copy_to_host()
        result = np.frombuffer(result_bytes, dtype=np.float32)
        np.testing.assert_array_equal(result, np.zeros(64, dtype=np.float32))
