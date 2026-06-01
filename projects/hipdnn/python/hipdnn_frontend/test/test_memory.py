# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for DeviceBuffer host/device transfers."""

import numpy as np
import pytest

import hipdnn_frontend as hipdnn


@pytest.mark.gpu
class TestDeviceBuffer:
    """Tests for DeviceBuffer allocation and copy round-trips."""

    def test_size_matches_allocation(self):
        """size() reports the allocated byte count."""
        buf = hipdnn.DeviceBuffer(256)

        assert buf.size() == 256

    def test_zeros_fills_buffer(self):
        """zeros() clears device memory to all zeros."""
        data = np.full(64, 7.0, dtype=np.float32)
        buf = hipdnn.DeviceBuffer(data.nbytes)
        buf.copy_from_host(data.tobytes())

        buf.zeros()

        result = np.frombuffer(buf.copy_to_host(), dtype=np.float32)
        assert np.all(result == 0)

    def test_host_device_round_trip(self):
        """copy_from_host()/copy_to_host() preserve data."""
        data = np.random.uniform(0.0, 1.0, 128).astype(np.float32)
        buf = hipdnn.DeviceBuffer(data.nbytes)

        buf.copy_from_host(data.tobytes())
        result = np.frombuffer(buf.copy_to_host(), dtype=np.float32)

        np.testing.assert_array_equal(result, data)
