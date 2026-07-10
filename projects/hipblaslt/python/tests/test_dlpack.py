# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""DLPack import/export tests for DeviceArray.

torch-ROCm is NOT a build-time dependency.  All tests in this file are gated
on ``pytest.importorskip("torch")`` so they skip cleanly when torch is absent.
On machines with torch-ROCm installed the tests exercise the full round-trip.
"""
import numpy as np
import pytest
import hipblaslt

c = hipblaslt._core
requires_gpu = pytest.mark.skipif(not c.hip_available(), reason="no HIP device")

torch = pytest.importorskip("torch")


@requires_gpu
def test_export_to_torch():
    """DeviceArray exported via __dlpack__ is consumable by torch.from_dlpack."""
    a = np.arange(8, dtype=np.float32)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    t = torch.from_dlpack(da)
    assert t.numel() == 8
    np.testing.assert_array_equal(t.cpu().numpy(), a)


@requires_gpu
def test_export_dlpack_device():
    """__dlpack_device__ returns (10, int) — kDLROCM device type."""
    a = np.zeros(4, dtype=np.float32)
    da = c.DeviceArray.from_numpy(a, c.DataType.R_32F)
    device_type, device_id = da.__dlpack_device__()
    assert device_type == 10, "Expected kDLROCM (10)"
    assert isinstance(device_id, int)


@requires_gpu
def test_import_from_torch():
    """DeviceArray.from_dlpack copies a torch-ROCm tensor into device memory."""
    t = torch.arange(8, dtype=torch.float32, device="cuda")
    da = c.DeviceArray.from_dlpack(t)
    np.testing.assert_array_equal(da.to_numpy(), np.arange(8, dtype=np.float32))


@requires_gpu
def test_import_from_torch_2d():
    """from_dlpack preserves shape for 2-D tensors."""
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = torch.from_numpy(data).to("cuda")
    da = c.DeviceArray.from_dlpack(t)
    assert tuple(da.shape) == (3, 4)
    np.testing.assert_array_equal(da.to_numpy(), data)
