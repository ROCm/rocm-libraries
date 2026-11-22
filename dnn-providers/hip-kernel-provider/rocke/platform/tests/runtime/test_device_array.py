# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.runtime.device_array``: the typed device array and the one array
description host-side launch checks read.

The first group needs no GPU: ``ArrayInfo`` arithmetic, the refusals a
``DeviceArray`` makes before it allocates, and the torch adapter, which reads
tensor attributes only (torch is not a test dependency, so the adapter is fed
an object with exactly the attributes a torch tensor has). The ``test_gpu_``
tests allocate on the current GPU and skip without one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rocke.numeric.storage import encode
from rocke.runtime.device_array import ArrayInfo, DeviceArray, array_info
from rocke.runtime.torch_interop import torch_array_info
from rocke.tests.gpu import current_device_arch


def _info(shape, strides, element_size=4, data_ptr=1024):
    return ArrayInfo(
        shape=shape,
        strides=strides,
        dtype="fp32",
        element_size=element_size,
        data_ptr=data_ptr,
        device=0,
    )


def test_contiguity_ignores_the_strides_of_unit_axes():
    assert _info((2, 3, 4), (12, 4, 1)).is_contiguous()
    assert _info((2, 1, 4), (4, 99, 1)).is_contiguous()
    assert not _info((2, 3, 4), (24, 8, 1)).is_contiguous()
    assert not _info((3, 4), (1, 3)).is_contiguous()


def test_byte_span_covers_the_holes_of_a_strided_view():
    # A [3, 4] column view of a [3, 10] fp32 buffer reaches row 2, column 3.
    view = _info((3, 4), (10, 1))
    assert view.numel == 12
    assert view.byte_span() == (1024, 1024 + (2 * 10 + 3 + 1) * 4)
    assert _info((0, 4), (4, 1)).byte_span() == (1024, 1024)


def test_array_info_refuses_anything_but_a_device_array_or_tensor():
    with pytest.raises(TypeError, match=r"array_info failed - array=ndarray"):
        array_info(np.zeros(4, np.float32))


class _TorchTensorAttributes:
    """The attributes ``torch_array_info`` reads from a torch tensor."""

    __module__ = "torch"
    device = SimpleNamespace(type="cuda", index=1)
    dtype = "torch.bfloat16"
    shape = (4, 8)

    def stride(self):
        return (16, 1)

    def element_size(self):
        return 2

    def data_ptr(self):
        return 4096


def test_torch_tensor_info_maps_dtype_and_device():
    info = array_info(_TorchTensorAttributes())
    assert info == ArrayInfo(
        shape=(4, 8),
        strides=(16, 1),
        dtype="bf16",
        element_size=2,
        data_ptr=4096,
        device=1,
    )
    assert not info.is_contiguous()


def test_torch_tensor_info_refuses_a_dtype_without_an_element_type():
    tensor = _TorchTensorAttributes()
    tensor.dtype = "torch.complex64"
    with pytest.raises(
        ValueError, match=r"torch_array_info failed - dtype=torch.complex64"
    ):
        torch_array_info(tensor)


def test_raw_bit_hosts_need_an_element_type():
    with pytest.raises(ValueError, match=r"element type failed - host dtype=uint16"):
        DeviceArray(np.zeros(4, np.uint16))


def test_element_type_must_match_the_host_width():
    with pytest.raises(ValueError, match=r"dtype='bf16' is 2 bytes, expected the host"):
        DeviceArray(np.zeros(4, np.float32), "bf16")


def test_view_shape_must_fit_the_host():
    with pytest.raises(
        ValueError, match=r"shape=\(2, 9\), expected one extent per host"
    ):
        DeviceArray(np.zeros((2, 8), np.int32), shape=(2, 9))


# --- on the GPU ----------------------------------------------------------------------------------


def test_gpu_round_trip_and_tensor_methods():
    current_device_arch()
    host = encode(np.arange(12.0).reshape(3, 4), "bf16")
    array = DeviceArray(host, "bf16")
    assert array.dtype == "bf16" and array.shape == (3, 4) and array.ndim == 2
    assert array.stride() == (4, 1) and array.stride(0) == 4
    assert array.element_size() == 2 and array.numel() == 12
    assert array.nbytes == host.nbytes == 24
    assert array.is_contiguous() and array.data_ptr() != 0
    np.testing.assert_array_equal(array.read(), host)
    array.write(host[::-1].copy())
    np.testing.assert_array_equal(array.read(), host[::-1])
    info = array_info(array)
    assert info.dtype == "bf16" and info.device == array.device


def test_gpu_column_view_keeps_the_host_strides():
    current_device_arch()
    table = DeviceArray(np.zeros((3, 10), np.int32), shape=(3, 4))
    assert table.shape == (3, 4) and table.stride(0) == 10
    assert table.nbytes == 3 * 4 * 4
    assert not table.is_contiguous()


def test_gpu_zeros():
    current_device_arch()
    array = DeviceArray.zeros((2, 3), "fp8e4m3")
    assert array.dtype == "fp8e4m3" and array.element_size() == 1
    np.testing.assert_array_equal(array.read(), np.zeros((2, 3), np.uint8))


def test_gpu_write_refuses_another_shape():
    current_device_arch()
    array = DeviceArray(np.zeros(4, np.float32))
    with pytest.raises(ValueError, match=r"DeviceArray.write failed - host \(5,\)"):
        array.write(np.zeros(5, np.float32))
