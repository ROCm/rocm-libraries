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
from rocke.runtime.device_array import (
    ArrayInfo,
    DescribedArray,
    DeviceArray,
    array_info,
    dense_strides,
    device_values,
    device_workspace,
    host_values,
    strided_extent,
    strided_view,
)
from rocke.runtime.torch_interop import torch_array_info
from rocke.tests.gpu import current_device_arch
from rocke.tests.host_arrays import HostArray


def _info(shape, strides, element_size=4, data_ptr=1024):
    return ArrayInfo(
        shape=shape,
        strides=strides,
        dtype="fp32",
        element_size=element_size,
        data_ptr=data_ptr,
        device=0,
        backed_elements=strided_extent(shape, strides),
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

    def untyped_storage(self):
        # 128 bytes (64 bf16) of storage; the tensor starts 8 elements in.
        return SimpleNamespace(nbytes=lambda: 128)

    def storage_offset(self):
        return 8


def test_torch_tensor_info_maps_dtype_and_device():
    info = array_info(_TorchTensorAttributes())
    assert info == ArrayInfo(
        shape=(4, 8),
        strides=(16, 1),
        dtype="bf16",
        element_size=2,
        data_ptr=4096,
        device=1,
        backed_elements=56,
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


def test_host_array_is_described_and_read_without_a_gpu():
    table = HostArray((2, 3), "i32", values=[[1, 2, 3], [4, 5, 6]], strides=(4, 1))
    assert isinstance(table, DescribedArray)
    info = array_info(table)
    assert (info.shape, info.strides, info.dtype, info.device) == (
        (2, 3),
        (4, 1),
        "i32",
        0,
    )
    assert info.element_size == 4 and not info.is_contiguous()
    np.testing.assert_array_equal(host_values(table), [[1, 2, 3], [4, 5, 6]])


def test_dense_strides_are_row_major():
    assert dense_strides((2, 3, 4)) == (12, 4, 1)
    assert dense_strides((0, 4)) == (4, 1)
    assert dense_strides(()) == ()


def test_host_arrays_never_overlap():
    first, second = HostArray((4,), "fp32"), HostArray((4,), "fp32")
    first_end = first.info().byte_span()[1]
    assert first_end <= second.info().byte_span()[0]


def test_host_array_without_values_refuses_a_read():
    with pytest.raises(ValueError, match=r"HostArray.values failed - built without"):
        host_values(HostArray((2,), "i32"))


def test_host_values_and_device_workspace_refuse_other_objects():
    with pytest.raises(TypeError, match=r"host_values failed - array=list"):
        host_values([1, 2])
    with pytest.raises(TypeError, match=r"device_workspace failed - like=HostArray"):
        device_workspace((2,), "fp32", like=HostArray((2,), "fp32"))
    with pytest.raises(TypeError, match=r"device_values failed - like=HostArray"):
        device_values(np.zeros(2, np.int32), like=HostArray((2,), "fp32"))
    with pytest.raises(TypeError, match=r"strided_view failed - array=HostArray"):
        strided_view(HostArray((2,), "fp32"), (2,), (1,))


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
    assert table.info().backed_elements == 30


def test_gpu_view_shares_the_buffer_with_its_own_layout():
    current_device_arch()
    host = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    array = DeviceArray(host)
    # The [2, 4, 3] transpose of the last two axes, starting one element in.
    view = array.view((2, 4, 2), (12, 1, 4), offset=1)
    assert view.stride() == (12, 1, 4) and not view.is_contiguous()
    assert view.data_ptr() == array.data_ptr() + 4
    assert view.info().backed_elements == 23
    expected = np.lib.stride_tricks.as_strided(
        host.reshape(-1)[1:], (2, 4, 2), (48, 4, 16)
    )
    np.testing.assert_array_equal(host_values(view), expected)
    array.write(host + 100)
    np.testing.assert_array_equal(view.values(), expected + 100)


def test_gpu_strided_view_keeps_the_view_start():
    current_device_arch()
    host = np.arange(12, dtype=np.int32)
    tail = DeviceArray(host).view((8,), (1,), offset=4)
    # A [4, 2] transpose of the tail starts where the tail starts.
    pairs = strided_view(tail, (4, 2), (1, 4))
    assert pairs.data_ptr() == tail.data_ptr()
    np.testing.assert_array_equal(host_values(pairs), host[4:].reshape(2, 4).T)


def test_gpu_device_values_of_a_device_array_kind():
    current_device_arch()
    lengths = device_values(
        np.array([3, 1, 2], np.int32), like=DeviceArray.zeros((1,), "fp16")
    )
    assert isinstance(lengths, DeviceArray) and lengths.dtype == "i32"
    np.testing.assert_array_equal(host_values(lengths), [3, 1, 2])


def test_gpu_view_defaults_to_dense_strides():
    current_device_arch()
    host = np.arange(12, dtype=np.int32)
    rows = DeviceArray(host).view((2, 3), offset=6)
    assert rows.stride() == (3, 1)
    np.testing.assert_array_equal(rows.values(), host[6:].reshape(2, 3))


def test_gpu_view_must_stay_inside_the_buffer():
    current_device_arch()
    array = DeviceArray(np.zeros(8, np.float32))
    with pytest.raises(ValueError, match=r"reaches past the 8-element buffer"):
        array.view((4,), (2,), offset=2)
    with pytest.raises(ValueError, match=r"expected one non-negative stride"):
        array.view((4,), (-1,))


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


def test_gpu_host_values_of_a_column_view():
    current_device_arch()
    host = np.arange(30, dtype=np.int32).reshape(3, 10)
    table = DeviceArray(host, shape=(3, 4))
    np.testing.assert_array_equal(host_values(table), host[:, :4])


def test_gpu_device_workspace_matches_a_device_array():
    current_device_arch()
    like = DeviceArray(np.zeros(4, np.float16))
    scratch = device_workspace((2, 3), "fp32", like=like)
    assert isinstance(scratch, DeviceArray)
    assert scratch.shape == (2, 3) and scratch.dtype == "fp32"
    assert scratch.device == like.device


# --- torch callers (optional: runs only where torch with a GPU is installed) ---------------------


def test_gpu_torch_tensors_get_torch_values_and_workspaces():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no GPU here")
    lengths = torch.arange(6, dtype=torch.int32, device="cuda").reshape(2, 3)
    np.testing.assert_array_equal(host_values(lengths[:, :2]), [[0, 1], [3, 4]])
    bits = torch.ones(2, dtype=torch.bfloat16, device="cuda")
    np.testing.assert_array_equal(host_values(bits), encode(np.ones(2), "bf16"))
    scratch = device_workspace((2, 3), "fp32", like=lengths)
    assert scratch.dtype == torch.float32 and scratch.device == lengths.device
    assert tuple(scratch.shape) == (2, 3)
    columns = strided_view(lengths[1:], (3, 1), (1, 3))
    np.testing.assert_array_equal(host_values(columns), [[3], [4], [5]])
    offsets = device_values(np.arange(3, dtype=np.int32), like=lengths)
    assert offsets.device == lengths.device and offsets.dtype == torch.int32
