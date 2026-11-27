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
from rocke.numeric.storage import element_type_of, encode
from rocke.runtime.device_array import (
    ArrayInfo,
    DescribedArray,
    DeviceArray,
    _names_each_element_once,
    _row_blocks,
    _Rows,
    array_info,
    copy_elements,
    dense_strides,
    device_values,
    device_workspace,
    host_values,
    strided_extent,
    strided_view,
    zero_elements,
)
from rocke.runtime.torch_interop import is_torch_tensor, torch_array_info
from rocke.tests.gpu import current_device_arch
from rocke.tests.host_arrays import HostArray
from rocke.tests.lanes import TICKET, import_or_skip_until


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


class _TensorSubclass(_TorchTensorAttributes):
    """A subclass of a torch type defined outside torch, as a user's tensor
    subclass is."""

    __module__ = __name__


def test_a_subclass_of_a_torch_type_is_a_torch_tensor():
    assert not is_torch_tensor(np.zeros(4, np.float32))
    assert is_torch_tensor(_TensorSubclass())
    assert array_info(_TensorSubclass()).shape == (4, 8)


def test_a_real_torch_tensor_subclass_is_a_torch_tensor():
    torch = import_or_skip_until("torch", TICKET)

    class Subclass(torch.Tensor):
        pass

    assert is_torch_tensor(torch.zeros(2).as_subclass(Subclass))


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


@pytest.mark.parametrize(
    "host, dtype, expected",
    [
        (np.float16, "bf16", "uint16 or a 2-byte"),
        (np.float32, "bf16", "uint16 or a 2-byte"),
        (np.float32, "i32", "int32 or a 4-byte"),
        (np.int8, "fp16", "float16 or a 2-byte"),
    ],
)
def test_element_type_must_match_the_host_dtype(host, dtype, expected):
    # float16 data named bf16 has the right width but would be read as bf16 bits.
    with pytest.raises(
        ValueError,
        match=rf"dtype={dtype!r}, host dtype={np.dtype(host)}, expected {expected} "
        "integer type holding its raw bits",
    ):
        DeviceArray(np.zeros(4, host), dtype)


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


def _host_info(array: np.ndarray) -> ArrayInfo:
    """The ``ArrayInfo`` of a host numpy array's memory (``device`` None)."""
    base = array.base if array.base is not None else array
    return ArrayInfo(
        shape=array.shape,
        strides=tuple(s // array.itemsize for s in array.strides),
        dtype=element_type_of(array.dtype),
        element_size=array.itemsize,
        data_ptr=array.ctypes.data,
        device=None,
        backed_elements=(base.ctypes.data + base.nbytes - array.ctypes.data)
        // array.itemsize,
    )


def test_a_view_describes_part_of_an_array_and_stays_inside_it():
    whole = _info((4, 6), (6, 1))
    rows = whole.view((2, 3), (6, 1), offset=7)
    assert rows.data_ptr == whole.data_ptr + 7 * 4
    assert rows.backed_elements == whole.backed_elements - 7
    assert (rows.shape, rows.strides, rows.dtype) == ((2, 3), (6, 1), "fp32")
    with pytest.raises(
        ValueError, match=r"reaches past the 24 elements from its origin"
    ):
        whole.view((2, 6), (6, 1), offset=13)
    with pytest.raises(ValueError, match=r"expected one non-negative stride"):
        whole.view((2,), (-1,))
    # A negative extent would reach backwards past the origin unchecked.
    with pytest.raises(
        ValueError, match=r"shape=\(-2, 3\), expected a non-negative extent per axis"
    ):
        whole.view((-2, 3), (6, 1), offset=7)


def test_dense_layouts_are_one_block_of_one_row():
    assert list(_row_blocks((2, 3, 4), ((12, 4, 1), (12, 4, 1)))) == [
        _Rows(width=24, height=1, pitches=(24, 24), starts=(0, 0))
    ]


def test_a_transpose_of_outer_axes_is_one_block_per_outer_position():
    # [2, 3, 4] read dense, written as [3, 2, 4]: rows of 4 along the longer of
    # the two outer axes (extent 3), one block per position of the other.
    assert list(_row_blocks((2, 3, 4), ((4, 8, 1), (12, 4, 1)))) == [
        _Rows(width=4, height=3, pitches=(8, 4), starts=(0, 0)),
        _Rows(width=4, height=3, pitches=(8, 4), starts=(4, 12)),
    ]


def test_a_transpose_of_the_inner_axis_moves_single_elements_along_the_longest_axis():
    # The [B, H, S, 1] stats against a token-major [S, H, 1] buffer: no axis is
    # contiguous in both, so rows are one element; they run along S (the
    # longest), one block per head instead of one call per element.
    blocks = list(_row_blocks((5, 3, 1), ((1, 8, 1), (3, 1, 1))))
    assert blocks == [
        _Rows(width=1, height=5, pitches=(1, 3), starts=(8 * head, head))
        for head in range(3)
    ]


def test_a_row_never_overlaps_the_next_of_its_block():
    # The middle axis's dst stride (2) is shorter than the 4-element row, so
    # it cannot carry rows; the outer axis does.
    assert list(_row_blocks((3, 2, 4), ((40, 2, 1), (8, 4, 1)))) == [
        _Rows(width=4, height=3, pitches=(40, 8), starts=(0, 0)),
        _Rows(width=4, height=3, pitches=(40, 8), starts=(2, 4)),
    ]


def test_a_broadcast_source_axis_reads_one_row_per_block():
    # The source repeats one row of 4 along the outer axis (stride 0): it cannot
    # carry rows (its stride is shorter than the row), so each outer position is
    # one block of one row.
    assert list(_row_blocks((3, 4), ((4, 1), (0, 1)))) == [
        _Rows(width=4, height=1, pitches=(4, 4), starts=(4 * row, 0))
        for row in range(3)
    ]


def test_the_longest_axis_cannot_carry_rows_when_its_stride_is_too_short():
    # Rows of 4; the longest outer axis (extent 5) steps 2 in the destination,
    # shorter than a row, so the rows run along the extent-3 axis instead.
    assert list(_row_blocks((3, 5, 4), ((40, 2, 1), (20, 4, 1)))) == [
        _Rows(width=4, height=3, pitches=(40, 20), starts=(2 * i, 4 * i))
        for i in range(5)
    ]


def test_a_layout_with_one_element_is_one_block():
    assert list(_row_blocks((1, 1), ((5, 7),))) == [
        _Rows(width=1, height=1, pitches=(1,), starts=(0,))
    ]


def test_host_copy_moves_strided_elements_and_zero_clears_them():
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    target = np.full((3, 2, 4), -1.0, np.float32)
    copy_elements(_host_info(target.transpose(1, 0, 2)), _host_info(source))
    np.testing.assert_array_equal(target, source.transpose(1, 0, 2))
    zero_elements(_host_info(target[:, 1, 1:]))
    expected = source.transpose(1, 0, 2).copy()
    expected[:, 1, 1:] = 0
    np.testing.assert_array_equal(target, expected)


def test_a_destination_must_name_each_element_once():
    """Every overlap of a destination's elements is refused, not only a zero
    stride: [2, 2] with strides (1, 1) names element 1 twice."""
    square = _info((2, 2), (2, 1))
    with pytest.raises(ValueError, match=r"names each element once"):
        copy_elements(_info((2, 2), (1, 1), data_ptr=4096), square)
    assert _names_each_element_once((2, 2), (2, 1))
    assert _names_each_element_once((2, 3), (1, 2))  # column-major
    assert _names_each_element_once((4, 1), (1, 0))  # a unit axis may repeat
    assert not _names_each_element_once((2, 2), (1, 1))
    # Sufficient, not exact: offsets 0, 2, 4, 3, 5, 7 are distinct, but the axis
    # of stride 3 starts inside the span (6) of the axis below it.
    assert not _names_each_element_once((3, 2), (2, 3))
    assert not _names_each_element_once((2,), (0,))


def test_copy_refuses_other_shapes_types_and_devices():
    four = _info((4,), (1,))
    with pytest.raises(ValueError, match=r"expected one shape and element type"):
        copy_elements(four, _info((5,), (1,)))
    with pytest.raises(ValueError, match=r"expected both on one device"):
        copy_elements(four, _host_info(np.zeros(4, np.float32)))
    with pytest.raises(ValueError, match=r"names each element once"):
        copy_elements(_info((4,), (0,), data_ptr=4096), four)
    with pytest.raises(ValueError, match=r"expected non-negative strides"):
        copy_elements(_info((4,), (-1,), data_ptr=4096), four)
    with pytest.raises(ValueError, match=r"expected non-negative strides"):
        copy_elements(_info((4,), (1,), data_ptr=4096), _info((4,), (-1,)))
    with pytest.raises(ValueError, match=r"expected non-negative strides"):
        zero_elements(_info((4,), (-1,)))
    with pytest.raises(ValueError, match=r"expected no overlap"):
        copy_elements(_info((4,), (1,), data_ptr=1028), four)


# --- on the GPU ----------------------------------------------------------------------------------


@pytest.mark.gpu
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


@pytest.mark.gpu
@pytest.mark.parametrize(
    "host, dtype",
    [(np.uint16, "bf16"), (np.int16, "fp16"), (np.uint32, "i32"), (np.int8, "fp8e4m3")],
)
def test_gpu_a_named_element_type_takes_its_raw_bits(host, dtype):
    current_device_arch()
    array = DeviceArray(np.zeros(4, host), dtype)
    assert array.dtype == dtype and array.element_size() == np.dtype(host).itemsize


@pytest.mark.gpu
def test_gpu_column_view_keeps_the_host_strides():
    current_device_arch()
    table = DeviceArray(np.zeros((3, 10), np.int32), shape=(3, 4))
    assert table.shape == (3, 4) and table.stride(0) == 10
    assert table.nbytes == 3 * 4 * 4
    assert not table.is_contiguous()
    assert table.info().backed_elements == 30


@pytest.mark.gpu
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


@pytest.mark.gpu
def test_gpu_strided_view_keeps_the_view_start():
    current_device_arch()
    host = np.arange(12, dtype=np.int32)
    tail = DeviceArray(host).view((8,), (1,), offset=4)
    # A [4, 2] transpose of the tail starts where the tail starts.
    pairs = strided_view(tail, (4, 2), (1, 4))
    assert pairs.data_ptr() == tail.data_ptr()
    np.testing.assert_array_equal(host_values(pairs), host[4:].reshape(2, 4).T)


@pytest.mark.gpu
@pytest.mark.parametrize("offset", [64, 65], ids=["offset_exact", "offset_short"])
def test_gpu_an_offset_view_backs_the_elements_from_its_start(offset):
    """A launch check bounds an index by ``backed_elements``, so a view that
    starts ``offset`` elements into a buffer backs only the elements from its
    start, never the buffer's whole size; and it cannot reach past the
    buffer."""
    current_device_arch()
    buffer = DeviceArray(np.zeros(320, np.float32))
    view = buffer.view((320 - offset,), offset=offset)
    assert view.info().backed_elements == 320 - offset
    assert array_info(view).backed_elements == 320 - offset
    with pytest.raises(
        ValueError, match=r"reaches past the 320 elements from its origin"
    ):
        buffer.view((321 - offset,), offset=offset)


@pytest.mark.gpu
def test_gpu_device_values_of_a_device_array_kind():
    current_device_arch()
    lengths = device_values(
        np.array([3, 1, 2], np.int32), like=DeviceArray.zeros((1,), "fp16")
    )
    assert isinstance(lengths, DeviceArray) and lengths.dtype == "i32"
    np.testing.assert_array_equal(host_values(lengths), [3, 1, 2])


@pytest.mark.gpu
def test_gpu_view_defaults_to_dense_strides():
    current_device_arch()
    host = np.arange(12, dtype=np.int32)
    rows = DeviceArray(host).view((2, 3), offset=6)
    assert rows.stride() == (3, 1)
    np.testing.assert_array_equal(rows.values(), host[6:].reshape(2, 3))


@pytest.mark.gpu
def test_gpu_view_must_stay_inside_the_buffer():
    current_device_arch()
    array = DeviceArray(np.zeros(8, np.float32))
    with pytest.raises(
        ValueError, match=r"reaches past the 8 elements from its origin"
    ):
        array.view((4,), (2,), offset=2)
    with pytest.raises(ValueError, match=r"expected one non-negative stride"):
        array.view((4,), (-1,))


@pytest.mark.gpu
def test_gpu_zeros():
    current_device_arch()
    array = DeviceArray.zeros((2, 3), "fp8e4m3")
    assert array.dtype == "fp8e4m3" and array.element_size() == 1
    np.testing.assert_array_equal(array.read(), np.zeros((2, 3), np.uint8))


@pytest.mark.gpu
def test_gpu_write_refuses_another_shape():
    current_device_arch()
    array = DeviceArray(np.zeros(4, np.float32))
    with pytest.raises(ValueError, match=r"DeviceArray.write failed - host \(5,\)"):
        array.write(np.zeros(5, np.float32))


@pytest.mark.gpu
def test_gpu_host_values_of_a_column_view():
    current_device_arch()
    host = np.arange(30, dtype=np.int32).reshape(3, 10)
    table = DeviceArray(host, shape=(3, 4))
    np.testing.assert_array_equal(host_values(table), host[:, :4])


@pytest.mark.gpu
def test_gpu_device_workspace_matches_a_device_array():
    current_device_arch()
    like = DeviceArray(np.zeros(4, np.float16))
    scratch = device_workspace((2, 3), "fp32", like=like)
    assert isinstance(scratch, DeviceArray)
    assert scratch.shape == (2, 3) and scratch.dtype == "fp32"
    assert scratch.device == like.device


@pytest.mark.gpu
def test_gpu_copy_and_zero_follow_each_layout():
    current_device_arch()
    host = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = DeviceArray(host)
    target = DeviceArray.zeros((3, 2, 4), "fp32")
    copy_elements(target.view((2, 3, 4), (4, 8, 1)).info(), source.info(), stream=0)
    np.testing.assert_array_equal(target.read(), host.transpose(1, 0, 2))
    zero_elements(target.info().view((3, 3), (8, 1), offset=5))
    expected = host.transpose(1, 0, 2).copy().reshape(3, 8)
    expected[:, 5:] = 0
    np.testing.assert_array_equal(target.read().reshape(3, 8), expected)
    # An inner-axis transpose: single elements in rows along the longest axis.
    stats = DeviceArray(np.arange(15, dtype=np.float32).reshape(3, 5))
    tokens = DeviceArray.zeros((5, 3), "fp32")
    copy_elements(tokens.view((3, 5), (1, 3)).info(), stats.info())
    np.testing.assert_array_equal(tokens.read(), stats.read().T)


# --- torch callers (optional: runs only where torch with a GPU is installed) ---------------------


@pytest.mark.gpu
def test_gpu_torch_tensors_get_torch_values_and_workspaces():
    torch = import_or_skip_until("torch", TICKET)
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
