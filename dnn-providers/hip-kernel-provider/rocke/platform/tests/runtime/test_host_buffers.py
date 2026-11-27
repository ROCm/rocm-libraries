# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``rocke.runtime.host_buffers``: page-locked host arrays.

The refusal needs no GPU. The ``test_gpu_`` tests allocate page-locked memory
through HIP and skip without a GPU.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
from rocke.numeric.storage import encode
from rocke.runtime.host_buffers import as_u8_buffer, pinned_copy, pinned_empty
from rocke.tests.gpu import current_device_arch


def test_pinned_empty_refuses_a_negative_extent():
    with pytest.raises(ValueError, match=r"pinned_empty failed - shape=\(2, -1\)"):
        pinned_empty((2, -1), np.float32)


def _allocation(array: np.ndarray) -> object:
    """The object at the root of ``array``'s base chain: the page-locked block."""
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


@pytest.mark.gpu
def test_gpu_pinned_arrays_round_trip_through_the_device():
    current_device_arch()
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import DeviceMem

    values = encode(np.linspace(-1.0, 1.0, 24).reshape(2, 3, 4), "bf16")
    source = pinned_copy(values)
    assert source.dtype == values.dtype and source.shape == values.shape
    assert source.flags["C_CONTIGUOUS"] and source.flags["WRITEABLE"]
    device = DeviceMem(source.nbytes)
    runtime = Runtime()
    runtime.memcpy_h2d(device.ptr(), as_u8_buffer(source), source.nbytes)
    back = pinned_empty(values.shape, values.dtype)
    runtime.memcpy_d2h(as_u8_buffer(back), device.ptr(), back.nbytes)
    np.testing.assert_array_equal(back, values)


@pytest.mark.gpu
def test_gpu_pinned_empty_takes_an_empty_shape():
    current_device_arch()
    assert pinned_empty((0, 5), np.float32).shape == (0, 5)


@pytest.mark.gpu
def test_gpu_the_allocation_lives_until_the_last_view_is_gone():
    current_device_arch()
    array = pinned_copy(np.arange(8, dtype=np.float32))
    allocation = weakref.ref(_allocation(array))
    view = array[2:4]
    del array
    gc.collect()
    assert allocation() is not None
    np.testing.assert_array_equal(view, [2.0, 3.0])
    del view
    gc.collect()
    assert allocation() is None
