# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A torch-free device buffer for the attention launch helpers.

``run_unified_attention_torch`` and ``KernelLauncher`` read only a few tensor
attributes (``shape``, ``stride``, ``data_ptr``, ``device``), so a
``rocke.runtime`` device copy of a numpy array can stand in for a torch tensor.
This lets numeric tests run in GPU lanes that have no torch.
"""

import ctypes

import numpy as np


def _u8(array):
    return (ctypes.c_uint8 * array.nbytes).from_buffer(array)


class DeviceArray:
    """A device copy of a numpy array, exposing the tensor attributes
    ``run_unified_attention_torch`` reads. ``shape`` may be narrower than the
    host array along the last axis, which models a column view of it."""

    device = None

    def __init__(self, rt, host, *, shape=None):
        from rocke.runtime.launcher import DeviceMem

        self._rt = rt
        self._host = np.ascontiguousarray(host)
        self._mem = DeviceMem(self._host.nbytes)
        self.write(self._host)
        self.shape = self._host.shape if shape is None else shape

    def stride(self, dim):
        return self._host.strides[dim] // self._host.itemsize

    def data_ptr(self):
        return self._mem.ptr()

    def read(self):
        """The device contents, as an array like the host one. The copy is
        blocking, so launches queued before it have finished."""
        out = np.empty_like(self._host)
        self._rt.memcpy_d2h(_u8(out), self._mem.ptr(), out.nbytes)
        return out

    def write(self, host):
        """Overwrite the device contents with ``host`` (same shape and dtype)."""
        host = np.ascontiguousarray(host)
        assert host.shape == self._host.shape and host.dtype == self._host.dtype
        self._rt.memcpy_h2d(self._mem.ptr(), _u8(host), host.nbytes)
