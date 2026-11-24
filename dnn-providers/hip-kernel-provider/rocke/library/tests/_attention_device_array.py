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


_TORCH_NAME = {
    np.dtype(np.float16): "torch.float16",
    np.dtype(np.float32): "torch.float32",
    np.dtype(np.float64): "torch.float64",
    np.dtype(np.int32): "torch.int32",
    np.dtype(np.int64): "torch.int64",
    np.dtype(np.uint8): "torch.uint8",
    np.dtype(np.uint16): "torch.bfloat16",
}


class DeviceArray:
    """A device copy of a numpy array, exposing the tensor attributes
    ``run_unified_attention_torch`` reads. ``shape`` may be narrower than the
    host array along the last axis, which models a column view of it.

    ``torch_dtype`` names the element type the kernels' host-side checks see
    (``uint16`` hosts default to ``bfloat16``); it defaults from the host dtype.
    """

    device = None
    is_cuda = True

    def __init__(self, rt, host, *, shape=None, torch_dtype=None):
        from rocke.runtime.launcher import DeviceMem

        self._rt = rt
        self._host = np.ascontiguousarray(host)
        self._mem = DeviceMem(self._host.nbytes)
        self.write(self._host)
        self.shape = self._host.shape if shape is None else shape
        name = torch_dtype or _TORCH_NAME.get(self._host.dtype, str(self._host.dtype))
        self.dtype = name if name.startswith("torch.") else f"torch.{name}"

    def is_contiguous(self):
        return self._host.flags.c_contiguous and tuple(self.shape) == self._host.shape

    def numel(self):
        return int(np.prod(self.shape))

    def element_size(self):
        return self._host.itemsize

    def dim(self):
        return len(self.shape)

    def stride(self, dim=None):
        strides = tuple(s // self._host.itemsize for s in self._host.strides)
        return strides if dim is None else strides[dim]

    def __getitem__(self, index):
        return self.read()[index]

    def tolist(self):
        return self.read().tolist()

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
