# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only stand-ins for the GPU tensors a paged dense launch takes.

It exposes just what the launcher's checks read (shape, dtype, device, strides,
indexing, values), so the paged launch validation is testable without torch or a GPU.
"""


class I32DeviceTensor:
    dtype = "torch.int32"

    def __init__(
        self, values, *, dtype=None, is_cuda=True, inner_stride=1, device="cuda:0"
    ):
        self._values = values
        if dtype is not None:
            self.dtype = dtype
        self.is_cuda = is_cuda
        self._inner_stride = inner_stride
        self.device = device

    @property
    def shape(self):
        rows = self._values
        if rows and isinstance(rows[0], list):
            return (len(rows), len(rows[0]))
        return (len(rows),)

    def stride(self, dim=None):
        strides = (
            (self.shape[1] * self._inner_stride, self._inner_stride)
            if len(self.shape) == 2
            else (self._inner_stride,)
        )
        return strides if dim is None else strides[dim]

    def __getitem__(self, index):
        if isinstance(index, tuple):  # [rows, cols] of a 2-D table
            rows, cols = index
            return I32DeviceTensor(
                [row[cols] for row in self._values[rows]], device=self.device
            )
        item = self._values[index]
        return (
            I32DeviceTensor(item, device=self.device)
            if isinstance(item, list)
            else item
        )

    def tolist(self):
        return self._values


class CacheStandIn:
    """A paged K or V cache: only its shape, dtype, layout and device are read."""

    def __init__(
        self, shape, *, contiguous=True, device="cuda:0", dtype="torch.float16"
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._contiguous = contiguous
        self.device = device

    def is_contiguous(self):
        return self._contiguous
