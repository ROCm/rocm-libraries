# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch-free tensors for the explicit paged binding tests.

``FakeTensor`` carries the metadata the binding validators read; ``paged_tensors``
is a valid paged decode set for the 1024-token, 32/8-head, d128 request.
"""

from __future__ import annotations

from unittest import mock

_ELEMENT_SIZE = {
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.bfloat16": 2,
    "torch.float16": 2,
}


class FakeTensor:
    """A stand-in for a torch tensor. Its storage backs exactly the elements
    its strides reach unless ``storage_elems`` says otherwise."""

    def __init__(
        self,
        shape,
        dtype,
        values=None,
        *,
        device="cuda:0",
        contiguous=True,
        strides=None,
        storage_elems=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._values = values
        self._contiguous = contiguous
        self._strides = strides
        self._storage_elems = storage_elems

    def element_size(self):
        return _ELEMENT_SIZE[self.dtype]

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        elems = self._storage_elems
        if elems is None:
            elems = 1 + sum(
                (extent - 1) * self.stride(dim) for dim, extent in enumerate(self.shape)
            )
        return mock.Mock(nbytes=mock.Mock(return_value=elems * self.element_size()))

    def is_contiguous(self):
        return self._contiguous

    def detach(self):
        return self

    def stride(self, dim=None):
        if dim is None:
            return tuple(self.stride(d) for d in range(len(self.shape)))
        if self._strides is not None:
            return self._strides[dim]
        stride = 1
        values = [0] * len(self.shape)
        for index in range(len(self.shape) - 1, -1, -1):
            values[index] = stride
            stride *= self.shape[index]
        return values[dim]

    def cpu(self):
        return self

    def tolist(self):
        if self._values is None:
            raise AssertionError("no host values")
        return self._values


def paged_tensors(*, num_blocks=64):
    return {
        "q": FakeTensor((1024, 32, 128), "torch.bfloat16"),
        "out": FakeTensor((1024, 32, 128), "torch.bfloat16"),
        "k": FakeTensor((num_blocks, 16, 8, 128), "torch.bfloat16"),
        "v": FakeTensor((num_blocks, 16, 8, 128), "torch.bfloat16"),
        "cu_seqlens_q": FakeTensor((2,), "torch.int32", [0, 1024]),
        "seqused_k": FakeTensor((1,), "torch.int32", [1024]),
        "block_table": FakeTensor((1, 64), "torch.int32", [list(range(64))]),
    }
