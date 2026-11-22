# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: host-only arrays that launch checks accept as device arrays.

A launch helper's host checks (dtype, shape, strides, device, offsets read back)
go through :func:`rocke.runtime.device_array.array_info` and
:func:`~rocke.runtime.device_array.host_values`, which accept any
:class:`~rocke.runtime.device_array.DescribedArray`. :class:`HostArray` is one that
allocates nothing, so those checks can be tested on a machine with no GPU and no
torch. It must never reach a launch: its address is made up.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from rocke.core.dtypes import dtype_info
from rocke.core.storage import element_bytes
from rocke.numeric.storage import host_dtype
from rocke.runtime.device_array import ArrayInfo, dense_strides, strided_extent

__all__ = ["HostArray"]

# Made-up device addresses, far apart, so two stand-ins never overlap.
_ADDRESS_STRIDE = 1 << 40
_addresses = itertools.count(1)


class HostArray:
    """A host-only stand-in for a device array of ``shape`` and element type ``dtype``.

    ``values`` (optional) are the elements :meth:`values` returns, as a host array
    of ``shape`` in ``dtype``'s host type (:func:`rocke.numeric.storage.host_dtype`);
    without them :meth:`values` raises. ``strides`` (elements) default to row-major
    dense; ``device`` is the HIP ordinal the array claims, or ``None`` for host
    memory. The claimed allocation is exactly what the strides reach.
    """

    def __init__(
        self,
        shape: Sequence[int],
        dtype: str,
        *,
        values: Sequence | np.ndarray | None = None,
        strides: Sequence[int] | None = None,
        device: int | None = 0,
    ) -> None:
        self.shape = tuple(int(extent) for extent in shape)
        self.dtype = dtype_info(dtype).name
        self.device = device
        self._strides = (
            tuple(int(stride) for stride in strides)
            if strides is not None
            else dense_strides(self.shape)
        )
        if len(self._strides) != len(self.shape):
            raise ValueError(
                f"HostArray failed - strides={self._strides}, expected one per axis "
                f"of shape {self.shape}"
            )
        self._values = None
        if values is not None:
            self._values = np.asarray(values, dtype=host_dtype(self.dtype))
            if self._values.shape != self.shape:
                raise ValueError(
                    f"HostArray failed - values shape {self._values.shape}, expected "
                    f"{self.shape}"
                )
        self._data_ptr = next(_addresses) * _ADDRESS_STRIDE

    def info(self) -> ArrayInfo:
        return ArrayInfo(
            shape=self.shape,
            strides=self._strides,
            dtype=self.dtype,
            element_size=element_bytes(self.dtype),
            data_ptr=self._data_ptr,
            device=self.device,
            backed_elements=strided_extent(self.shape, self._strides),
        )

    def values(self) -> np.ndarray:
        if self._values is None:
            raise ValueError(
                "HostArray.values failed - built without values, expected values= "
                "for a check that reads the elements"
            )
        return self._values

    def __repr__(self) -> str:
        return f"HostArray(shape={self.shape}, dtype={self.dtype!r})"
