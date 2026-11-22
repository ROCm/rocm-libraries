# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed device arrays without torch, and the one description launch checks read.

Three pieces, one per responsibility:

* :class:`ArrayInfo` describes an array a kernel argument points at: shape,
  element strides, rocKE element type (``"fp16"``, ``"bf16"``, ``"fp32"``,
  ``"i32"``, ... -- the :func:`rocke.core.dtypes.dtype_info` names), element
  size, device pointer and HIP device. :func:`array_info` builds it from a
  :class:`DescribedArray` (a :class:`DeviceArray`, or a host-side stand-in in a
  test) or from a torch tensor (through
  :func:`rocke.runtime.torch_interop.torch_array_info`, which never imports
  torch). Host-side launch checks (dtype, shape, contiguity, same device, no
  overlap) read an ``ArrayInfo`` and so accept any of these.
* :func:`host_values` copies an array's elements to the host, and
  :func:`device_workspace` allocates a scratch array of the same kind as a
  caller's array, so a launch helper that reads offsets or needs scratch stays
  torch-neutral too.
* :class:`DeviceArray` is a device copy of a host numpy array, allocated and
  copied by the rocKE runtime. It also answers the few tensor methods the
  launchers read (``shape``, ``stride``, ``data_ptr``, ``nbytes``), so it can
  be passed wherever a launch helper takes a device tensor.

numpy has no bf16 and no fp8, so a :class:`DeviceArray` of those types holds
the raw bits on the host (``uint16`` / ``uint8``) and names its element type
explicitly; the element type <-> host dtype map is
:func:`rocke.numeric.storage.host_dtype` / :func:`~rocke.numeric.storage.element_type_of`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.dtypes import dtype_info
from ..numeric.storage import element_type_of, host_dtype
from .hip_module import Runtime, current_device
from .host_buffers import as_u8_buffer
from .launcher import DeviceMem

__all__ = [
    "ArrayInfo",
    "DescribedArray",
    "DeviceArray",
    "array_info",
    "device_workspace",
    "host_values",
]


@dataclass(frozen=True)
class ArrayInfo:
    """What a launch check needs to know about one device array.

    ``strides`` and every extent are in elements; ``device`` is the HIP device
    ordinal the memory lives on, or ``None`` for host memory."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_size: int
    data_ptr: int
    device: int | None

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def is_contiguous(self) -> bool:
        """Row-major dense, ignoring the strides of size-1 axes (as torch does)."""
        expected = 1
        for extent, stride in zip(reversed(self.shape), reversed(self.strides)):
            if extent != 1 and stride != expected:
                return False
            expected *= extent
        return True

    def byte_span(self) -> tuple[int, int]:
        """``[start, end)`` of the bytes the array can touch, holes included
        (strides are non-negative)."""
        if not self.numel:
            return self.data_ptr, self.data_ptr
        last = sum((e - 1) * s for e, s in zip(self.shape, self.strides))
        return self.data_ptr, self.data_ptr + (last + 1) * self.element_size


@runtime_checkable
class DescribedArray(Protocol):
    """An array that describes itself and can be read back to the host.

    :class:`DeviceArray` is one. Host-only test stand-ins implement the same two
    methods so launch checks can be exercised without a GPU."""

    def info(self) -> ArrayInfo:
        """The array's :class:`ArrayInfo`."""
        ...

    def read(self) -> np.ndarray:
        """A host array whose leading ``info().shape`` part along each axis holds
        the array's elements (it may be wider, as a :class:`DeviceArray` column
        view's buffer is)."""
        ...


def _is_torch_tensor(array: object) -> bool:
    # By module name, so torch is never imported to answer the question.
    return type(array).__module__.split(".", 1)[0] == "torch"


def array_info(array: object) -> ArrayInfo:
    """The :class:`ArrayInfo` of a :class:`DescribedArray` (e.g. a
    :class:`DeviceArray`) or a torch tensor.

    Anything else raises ``TypeError``: a launch check must know exactly what
    memory it is looking at."""
    if isinstance(array, DescribedArray):
        return array.info()
    if _is_torch_tensor(array):
        from .torch_interop import torch_array_info

        return torch_array_info(array)
    raise TypeError(
        f"array_info failed - array={type(array).__name__}, expected a "
        "rocke.runtime.DeviceArray or a torch tensor"
    )


def host_values(array: object) -> np.ndarray:
    """The elements of ``array`` as a host numpy array of its shape.

    The copy is blocking. A torch tensor is copied by torch, ordered on its
    current stream as ``Tensor.cpu()`` is. Element types numpy has no type for
    (bf16, fp8) come back as raw bits (:func:`rocke.numeric.storage.host_dtype`).
    Anything :func:`array_info` refuses raises ``TypeError``."""
    if _is_torch_tensor(array):
        from .torch_interop import torch_host_values

        return torch_host_values(array)
    info = array_info(array)
    return array.read()[tuple(slice(0, extent) for extent in info.shape)]


def device_workspace(shape: Sequence[int], dtype: str, *, like: object) -> object:
    """Device scratch of ``shape`` and element type ``dtype``, of the same kind
    as the caller's array ``like``.

    A torch tensor gets an uninitialized torch tensor on its device (torch's
    caching allocator, ordered on torch's current stream); a
    :class:`DeviceArray`, which must be on HIP's current device, gets a
    zero-filled :class:`DeviceArray`. The contents are therefore unspecified:
    read nothing from the scratch before a kernel writes it."""
    if _is_torch_tensor(like):
        from .torch_interop import torch_empty

        return torch_empty(shape, dtype, device=like.device)
    if isinstance(like, DeviceArray):
        if like.device != current_device():
            raise ValueError(
                f"device_workspace failed - like is on device {like.device}, "
                f"expected HIP's current device {current_device()}"
            )
        return DeviceArray.zeros(shape, dtype)
    raise TypeError(
        f"device_workspace failed - like={type(like).__name__}, expected a "
        "rocke.runtime.DeviceArray or a torch tensor"
    )


class DeviceArray:
    """A device copy of a host numpy array.

    ``dtype`` names the element type the device sees; it defaults from the host
    dtype (``float16`` -> ``"fp16"``, ``int32`` -> ``"i32"``, ...) and must be
    given for raw-bit hosts such as bf16 (``uint16``). ``shape`` may be narrower
    than the host array along its axes, which views the leading part of each
    axis with the host's strides (e.g. a column slice); it defaults to the host
    shape. The device memory is freed with the object.
    """

    def __init__(
        self,
        host: np.ndarray,
        dtype: str | None = None,
        *,
        shape: Sequence[int] | None = None,
    ) -> None:
        self._host = np.ascontiguousarray(host)
        self.dtype = self._element_type(self._host.dtype, dtype)
        self.shape = self._view_shape(self._host.shape, shape)
        self.device = current_device()
        self._memory = DeviceMem(self._host.nbytes)
        self.write(self._host)

    @classmethod
    def zeros(cls, shape: Sequence[int], dtype: str) -> DeviceArray:
        """A zero-filled array of ``shape`` and element type ``dtype``."""
        return cls(np.zeros(tuple(shape), dtype=host_dtype(dtype)), dtype)

    @staticmethod
    def _element_type(array_dtype: np.dtype, dtype: str | None) -> str:
        if dtype is None:
            return element_type_of(array_dtype)
        name = dtype_info(dtype).name
        if host_dtype(name).itemsize != array_dtype.itemsize:
            raise ValueError(
                f"DeviceArray failed - dtype={name!r} is {host_dtype(name).itemsize} "
                f"bytes, expected the host element size {array_dtype.itemsize} "
                f"({array_dtype})"
            )
        return name

    @staticmethod
    def _view_shape(host_shape, shape) -> tuple[int, ...]:
        if shape is None:
            return tuple(host_shape)
        view = tuple(int(extent) for extent in shape)
        if len(view) != len(host_shape) or any(
            not 0 <= v <= h for v, h in zip(view, host_shape)
        ):
            raise ValueError(
                f"DeviceArray failed - shape={view}, expected one extent per host "
                f"axis, each at most the host extent {tuple(host_shape)}"
            )
        return view

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def stride(self, dim: int | None = None) -> int | tuple[int, ...]:
        """Element stride of axis ``dim``, or of every axis when ``dim`` is None."""
        strides = tuple(s // self._host.itemsize for s in self._host.strides)
        return strides if dim is None else strides[dim]

    def data_ptr(self) -> int:
        return self._memory.ptr()

    def element_size(self) -> int:
        return self._host.itemsize

    def numel(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        """Bytes of the elements ``shape`` covers (``numel() * element_size()``,
        as a torch tensor's ``nbytes``)."""
        return self.numel() * self.element_size()

    def is_contiguous(self) -> bool:
        return self.info().is_contiguous()

    def info(self) -> ArrayInfo:
        return ArrayInfo(
            shape=self.shape,
            strides=self.stride(),
            dtype=self.dtype,
            element_size=self.element_size(),
            data_ptr=self.data_ptr(),
            device=self.device,
        )

    def read(self) -> np.ndarray:
        """The whole device buffer as a host array like the one uploaded. The
        copy is blocking, so launches queued before it have finished."""
        out = np.empty_like(self._host)
        Runtime().memcpy_d2h(as_u8_buffer(out), self.data_ptr(), out.nbytes)
        return out

    def write(self, host: np.ndarray) -> None:
        """Overwrite the device buffer with ``host`` (same shape and dtype as
        the array this one was built from)."""
        host = np.ascontiguousarray(host)
        if host.shape != self._host.shape or host.dtype != self._host.dtype:
            raise ValueError(
                f"DeviceArray.write failed - host {host.shape} {host.dtype}, expected "
                f"{self._host.shape} {self._host.dtype}"
            )
        Runtime().memcpy_h2d(self.data_ptr(), as_u8_buffer(host), host.nbytes)

    def __repr__(self) -> str:
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype!r})"
