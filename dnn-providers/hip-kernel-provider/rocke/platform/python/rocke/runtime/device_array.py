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
* :func:`copy_elements` and :func:`zero_elements` move or clear the elements
  an ``ArrayInfo`` describes, strided or not, with the runtime's 2D memcpy and
  memset (``ctypes.memmove`` / ``memset`` for host memory), and
  :meth:`ArrayInfo.view` describes part of an array for them.
* :func:`host_values` copies an array's elements to the host;
  :func:`device_workspace`, :func:`device_values` and :func:`strided_view` make a
  scratch array, a filled array and a re-described view of the same kind as a
  caller's array, so a launch helper that reads offsets, needs scratch or
  adapts a layout stays torch-neutral too.
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

import ctypes
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.dtypes import dtype_info
from ..numeric.storage import element_type_of, host_dtype
from .hip_module import Runtime, current_device
from .host_buffers import as_u8_buffer
from .launcher import DeviceMem
from .torch_interop import (
    is_torch_tensor,
    torch_array_info,
    torch_device_values,
    torch_empty,
    torch_host_values,
    torch_strided_view,
)

__all__ = [
    "ArrayInfo",
    "DescribedArray",
    "DeviceArray",
    "array_info",
    "copy_elements",
    "dense_strides",
    "device_values",
    "device_workspace",
    "host_values",
    "strided_extent",
    "strided_view",
    "zero_elements",
]


@dataclass(frozen=True)
class ArrayInfo:
    """What a launch check needs to know about one device array.

    ``dtype`` is a :func:`rocke.core.dtypes.dtype_info` name, or one of the FNUZ
    fp8 storage encodings ``"fp8e4m3fnuz"`` / ``"bf8e5m2fnuz"`` that only a torch
    tensor holds (the kernels read them as bytes; the IR has no type for them).
    ``strides`` and every extent are in elements; ``device`` is the HIP device
    ordinal the memory lives on, or ``None`` for host memory. ``backed_elements``
    is how many elements the allocation holds from ``data_ptr`` on, so a check
    can bound an index computed from strides passed separately from the array."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_size: int
    data_ptr: int
    device: int | None
    backed_elements: int

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
        reach = strided_extent(self.shape, self.strides)
        return self.data_ptr, self.data_ptr + reach * self.element_size

    def view(
        self, shape: Sequence[int], strides: Sequence[int], offset: int = 0
    ) -> ArrayInfo:
        """The same memory, from ``offset`` elements past **this array's first
        element**, described as ``shape`` with element ``strides``
        (:meth:`DeviceArray.view` counts its offset from the start of the
        buffer instead). Strides and the offset are non-negative and the view
        must stay inside :attr:`backed_elements` (:func:`_checked_layout`)."""
        shape, strides, offset = _checked_layout(
            "ArrayInfo.view", shape, strides, offset, self.backed_elements
        )
        return replace(
            self,
            shape=shape,
            strides=strides,
            data_ptr=self.data_ptr + offset * self.element_size,
            backed_elements=self.backed_elements - offset,
        )


def _checked_layout(
    caller: str,
    shape: Sequence[int],
    strides: Sequence[int],
    offset: int,
    room: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """``(shape, strides, offset)`` as ints, for a view of ``room`` elements
    from its origin: one non-negative stride per axis, a non-negative offset,
    and every element inside ``room``. Raises ``ValueError`` naming ``caller``
    otherwise. The one check both :meth:`ArrayInfo.view` and
    :meth:`DeviceArray.view` make."""
    shape = tuple(int(extent) for extent in shape)
    strides = tuple(int(stride) for stride in strides)
    offset = int(offset)
    if len(strides) != len(shape) or min((*strides, offset), default=0) < 0:
        raise ValueError(
            f"{caller} failed - shape={shape} strides={strides} offset={offset}, "
            "expected one non-negative stride per axis and a non-negative offset"
        )
    if offset + strided_extent(shape, strides) > room:
        raise ValueError(
            f"{caller} failed - shape={shape} strides={strides} offset={offset} "
            f"reaches past the {room} elements from its origin"
        )
    return shape, strides, offset


def _names_each_element_once(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether a layout reaches a distinct element for every index: with its
    axes longer than 1 sorted by stride, the first stride is at least 1 and
    each other stride at least the span (extent times stride) of the axis
    before it. A sufficient check: a layout it refuses may still be
    one-to-one, but none it accepts names an element twice."""
    axes = sorted(
        (stride, extent) for extent, stride in zip(shape, strides) if extent > 1
    )
    reach = 1
    for stride, extent in axes:
        if stride < reach:
            return False
        reach = stride * extent
    return True


def dense_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Row-major element strides of a dense array of ``shape``."""
    strides, step = [], 1
    for extent in reversed(tuple(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return tuple(reversed(strides))


def strided_extent(shape: Sequence[int], strides: Sequence[int]) -> int:
    """Elements from an array's first element through its last, holes included:
    0 for an empty array (strides are non-negative)."""
    if not all(shape):
        return 0
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


@dataclass(frozen=True)
class _Rows:
    """One 2D call of a copy or fill: ``height`` rows of ``width`` elements,
    row ``r`` starting ``starts[k] + r * pitches[k]`` elements into layout
    ``k``. Every row is contiguous in every layout, and every pitch is at least
    ``width``, so the rows of one layout never overlap."""

    width: int
    height: int
    pitches: tuple[int, ...]
    starts: tuple[int, ...]


def _row_blocks(
    shape: tuple[int, ...], strides: Sequence[tuple[int, ...]]
) -> Iterator[_Rows]:
    """Walk ``shape`` in each of the stride sets ``strides`` at once, as 2D
    blocks of rows: every element lies in exactly one row of one block.

    Axes of extent 1 are dropped and neighbours contiguous in every set
    merged. The row is the innermost merged axis when it is contiguous in
    every set (one element otherwise); the rows of a block run along the
    longest other merged axis whose stride, in every set, is at least the row
    (an axis's stride is uniform by construction); each remaining position of
    the other axes is one block. So two dense layouts are one block of one
    row, a transpose of the outer axes is one block per outer position, and a
    transpose of the innermost axis is one block of single elements per
    position of all but the longest axis."""
    axes = [
        (extent, tuple(s[axis] for s in strides))
        for axis, extent in enumerate(shape)
        if extent != 1
    ]
    merged: list[tuple[int, tuple[int, ...]]] = []
    for extent, steps in axes:
        if merged and all(
            outer == extent * inner for outer, inner in zip(merged[-1][1], steps)
        ):
            merged[-1] = (merged[-1][0] * extent, steps)
        else:
            merged.append((extent, steps))
    width = 1
    if merged and all(step == 1 for step in merged[-1][1]):
        width = merged.pop()[0]
    rows = [axis for axis in merged if all(step >= width for step in axis[1])]
    height, pitches = 1, tuple(width for _ in strides)
    if rows:
        longest = max(rows, key=lambda axis: axis[0])
        merged.remove(longest)
        height, pitches = longest
    for position in itertools.product(*(range(extent) for extent, _ in merged)):
        yield _Rows(
            width,
            height,
            pitches,
            tuple(
                sum(index * steps[k] for index, (_, steps) in zip(position, merged))
                for k in range(len(strides))
            ),
        )


def copy_elements(dst: ArrayInfo, src: ArrayInfo, *, stream: int = 0) -> None:
    """Copy the elements ``src`` describes into those ``dst`` describes.

    The two have one shape and element type and live on one device (copied
    by ``hipMemcpy2DAsync`` queued on ``stream``) or both in host memory
    (copied at once with ``ctypes.memmove``, one row at a time). Each 2D block
    of rows contiguous in both (:func:`_row_blocks`) is one device call, so
    two dense arrays are one call and a transposing copy is one per outer
    position. Negative strides, a destination that may name one element twice
    (:func:`_names_each_element_once`) and arrays whose byte spans overlap are
    refused: the result would depend on the order of the copies."""
    if (dst.shape, dst.dtype, dst.element_size) != (
        src.shape,
        src.dtype,
        src.element_size,
    ):
        raise ValueError(
            f"copy_elements failed - dst {dst.shape} {dst.dtype}, src {src.shape} "
            f"{src.dtype}, expected one shape and element type"
        )
    if dst.device != src.device:
        raise ValueError(
            f"copy_elements failed - dst on device {dst.device}, src on device "
            f"{src.device}, expected both on one device or both in host memory"
        )
    if min((*dst.strides, *src.strides), default=0) < 0:
        raise ValueError(
            f"copy_elements failed - dst strides {dst.strides}, src strides "
            f"{src.strides}, expected non-negative strides"
        )
    if not _names_each_element_once(dst.shape, dst.strides):
        raise ValueError(
            f"copy_elements failed - dst shape {dst.shape} strides {dst.strides}, "
            "expected a destination that names each element once"
        )
    (dst_start, dst_end), (src_start, src_end) = dst.byte_span(), src.byte_span()
    if dst_start < src_end and src_start < dst_end:
        raise ValueError(
            f"copy_elements failed - dst bytes [{dst_start:#x}, {dst_end:#x}) and src "
            f"bytes [{src_start:#x}, {src_end:#x}), expected no overlap"
        )
    if not all(dst.shape):
        return
    size = dst.element_size
    runtime = None if dst.device is None else Runtime()
    for block in _row_blocks(dst.shape, (dst.strides, src.strides)):
        (dst_pitch, src_pitch), (to, start) = block.pitches, block.starts
        to_ptr, from_ptr = dst.data_ptr + to * size, src.data_ptr + start * size
        if runtime is not None:
            runtime.memcpy2d_d2d_async(
                to_ptr,
                dst_pitch * size,
                from_ptr,
                src_pitch * size,
                block.width * size,
                block.height,
                stream,
            )
            continue
        for row in range(block.height):
            ctypes.memmove(
                to_ptr + row * dst_pitch * size,
                from_ptr + row * src_pitch * size,
                block.width * size,
            )


def zero_elements(dst: ArrayInfo, *, stream: int = 0) -> None:
    """Set the elements ``dst`` describes to zero bytes: ``hipMemset2DAsync``
    queued on ``stream`` for device memory, ``ctypes.memset`` at once for host
    memory, one call per 2D block of rows (:func:`_row_blocks`). Negative
    strides are refused."""
    if min(dst.strides, default=0) < 0:
        raise ValueError(
            f"zero_elements failed - dst strides {dst.strides}, expected "
            "non-negative strides"
        )
    if not all(dst.shape):
        return
    size = dst.element_size
    runtime = None if dst.device is None else Runtime()
    for block in _row_blocks(dst.shape, (dst.strides,)):
        ((pitch,), (to,)) = block.pitches, block.starts
        to_ptr = dst.data_ptr + to * size
        if runtime is not None:
            runtime.memset2d_async(
                to_ptr, pitch * size, 0, block.width * size, block.height, stream
            )
            continue
        for row in range(block.height):
            ctypes.memset(to_ptr + row * pitch * size, 0, block.width * size)


@runtime_checkable
class DescribedArray(Protocol):
    """An array that describes itself and can be read back to the host.

    :class:`DeviceArray` is one. Host-only test stand-ins implement the same two
    methods so launch checks can be exercised without a GPU."""

    def info(self) -> ArrayInfo:
        """The array's :class:`ArrayInfo`."""
        ...

    def values(self) -> np.ndarray:
        """The array's elements as a host array of ``info().shape``."""
        ...


def array_info(array: object) -> ArrayInfo:
    """The :class:`ArrayInfo` of a :class:`DescribedArray` (e.g. a
    :class:`DeviceArray`) or a torch tensor.

    Anything else raises ``TypeError``: a launch check must know exactly what
    memory it is looking at."""
    if isinstance(array, DescribedArray):
        return array.info()
    if is_torch_tensor(array):
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
    if is_torch_tensor(array):
        return torch_host_values(array)
    if isinstance(array, DescribedArray):
        return array.values()
    raise TypeError(
        f"host_values failed - array={type(array).__name__}, expected a "
        "rocke.runtime.DeviceArray or a torch tensor"
    )


def device_workspace(shape: Sequence[int], dtype: str, *, like: object) -> object:
    """Device scratch of ``shape`` and element type ``dtype``, of the same kind
    as the caller's array ``like``.

    A torch tensor gets an uninitialized torch tensor on its device (torch's
    caching allocator, ordered on torch's current stream); a
    :class:`DeviceArray`, which must be on HIP's current device, gets a
    zero-filled :class:`DeviceArray`. The contents are therefore unspecified:
    read nothing from the scratch before a kernel writes it."""
    if is_torch_tensor(like):
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


def strided_view(array: object, shape: Sequence[int], strides: Sequence[int]) -> object:
    """``array``'s memory, from its first element, described as ``shape`` with
    element ``strides``: a torch tensor view of a torch tensor
    (``torch.as_strided``), a :meth:`DeviceArray.view` of a
    :class:`DeviceArray`. Anything else raises ``TypeError``."""
    if is_torch_tensor(array):
        return torch_strided_view(array, shape, strides)
    if isinstance(array, DeviceArray):
        return array.view(shape, strides)
    raise TypeError(
        f"strided_view failed - array={type(array).__name__}, expected a "
        "rocke.runtime.DeviceArray or a torch tensor"
    )


def device_values(values: np.ndarray, *, like: object) -> object:
    """A device array holding the host ``values``, of the same kind as ``like``:
    a torch tensor on ``like``'s device, or a :class:`DeviceArray` (whose element
    type follows ``values``' dtype). Anything else raises ``TypeError``."""
    if is_torch_tensor(like):
        return torch_device_values(values, device=like.device)
    if isinstance(like, DeviceArray):
        return DeviceArray(values)
    raise TypeError(
        f"device_values failed - like={type(like).__name__}, expected a "
        "rocke.runtime.DeviceArray or a torch tensor"
    )


class DeviceArray:
    """A device copy of a host numpy array.

    ``dtype`` names the element type the device sees; it defaults from the host
    dtype (``float16`` -> ``"fp16"``, ``int32`` -> ``"i32"``, ...) and must be
    given for raw-bit hosts such as bf16 (``uint16``). ``shape`` may be narrower
    than the host array along its axes, which views the leading part of each
    axis with the host's strides (e.g. a column slice); it defaults to the host
    shape. :meth:`view` describes the same buffer another way (a transposed or
    padded layout, an offset). The device memory is freed with the last array
    that uses it.
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
        self._strides = tuple(s // self._host.itemsize for s in self._host.strides)
        self._offset = 0
        self.device = current_device()
        self._memory = DeviceMem(self._host.nbytes)
        self.write(self._host)

    def view(
        self,
        shape: Sequence[int],
        strides: Sequence[int] | None = None,
        *,
        offset: int | None = None,
    ) -> DeviceArray:
        """This array's buffer as ``shape`` with element ``strides`` (``None``:
        row-major dense), starting ``offset`` elements into **the buffer**
        (``None``: where this array starts), as ``torch.as_strided`` describes a
        storage (:meth:`ArrayInfo.view` counts its offset from the array's first
        element instead). The view shares the device memory, so a kernel's
        writes through either array show in both. Strides and the offset are
        non-negative and the view must stay inside the buffer
        (:func:`_checked_layout`)."""
        shape, strides, offset = _checked_layout(
            "DeviceArray.view",
            shape,
            dense_strides(shape) if strides is None else strides,
            self._offset if offset is None else offset,
            self._host.size,
        )
        view = object.__new__(DeviceArray)
        view._host, view._memory = self._host, self._memory
        view.dtype, view.device = self.dtype, self.device
        view.shape, view._strides, view._offset = shape, strides, offset
        return view

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
        return self._strides if dim is None else self._strides[dim]

    def data_ptr(self) -> int:
        return self._memory.ptr() + self._offset * self._host.itemsize

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
            backed_elements=self._host.size - self._offset,
        )

    def read(self) -> np.ndarray:
        """The whole device buffer as a host array like the one uploaded. The
        copy is blocking, so launches queued before it have finished."""
        out = np.empty_like(self._host)
        Runtime().memcpy_d2h(as_u8_buffer(out), self._memory.ptr(), out.nbytes)
        return out

    def values(self) -> np.ndarray:
        """This array's elements (its shape, strides and offset applied to the
        buffer :meth:`read` copies back), as a new host array."""
        buffer = self.read().reshape(-1)
        itemsize = buffer.itemsize
        return np.lib.stride_tricks.as_strided(
            buffer[self._offset :],
            shape=self.shape,
            strides=tuple(stride * itemsize for stride in self._strides),
        ).copy()

    def write(self, host: np.ndarray) -> None:
        """Overwrite the device buffer with ``host`` (same shape and dtype as
        the array this one was built from)."""
        host = np.ascontiguousarray(host)
        if host.shape != self._host.shape or host.dtype != self._host.dtype:
            raise ValueError(
                f"DeviceArray.write failed - host {host.shape} {host.dtype}, expected "
                f"{self._host.shape} {self._host.dtype}"
            )
        Runtime().memcpy_h2d(self._memory.ptr(), as_u8_buffer(host), host.nbytes)

    def __repr__(self) -> str:
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype!r})"
