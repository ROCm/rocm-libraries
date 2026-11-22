# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side byte helpers for torch-free numpy / manifest data flows.

These sit on the *host* side of a device transfer: they turn a host numpy
array into the raw byte view that :meth:`Runtime.memcpy_h2d` /
:meth:`Runtime.memcpy_d2h` consume. The device side is elsewhere -- the RAII
buffer :class:`rocke.runtime.launcher.DeviceMem` owns the allocation and the
transfer itself lives on :class:`rocke.runtime.hip_module.Runtime`. Together
they let a harness allocate, upload, launch, and read back without pulling
torch into the process (they import only ``ctypes`` + ``numpy`` on demand).

:func:`pinned_empty` / :func:`pin` give a host array in page-locked memory
(:meth:`Runtime.host_alloc`), for a harness whose copies must be DMA
transfers rather than pageable ones; the memory is released when the array
and every view of it are gone.

``as_u8_buffer`` returns a ``ctypes`` *view* over the array's storage via
``from_buffer``; the array must therefore be C-contiguous and must outlive the
copy. Callers that build fresh arrays (``np.zeros(...)`` /
``rng.standard_normal(...).astype(...)``) already satisfy this; pass
``np.ascontiguousarray(a)`` first only when the source may be non-contiguous.

The array must also carry a native numpy dtype (a real ``itemsize``): stock
numpy has no ``bfloat16``, so a bf16 host encoding needs a hand-rolled dtype --
``ml_dtypes`` is deliberately *not* a dependency of this path. Torch-free
harnesses that need bf16 must encode it themselves rather than reach for it.
"""

from __future__ import annotations

import ctypes
import math
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["as_u8_buffer", "nbytes", "pin", "pinned_empty", "require_numpy"]


def require_numpy():
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("rocke host-buffer helpers require numpy") from e
    return np


def nbytes(a) -> int:
    return int(a.nbytes)


def as_u8_buffer(a):
    return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)


def pinned_empty(shape: Sequence[int], dtype: npt.DTypeLike) -> np.ndarray:
    """An uninitialised C-contiguous ``dtype`` array of ``shape`` in page-locked
    host memory.

    The memory comes from :meth:`Runtime.host_alloc` and goes back through
    :meth:`Runtime.host_free` once the array and every view of it are
    collected (not at interpreter exit, when the process releases it)."""
    np = require_numpy()
    from .hip_module import Runtime

    extents = tuple(int(extent) for extent in shape)
    if any(extent < 0 for extent in extents):
        raise ValueError(
            f"pinned_empty failed - shape={extents}, expected every extent >= 0"
        )
    array_dtype = np.dtype(dtype)
    count = math.prod(extents)
    # hipHostMalloc of 0 bytes returns no memory; a 1-byte block keeps the
    # empty array's buffer a real allocation like any other.
    block_bytes = max(count * array_dtype.itemsize, 1)
    runtime = Runtime()
    ptr = runtime.host_alloc(block_bytes)
    block = (ctypes.c_uint8 * block_bytes).from_address(ptr)
    release = weakref.finalize(block, runtime.host_free, ptr)
    release.atexit = False
    return np.frombuffer(block, dtype=array_dtype, count=count).reshape(extents)


def pin(array: npt.ArrayLike) -> np.ndarray:
    """A copy of ``array`` in page-locked host memory (:func:`pinned_empty`)."""
    np = require_numpy()
    source = np.asarray(array)
    pinned = pinned_empty(source.shape, source.dtype)
    pinned[...] = source
    return pinned
