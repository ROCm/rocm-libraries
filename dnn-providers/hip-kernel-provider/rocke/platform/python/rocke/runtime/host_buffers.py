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

__all__ = ["require_numpy", "nbytes", "as_u8_buffer"]


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
