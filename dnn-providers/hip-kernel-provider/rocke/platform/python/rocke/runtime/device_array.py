# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A torch-free device buffer backed by :class:`~rocke.runtime.launcher.DeviceMem`.

Exposes the few tensor attributes the launch helpers read (``data_ptr``,
``numel``, ``shape``) plus host transfers, so benchmarks and numeric tests can
run in GPU lanes that carry no torch.  Elements are stored on the device in
``dtype`` (``f16``, ``bf16``, ``f32``, ``i32``); the host side of :meth:`write` /
:meth:`read` is always a float64 / float32 / int32 numpy array that is rounded to
or widened from the storage dtype (bf16 uses the raw ``uint16`` codec of
:mod:`rocke.numeric.bf16`).
"""

from __future__ import annotations

import ctypes

_ALIASES = {"fp16": "f16", "fp32": "f32", "float16": "f16", "float32": "f32"}
_ITEMSIZE = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4}


def _u8(array):
    return (ctypes.c_uint8 * array.nbytes).from_buffer(array)


def _encode(host, dtype):
    import numpy as np

    from ..numeric.bf16 import f32_to_bf16

    if dtype == "f16":
        return np.ascontiguousarray(host, dtype=np.float16)
    if dtype == "f32":
        return np.ascontiguousarray(host, dtype=np.float32)
    if dtype == "i32":
        return np.ascontiguousarray(host, dtype=np.int32)
    return f32_to_bf16(np.asarray(host, dtype=np.float32))


def _decode(raw, dtype):
    import numpy as np

    from ..numeric.bf16 import bf16_to_f32

    if dtype == "bf16":
        return bf16_to_f32(raw)
    if dtype == "i32":
        return raw
    return raw.astype(np.float32)


class DeviceArray:
    """A device array of ``shape`` and storage ``dtype``."""

    def __init__(self, shape, dtype="f32", host=None):
        from .launcher import DeviceMem

        self.dtype = _ALIASES.get(dtype, dtype)
        if self.dtype not in _ITEMSIZE:
            raise ValueError(f"unsupported dtype {dtype!r}")
        self.shape = tuple(int(s) for s in shape)
        n = 1
        for s in self.shape:
            n *= s
        self._numel = n
        self._mem = DeviceMem(n * _ITEMSIZE[self.dtype])
        if host is None:
            self.zero_()
        else:
            self.write(host)

    @classmethod
    def from_host(cls, host, dtype="f32"):
        """Device copy of ``host`` rounded to ``dtype``."""
        import numpy as np

        host = np.asarray(host)
        return cls(host.shape, dtype, host=host)

    @classmethod
    def uniform(cls, shape, dtype, lo, hi, rng):
        """Device array of values drawn uniformly from ``[lo, hi)`` with numpy ``rng``."""
        return cls.from_host(rng.uniform(lo, hi, size=shape).astype("float32"), dtype)

    def data_ptr(self):
        return self._mem.ptr()

    def numel(self):
        return self._numel

    def zero_(self):
        from .launcher import _runtime

        if self._numel:
            _runtime().memset(self._mem.ptr(), 0, self._mem.nbytes())
        return self

    def write(self, host):
        """Overwrite the device contents with ``host`` (same element count)."""
        import numpy as np

        from .launcher import _runtime

        host = np.asarray(host)
        assert host.size == self._numel, (host.size, self._numel)
        enc = _encode(host.reshape(-1), self.dtype)
        _runtime().memcpy_h2d(self._mem.ptr(), _u8(enc), enc.nbytes)

    def read(self):
        """The device contents as a float32 (int32 for ``i32``) array of ``shape``.
        The copy is blocking, so launches queued before it have finished."""
        import numpy as np

        from .launcher import _runtime

        np_dtype = {
            "f16": np.float16,
            "bf16": np.uint16,
            "f32": np.float32,
            "i32": np.int32,
        }[self.dtype]
        raw = np.empty(self._numel, dtype=np_dtype)
        _runtime().memcpy_d2h(_u8(raw), self._mem.ptr(), raw.nbytes)
        return _decode(raw, self.dtype).reshape(self.shape)
