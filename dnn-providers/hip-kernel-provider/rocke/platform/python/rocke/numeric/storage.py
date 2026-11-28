# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free host storage codec for the element types a kernel reads and writes.

A verify harness builds its operands as fp32 values, stores them in the kernel's
element type, and computes its reference from exactly the values the device
reads back. This module is the one place that maps a storage dtype name to that
host representation, so harnesses do not each grow their own copy:

=============  ===================  ============================================
``dtype``      host array           rounding from fp32
=============  ===================  ============================================
``"fp32"``     ``float32``          none
``"fp16"``     ``float16``          numpy's round-to-nearest-even cast
``"bf16"``     ``uint16`` raw bits  round-to-nearest-even (:mod:`rocke.numeric.bf16`)
``"fp8e4m3"``  ``uint8`` raw bits   OCP e4m3, round-to-nearest-even, saturating
                                    at 448; NaN refused (:mod:`rocke.numeric.fp8`)
=============  ===================  ============================================

numpy has no ``bfloat16`` or fp8, so those travel as their raw bit patterns; the
bit codecs themselves live in :mod:`rocke.numeric.bf16` and
:mod:`rocke.numeric.fp8` and are only dispatched to from here.
``xf32_truncate`` is the operand rounding of a native XF32 matrix core (it reads
the top 19 bits of each fp32 operand); which architectures do that is the
architecture catalog's question, not this module's.

:func:`host_dtype` and :func:`element_type_of` are the one map between a rocKE
element type (:func:`rocke.core.dtypes.dtype_info` names) and the numpy dtype of
its host array, for every whole-byte type: numpy's own type where it has one,
else an unsigned integer of the same width holding the raw bits.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.dtypes import dtype_info
from ..core.storage import element_bytes
from .bf16 import bf16_to_f32, f32_to_bf16
from .fp8 import fp8e4m3_decode_table, fp8e4m3_round

__all__ = [
    "STORAGE_DTYPES",
    "decode",
    "element_type_of",
    "encode",
    "host_dtype",
    "round_to",
    "xf32_truncate",
]

#: Element types numpy has a type of its own for.
_NUMPY_NATIVE = {
    "fp16": np.dtype(np.float16),
    "fp32": np.dtype(np.float32),
    "fp64": np.dtype(np.float64),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "u8": np.dtype(np.uint8),
}

#: Every storage dtype name the value codec (encode / decode / round_to) accepts.
STORAGE_DTYPES = ("fp32", "fp16", "bf16", "fp8e4m3")

#: An fp32 bit pattern with the 13 mantissa bits a native XF32 MMA ignores cleared.
_XF32_MASK = np.uint32(0xFFFFE000)


def host_dtype(dtype: str) -> np.dtype:
    """The numpy dtype of a ``dtype`` host array (``uint16`` for ``"bf16"``,
    ``uint8`` for the fp8 types). A sub-byte type raises ``ValueError``."""
    name = dtype_info(dtype).name
    return _NUMPY_NATIVE.get(name, np.dtype(f"u{element_bytes(name)}"))


def element_type_of(array_dtype: npt.DTypeLike) -> str:
    """The element type a host array of ``array_dtype`` holds, for the numpy
    types that name one. Raw-bit hosts (``uint16``, ...) are ambiguous and raise
    ``ValueError``: the caller must name the element type."""
    array_dtype = np.dtype(array_dtype)
    for name, native in _NUMPY_NATIVE.items():
        if native == array_dtype:
            return name
    raise ValueError(
        f"element type failed - host dtype={array_dtype} has no element type of its "
        "own, expected the caller to name one (e.g. 'bf16' for uint16 bits)"
    )


def _require_storage_dtype(dtype: str) -> np.dtype:
    if dtype not in STORAGE_DTYPES:
        raise ValueError(
            f"storage codec failed - dtype={dtype!r}, expected one of {STORAGE_DTYPES}"
        )
    return host_dtype(dtype)


def _fp32(values: npt.ArrayLike) -> np.ndarray:
    """``values`` as a contiguous fp32 array of their own shape (a scalar stays
    0-d: ``np.ascontiguousarray`` alone would make it 1-d)."""
    fp32 = np.asarray(values, dtype=np.float32)
    return np.ascontiguousarray(fp32).reshape(fp32.shape)


def encode(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """``values`` (cast to fp32 first) as a contiguous ``dtype`` host array of
    their shape (0-d for a scalar)."""
    array_dtype = _require_storage_dtype(dtype)
    fp32 = _fp32(values)
    if dtype == "bf16":
        encoded = f32_to_bf16(fp32)
    elif dtype == "fp8e4m3":
        encoded = fp8e4m3_round(fp32)
    else:
        encoded = fp32.astype(array_dtype)
    return encoded.reshape(fp32.shape)


def decode(stored: np.ndarray, dtype: str) -> np.ndarray:
    """The fp32 values of an :func:`encode` array.

    The array must already carry ``dtype``'s host dtype, so a ``uint16`` buffer is
    never silently read as fp16 (or the reverse)."""
    array_dtype = _require_storage_dtype(dtype)
    if stored.dtype != array_dtype:
        raise ValueError(
            f"storage decode failed - array dtype={stored.dtype}, expected "
            f"{array_dtype} for dtype={dtype!r}"
        )
    if dtype == "bf16":
        decoded = bf16_to_f32(stored)
    elif dtype == "fp8e4m3":
        decoded = fp8e4m3_decode_table()[stored]
    else:
        decoded = stored.astype(np.float32)
    return decoded.reshape(stored.shape)


def round_to(values: npt.ArrayLike, dtype: str) -> np.ndarray:
    """``values`` rounded to ``dtype`` and returned as fp32: what the kernel reads."""
    return decode(encode(values, dtype), dtype)


def xf32_truncate(values: npt.ArrayLike) -> np.ndarray:
    """fp32 ``values`` as a native XF32 MMA consumes them: the low 13 mantissa
    bits cleared (truncation, not rounding)."""
    fp32 = _fp32(values)
    return (fp32.view(np.uint32) & _XF32_MASK).view(np.float32)
