# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host data helpers shared by the parity harnesses.

The harnesses (``ck_tile_parity``, ``parity_extended_kernels`` and the
library's ``builders.common.parity_fmha_extended``) draw numpy inputs, store
them in the kernel's element type, launch on
:class:`~rocke.runtime.device_array.DeviceArray` buffers, and compare the
kernel's output values with a float64 reference computed from the stored
inputs. This module is the one home of those steps.
"""

from __future__ import annotations

import numpy as np

from rocke.numeric import storage
from rocke.runtime.device_array import DeviceArray

__all__ = [
    "device_values",
    "draw",
    "round_to_f16",
    "values",
]


def draw(
    rng: np.random.Generator,
    *shape: int,
    dtype: str = "fp16",
    scale: float = 1.0,
    shift: float = 0.0,
) -> np.ndarray:
    """``shift + scale * N(0, 1)`` draws stored as ``dtype`` (a host storage
    array)."""
    return storage.encode(shift + scale * rng.standard_normal(shape), dtype)


def values(stored: np.ndarray, dtype: str) -> np.ndarray:
    """The float64 values of a ``dtype`` host storage array."""
    return storage.decode(stored, dtype).astype(np.float64)


def device_values(array: DeviceArray) -> np.ndarray:
    """The float64 values of a floating-point :class:`DeviceArray`."""
    return values(array.values(), array.dtype)


def round_to_f16(reference: np.ndarray) -> np.ndarray:
    """``reference`` rounded to fp16, as an fp16 kernel stores its output, in
    float64."""
    return storage.round_to(reference, "fp16").astype(np.float64)
