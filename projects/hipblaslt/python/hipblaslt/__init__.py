# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level Python bindings for the hipBLASLt GEMM API."""

from . import _core
import numpy as _np

__all__ = ["_core"]
__version__ = _core.__version__

# Minimal numpy-native dtype map; extended with ml_dtypes in Phase 4.
_DTYPE_TO_NP = {
    _core.DataType.R_32F: _np.float32,
    _core.DataType.R_64F: _np.float64,
    _core.DataType.R_16F: _np.float16,
    _core.DataType.R_32I: _np.int32,
    _core.DataType.R_8I: _np.int8,
}


def _device_array_to_numpy(self):
    np_dtype = _DTYPE_TO_NP[self.dtype]
    out = _np.empty(tuple(self.shape), dtype=np_dtype)
    self.copy_to_host(out)
    return out


_core.DeviceArray.to_numpy = _device_array_to_numpy
