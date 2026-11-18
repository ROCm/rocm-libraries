# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Low-level Python bindings for the hipBLASLt GEMM API.

This is a development and testing tool for hipBLASLt and its consumers, not a ROCm deliverable.
"""

from . import _build_info, _loader

library = _loader.load(
    _build_info.LIB_DIR,
    _build_info.BIN_DIR,
    _build_info.LIB_NAME,
    _build_info.HIP_RUNTIME,
)

from . import _core  # noqa: E402 - libhipblaslt must be preloaded first

_loader.check_after_import(library, _core)

from ._dtypes import DTYPE_TO_NUMPY, from_numpy, to_numpy  # noqa: E402

_core.DeviceArray.to_numpy = to_numpy

from ._gemm import gemm  # noqa: E402

__version__ = _core.__version__
HipblasLtError = _core.HipblasLtError
HipblasLtLoadError = _loader.HipblasLtLoadError
HipblasLtLoadWarning = _loader.HipblasLtLoadWarning

__all__ = [
    "DTYPE_TO_NUMPY",
    "HipblasLtError",
    "HipblasLtLoadError",
    "HipblasLtLoadWarning",
    "__version__",
    "from_numpy",
    "gemm",
    "library",
]
