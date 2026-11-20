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

__version__ = _core.__version__
HipblasLtError = _core.HipblasLtError
HipblasLtLoadError = _loader.HipblasLtLoadError
HipblasLtLoadWarning = _loader.HipblasLtLoadWarning

__all__ = [
    "HipblasLtError",
    "HipblasLtLoadError",
    "HipblasLtLoadWarning",
    "__version__",
    "library",
]
