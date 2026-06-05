# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipDNN Frontend Python Bindings

This module provides Python bindings for the hipDNN frontend library,
enabling GPU-accelerated deep neural network operations through a
high-level Python interface.
"""

import os
import platform

is_windows = platform.system() == "Windows"

# Windows extension modules cannot resolve dependent DLLs from PATH: CPython
# loads them with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes PATH and has
# no RPATH equivalent. HIPDNN_DLL_DIRECTORIES lets a caller register absolute
# directories (e.g. a raw ROCm artifact tree's bin/) via os.add_dll_directory
# before the extension imports, without requiring rocm_sdk to be installed.
if is_windows:
    for _dll_dir in os.environ.get("HIPDNN_DLL_DIRECTORIES", "").split(os.pathsep):
        if _dll_dir and os.path.isdir(_dll_dir):
            os.add_dll_directory(_dll_dir)

# Preload ROCm libraries when installed via ROCm wheels. The compiled extension
# (hipdnn_frontend_python) and hipdnn_backend live in separate wheel package
# directories, not on LD_LIBRARY_PATH. rocm_sdk loads them by absolute path so
# the extension resolves them at import time.
try:
    import rocm_sdk

    core_shortnames = ["hipdnn"]
    if is_windows:
        core_shortnames = [
            "amd_comgr",
            "amdhip64",
            "hiprtc",
            "hipdnn",
            "hipblaslt",
            "miopen",
        ]

    rocm_sdk.initialize_process(preload_shortnames=core_shortnames)
except ImportError:
    # rocm_sdk is not installed. Non-wheel installs (source builds, system
    # ROCm) resolve the runtime via RPATH/LD_LIBRARY_PATH/PATH on Linux, or via
    # the HIPDNN_DLL_DIRECTORIES os.add_dll_directory registration above on
    # Windows, and do not need preloading, so there is nothing to do.
    pass
except Exception:
    # Preload is best-effort. initialize_process can raise when a requested
    # library is unavailable (ModuleNotFoundError if the providing wheel is not
    # installed, FileNotFoundError if the wheel is present but the DLL is
    # missing). The library may still be resolvable by other means; a genuine
    # miss surfaces as a clear dlopen/ImportError from the extension import
    # below.
    pass

# Import everything from the compiled extension module
try:
    # The compiled extension module
    from hipdnn_frontend_python import *
except ImportError as e:
    # Fallback for development/editable installs
    try:
        from .hipdnn_frontend_python import *
    except ImportError:
        raise ImportError(
            "Could not import the hipdnn_frontend_python compiled extension. "
            "Please ensure the package is properly installed.\n"
            f"Original error: {e}"
        )

# Package metadata
__version__ = "0.1.0"
__author__ = "Advanced Micro Devices, Inc."

# Define what should be available when using "from hipdnn_frontend import *"
# This will be populated by the compiled extension's exports
__all__ = [
    # These will be defined by the C++ bindings
    "Graph",
    "Tensor",
    "TensorAttributes",
    "ConvolutionForwardAttributes",
    "ActivationAttributes",
    "BatchnormForwardInferenceAttributes",
    "BatchnormBackwardAttributes",
    "PoolingForwardAttributes",
    "MatmulAttributes",
    "DataType",
    "TensorLayout",
    "ConvolutionMode",
    "ActivationMode",
    "PoolingMode",
    "BatchnormMode",
    "Handle",
    "create_handle",
    "destroy_handle",
    "set_stream",
    "get_stream",
]
