# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipDNN Frontend Python Bindings

This module provides Python bindings for the hipDNN frontend library,
enabling GPU-accelerated deep neural network operations through a
high-level Python interface.
"""

# Preload ROCm libraries when installed via ROCm wheels. The compiled extension
# (hipdnn_frontend_python) and hipdnn_backend live in separate wheel package
# directories, not on LD_LIBRARY_PATH. rocm_sdk loads them by absolute path so
# the extension resolves them at import time.
try:
    import rocm_sdk
except ImportError:
    rocm_sdk = None

if rocm_sdk is not None:
    import platform

    # On Windows, CPython >= 3.8 loads extension modules with
    # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes PATH. The extension and
    # hipdnn_backend.dll resolve their ROCm runtime imports (amdhip64, hiprtc,
    # amd_comgr) by base name, so those DLLs must already be in the process. The
    # runtime libs are listed before hipdnn so they load first.
    preload_shortnames = ["hipdnn"]
    if platform.system() == "Windows":
        preload_shortnames = ["amd_comgr", "amdhip64", "hiprtc", "hipdnn"]

    try:
        rocm_sdk.initialize_process(preload_shortnames=preload_shortnames)
    except Exception:
        # Preload is best-effort: the libraries may already be resolvable
        # (source builds, system installs, LD_LIBRARY_PATH). A genuine miss
        # surfaces as a clear dlopen/ImportError below.
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
