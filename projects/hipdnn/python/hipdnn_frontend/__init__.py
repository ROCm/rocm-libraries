# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipDNN Frontend Python Bindings

This module provides Python bindings for the hipDNN frontend library,
enabling GPU-accelerated deep neural network operations through a
high-level Python interface.
"""

# Preload hipDNN backend library when installed via ROCm wheels.
# The Python extension (hipdnn_frontend_python.so) depends on libhipdnn_backend.so
# which lives in a separate wheel package directory — not on LD_LIBRARY_PATH.
# rocm_sdk.preload_libraries loads it with RTLD_GLOBAL so the extension finds it.
try:
    import rocm_sdk
except ImportError:
    rocm_sdk = None

if rocm_sdk is not None:
    try:
        rocm_sdk.preload_libraries("hipdnn")
    except Exception:
        # Preload is best-effort: the library may already be on LD_LIBRARY_PATH
        # (e.g., source builds, system installs). If it's truly missing, the
        # extension import below will fail with a clear dlopen error.
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
