# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
hipDNN Frontend Python Bindings

This module provides Python bindings for the hipDNN frontend library,
enabling GPU-accelerated deep neural network operations through a
high-level Python interface.
"""

import sys
import ctypes
import signal

# Save original flags
_original_flags = sys.getdlopenflags()

# Set RTLD_GLOBAL. This is required for MIOpen JIT compilation to find 
# symbols from the loaded backend libraries (like libMIOpen.so).
sys.setdlopenflags(_original_flags | ctypes.RTLD_GLOBAL)

# Explicitly load backend libraries with RTLD_GLOBAL.
# This ensures symbols are visible to MIOpen's JIT compiler, which is required
# for the "First Run" initialization to succeed.
try:
    # Try standard ROCm location first
    ctypes.CDLL("/opt/rocm/lib/libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/opt/rocm/lib/libMIOpen.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    # Fallback to system search path if /opt/rocm isn't standard
    try:
        ctypes.CDLL("libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL("libMIOpen.so", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # If we can't load them here, we proceed and let the extension module
        # try to handle dependencies, though JIT might fail.
        pass

# Reset SIGCHLD handler to default.
# MIOpen's JIT compilation spawns subprocesses. Python's custom SIGCHLD 
# handler can intercept the termination signals of these subprocesses,
# causing the JIT compilation to hang or fail to write the DB files.
signal.signal(signal.SIGCHLD, signal.SIG_DFL)

# Import everything from the compiled extension module
try:
    # The compiled extension module
    from hipdnn_frontend_python import *
except ImportError as e:
    # Fallback for development/editable installs
    try:
        from .hipdnn_frontend_python import *
    except ImportError:
        # Restore original flags before raising error
        sys.setdlopenflags(_original_flags)
        raise ImportError(
            "Could not import the hipdnn_frontend compiled extension. "
            "Please ensure the package is properly installed.\n"
            f"Original error: {e}"
        )

# Restore original flags to avoid polluting the namespace for future imports
sys.setdlopenflags(_original_flags)

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
    "DataType",
    "TensorLayout",
    "ConvolutionMode",
    "ActivationMode",
    "PoolingMode",
    "BatchnormMode",
    # Add other exported symbols as needed
]
