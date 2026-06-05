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

_IS_WINDOWS = platform.system() == "Windows"

# ROCm runtime libraries the compiled extension and hipdnn_backend link against.
# On Windows every dependent DLL (runtime, runtime compiler, and engine-provider
# deps) resolves by base name under a restricted loader search, so all must be
# in the process beforehand. On Linux, RPATH/ldconfig pull the transitive deps
# once libhipdnn itself is loadable, so only hipdnn needs to be located.
_ROCM_CORE_SHORTNAMES = (
    ["amd_comgr", "amdhip64", "hiprtc", "hipdnn"] if _IS_WINDOWS else ["hipdnn"]
)
# Engine provider plugins are loaded by absolute path at handle-create time, but
# their transitive ROCm deps resolve by base name under the same restricted
# Windows search, so preload them too. Optional: a distribution may ship only one.
_ROCM_PROVIDER_SHORTNAMES = ["hipblaslt", "miopen"] if _IS_WINDOWS else []


def _preload_via_rocm_sdk():
    """Wheel install: ROCm libs ship as sibling rocm_sdk packages, off the
    loader path. rocm_sdk resolves them by absolute path and preloads them so
    the extension's by-name imports resolve at import time. Raises ImportError
    when rocm_sdk is not installed (i.e. this is not a ROCm-wheel environment).
    """
    import rocm_sdk

    # Core first so a missing optional provider can never block it. Each group is
    # best-effort: initialize_process raises if a requested wheel/DLL is absent,
    # but the library may still be resolvable by other means, and a genuine miss
    # surfaces as a clear ImportError from the extension import below.
    for shortnames in (_ROCM_CORE_SHORTNAMES, _ROCM_PROVIDER_SHORTNAMES):
        if not shortnames:
            continue
        try:
            rocm_sdk.initialize_process(preload_shortnames=shortnames)
        except Exception:
            pass


def _preload_via_rocm_path():
    """Non-wheel installs (system .deb / /opt/rocm, and build/artifact trees):
    locate the ROCm library directory from the standard ROCM_PATH/ROCM_HOME env
    vars and make its libraries resolvable for the extension import.

    On Windows the directory is registered via os.add_dll_directory: extension
    modules load with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes PATH and
    has no RPATH equivalent. On Linux libhipdnn is ctypes-preloaded with
    RTLD_GLOBAL so its RPATH pulls the transitive ROCm deps and the soname is
    already resolved when the extension imports; LD_LIBRARY_PATH cannot be set
    from here because the dynamic loader reads it once at process start.
    """
    for var in ("ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if not (root and os.path.isdir(root)):
            continue
        lib_dir = os.path.join(root, "bin" if _IS_WINDOWS else "lib")
        if not os.path.isdir(lib_dir):
            return
        if _IS_WINDOWS:
            os.add_dll_directory(lib_dir)
        else:
            import ctypes
            from glob import glob

            # Base symlink sorts before its versioned aliases; loading any of
            # them registers the lib under its soname for the extension.
            matches = sorted(glob(os.path.join(lib_dir, "libhipdnn.so*")))
            if matches:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
        return


try:
    _preload_via_rocm_sdk()
except ImportError:
    _preload_via_rocm_path()

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
