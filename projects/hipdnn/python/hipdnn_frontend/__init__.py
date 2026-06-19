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

# ROCm runtime libraries to preload via rocm_sdk before importing the compiled
# extension. On Windows every dependent DLL resolves by base name under a
# restricted loader search, so the runtime, runtime compiler, and engine-provider
# deps -- including hipdnn -- must all be named. On Linux the libraries prefix is
# already on the loader path (LD_LIBRARY_PATH, via the venv), so hipDNN resolves
# on its own; only the HIP runtime, which lives off-path in the core wheel, needs
# preloading (amdhip64 pulls comgr/hiprtc and the sysdeps via its own runpath).
_ROCM_WHEEL_SHORTNAMES = (
    ["amd_comgr", "amdhip64", "hiprtc", "hipdnn", "hipblaslt", "miopen"]
    if _IS_WINDOWS
    else ["amd_comgr", "amdhip64", "hiprtc"]
)


def _preload_via_rocm_sdk() -> bool:
    """Wheel-install path. ROCm libs ship inside sibling _rocm_sdk_* packages,
    off the loader path; their package names carry a build-time version nonce and
    GPU target family, so only rocm_sdk knows their absolute locations. Use its
    public API rather than reimplementing that discovery.

    Returns True when rocm_sdk is installed and drove the preload (i.e. this is a
    ROCm-wheel environment), False otherwise so the caller can fall back.
    """
    try:
        import rocm_sdk
    except ImportError:
        return False

    # Best-effort: initialize_process raises if a requested library is absent
    # (ModuleNotFoundError for a missing provider wheel, FileNotFoundError for a
    # missing DLL), but it may still be resolvable by other means; a genuine miss
    # surfaces as a clear ImportError from the extension import below. Core libs
    # lead the list so a missing optional provider cannot block them.
    try:
        rocm_sdk.initialize_process(preload_shortnames=_ROCM_WHEEL_SHORTNAMES)
    except Exception:
        pass
    return True


def _register_rocm_path_dir() -> None:
    """Non-wheel installs: a system /opt/rocm, a .deb, the Windows HIP SDK, or a
    build/artifact tree, where the runtime sits in one directory named by the
    standard ROCM_PATH/HIP_PATH/ROCM_HOME env vars.

    On Windows that directory's bin/ must be registered via os.add_dll_directory:
    extension modules load with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes
    PATH and has no RPATH equivalent. On Linux the dynamic loader already
    searches RPATH/ldconfig/LD_LIBRARY_PATH, so there is nothing to do.
    """
    if not _IS_WINDOWS:
        return
    for var in ("ROCM_PATH", "HIP_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                return


if not _preload_via_rocm_sdk():
    _register_rocm_path_dir()

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
            "Could not load the hipdnn_frontend_python compiled extension. Its "
            "ROCm dependencies were not found. Install the ROCm wheels "
            "(`pip install rocm[libraries]`), or set ROCM_PATH/HIP_PATH to a "
            "ROCm install or build tree (on Windows the directory containing the "
            f"ROCm DLLs under bin/).\nOriginal error: {e}"
        ) from e

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
    "HipEvent",
    "hip_stream_synchronize",
    "hip_get_device_count",
]
