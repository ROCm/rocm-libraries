# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
tilewright: standalone, framework-agnostic GEMM kernel recommender.

Python bindings for the tilewright C++ library.
"""

try:
    # Import the compiled extension module (tilewright/tilewright.<ext>).
    from .tilewright import (
        # Enums
        DataType,
        Transpose,
        # Data structures
        Dim3,
        Problem,
        Config,
        Hardware,
        Result,
        # Free functions
        load_weights,
        weights_loaded,
        route,
        rank_configs,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import tilewright extension module: {e}. "
        "Please ensure the package is properly installed."
    ) from e

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Enums
    "DataType",
    "Transpose",
    # Data structures
    "Dim3",
    "Problem",
    "Config",
    "Hardware",
    "Result",
    # Free functions
    "load_weights",
    "weights_loaded",
    "route",
    "rank_configs",
]
