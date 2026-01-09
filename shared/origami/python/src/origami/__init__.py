# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""

try:
    # Import the compiled extension module
    from .origami import *
except ImportError as e:
    raise ImportError(
        f"Failed to import origami extension module: {e}. "
        "Please ensure the package is properly installed."
    ) from e

__version__ = "0.1.0"

