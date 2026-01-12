################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

"""
Origami: Analytical GEMM Solution Selection

Python bindings for the Origami C++ library.
"""

# Import the compiled extension module
HAS_CORE = False
try:
    from .origami import *
    HAS_CORE = True
except ImportError as e:
    raise ImportError(
        f"Failed to import origami extension module: {e}. "
        "Please ensure the package is properly installed."
    ) from e

# Import the torch heuristic selection module if possible
HAS_PYTHON_SELECTION = False
try:
    from .selector import *
    HAS_PYTHON_SELECTION = True
except ImportError:
    pass

__all__ = ["HAS_CORE", "HAS_PYTHON_SELECTION"]
__version__ = "0.1.0"

