################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

"""
Library module.

This module provides tools to manage and manipulate Tensile solution libraries including
loading, merging, and creating optimized GEMM solution libraries. It handles YAML
manipulation, solution library operations, and integration with the Tensile framework.

The library module enables the final step of the optimization workflow by merging
individual optimized solutions into hipBLASLt libraries.

Modules:
    library: Defines the Library and LibraryCollection classes.
    operations: Functions for loading, merging optimized solutions, creating, merging, and others.

Example:
    >>> from library import Library, LibraryCollection
    >>> from library import operations
    >>> lib = operations.load_library("path/to/lib.yaml")
    >>> collection = operations.load_collection("path/to/folder")
    >>> merged = operations.merge_solutions("path/to/folder")
"""

from .library import *
from .operations import *
