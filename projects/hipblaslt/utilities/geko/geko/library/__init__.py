# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
