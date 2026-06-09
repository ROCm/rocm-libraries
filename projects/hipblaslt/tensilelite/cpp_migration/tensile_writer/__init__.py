# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer — C++ migration scaffold for the TensileLite KernelWriter.

The compiled ``_tensile_writer`` nanobind extension hosts the pure subtile
geometry math. This package re-exports it under a stable Python namespace so
callers (and the optional delegation path in
``Tensile.Components.Subtile.SubtileGeometry``) can import
``tensile_writer.subtile.geometry`` without touching the extension directly.
"""

from . import _tensile_writer  # noqa: F401  (compiled nanobind extension)
from . import subtile  # noqa: F401

__all__ = ["subtile"]
