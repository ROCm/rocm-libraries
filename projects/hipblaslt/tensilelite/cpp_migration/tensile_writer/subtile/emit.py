# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.emit — C++-backed subtile emit-leaf decisions.

Thin re-export of the ``_tensile_writer.subtile.emit`` nanobind submodule.
Only the *data-only* decisions of the subtile emit leaves live here (the MFMA
F8F6F4 instType selection); no rocisa objects are constructed in C++. The
Python emit functions in ``Tensile.Components.Subtile`` build the rocisa Module
from these decisions, calling this mapping unconditionally for the supported
F8/F6/F4 cases (there is no Python fallback; unsupported combinations raise).

The single-buffer-load / single-ds-read instruction-shape plans are exposed as
methods on ``tensile_writer.subtile.tile_info.ABTileInfoQuery`` (they reuse the
read-only TileInfo query layer).
"""

from tensile_writer import _tensile_writer as _ext

_emit = _ext.subtile.emit

mfma_f8f6f4_inst_type = _emit.mfma_f8f6f4_inst_type

__all__ = [
    "mfma_f8f6f4_inst_type",
]
