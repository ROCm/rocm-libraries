# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.geometry — C++-backed subtile geometry.

Thin re-export of the ``_tensile_writer.subtile.geometry`` nanobind submodule.
Everything here is pure geometry math (value classes + query methods); no
writer state, register allocation, or instruction emission lives in C++.

The names mirror ``Tensile.Components.Subtile.SubtileGeometry``, whose dataclass
facade forwards its pure-math value/query methods here unconditionally — this
module is the single source of truth for the geometry formulas.
"""

from tensile_writer import _tensile_writer as _ext

_g = _ext.subtile.geometry

# Value / layout classes
LoadShape = _g.LoadShape
MMALayout = _g.MMALayout
MMAScaleLayout = _g.MMAScaleLayout

# Tile geometries
ABGRGeometry = _g.ABGRGeometry
ABLRGeometry = _g.ABLRGeometry
CDTileGeometry = _g.CDTileGeometry
MXScaleGRGeometry = _g.MXScaleGRGeometry
MXScaleLRGeometry = _g.MXScaleLRGeometry

# Tag marker identities
GRTag_1x1 = _g.GRTag_1x1
GRTag_1x2 = _g.GRTag_1x2
GRTag_2x2 = _g.GRTag_2x2
GRTag_TLU1 = _g.GRTag_TLU1
LRTag_1x1 = _g.LRTag_1x1
LRTag_1x2 = _g.LRTag_1x2
LRTag_TLU1 = _g.LRTag_TLU1

# Pre-defined gfx950 layout constants
MFMA_16x16_1B_4K_4V = _g.MFMA_16x16_1B_4K_4V
MFMA_16x16_1B_4K_8V = _g.MFMA_16x16_1B_4K_8V
MFMA_16x16_1B_4N_4V = _g.MFMA_16x16_1B_4N_4V
MFMA_SCALE_16x16_1B_MX32_8V = _g.MFMA_SCALE_16x16_1B_MX32_8V

__all__ = [
    "LoadShape",
    "MMALayout",
    "MMAScaleLayout",
    "ABGRGeometry",
    "ABLRGeometry",
    "CDTileGeometry",
    "MXScaleGRGeometry",
    "MXScaleLRGeometry",
    "GRTag_1x1",
    "GRTag_1x2",
    "GRTag_2x2",
    "GRTag_TLU1",
    "LRTag_1x1",
    "LRTag_1x2",
    "LRTag_TLU1",
    "MFMA_16x16_1B_4K_4V",
    "MFMA_16x16_1B_4K_8V",
    "MFMA_16x16_1B_4N_4V",
    "MFMA_SCALE_16x16_1B_MX32_8V",
]
