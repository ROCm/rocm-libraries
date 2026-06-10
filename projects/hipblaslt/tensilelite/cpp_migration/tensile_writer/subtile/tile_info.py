# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.tile_info — C++-backed TileInfo query layer.

Thin re-export of the ``_tensile_writer.subtile.tile_info`` nanobind submodule.
Everything here is the read-only TileInfo construction + grid/index query layer
(derived grids, load ratios, and index helpers); no writer state, register
allocation, or instruction emission lives in C++.

The names mirror the ABTilePair branch of
``Tensile.Components.Subtile.Kernel.TileInfo``, which delegates its read-only
query methods and emit-leaf plans here unconditionally for the AB case (no
parallel Python formula). The opt-in GR/LR offset-assignment plans
(``grOffsetAssignPlan`` / ``lrOffsetAssignPlan``) are also exposed here.
"""

from tensile_writer import _tensile_writer as _ext

_ti = _ext.subtile.tile_info

ABTileInfoQuery = _ti.ABTileInfoQuery
# Data-only emit-leaf plans returned by ABTileInfoQuery.singleBufferLoadPlan /
# singleDsReadPlan (instruction shape only — no rocisa objects).
SingleBufferLoadPlan = _ti.SingleBufferLoadPlan
SingleDsReadPlan = _ti.SingleDsReadPlan
DsReadEntry = _ti.DsReadEntry
# Data-only offset-assignment scalar math for graTileAssignment /
# lraTileAssignment (B16/TLU0). No rocisa objects, no writer register state.
GROffsetAssignPlan = _ti.GROffsetAssignPlan
LROffsetAssignPlan = _ti.LROffsetAssignPlan

__all__ = [
    "ABTileInfoQuery",
    "SingleBufferLoadPlan",
    "SingleDsReadPlan",
    "DsReadEntry",
    "GROffsetAssignPlan",
    "LROffsetAssignPlan",
]
