# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test kernel: a one-wave tile copy through ``load_fragment`` and ``store_fragment``.

A :class:`CopyCase` names a layout and a tensor; :func:`copy_kernel` builds the kernel that loads
the case's tile from ``src`` and stores it to ``dst`` through one window. The tests that check how
the verbs address a lane's run (its IR form, the widths the backend reaches) read it.

Layout: 16 lanes along the stride-1 column axis, each holding ``run`` consecutive columns; the
wave's other lanes stack along the rows, and each lane repeats over ``rows_per_lane`` row blocks,
so register ``r * run + i`` holds row ``lane_row + lane_rows * r`` and column
``lane_column * run + i`` of the tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType, Type
from rocke.helpers.tiling import (
    TensorWindow,
    TileDesc,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)

__all__ = ["LANES_PER_ROW", "CopyCase", "copy_kernel"]

# Lanes side by side along a tile row (the stride-1 column axis), each holding `run` columns.
LANES_PER_ROW = 16


@dataclass(frozen=True)
class CopyCase:
    """One copy kernel: the layout (`run`, `rows_per_lane`, `wave_size`) and the tensor (`rows` x
    `columns`, rows `row_stride` elements apart). `column_origin` places the tile's columns at
    ``block_id_x * tile width`` (``"block"``) or at a kernel parameter ``col0`` (``"param"``, an
    origin whose alignment the compiler cannot prove); its rows are always at
    ``block_id_y * tile height``. A `rows` that is not a multiple of the tile height makes every
    access masked."""

    name: str
    dtype: Type
    run: int
    rows_per_lane: int = 2
    rows: int = 64
    columns: int = 256
    row_stride: int | None = None  # None: the rows are packed, `columns` apart
    column_origin: Literal["block", "param"] = "block"
    wave_size: int = 64

    @property
    def lane_rows(self) -> int:
        """Lanes stacked along the rows."""
        return self.wave_size // LANES_PER_ROW

    @property
    def stride(self) -> int:
        """Elements between consecutive rows."""
        return self.columns if self.row_stride is None else self.row_stride

    def tile(self) -> TileDesc:
        """The tile one wave copies: `lane_rows * rows_per_lane` rows of `LANES_PER_ROW * run`
        columns."""
        return make_tile_desc(
            shape=[self.lane_rows * self.rows_per_lane, LANES_PER_ROW * self.run],
            thread_tile=[1, self.run],
            thread_dist=[self.lane_rows, LANES_PER_ROW],
            block_repeat=[self.rows_per_lane, 1],
            wave_size=self.wave_size,
        )


def copy_kernel(case: CopyCase) -> tuple[KernelDef, TileDesc, TensorWindow]:
    """The copy kernel of `case`, with its tile and the window both verbs access."""
    b = IRBuilder(f"copy_{case.name}")
    b.kernel.attrs["max_workgroup_size"] = case.wave_size
    src = b.param(
        "src", PtrType(case.dtype, "global"), noalias=True, readonly=True, align=16
    )
    dst = b.param(
        "dst", PtrType(case.dtype, "global"), noalias=True, writeonly=True, align=16
    )
    tile = case.tile()
    tile_rows, tile_columns = tile.shape
    row_origin = b.mul(b.block_id_y(), b.const_i32(tile_rows))
    if case.column_origin == "param":
        column_origin = b.param("col0", I32, ge=0, le=case.columns - tile_columns)
    else:
        column_origin = b.mul(b.block_id_x(), b.const_i32(tile_columns))
    tensor = make_tensor_desc((case.rows, case.columns), (case.stride, 1), case.dtype)
    window = make_window(tensor, (row_origin, column_origin))
    lane = b.thread_id_x()
    store_fragment(b, dst, window, load_fragment(b, src, window, tile, lane), lane)
    b.ret()
    return b.kernel, tile, window
