# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared test kernel: a one-wave tile copy through ``load_fragment`` and ``store_fragment``.

A :class:`CopyCase` names a layout and a tensor; :func:`copy_kernel` builds the kernel that loads
the case's tile from ``src`` and stores it to ``dst`` through one window, at origins only known at
run time. :func:`window_copy_kernel` does the same for any tile and tensor at a compile-time origin.
The tests that check how the verbs address a lane's run (its IR form, the widths the backend
reaches) read them.

A :class:`CopyCase`'s layout: 16 lanes along the stride-1 column axis, each holding ``run`` consecutive columns; the
wave's other lanes stack along the rows, and each lane repeats over ``rows_per_lane`` row blocks,
so register ``r * run + i`` holds row ``lane_row + lane_rows * r`` and column
``lane_column * run + i`` of the tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType, Type, Value
from rocke.helpers.tiling import (
    TensorDesc,
    TensorWindow,
    TileDesc,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)

__all__ = ["LANES_PER_ROW", "CopyCase", "copy_kernel", "window_copy_kernel"]

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
    b, source, destination = _copy_kernel_builder(
        f"copy_{case.name}", case.dtype, case.wave_size
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
    _copy_tile(b, source, destination, window, tile)
    return b.kernel, tile, window


def window_copy_kernel(
    name: str,
    tile: TileDesc,
    tensor: TensorDesc,
    origin: tuple[int, ...],
    element_type: Type,
    *,
    wave_size: int = 64,
) -> KernelDef:
    """A one-wave kernel that copies `tile` of `tensor` at the compile-time `origin` from ``src``
    to ``dst``, both pointers to `element_type` (``i32`` for a packed sub-byte tensor, whose
    accesses move 32-bit words)."""
    b, source, destination = _copy_kernel_builder(name, element_type, wave_size)
    _copy_tile(b, source, destination, make_window(tensor, origin), tile)
    return b.kernel


def _copy_kernel_builder(
    name: str, element_type: Type, wave_size: int
) -> tuple[IRBuilder, Value, Value]:
    """A one-wave kernel builder with the copy's ``src`` and ``dst`` parameters."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = wave_size
    source = b.param(
        "src", PtrType(element_type, "global"), noalias=True, readonly=True, align=16
    )
    destination = b.param(
        "dst", PtrType(element_type, "global"), noalias=True, writeonly=True, align=16
    )
    return b, source, destination


def _copy_tile(
    b: IRBuilder,
    source: Value,
    destination: Value,
    window: TensorWindow,
    tile: TileDesc,
) -> None:
    """Load `tile` of `window` from `source`, store it to `destination` and end the kernel."""
    lane = b.thread_id_x()
    store_fragment(
        b, destination, window, load_fragment(b, source, window, tile, lane), lane
    )
    b.ret()
