# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Sweep iteration helpers, modelled on CK Tile's ``sweep_tile``.

CK Tile's :file:`include/ck_tile/core/tensor/sweep_tile.hpp` exposes a
lambda-iteration helper that "loads X once, sweeps Y positions" -- the
canonical pattern for any kernel that streams a tile through a
register cache, accumulating or scattering on each Y step. From the
:ref:`sweep_tile docs <ck_tile_sweep_tile>`:

  Sweep operations are similar to ``forEach()``. They call a function
  for every data element. Sweep operations use the "load once, use many
  times" pattern: load X data into registers, then sweep through Y
  positions while keeping X in fast memory.

The Python analogue here unrolls a thread's chunked vec-loop into a
lambda body. Without ``sweep_tile`` every small-op kernel had to write::

    for k in range(chunks_per_thread):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        for xi in x_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC):
            ...

With it::

    @sweep_row_chunks(b, x_tile, tid=tid, block_size=BS, vec=VEC,
                      elems_per_thread=elems_per_thread,
                      row=row, dtype=spec.dtype)
    def body(n_off, x_scalars):
        for xi in x_scalars:
            ...

The lambda body is invoked once per chunk; the helper handles the
``n_off`` arithmetic, the vec load, the f32 promotion, and the
``cached`` buffer (when the caller asks for one for a second pass).

This module is opt-in -- the existing :class:`TileWindow` API is the
escape hatch when a kernel needs custom iteration. The
:func:`sweep_row_chunks` and :func:`pass2_row_chunks` helpers cover the
single-axis-per-row patterns used by elementwise / norm / reduce
kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from ..core.ir import IRBuilder, Value
from .tensor_view import TileWindow


__all__ = [
    "RowChunkSweepResult",
    "pass2_row_chunks",
    "sweep_row_chunks",
]


@dataclass
class RowChunkSweepResult:
    """Bookkeeping returned by :func:`sweep_row_chunks`.

    * ``cached`` is the per-thread list of f32 :class:`Value`\\s loaded
      during pass 1. When the caller passes ``cache=True``, the body
      lambda's appended values are recorded here so a follow-up
      :func:`pass2_row_chunks` (or any second pass over the same tile)
      can re-use them without re-reading HBM.

    * ``chunks_per_thread`` is the compile-time chunk count -- handy for
      the caller to size accumulator lists or assert against
      ``elems_per_thread / vec``.
    """

    cached: List[Value]
    chunks_per_thread: int


def sweep_row_chunks(
    b: IRBuilder,
    tile: TileWindow,
    *,
    tid: Value,
    block_size: int,
    vec: int,
    elems_per_thread: int,
    row: Optional[Value] = None,
    body: Optional[Callable[[Value, List[Value]], None]] = None,
    cache: bool = False,
) -> RowChunkSweepResult:
    """Sweep one block's worth of ``(block_size * vec)``-element chunks
    along the row at ``row``, invoking ``body(n_off, x_scalars)`` per
    chunk.

    The :class:`TileWindow` ``tile`` must be 2D with the row dim first;
    each call to ``tile.load_vec_as_f32(b, 0, n_off, n=vec)`` reads a
    contiguous vec-wide slice of the current row, promoted to f32. The
    ``row`` argument shifts the tile's row origin in-place (matching
    CK Tile's ``set_window_origin``); pass ``None`` to leave the
    current origin alone (e.g. when the caller already moved the tile
    to the right row outside the sweep).

    ``body`` is called ``elems_per_thread / vec`` times. The two
    arguments are the per-chunk ``n_off`` SSA :class:`Value` (the
    starting column index along the row) and the list of ``vec`` f32
    scalars freshly loaded. If ``cache=True``, the helper stores those
    scalars in the result's ``cached`` list, in element order across
    chunks -- the standard "pass 1 reads + accumulates + caches; pass 2
    re-uses without re-loading" two-pass pattern.

    ``body`` may be omitted entirely when the caller just wants to
    populate ``cached`` and the body is empty.
    """
    if tile.rank != 2:
        raise ValueError(
            f"sweep_row_chunks expects a 2D TileWindow (got rank {tile.rank})"
        )
    if elems_per_thread % vec:
        raise ValueError(
            f"elems_per_thread ({elems_per_thread}) not divisible by vec ({vec})"
        )

    chunks_per_thread = elems_per_thread // vec
    if row is not None:
        # Shift the tile origin to the target row, keeping the column
        # origin untouched. This is the analogue of CK Tile's
        # ``move_tile_window`` for the 2D-row case.
        old_origin = tile.origin
        tile = TileWindow(
            view=tile.view,
            lengths=tile.lengths,
            origin=(row, old_origin[1]),
        )

    cached: List[Value] = []
    c_vec = b.const_i32(vec)
    for k in range(chunks_per_thread):
        n_off = b.add(
            b.mul(b.const_i32(k * block_size), c_vec),
            b.mul(tid, c_vec),
        )
        x_scalars = tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=vec)
        if cache:
            cached.extend(x_scalars)
        if body is not None:
            body(n_off, x_scalars)
    return RowChunkSweepResult(cached=cached, chunks_per_thread=chunks_per_thread)


def pass2_row_chunks(
    b: IRBuilder,
    tile: TileWindow,
    *,
    tid: Value,
    block_size: int,
    vec: int,
    elems_per_thread: int,
    row: Optional[Value] = None,
    body: Optional[Callable[[Value, int, List[Value]], List[Value]]] = None,
    cached_f32: Sequence[Value] = (),
) -> None:
    """Second-pass sweep that *writes* one block's worth of chunks.

    ``body(n_off, k, x_scalars)`` is invoked ``elems_per_thread / vec``
    times; it returns a list of ``vec`` f32 scalars that the helper
    truncates back to the tile's dtype and stores via
    :meth:`TileWindow.store_vec_from_f32`.

    ``cached_f32`` is the optional per-thread cache returned by
    :func:`sweep_row_chunks` -- when present, the helper indexes into
    it via the k/chunk position so the body can read the pre-loaded
    scalar for the i-th lane as ``cached_f32[k * vec + i]``. This
    matches CK Tile's "load X once / sweep Y positions" idiom on the
    write side.
    """
    if tile.rank != 2:
        raise ValueError(
            f"pass2_row_chunks expects a 2D TileWindow (got rank {tile.rank})"
        )
    if body is None:
        raise ValueError("pass2_row_chunks requires a body callback")
    if elems_per_thread % vec:
        raise ValueError(
            f"elems_per_thread ({elems_per_thread}) not divisible by vec ({vec})"
        )

    if row is not None:
        old_origin = tile.origin
        tile = TileWindow(
            view=tile.view,
            lengths=tile.lengths,
            origin=(row, old_origin[1]),
        )

    chunks_per_thread = elems_per_thread // vec
    c_vec = b.const_i32(vec)
    for k in range(chunks_per_thread):
        n_off = b.add(
            b.mul(b.const_i32(k * block_size), c_vec),
            b.mul(tid, c_vec),
        )
        if cached_f32:
            x_scalars = list(cached_f32[k * vec : (k + 1) * vec])
        else:
            x_scalars = []
        out = body(n_off, k, x_scalars)
        if len(out) != vec:
            raise ValueError(
                f"pass2_row_chunks body must return {vec} f32 scalars (got {len(out)})"
            )
        tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=out)
