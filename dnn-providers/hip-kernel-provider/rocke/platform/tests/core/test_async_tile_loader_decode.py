# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the AsyncTileLoader chunk -> (row, col) decode.

Regression cover for a defect that shipped silently: ``cols_per_chunk`` returned
``elems_per_chunk`` while :meth:`AsyncTileLoaderSlot.issue` used it as the number
of *chunks per tile row*. Those two coincide only when
``tile_cols == elems_per_chunk ** 2`` (tile_cols=64 for a 2-byte dtype at
dwords=4), so tile_cols=64 loaded correctly and every other width decoded to the
wrong row and to a column past the end of the tile. Nothing in the suite caught
it because no test exercised the async path's addressing at all.

These tests are pure integer arithmetic -- no GPU, no torch, no IR build -- so
they run in the fast tier and pin the decode for every admissible tile shape.
"""

from __future__ import annotations

import pytest

from rocke.core.ir import BF16, F16
from rocke.helpers.loads import AsyncTileLoader


# Tile widths that the loader must handle. 64 is the historically-working one;
# the others are the widths that used to be silently corrupted.
_TILE_COLS = (16, 32, 64, 128)
_TILE_ROWS = (32, 64, 128)
_BLOCK_SIZES = (64, 128, 256)


def _loader(tile_rows, tile_cols, block_size, dtype=BF16):
    return AsyncTileLoader.from_tile(
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        block_size=block_size,
        wave_size=64,
        elem_dtype=dtype,
    )


def _decode(loader, chunk_idx):
    """Mirror of the integer decode in AsyncTileLoaderSlot.issue."""
    row = chunk_idx // loader.chunks_per_row
    col = (chunk_idx % loader.chunks_per_row) * loader.elems_per_chunk
    return row, col


class TestChunksPerRow:
    def test_chunks_per_row_tiles_the_row_exactly(self):
        for tr in _TILE_ROWS:
            for tc in _TILE_COLS:
                for bs in _BLOCK_SIZES:
                    try:
                        L = _loader(tr, tc, bs)
                    except ValueError:
                        continue  # shape not admissible for this block size
                    assert L.chunks_per_row * L.elems_per_chunk == tc, (
                        f"tile {tr}x{tc} bs={bs}: chunks_per_row="
                        f"{L.chunks_per_row} * elems={L.elems_per_chunk} != {tc}"
                    )

    def test_regression_not_confused_with_elems_per_chunk(self):
        """The two must differ wherever tile_cols != elems_per_chunk ** 2.

        This is the exact confusion that produced wrong LDS tiles.
        """
        L64 = _loader(64, 64, 256)
        assert L64.chunks_per_row == L64.elems_per_chunk, (
            "tile_cols=64 is the coincidental case where the old and new "
            "divisors agree; if this breaks, the byte-identity of the one "
            "previously-correct async config has moved"
        )
        L32 = _loader(64, 32, 256)
        assert L32.chunks_per_row == 32 // L32.elems_per_chunk
        assert L32.chunks_per_row != L32.elems_per_chunk


class TestDecodeCoversTileExactlyOnce:
    """Every element of the tile must be written exactly once, by one chunk."""

    @pytest.mark.parametrize("tile_cols", _TILE_COLS)
    @pytest.mark.parametrize("dtype", [F16, BF16])
    def test_full_cover_no_overlap_no_oob(self, tile_cols, dtype):
        tile_rows, block_size = 64, 256
        try:
            L = _loader(tile_rows, tile_cols, block_size, dtype)
        except ValueError:
            pytest.skip(f"tile 64x{tile_cols} not admissible at bs={block_size}")

        seen = set()
        for chunk_idx in range(L.chunks_total):
            row, col = _decode(L, chunk_idx)
            assert 0 <= row < tile_rows, (
                f"chunk {chunk_idx} decoded to row {row}, outside "
                f"[0,{tile_rows}) -- the decode overruns the tile"
            )
            assert 0 <= col <= tile_cols - L.elems_per_chunk, (
                f"chunk {chunk_idx} decoded to col {col}, which runs past "
                f"tile_cols={tile_cols} with {L.elems_per_chunk} elems/chunk"
            )
            for e in range(L.elems_per_chunk):
                key = (row, col + e)
                assert key not in seen, f"element {key} written twice"
                seen.add(key)

        assert len(seen) == tile_rows * tile_cols, (
            f"decode covered {len(seen)} of {tile_rows * tile_cols} elements"
        )

    def test_lane_payload_is_lds_contiguous(self):
        """Adjacent chunk indices must land at adjacent LDS byte offsets.

        The intrinsic writes lane-contiguously (lane i at base + i*bytes), so the
        decode is only sound if consecutive chunk indices are consecutive in the
        tile's row-major byte order.
        """
        L = _loader(64, 128, 256)
        for chunk_idx in range(L.chunks_total - 1):
            r0, c0 = _decode(L, chunk_idx)
            r1, c1 = _decode(L, chunk_idx + 1)
            flat0 = r0 * 128 + c0
            flat1 = r1 * 128 + c1
            assert flat1 - flat0 == L.elems_per_chunk, (
                f"chunks {chunk_idx},{chunk_idx + 1} are not adjacent in the "
                f"row-major tile: flat {flat0} -> {flat1}"
            )
