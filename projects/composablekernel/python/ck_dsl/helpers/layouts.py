# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared LDS layout descriptions.

Bank-conflict avoidance on AMDGPU LDS (32 banks, 32-bit each) admits
two strategies:

1. **Row-stride padding** (``LdsLayout.padded_k``): add a few halves
   of trailing dead space so adjacent rows hit different banks. Cheap
   and works for any (rows, cols) but wastes ~6% LDS for a typical
   ``+8`` pad on a 128-element row.

2. **XOR swizzle** (``LdsLayout.xor_*``): permute the byte offset with
   ``off ^ (((off % P) >> S) << R)`` so adjacent thread reads still
   land on different banks without growing the row stride. Closed-form
   per (tile-shape, dtype) pair. These are the canonical AMDGPU LDS
   swizzles used by CK Tile (the ``st_16x32`` / ``st_32x16`` /
   ``st_32x32`` family) and have been independently rediscovered by
   most high-performance AMD matmul libraries.

Both are valid for the sync-load path. The async-DMA path
(``raw_ptr_buffer_load_lds``) writes lane-contiguous LDS, so swizzles
must move into the *consumer* (the MFMA's ds_read address math) —
``validate_for_async`` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# Closed-form XOR swizzle parameters per tile shape, for fp16/bf16 (2-byte)
# data. Read these as: for a byte offset ``off`` into the tile, the
# swizzled offset is ``off ^ (((off % period) >> shift) << bits)``.
# Each entry is bank-conflict-free for the corresponding
# (tile_rows × tile_cols × 2-byte) physical layout when consumed by the
# matching MFMA atom (16x16, 16x32, 32x16, 32x32, ...).
XOR_SWIZZLE_TABLE = {
    # (tile_rows, tile_cols, elem_bytes) -> [(period, shift, bits), ...]
    # Multiple stages compose by left-to-right XOR application.
    (16, 16, 2): [(512, 7, 3)],  # st_16x16_swizzled
    (16, 32, 2): [(1024, 9, 5)],  # st_16x32
    (32, 16, 2): [(1024, 9, 4)],  # st_32x16
    (32, 32, 2): [(1024, 9, 5), (2048, 10, 4)],  # st_32x32 (two-stage)
    (16, 128, 1): [(16 * 128, 8, 4)],  # st_16x128 (fp8)
}


def xor_swizzle_bytes(off_bytes: int, stages) -> int:
    """Apply a multi-stage XOR swizzle to a byte offset.

    Used in unit-tests and in the IR-builder path that emits the
    swizzled LDS-store / LDS-read offsets. The stage list is the
    one stored in :data:`XOR_SWIZZLE_TABLE`.
    """
    result = int(off_bytes)
    for period, shift, bits in stages:
        result ^= ((result % period) >> shift) << bits
    return result


@dataclass(frozen=True)
class LdsLayout:
    """2D LDS layout for fp16 tiles.

    `logical_cols` is the number of useful columns. `k_pad` adds trailing
    padding to the physical row stride. `requires_packed_async` marks layouts
    that can be directly written by `raw_ptr_buffer_load_lds`.

    `swizzle` selects an addressing scheme: ``None`` (identity),
    ``'xor'`` (closed-form bank-permuting XOR per tile shape;
    requires ``swizzle_stages`` to be populated), or ``'cyclic'``
    (legacy placeholder).
    """

    logical_cols: int
    k_pad: int = 0
    swizzle: Optional[str] = None
    requires_packed_async: bool = False
    swizzle_stages: Tuple[Tuple[int, int, int], ...] = ()

    @classmethod
    def padded_k(cls, logical_cols: int, k_pad: int = 8) -> "LdsLayout":
        return cls(logical_cols=int(logical_cols), k_pad=int(k_pad), swizzle=None)

    @classmethod
    def packed_async(cls, logical_cols: int) -> "LdsLayout":
        return cls(
            logical_cols=int(logical_cols),
            k_pad=0,
            swizzle=None,
            requires_packed_async=True,
        )

    @classmethod
    def xor_swizzled(
        cls,
        *,
        tile_rows: int,
        tile_cols: int,
        elem_bytes: int = 2,
    ) -> "LdsLayout":
        """Closed-form XOR swizzle for one of the canonical CK Tile
        tile shapes (16x16, 16x32, 32x16, 32x32, 16x128).

        Picks the swizzle stages from :data:`XOR_SWIZZLE_TABLE` and
        returns an :class:`LdsLayout` with ``k_pad=0`` and
        ``swizzle='xor'``. Caller is responsible for ensuring the
        producer (LDS store / async DMA) and the consumer (ds_read)
        both apply the same swizzle to their byte offsets — see
        :func:`xor_swizzle_bytes`.

        For tile shapes not in the table the caller should fall back
        to :meth:`padded_k` or supply ``swizzle_stages`` directly.
        """
        key = (int(tile_rows), int(tile_cols), int(elem_bytes))
        if key not in XOR_SWIZZLE_TABLE:
            raise ValueError(
                f"no canonical XOR swizzle for tile shape {tile_rows}x{tile_cols} "
                f"with {elem_bytes}-byte elements; supported: "
                f"{sorted(XOR_SWIZZLE_TABLE.keys())}"
            )
        stages = tuple(XOR_SWIZZLE_TABLE[key])
        return cls(
            logical_cols=int(tile_cols),
            k_pad=0,
            swizzle="xor",
            swizzle_stages=stages,
        )

    @property
    def row_stride(self) -> int:
        return self.logical_cols + self.k_pad

    def storage_shape(self, rows: int) -> Tuple[int, int]:
        return int(rows), self.row_stride

    def apply_swizzle_bytes(self, off_bytes: int) -> int:
        """Apply the layout's swizzle (if any) to a byte offset.

        ``None`` and ``'cyclic'`` are identity; ``'xor'`` applies
        :func:`xor_swizzle_bytes` with the stored stages.
        """
        if self.swizzle == "xor" and self.swizzle_stages:
            return xor_swizzle_bytes(off_bytes, self.swizzle_stages)
        return int(off_bytes)

    def validate_for_async(self) -> None:
        if self.k_pad != 0:
            raise ValueError("async LDS layout must be packed: k_pad must be 0")
        if self.swizzle is not None:
            raise ValueError(
                "async LDS layout cannot use arbitrary per-lane swizzle; "
                "express swizzle in consumer read math instead"
            )

    def validate(self) -> None:
        if self.logical_cols <= 0:
            raise ValueError("logical_cols must be positive")
        if self.k_pad < 0:
            raise ValueError("k_pad must be >= 0")
        if self.swizzle not in (None, "xor", "cyclic"):
            raise ValueError(f"unsupported LDS swizzle {self.swizzle!r}")
        if self.swizzle == "xor" and not self.swizzle_stages:
            raise ValueError(
                "xor swizzle requires non-empty swizzle_stages; "
                "use LdsLayout.xor_swizzled(...) to populate it"
            )
