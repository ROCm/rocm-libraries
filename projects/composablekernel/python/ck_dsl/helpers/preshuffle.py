# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Preshuffled-B layout helpers.

CK Tile's preshuffled-B GEMM family (``03_gemm`` Preshuffle pipeline,
``18_flatmm``, the ``38_block_scale_gemm`` ``preshuffleb`` columns)
reorders the B-matrix tiles on the host side into a "tile-major"
layout::

 canonical [N, K] row-major; per-element stride (K, 1)
 preshuffled [N_outer, K_outer, N_inner, K_inner]
 with (N_inner * K_inner) == (block_n * block_k)
 packed contiguously

so the per-K-iter load issues *one* aligned ``buffer_load_dwordx4``
per warp (no scattered ds_write strides on the load side, no
``ds_read_b32`` strides on the consumer side). The downside is a host-
side permutation of the B matrix; the upside is a 1.3-1.8x speedup on
fp8 / bf8 / i4 GEMMs where the standard pipeline is bandwidth-bound on
the consumer-side LDS reads.

This module is the *layout descriptor* + load-distribution helper. It
does not own the per-element load -- that lives in
``CoalescedTileLoader`` / ``AsyncTileLoader`` -- but it does compute
the per-thread byte offset into the preshuffled buffer.

What v1 ships:

* :class:`PreshuffleBSpec`: tile shape + element bytes; computes the
 per-tile element stride into the preshuffled layout.
* :func:`emit_preshuffleb_offset`: per-lane byte offset for one
 ``(n_tile, k_tile, n_in_tile, k_in_tile)`` quad.
* :func:`host_preshuffle_layout`: pure-Python description of the
 layout for the host-side permutation pass (returns a numpy-style
 shape + strides tuple the launcher can use to build the
 permuted-B torch tensor).

Pairs with the FP8 / BF8 MFMA atoms in :mod:`ck_dsl.helpers.atoms`:
the per-lane operand width (``a_per_lane`` / ``b_per_lane`` = 8 fp8 /
bf8 bytes = ``<2 x i32>``) determines the contiguous-load size the
preshuffle layout has to package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import IRBuilder, Value


__all__ = [
    "PreshuffleBSpec",
    "emit_preshuffleb_offset",
    "host_preshuffle_layout",
]


@dataclass(frozen=True)
class PreshuffleBSpec:
    """One concrete preshuffled-B tile shape.

    ``block_n`` / ``block_k`` are the GEMM block tile sizes. The
    preshuffled layout packs each ``(block_n, block_k)`` tile
    contiguously, so the per-block byte stride is
    ``block_n * block_k * elem_bytes``.
    """

    block_n: int
    block_k: int
    elem_bytes: int = 1  # 1 for fp8/bf8/i8; 2 for f16/bf16; 0.5 for i4 (use 1 with 2-per-byte packing)

    @property
    def tile_bytes(self) -> int:
        """``block_n * block_k * elem_bytes`` -- bytes per preshuffled tile."""
        return self.block_n * self.block_k * self.elem_bytes


def emit_preshuffleb_offset(
    b: IRBuilder,
    spec: PreshuffleBSpec,
    *,
    n_tile: Value,
    k_tile: Value,
    n_in_tile: Value,
    k_in_tile: Value,
    n_tile_count: Value,
) -> Value:
    """Per-lane byte offset for one ``(n_tile, k_tile, n_in_tile, k_in_tile)`` quad.

    Layout (matches CK Tile's preshuffle-B writer):

    .. code-block:: text

    offset = (k_tile * n_tile_count + n_tile) * tile_bytes
    + (n_in_tile * block_k + k_in_tile) * elem_bytes

    The outer dim is K (so consecutive K-tiles in the K-loop touch
    contiguous tiles), then N, then the per-tile (N, K) layout. This
    is the canonical order CK Tile's ``permute_b`` writer produces.

    ``n_tile_count`` is the number of N-tiles per block-N stripe
    (i.e., ``ceil(N / block_n)`` in the standard non-stripe case;
    larger when the spec's stripe size differs).
    """
    c_tile_bytes = b.const_i32(spec.tile_bytes)
    c_block_k = b.const_i32(spec.block_k)
    c_elem_bytes = b.const_i32(spec.elem_bytes)

    tile_id = b.add(b.mul(k_tile, n_tile_count), n_tile)
    tile_base = b.mul(tile_id, c_tile_bytes)
    inner = b.add(b.mul(n_in_tile, c_block_k), k_in_tile)
    inner_bytes = b.mul(inner, c_elem_bytes)
    return b.add(tile_base, inner_bytes)


def host_preshuffle_layout(
    spec: PreshuffleBSpec, *, n: int, k: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Pure-Python layout descriptor for the host-side permutation.

    Returns ``(shape, strides)`` for a torch / numpy view of the
    preshuffled buffer::

    shape = (k_tiles, n_tiles, block_n, block_k)
    strides = (n_tiles * block_n * block_k,
    block_n * block_k,
    block_k,
    1)

    The launcher uses this to build a contiguous-after-permute torch
    tensor: ``b_pre = b_natural.unfold(...).permute(...).contiguous()``.
    """
    n_tiles = (n + spec.block_n - 1) // spec.block_n
    k_tiles = (k + spec.block_k - 1) // spec.block_k
    if n_tiles * spec.block_n != n or k_tiles * spec.block_k != k:
        raise ValueError(
            f"preshuffle requires N / K to divide block_n / block_k "
            f"(got N={n}, block_n={spec.block_n}, "
            f"K={k}, block_k={spec.block_k})"
        )
    shape = (k_tiles, n_tiles, spec.block_n, spec.block_k)
    strides = (
        n_tiles * spec.block_n * spec.block_k,
        spec.block_n * spec.block_k,
        spec.block_k,
        1,
    )
    return shape, strides
