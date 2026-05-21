# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""StreamK tile partitioner helpers (CK Tile parity).

CK Tile's :class:`StreamKTilePartitioner` decomposes a GEMM workload
into ``(m_tile, n_tile, k_iter)`` triples and decides how each
partial-K accumulation feeds the output. Two reduction strategies are
exposed:

* ``ReductionStrategy.Atomic`` -- each CTA atomically adds its
 partial K-sum into a shared f32 workspace at the output's
 ``(m_tile, n_tile)`` position. Simpler; relies on
 ``global_atomic_add_f32`` (gfx940+). Requires a finalisation pass
 to convert f32 -> output dtype.

* ``ReductionStrategy.Reduction`` -- the CTAs that contribute to the
 same ``(m_tile, n_tile)`` cooperate through a tile-major workspace
 + a flag table; the *last* contributor performs the reduction +
 finalisation in-kernel. More complex but avoids the second launch
 and is the path CK Tile defaults to.

This module ships the partitioner math (a pure-Python view) and the
IR-side glue that
``instances/streamk_gemm.py`` uses. The kernel-shape primitives stay
in :mod:`ck_dsl.helpers.atoms` / :mod:`ck_dsl.helpers.loads` /
:mod:`ck_dsl.helpers.epilogues`; this file is purely about
*partitioning* the work, not about running the per-tile GEMM.

What we ship today:

* :class:`StreamKReductionStrategy` -- enum of supported strategies.
* :class:`StreamKPartition` -- the decoded ``(m_tile, n_tile, k_iter,
 is_first, is_last)`` for a given linear partition id.
* :func:`compute_streamk_grid_size` -- worst-case CTA count for a
 spec.
* :func:`emit_streamk_decode` -- IR-side decode of a linear partition
 id into the SSA ``(m_tile, n_tile, k_iter, is_first, is_last)``
 bundle the kernel needs.

v1 implements the ``Atomic`` strategy end-to-end and the
``Reduction`` strategy's *decode* surface (so the spec layer can
emit the right kernel shape); the per-tile reduction pass that the
``Reduction`` strategy needs lands with the StreamK GEMM kernel
itself in .
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from ..core.ir import IRBuilder, Value


__all__ = [
    "StreamKPartition",
    "StreamKReductionStrategy",
    "compute_streamk_grid_size",
    "emit_streamk_decode",
    "streamk_num_macro_tiles",
]


class StreamKReductionStrategy(Enum):
    """Per CK Tile naming."""

    Atomic = "atomic"  # global_atomic_add_f32 into workspace
    Reduction = "reduction"  # cooperative + flag-table reduction


@dataclass(frozen=True)
class StreamKPartition:
    """The compile-time-fixed shape inputs to the StreamK partitioner.

    The actual ``(m_tile, n_tile, k_iter, is_first, is_last)``
    decoding happens at runtime via :func:`emit_streamk_decode`;
    this dataclass just collects the compile-time constants the
    decode helper needs.
    """

    m_tiles: int  # ceil(M / tile_m); the number of M tiles
    n_tiles: int  # ceil(N / tile_n)
    k_iters: int  # K / tile_k

    @property
    def num_macro_tiles(self) -> int:
        """``m_tiles * n_tiles * k_iters`` -- the total chunk count."""
        return self.m_tiles * self.n_tiles * self.k_iters

    @property
    def k_iters_per_output_tile(self) -> int:
        """``k_iters`` -- the number of CTAs that touch each
        ``(m, n)`` output tile in the Atomic strategy.
        """
        return self.k_iters


class _DecodedTile(NamedTuple):
    """The SSA bundle returned by :func:`emit_streamk_decode`."""

    m_tile: Value  # i32
    n_tile: Value  # i32
    k_iter: Value  # i32 in [0, k_iters)
    is_first: Value  # i1: k_iter == 0
    is_last: Value  # i1: k_iter == k_iters - 1


def streamk_num_macro_tiles(spec: StreamKPartition) -> int:
    """Plain Python view of :attr:`StreamKPartition.num_macro_tiles`."""
    return spec.num_macro_tiles


def compute_streamk_grid_size(
    spec: StreamKPartition,
    *,
    num_cus: int = 304,
    blocks_per_cu: int = 1,
) -> int:
    """Recommended persistent-launch grid size for ``spec``.

    Cap = ``min(num_macro_tiles, num_cus * blocks_per_cu)``. The
    persistent kernel re-fetches macro tiles via atomic_add until
    the global counter exhausts; the launch grid stays small and
    constant so launch overhead is one-shot.

    Default ``num_cus = 304`` matches MI300X (the canonical CK Tile
    deployment target). Override for MI355X (304) or other parts.
    """
    if spec.num_macro_tiles <= 0:
        raise ValueError("spec has zero macro tiles")
    return min(spec.num_macro_tiles, num_cus * blocks_per_cu)


def emit_streamk_decode(
    b: IRBuilder, linear_id: Value, spec: StreamKPartition
) -> _DecodedTile:
    """Decode a linear macro-tile id into ``(m_tile, n_tile, k_iter,
    is_first, is_last)``.

    Layout (matches CK Tile's "K-major within a (m, n) tile" walk):

    .. code-block:: text

    k_iter = linear_id % k_iters
    nn = linear_id // k_iters
    n_tile = nn % n_tiles
    m_tile = nn // n_tiles

    The ``is_first`` / ``is_last`` predicates let the kernel decide
    whether it owns the *seed* of the output (must clear) or the
    *finalize* (must write the converted output dtype to ``C``); the
    middle iterations atomic-add their partial into the workspace.
    """
    c_k_iters = b.const_i32(spec.k_iters)
    c_n_tiles = b.const_i32(spec.n_tiles)

    k_iter = b.mod(linear_id, c_k_iters)
    nn = b.div(linear_id, c_k_iters)
    n_tile = b.mod(nn, c_n_tiles)
    m_tile = b.div(nn, c_n_tiles)

    is_first = b.cmp_eq(k_iter, b.const_i32(0))
    is_last = b.cmp_eq(k_iter, b.const_i32(spec.k_iters - 1))
    return _DecodedTile(
        m_tile=m_tile,
        n_tile=n_tile,
        k_iter=k_iter,
        is_first=is_first,
        is_last=is_last,
    )
