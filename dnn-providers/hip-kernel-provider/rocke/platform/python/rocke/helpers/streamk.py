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
IR-side glue that ``instances/streamk_gemm.py`` and the stream-K
implicit-GEMM wgrad conv kernel use. The kernel-shape primitives stay
in :mod:`rocke.helpers.atoms` / :mod:`rocke.helpers.loads` /
:mod:`rocke.helpers.epilogues`; this file is purely about
*partitioning* the work, not about running the per-tile GEMM or its
cross-CTA fixup.

What we ship today:

* :class:`StreamKReductionStrategy` -- enum of supported strategies.
* :class:`StreamKPartition` -- the decoded ``(m_tile, n_tile, k_iter,
 is_first, is_last)`` for a given linear partition id.
* :func:`compute_streamk_grid_size` -- worst-case CTA count for a
 spec.
* :func:`emit_streamk_decode` -- IR-side decode of a linear partition
 id into the SSA ``(m_tile, n_tile, k_iter, is_first, is_last)``
 bundle the kernel needs.
* :class:`StreamKIterPartition` and friends -- CK Tile's
 iteration-balanced (DP + stream-K) partitioner, plus the SSA emitters
 that map a CTA onto its MAC-iteration range.

The ``Reduction`` strategy's cross-CTA fixup (partial slots + a
per-(CTA, wave) flag table, Linear or Tree fan-in) is kernel-specific
and lives with its kernel; the reference implementation is the nested
``_emit_streamk_tiles`` of ``build_implicit_gemm_conv_wgrad`` in
``library/kernels/common/conv_implicit_gemm_wgrad.py`` (C++ twin:
``wgrad_emit_streamk`` in ``conv_implicit_gemm_wgrad.cpp``).
:meth:`StreamKIterPartition.flags_bytes` sizes its flag table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from ..core.ir import IRBuilder, Value
from .spec import align_up, ceil_div


__all__ = [
    "StreamKIterArgs",
    "StreamKIterPartition",
    "StreamKPartition",
    "StreamKReductionStrategy",
    "compute_streamk_grid_size",
    "emit_streamk_decode",
    "emit_streamk_iter_range",
    "emit_streamk_sk_start_iter",
    "streamk_end_iter",
    "streamk_iter_partition",
    "streamk_num_macro_tiles",
    "streamk_start_iter",
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


# ---------------------------------------------------------------------------
# Iteration-balanced stream-K (CK Tile StreamKTilePartitioner)
# ---------------------------------------------------------------------------
#
# The partition above hands every CTA exactly one ``tile_k`` slice. CK Tile's
# production partitioner instead balances *iterations*: output tiles that
# divide evenly across the CTA pool run data-parallel (one CTA per tile, the
# full K loop), and only the remainder is spread over ``max_active_wgs``
# stream-K CTAs, each owning a contiguous range of MAC-loop iterations that
# may straddle output-tile boundaries. rocKE compiles one kernel per shape, so
# the whole partition is resolved host-side and folded into the IR as
# constants; the device only maps its linear CTA id onto an iteration range.


@dataclass(frozen=True)
class StreamKIterPartition:
    """CK Tile ``StreamKTilePartitionerBase``, resolved at build time.

    ``m_tiles`` / ``n_tiles`` count output tiles (a caller folding a batch
    dimension into M passes the folded count). ``iters_per_tile`` is the
    number of ``tile_k`` MAC iterations per output tile. ``max_active_wgs``
    is the CTA pool the stream-K remainder is spread over. It need not fit on
    the device at once: a CTA only waits on higher-index CTAs and publishes
    its own partial before it waits, so with workgroups dispatched in
    ascending order the waiting CTAs cannot starve their producers.
    """

    m_tiles: int
    n_tiles: int
    iters_per_tile: int
    max_active_wgs: int
    persistent: bool = False

    @property
    def num_tiles(self) -> int:
        return self.m_tiles * self.n_tiles

    @property
    def sk_tiles(self) -> int:
        """Output tiles handled by stream-K CTAs (0 means all-DP).

        CK's "DP + 2-tile SK": when the tiles do not divide the CTA pool,
        one full wave of tiles plus the remainder go stream-K so the last
        wave is balanced; a remainder too small to give every stream-K CTA
        at least one iteration falls back to all-DP.
        """
        rem = self.num_tiles % self.max_active_wgs
        if rem == 0:
            return 0
        if self.num_tiles > self.max_active_wgs:
            tiles = self.max_active_wgs + rem
        else:
            tiles = self.num_tiles
        if tiles * self.iters_per_tile < self.max_active_wgs:
            return 0
        return tiles

    @property
    def sk_ctas(self) -> int:
        return self.max_active_wgs if self.sk_tiles else 0

    @property
    def total_sk_iters(self) -> int:
        return self.sk_tiles * self.iters_per_tile

    @property
    def iters_per_sk_cta(self) -> int:
        """Base MAC iterations per stream-K CTA; always >= 1.

        Kernels divide by this value (the owner lookup) on every path,
        including an all-DP launch, so it is clamped to 1 when there are no
        stream-K CTAs. With ``sk_ctas > 0`` it is already >= 1 because
        :attr:`sk_tiles` falls back to all-DP otherwise.
        """
        if not self.sk_ctas:
            return 1
        return self.total_sk_iters // self.sk_ctas

    @property
    def extra_iters(self) -> int:
        """Stream-K CTAs ``[0, extra_iters)`` own one extra iteration."""
        return self.total_sk_iters % self.sk_ctas if self.sk_ctas else 0

    @property
    def dp_tiles(self) -> int:
        return self.num_tiles - self.sk_tiles

    @property
    def total_dp_iters(self) -> int:
        return self.dp_tiles * self.iters_per_tile

    @property
    def grid_size(self) -> int:
        """Launch grid (x): the CTA pool when persistent, else DP + SK CTAs."""
        if self.persistent:
            return self.max_active_wgs
        return self.dp_tiles + self.sk_ctas

    @property
    def max_linear_partners(self) -> int:
        """Upper bound on the CTAs a tile owner has to fold in (Linear).

        The owner is whichever CTA covers the tile's first iteration; it may
        have started in the previous tile, so it is only guaranteed one
        iteration here. Every later contributor covers at least
        ``iters_per_sk_cta`` iterations until the tile ends.
        """
        if not self.sk_ctas:
            return 0
        return ceil_div(self.iters_per_tile - 1, self.iters_per_sk_cta)

    @property
    def tree_rounds(self) -> int:
        """Pairwise fan-in rounds bounding the Tree fixup: ``ceil(log2)`` of
        the most CTAs that can contribute to one tile."""
        contributors = self.max_linear_partners + 1
        return (contributors - 1).bit_length()

    def flags_bytes(self, n_waves: int) -> int:
        """Bytes of the fixup flag table: one i32 flag per (stream-K CTA,
        wave), padded to 256 bytes (CK layout).

        ``n_waves`` is the waves per CTA that publish/wait independently
        (``warp_m * warp_n`` for the wgrad kernel); pass 1 for one flag per
        CTA. 0 for an all-DP partition.
        """
        if n_waves <= 0:
            raise ValueError(f"stream-K flags: n_waves must be > 0 (got {n_waves})")
        return align_up(4 * self.sk_ctas * n_waves, 256)


def streamk_iter_partition(
    *,
    m_tiles: int,
    n_tiles: int,
    iters_per_tile: int,
    max_active_wgs: int,
    persistent: bool = False,
) -> StreamKIterPartition:
    """Validated constructor for :class:`StreamKIterPartition`."""
    for name, v in (
        ("m_tiles", m_tiles),
        ("n_tiles", n_tiles),
        ("iters_per_tile", iters_per_tile),
        ("max_active_wgs", max_active_wgs),
    ):
        if v <= 0:
            raise ValueError(f"stream-K partition: {name} must be > 0 (got {v})")
    return StreamKIterPartition(
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        iters_per_tile=iters_per_tile,
        max_active_wgs=max_active_wgs,
        persistent=persistent,
    )


def streamk_start_iter(part: StreamKIterPartition, sk_cta: int) -> int:
    """Global first MAC iteration of stream-K CTA ``sk_cta``."""
    return (
        part.total_dp_iters
        + sk_cta * part.iters_per_sk_cta
        + min(sk_cta, part.extra_iters)
    )


def streamk_end_iter(part: StreamKIterPartition, sk_cta: int) -> int:
    """One past the last MAC iteration of stream-K CTA ``sk_cta``."""
    return (
        streamk_start_iter(part, sk_cta)
        + part.iters_per_sk_cta
        + (1 if sk_cta < part.extra_iters else 0)
    )


class StreamKIterArgs(NamedTuple):
    """The runtime stream-K partition a shape-agnostic kernel reads.

    The host computes a :class:`StreamKIterPartition` per launch and passes
    these fields as kernel arguments; the device emitters below only ever
    see them as SSA values.
    """

    iters_per_tile: Value  # i32 MAC iterations per output tile
    dp_tiles: Value  # i32 tiles owned whole by data-parallel CTAs
    total_dp_iters: Value  # i32 dp_tiles * iters_per_tile
    iters_per_sk_cta: Value  # i32 base iterations per stream-K CTA (>= 1)
    extra_iters: Value  # i32 stream-K CTAs that own one more iteration


def emit_streamk_sk_start_iter(
    b: IRBuilder, sk_cta: Value, args: StreamKIterArgs
) -> Value:
    """SSA :func:`streamk_start_iter` for a runtime stream-K CTA index."""
    body = b.mul(sk_cta, args.iters_per_sk_cta)
    lead = b.smin(sk_cta, args.extra_iters)
    return b.add(args.total_dp_iters, b.add(body, lead))


class _IterRange(NamedTuple):
    """The SSA bundle returned by :func:`emit_streamk_iter_range`."""

    start: Value  # i32 global first iteration (SGPR)
    end: Value  # i32 one past the last iteration (SGPR)
    sk_cta: Value  # i32 stream-K CTA index; meaningless for a DP CTA


def emit_streamk_iter_range(
    b: IRBuilder, cta: Value, args: StreamKIterArgs, *, dp: bool
) -> _IterRange:
    """Map a CTA onto its contiguous range of global MAC iterations.

    ``dp=True`` (non-persistent launch): CTAs ``[0, dp_tiles)`` each own one
    whole tile and the rest are stream-K CTAs, selected branch-free so the
    GEMM body is emitted once. ``dp=False``: ``cta`` already is a stream-K
    CTA index (the persistent launch runs its DP sweep separately).
    """
    if dp:
        sk_cta = b.sub(cta, args.dp_tiles)
    else:
        sk_cta = cta
    sk_start = emit_streamk_sk_start_iter(b, sk_cta, args)
    c_one = b.const_i32(1)
    c_zero = b.const_i32(0)
    sk_len = b.add(
        args.iters_per_sk_cta,
        b.select(b.cmp_lt(sk_cta, args.extra_iters), c_one, c_zero),
    )
    if dp:
        is_dp = b.cmp_lt(cta, args.dp_tiles)
        start = b.select(is_dp, b.mul(cta, args.iters_per_tile), sk_start)
        length = b.select(is_dp, args.iters_per_tile, sk_len)
    else:
        start = sk_start
        length = sk_len
    start = b.to_sgpr_u32(start)
    end = b.to_sgpr_u32(b.add(start, length))
    return _IterRange(start=start, end=end, sk_cta=sk_cta)
