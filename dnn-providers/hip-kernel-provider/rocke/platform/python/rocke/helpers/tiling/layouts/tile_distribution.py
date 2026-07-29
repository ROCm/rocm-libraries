# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Quantity-major tile-distribution authoring -- the human-approachable surface.

The public factory is :func:`make_tile_desc`: it takes the geometric quantities AND the tile
shape, so it returns a ready :class:`~rocke.helpers.tiling.fragments.TileDesc` (shape + derived layout) in
one call -- no separate wrap step. A tile is authored as a STRUCT-OF-ARRAYS: one axes-ordered list
per geometric quantity, so the COLUMNS are the LOGICAL matrix axes -- column ``i`` is axis ``i``
in every list. It reads as a geometric table (quantities = rows, axes = columns) instead of six
cross-referenced integer sequences (the raw ``lane_to_rh_*`` / ``register_to_rh_*`` encoding).

Quantities (all per-axis lists unless noted); the surface is symmetric in THREAD (lane) vs WAVE:

    shape            -- the overall tile size per axis.
    thread_tile      -- contiguous elements a thread holds per axis (stride-1 = the vector).
    thread_dist      -- how the wave's LANES spread over each axis; product == wave_size.
    thread_order     -- lane-carrying axes, fastest-moving axis RIGHT-MOST (default = axis order).
    thread_broadcast -- duplicate the tile across LANES (int count, or [size, count]; R in the lane P).
    block_repeat     -- the whole lane tile STAMPED (strided registers) per axis.
    wave_dist        -- how the block's WAVES spread over each axis.
    wave_order       -- wave-carrying axes, fastest-moving axis RIGHT-MOST (default = axis order).
    wave_broadcast   -- duplicate the tile across WAVES (int count, or [size, count]; R in the wave P).

Per axis the extent factors as ``thread_dist * wave_dist * thread_tile * block_repeat`` (checked).

ORDER quantities pick which axis moves fastest, FASTEST-MOVING axis RIGHT-MOST (the standard
stride convention). ``thread_order=[0,1]`` -> thread_1=(0,1); ``[1,0]`` -> thread_1=(1,0). Both
default to axis order (the highest-index axis fastest) -- a sensible default that never needs to be
set for a plain layout. ``wave_order`` is the same for waves and permutes which wave gets which slice.

REGISTER order is CANONICAL (no knob): ``block_repeat`` is the MAJOR register (outer, slowest VGPR),
``thread_tile`` the MINOR (inner, the stride-1 contiguous vector), taken in axis order.

BROADCAST (thread_broadcast / wave_broadcast): thread-only duplication (the encoding's R space).
Int COUNT -> duplicate the WHOLE tile that many times (R at the most-significant lane/wave position:
the "half" split); ``[size, count]`` -> duplicate a block of ``size`` consecutive lanes/waves
``count`` times (``size==1`` puts the copies ADJACENT, e.g. W0==W1). ``size`` must align to a
lane/wave boundary. Cross-wave duplication is a distribution statement (waves hold the same tile);
physically it is an LDS / redundant-load, not a register copy.

Reproduces ck_tile's ``MakeADramTileDistribution`` field-for-field (see the tests).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..encoding import WarpDistributionEncoding
from ..fragments import TileDesc

__all__ = ["make_tile_desc"]

def _axis_tuple(name: str, value: Sequence[int] | None, n_axes: int) -> tuple[int, ...]:
    """Normalize a per-axis quantity to a validated rank-``n_axes`` tuple (``None`` -> all 1s)."""
    if value is None:
        return (1,) * n_axes
    out = tuple(value)
    if len(out) != n_axes:
        raise ValueError(
            f"{name} must have one entry per axis ({n_axes}) -- got {out!r}"
        )
    for entry in out:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise TypeError(f"{name} entries must be ints -- got {out!r}")
        if entry <= 0:
            raise ValueError(f"{name} entries must be positive -- got {out!r}")
    return out

def _parse_broadcast(name: str, value: int | Sequence[int], whole_size: int) -> tuple[int, int]:
    """Return ``(size, count)`` for a broadcast: a bare int -> whole-tile (size = ``whole_size``,
    the full lane/wave extent); else the ``[size, count]`` pair."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int or [size, count] -- got {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} count must be positive -- got {value!r}")
        return whole_size, value
    pair = tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int (count) or a [size, count] pair -- got {value!r}")
    for entry in pair:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0:
            raise ValueError(f"{name} [size, count] entries must be positive ints -- got {value!r}")
    size, count = pair
    return size, count

def _broadcast_insert_index(
    bucket: list[tuple[int, int]],
    hierarchical: list[tuple[int, ...]],
    size: int,
    name: str,
    dist_name: str,
    boundary: str,
) -> int:
    """Index into a partition bucket where R sits so the entries BELOW it multiply to ``size``.
    ``size == 1`` -> least-significant (append, ADJACENT copies); ``size == whole`` -> index 0
    (most-significant, the half split). Must land on a boundary (a trailing run of ``dist_name``)."""
    if size == 1:
        return len(bucket)
    extents = [hierarchical[major - 1][minor] for major, minor in bucket]
    suffix = 1
    for i in range(len(extents) - 1, -1, -1):
        suffix *= extents[i]
        if suffix == size:
            return i
        if suffix > size:
            break
    raise ValueError(
        f"{name} size {size} does not align to a {boundary} boundary -- it must equal the product "
        f"of a trailing run of {dist_name} (extents {extents})"
    )

def _reorder_bucket(
    bucket: list[tuple[int, int]], axes: Sequence[int], name: str, noun: str
) -> list[tuple[int, int]]:
    """Reorder a partition bucket into ``axes`` (axis indices, fastest-moving RIGHT-MOST). Must be
    a permutation of the axes that actually carry that partition (``noun`` = 'threads'/'waves')."""
    order = tuple(axes)
    for axis in order:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError(f"{name} entries must be ints -- got {order!r}")
    present = [major - 1 for major, _ in bucket]
    if sorted(order) != sorted(present):
        raise ValueError(
            f"{name} {order} must be a permutation of the axes that carry {noun} ({sorted(present)})"
        )
    by_axis = {major - 1: (major, minor) for major, minor in bucket}
    return [by_axis[axis] for axis in order]

def make_tile_desc(
    *,
    shape: Sequence[int],
    thread_tile: Sequence[int] | None = None,
    thread_dist: Sequence[int] | None = None,
    thread_order: Sequence[int] | None = None,
    thread_broadcast: int | Sequence[int] = 1,
    block_repeat: Sequence[int] | None = None,
    wave_dist: Sequence[int] | None = None,
    wave_order: Sequence[int] | None = None,
    wave_broadcast: int | Sequence[int] = 1,
    wave_size: int,
) -> TileDesc:
    """Author a :class:`~rocke.helpers.tiling.fragments.TileDesc` (``shape`` + the derived warp-distribution
    layout) from axes-ordered geometric quantities -- one call. See the module docstring for the
    quantity glossary. ``thread_order`` / ``wave_order`` default to axis order (fastest right-most).

    Enforced (fail-fast, in author vocabulary):
      * every quantity list has one entry per axis (== ``len(shape)``);
      * each column factors its extent: ``thread_dist*wave_dist*thread_tile*block_repeat == shape[axis]``;
      * lanes cover the wave exactly: ``product(thread_dist) * thread_broadcast_count == wave_size``;
      * ``thread_order`` / ``wave_order`` (if given) permute the axes that carry lanes / waves;
      * broadcast ``size`` (if a pair) aligns to a lane / wave boundary.
    """
    shape = _axis_tuple("shape", shape, len(tuple(shape)))
    n_axes = len(shape)
    if n_axes == 0:
        raise ValueError("make_tile_desc needs at least one axis -- shape=()")
    thread_dist = _axis_tuple("thread_dist", thread_dist, n_axes)
    wave_dist = _axis_tuple("wave_dist", wave_dist, n_axes)
    thread_tile = _axis_tuple("thread_tile", thread_tile, n_axes)
    block_repeat = _axis_tuple("block_repeat", block_repeat, n_axes)

    for axis in range(n_axes):
        product = thread_dist[axis] * wave_dist[axis] * thread_tile[axis] * block_repeat[axis]
        if product != shape[axis]:
            raise ValueError(
                f"axis {axis}: thread_dist*wave_dist*thread_tile*block_repeat = {product} != shape "
                f"{shape[axis]} -- (thread_dist={thread_dist[axis]}, wave_dist={wave_dist[axis]}, "
                f"thread_tile={thread_tile[axis]}, block_repeat={block_repeat[axis]})"
            )

    lane_product = 1
    for count in thread_dist:
        lane_product *= count
    wave_product = 1
    for count in wave_dist:
        wave_product *= count
    t_size, t_count = _parse_broadcast("thread_broadcast", thread_broadcast, lane_product)
    w_size, w_count = _parse_broadcast("wave_broadcast", wave_broadcast, wave_product)
    if lane_product * t_count != wave_size:
        raise ValueError(
            f"product(thread_dist) * thread_broadcast_count = {lane_product}*{t_count} = "
            f"{lane_product * t_count} != wave_size={wave_size} -- the lanes of a wave "
            f"must cover it exactly"
        )

    # Build Hs per axis in canonical slot order [block_repeat, wave, lane, thread_tile], >1 only.
    hierarchical: list[tuple[int, ...]] = []
    wave_bucket: list[tuple[int, int]] = []
    lane_bucket: list[tuple[int, int]] = []
    block_repeat_bucket: list[tuple[int, int]] = []
    thread_tile_bucket: list[tuple[int, int]] = []
    for axis in range(n_axes):
        major = axis + 1  # major 0 = R; X-dims are 1-indexed
        slots: list[int] = []
        if block_repeat[axis] > 1:
            block_repeat_bucket.append((major, len(slots)))
            slots.append(block_repeat[axis])
        if wave_dist[axis] > 1:
            wave_bucket.append((major, len(slots)))
            slots.append(wave_dist[axis])
        if thread_dist[axis] > 1:
            lane_bucket.append((major, len(slots)))
            slots.append(thread_dist[axis])
        if thread_tile[axis] > 1:
            thread_tile_bucket.append((major, len(slots)))
            slots.append(thread_tile[axis])
        hierarchical.append(tuple(slots))

    # Order overrides: reorder the lane / wave merges (fastest right-most) when the author sets them.
    if thread_order is not None:
        lane_bucket = _reorder_bucket(lane_bucket, thread_order, "thread_order", "threads")
    if wave_order is not None:
        wave_bucket = _reorder_bucket(wave_bucket, wave_order, "wave_order", "waves")

    # R (replication) buckets: thread_broadcast is referenced by the lane P entry, wave_broadcast by
    # the wave P entry. Each gets a distinct minor index in the R space.
    replication: list[int] = []
    thread_r: int | None = None
    wave_r: int | None = None
    if t_count > 1:
        thread_r = len(replication)
        replication.append(t_count)
    if w_count > 1:
        wave_r = len(replication)
        replication.append(w_count)

    # P: the wave entry (only when waves exist or a wave broadcast is present -> NDimP == 2), then
    # the lane entry. Each R bucket is inserted into its partition at place-value `size`.
    lane_to_rh_major: list[tuple[int, ...]] = []
    lane_to_rh_minor: list[tuple[int, ...]] = []
    if wave_bucket or wave_r is not None:
        wave_majors = [major for major, _ in wave_bucket]
        wave_minors = [minor for _, minor in wave_bucket]
        if wave_r is not None:
            at = _broadcast_insert_index(
                wave_bucket, hierarchical, w_size, "wave_broadcast", "wave_dist", "wave"
            )
            wave_majors.insert(at, 0)
            wave_minors.insert(at, wave_r)
        lane_to_rh_major.append(tuple(wave_majors))
        lane_to_rh_minor.append(tuple(wave_minors))
    lane_majors = [major for major, _ in lane_bucket]
    lane_minors = [minor for _, minor in lane_bucket]
    if thread_r is not None:
        at = _broadcast_insert_index(
            lane_bucket, hierarchical, t_size, "thread_broadcast", "thread_dist", "lane"
        )
        lane_majors.insert(at, 0)
        lane_minors.insert(at, thread_r)
    lane_to_rh_major.append(tuple(lane_majors))
    lane_to_rh_minor.append(tuple(lane_minors))

    # Y (registers), MAJOR -> MINOR: block_repeats (axis order) then thread_tiles (axis order), so
    # the thread_tile lands innermost / contiguous (stride-1).
    register = block_repeat_bucket + thread_tile_bucket

    layout = WarpDistributionEncoding(
        replication_lengths=tuple(replication),
        hierarchical_lengths=tuple(hierarchical),
        lane_to_rh_major=tuple(lane_to_rh_major),
        lane_to_rh_minor=tuple(lane_to_rh_minor),
        register_to_rh_major=tuple(major for major, _ in register),
        register_to_rh_minor=tuple(minor for _, minor in register),
    )
    return TileDesc(tuple(shape), layout)
