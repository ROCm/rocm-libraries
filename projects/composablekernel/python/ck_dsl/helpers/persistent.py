# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Persistent-kernel pattern helper.

CK Tile (and AITER, FlashAttention-3, etc.) commonly launches "persistent"
kernels: a small number of CTAs (~num_cus * waves_per_cu) that pull
their work-items from a global counter via ``atomic_add(1)`` until the
counter exhausts the total tile count. This decouples the launch grid
from the problem size:

* The launch grid is **constant** (sized to the GPU's CU count),
 so the kernel hits steady state immediately and there is no
 per-tile launch overhead.
* Each CTA can dynamically rebalance its work without the scheduler
 reordering tiles -- useful for irregular workloads (StreamK GEMM,
 MoE sort, persistent attention).

The pattern this helper emits:

.. code-block:: python

 # Outside the loop: every CTA grabs its first tile.
 tile_idx_init = b.global_atomic_add(Counter, c_zero, c_one)

 # Bounded scf.for whose trip count is sized to the worst case
 # (ceil(num_tiles / launch_grid_size)); the in-range predicate
 # guards the processing so the final tail iterations are no-ops.
 for tile_idx in persistent_tile_loop(b, counter=Counter, num_tiles=N,
 launch_grid_size=G,
 tile_idx_init=tile_idx_init):
 # process tile at index ``tile_idx``
 ...

The helper handles:

* Sizing the bounded ``scf.for`` so it can statically unroll if the
 caller passes a small ``launch_grid_size`` (compile-time wave
 budget).
* Threading the per-iteration ``tile_idx`` as a loop-carried value
 so each iteration's atomic_add result feeds the next.
* Skipping (via ``in_range``) all over-fetched tiles past
 ``num_tiles``; the counter bumps past ``num_tiles`` are harmless
 for correctness (just spurious atomic traffic at the tail).

What this helper does NOT do:

* Workgroup-level cooperation across CTAs (we don't need it for
 any of the + kernels).
* Tile-level priority scheduling.
* Coarse / fine StreamK splits (those live in
 :mod:`ck_dsl.helpers.streamk` when we ship ).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional

from ..core.ir import IRBuilder, Value


__all__ = [
    "build_persistent_counter_init",
    "persistent_tile_loop",
    "persistent_tile_for_each",
]


def build_persistent_counter_init(
    b: IRBuilder,
    counter: Value,
    *,
    counter_idx: Optional[Value] = None,
    increment: int = 1,
    cooperative: bool = True,
    broadcast_slot: Optional[Value] = None,
) -> Value:
    """Atomic-fetch the first tile id for this CTA from ``counter``.

    Returns an i32 SSA value: the slot's pre-increment value (i.e. the
    tile index this CTA owns first). Caller threads this through the
    :func:`persistent_tile_loop` helper's ``tile_idx_init`` argument.

    ``counter`` is a pointer to an i32 global slot the kernel author
    pre-cleared to 0 from the host side; ``counter_idx`` indexes into
    that buffer (defaults to slot 0). ``increment`` is the per-fetch
    stride -- always 1 for the canonical pattern; values > 1 are for
    fancy "fetch a chunk at a time" variants that we don't ship yet.

    When ``cooperative`` is True (default), only thread 0 performs the
    atomic; the result is broadcast to all threads in the workgroup via
    an LDS slot (``broadcast_slot``, allocated by the caller at CTA
    scope) + ``sync``. Pass ``cooperative=False`` for the "every thread
    is its own worker" variant (rare; the MoE-sort histogram pass uses
    it because each thread owns its own work item).
    """
    from ..core.ir import I32

    if counter_idx is None:
        counter_idx = b.const_i32(0)
    if not cooperative:
        return b.global_atomic_add(counter, counter_idx, b.const_i32(increment))
    if broadcast_slot is None:
        broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
    tid = b.thread_id_x()
    is_lead = b.cmp_eq(tid, b.const_i32(0))
    with b.scf_if(is_lead):
        v = b.global_atomic_add(counter, counter_idx, b.const_i32(increment))
        b.smem_store_vN(broadcast_slot, [b.const_i32(0)], v, 1)
    b.sync()
    return b.vec_extract(
        b.smem_load_vN(broadcast_slot, b.const_i32(0), dtype=I32, n=1),
        0,
    )


@contextmanager
def persistent_tile_loop(
    b: IRBuilder,
    *,
    counter: Value,
    num_tiles: Value,
    max_iters: int,
    tile_idx_init: Value,
    counter_idx: Optional[Value] = None,
    cooperative: bool = True,
    broadcast_slot: Optional[Value] = None,
) -> Iterator[tuple]:
    """Context manager wrapping the persistent-kernel body.

    Yields ``(tile_idx, in_range)``: the current tile id (the
    pre-incremented counter value, threaded as a loop-carried SSA
    value) and an i1 predicate the caller wraps its processing in. The
    helper handles the per-iteration atomic-add that fetches the
    *next* tile id; the caller just consumes ``tile_idx`` and emits
    its processing under ``scf.if(in_range)``.

    ``max_iters`` is the worst-case per-CTA iteration count: typically
    ``ceil(num_tiles_total / launch_grid_size)`` rounded up to the
    next power of two for a clean scf.for trip count. The helper does
    not validate ``max_iters`` against ``num_tiles`` at codegen time;
    the in-range guard makes any over-estimation correct (and any
    under-estimation a latent bug the caller has to catch).

    Usage::

    with persistent_tile_loop(b, counter=Counter, num_tiles=N,
    max_iters=64, tile_idx_init=t0) as (
    tile_idx, in_range
    ):
    with b.scf_if(in_range):
    # process tile at index ``tile_idx``
    ...

    The yielded ``in_range`` is recomputed on every iteration (it
    depends on the loop-carried ``tile_idx``); the caller does not
    need to recompute it itself.
    """
    if counter_idx is None:
        counter_idx = b.const_i32(0)
    for_op = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(max_iters),
        b.const_i32(1),
        [("tile_idx", tile_idx_init)],
        iv_name="pers_iter",
    )
    with for_op as (_iter_v, (tile_idx,)):
        in_range = b.cmp_lt(tile_idx, num_tiles)
        yield tile_idx, in_range
        # After the caller's body, fetch the next tile id for this CTA
        # and yield it as the loop-carried value. Cooperative path uses
        # one atomic + LDS broadcast per iteration.
        next_tile = build_persistent_counter_init(
            b,
            counter,
            counter_idx=counter_idx,
            increment=1,
            cooperative=cooperative,
            broadcast_slot=broadcast_slot,
        )
        b.scf_yield(next_tile)


def persistent_tile_for_each(
    b: IRBuilder,
    *,
    counter: Value,
    num_tiles: Value,
    max_iters: int,
    body: Callable[[Value], None],
    counter_idx: Optional[Value] = None,
    cooperative: bool = True,
) -> None:
    """Functional sugar over :func:`persistent_tile_loop`.

    ``body(tile_idx)`` is invoked once per iteration; the helper wraps
    it in the ``in_range`` guard automatically and handles the
    per-iteration atomic_add bookkeeping.

    Use the context-manager form (:func:`persistent_tile_loop`)
    directly when the caller needs to interleave non-tile work
    (e.g. epilogue prologue) inside the iteration.

    When ``cooperative`` is True (default), the CTA pulls tile ids via
    one atomic + LDS broadcast (so every thread in the workgroup
    processes the same tile in lockstep). The broadcast slot is
    allocated once here at CTA scope and threaded through.
    """
    from ..core.ir import I32

    broadcast_slot = None
    if cooperative:
        broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
    tile_idx0 = build_persistent_counter_init(
        b,
        counter,
        counter_idx=counter_idx,
        cooperative=cooperative,
        broadcast_slot=broadcast_slot,
    )
    with persistent_tile_loop(
        b,
        counter=counter,
        num_tiles=num_tiles,
        max_iters=max_iters,
        tile_idx_init=tile_idx0,
        counter_idx=counter_idx,
        cooperative=cooperative,
        broadcast_slot=broadcast_slot,
    ) as (tile_idx, in_range):
        with b.scf_if(in_range):
            body(tile_idx)
