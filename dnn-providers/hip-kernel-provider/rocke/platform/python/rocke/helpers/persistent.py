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
 :mod:`rocke.helpers.streamk` when we ship ).
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
    wave_size: int = 64,
    block_size: int = 64,
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

    Cooperative broadcast (P35 fix)
    -------------------------------

    When ``cooperative`` is True (default), only thread 0 performs the
    atomic; the result must reach every thread in the workgroup so
    they all process the same tile. The historical implementation used
    an LDS slot + ``s_barrier`` for the broadcast, but at
    ``max_iters > 1`` and ``block_size <= 64`` (single-wave CTA) the
    AMDGPU optimizer elided the ``s_barrier`` and ~1.7% of in-range
    body executions were skipped because lanes 1..63 raced ahead of
    lane 0's atomic.

    The fix uses :meth:`IRBuilder.ds_bpermute` for the broadcast.
    ``ds_bpermute(addr=0, val)`` returns lane 0's ``val`` to every
    lane in the wave, with ZERO need for an LDS round-trip or
    ``s_barrier`` — wave-internal cross-lane traffic is sequenced by
    the SIMD pipeline directly.

    For multi-wave workgroups (``block_size > 64``), the ds_bpermute
    is still wave-internal, so we keep the LDS broadcast as a fallback
    when any other wave needs the value. The ``broadcast_slot``
    argument is preserved on the public surface for backwards
    compatibility but is unused on the SGPR-broadcast path.
    """
    from ..core.ir import I32

    if counter_idx is None:
        counter_idx = b.const_i32(0)
    if not cooperative:
        return b.global_atomic_add(counter, counter_idx, b.const_i32(increment))

    tid = b.thread_id_x()
    is_lead = b.cmp_eq(tid, b.const_i32(0))

    # P35 fix: choose a broadcast medium that survives at all
    # ``block_size`` values.
    #
    # * **Single-wave CTA** (``block_size <= wave_size``): use
    #   ``ds_bpermute(0, fetched)`` — every lane issues the atomic
    #   with a per-lane ``increment`` (only lane 0 contributes a
    #   non-zero value), the wave's lane 0 atomic-result is broadcast
    #   to every lane via wave-internal SIMD-pipeline traffic. No
    #   ``s_barrier`` needed (the optimiser had been eliding the
    #   barrier on the single-wave path, causing ~1.7% of in-range
    #   body executions to be skipped at ``max_iters > 1``).
    #
    #   AMDGPU coalesces the 64 lane-atomics targeting the same
    #   address into a single VMEM atomic transaction at the
    #   hardware level so the redundant lane atomics are nearly
    #   free.
    #
    # * **Multi-wave CTA** (``block_size > wave_size``): keep the
    #   LDS slot + ``s_barrier`` shape, but the barrier is REAL
    #   (other waves are observers) so the optimiser cannot elide
    #   it.
    if block_size <= wave_size:
        inc_per_lane = b.select(is_lead, b.const_i32(increment), b.const_i32(0))
        fetched = b.global_atomic_add(counter, counter_idx, inc_per_lane)
        return b.ds_bpermute(b.const_i32(0), fetched)

    if broadcast_slot is None:
        broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
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
    wave_size: int = 64,
    block_size: int = 64,
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
        # one atomic + (ds_bpermute | LDS) broadcast per iteration; the
        # broadcast medium follows the P35 dispatch in
        # :func:`build_persistent_counter_init`.
        next_tile = build_persistent_counter_init(
            b,
            counter,
            counter_idx=counter_idx,
            increment=1,
            cooperative=cooperative,
            broadcast_slot=broadcast_slot,
            wave_size=wave_size,
            block_size=block_size,
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
    wave_size: int = 64,
    block_size: int = 64,
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
    if cooperative and block_size > wave_size:
        # Multi-wave CTA still uses LDS broadcast; allocate the slot
        # once at CTA scope so every iteration reuses it.
        broadcast_slot = b.smem_alloc(I32, [1], name_hint="pers_brd")
    tile_idx0 = build_persistent_counter_init(
        b,
        counter,
        counter_idx=counter_idx,
        cooperative=cooperative,
        broadcast_slot=broadcast_slot,
        wave_size=wave_size,
        block_size=block_size,
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
        wave_size=wave_size,
        block_size=block_size,
    ) as (tile_idx, in_range):
        with b.scf_if(in_range):
            body(tile_idx)
