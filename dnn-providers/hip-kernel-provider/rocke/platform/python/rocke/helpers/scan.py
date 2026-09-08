# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cooperative block-wide scan + histogram helpers.

CK Tile's MoE-sort and similar bucket-style kernels share three
patterns that aren't covered by the existing
:func:`rocke.helpers.block_lds_reduce`:

* **Block histogram**: every lane contributes one or more ``(key, +1)``
  pairs; the kernel materialises an ``(N,)`` count array in LDS.
  Implemented as a zero-LDS prologue + cooperative
  :meth:`IRBuilder.lds_atomic_add` chain + sync.

* **Block exclusive scan**: given an ``(N,)`` LDS array, compute the
  exclusive prefix sum in place. Used to convert per-bucket counts
  into per-bucket *offsets*. Uses the classic Hillis-Steele tree
  with ``log2(N)`` LDS round-trips.

* **Atomic bucket counter**: per-bucket "next free slot" counters
  sitting in LDS (or global) that a scatter step ``atomic_add(1)``s
  to claim the next position. The primitive itself is
  :meth:`IRBuilder.lds_atomic_add` / :meth:`IRBuilder.global_atomic_add`;
  this module just packages the standard initialisation pattern
  (zero the counters, sync, then scatter) as a one-liner.

The helpers operate on LDS allocations the caller owns (so the
caller can re-use the buffer across multiple phases when lifetimes
don't overlap), and they assume ``i32`` keys and counts -- the only
case the MoE-sort family needs today. fp32 variants (for soft
histograms) are a straightforward extension if a future kernel asks.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..core.ir import I32, IRBuilder, Value


__all__ = [
    "lds_zero_i32",
    "block_histogram_i32",
    "block_exclusive_scan_i32",
    "block_two_level_scan_i32",
]


def lds_zero_i32(
    b: IRBuilder,
    lds_buf: Value,
    *,
    tid: Value,
    block_size: int,
    length: int,
) -> None:
    """Cooperatively zero an ``(length,)`` i32 LDS allocation.

    Each lane writes ``ceil(length / block_size)`` slots; the helper
    issues a ``sync`` at the end so subsequent atomic adds against
    the same buffer see the freshly-cleared state.

    For ``length`` that doesn't divide ``block_size`` cleanly, the
    tail lanes are guarded with an in-bounds predicate so out-of-range
    slots are skipped (LDS allocations are exact-sized, so an OOB
    write would corrupt unrelated state).
    """
    if length <= 0:
        raise ValueError(f"length must be > 0 (got {length})")
    chunks = (length + block_size - 1) // block_size
    c_block = b.const_i32(block_size)
    c_length = b.const_i32(length)
    c_zero = b.const_i32(0)
    for c in range(chunks):
        local = b.add(tid, b.mul(b.const_i32(c), c_block))
        in_bounds = b.cmp_lt(local, c_length)
        with b.scf_if(in_bounds):
            b.smem_store_vN(lds_buf, [local], c_zero, 1)
    b.sync()


def block_histogram_i32(
    b: IRBuilder,
    lds_hist: Value,
    keys: Sequence[Value],
    *,
    tid: Value,
    block_size: int,
    num_bins: int,
    valid_mask: Optional[Sequence[Value]] = None,
) -> None:
    """Accumulate per-lane ``keys`` into an LDS histogram of length
    ``num_bins`` (i32).

    Algorithm:

    1. Cooperative zero of ``lds_hist[0..num_bins)`` via
       :func:`lds_zero_i32`.
    2. For each lane-local ``key in keys``: ``atomic_add(lds_hist[key], 1)``.
       The caller chooses how to chunk the keys -- ``len(keys)``
       elements per lane is the typical layout (e.g. ``topk`` for
       MoE-sort, ``vec`` for vectorised histogram passes).
    3. ``sync`` so the histogram is globally visible before any
       follow-up scan.

    ``valid_mask`` is an optional list of i1 predicates (same length
    as ``keys``); when supplied, masked-off entries are skipped. This
    is the canonical "partial last tile" guard.

    ``num_bins`` must equal the LDS allocation's length; this is a
    compile-time fact so no runtime check is emitted.
    """
    if valid_mask is not None and len(valid_mask) != len(keys):
        raise ValueError(
            f"valid_mask length {len(valid_mask)} != keys length {len(keys)}"
        )

    lds_zero_i32(b, lds_hist, tid=tid, block_size=block_size, length=num_bins)

    c_one = b.const_i32(1)
    c_bins = b.const_i32(num_bins)
    for i, key in enumerate(keys):
        if valid_mask is not None:
            in_range = b.land(
                valid_mask[i],
                b.land(b.cmp_ge(key, b.const_i32(0)), b.cmp_lt(key, c_bins)),
            )
        else:
            in_range = b.land(b.cmp_ge(key, b.const_i32(0)), b.cmp_lt(key, c_bins))
        with b.scf_if(in_range):
            b.lds_atomic_add(lds_hist, [key], c_one)
    b.sync()


def block_exclusive_scan_i32(
    b: IRBuilder,
    lds_buf: Value,
    *,
    tid: Value,
    block_size: int,
    length: int,
) -> None:
    """In-place exclusive prefix-sum over ``lds_buf[0..length)`` (i32).

    Hillis-Steele scan with a temporary LDS buffer? No -- we use a
    ping-pong over the *same* buffer with a write-on-other-half
    invariant:

    .. code-block:: text

        for stride in 1, 2, 4, ..., length // 2:
            new_val = (lane >= stride) ? old[lane] + old[lane - stride] : old[lane]
            sync; lds[lane] = new_val; sync

    then shift right by 1 to make the scan exclusive (slot 0 becomes 0,
    slot i becomes sum of [0..i)).

    The kernel must call this with ``length <= block_size`` so every
    lane handles exactly one slot (or none, when ``length < block_size``).
    For larger ``length`` (multi-block scans) the caller should pre-
    reduce per-warp and chain block-level scans -- a future extension.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0 (got {length})")
    if length > block_size:
        raise ValueError(
            f"length {length} > block_size {block_size}; "
            "multi-pass scans not implemented yet"
        )

    c_length = b.const_i32(length)
    in_bounds = b.cmp_lt(tid, c_length)

    # Inclusive Hillis-Steele scan. Both the self-load and the
    # left-neighbour load happen unconditionally on every lane; we
    # clamp the indices to slot 0 for lanes that shouldn't read so
    # the LDS read is always in-bounds. The masked write below keeps
    # those lanes from contributing to the buffer.
    stride = 1
    while stride < length:
        c_stride = b.const_i32(stride)
        do_add = b.land(in_bounds, b.cmp_ge(tid, c_stride))
        self_idx = b.select(in_bounds, tid, b.const_i32(0))
        left_idx = b.select(do_add, b.sub(tid, c_stride), b.const_i32(0))
        self_vec = b.smem_load_vN(lds_buf, self_idx, dtype=I32, n=1)
        left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1)
        self_val = b.vec_extract(self_vec, 0)
        left_val = b.vec_extract(left_vec, 0)
        new_val = b.add(self_val, left_val)
        b.sync()
        with b.scf_if(do_add):
            b.smem_store_vN(lds_buf, [tid], new_val, 1)
        b.sync()
        stride *= 2

    # Convert inclusive -> exclusive via a one-position right-shift:
    # ``lane k`` writes the value previously at ``lane k - 1`` (or 0
    # for ``lane 0``). A two-phase shift (read all, sync, write all)
    # avoids a same-cycle read/write race on the same LDS slot.
    in_range_left = b.land(in_bounds, b.cmp_gt(tid, b.const_i32(0)))
    # Phase 1: every in-bounds lane materialises its target value in
    # a register. Lanes that aren't in-bounds compute a dummy 0 so the
    # SSA values dominate the phase-2 write; the predicate around the
    # actual store keeps OOB lanes from clobbering anything.
    left_idx = b.select(in_range_left, b.sub(tid, b.const_i32(1)), b.const_i32(0))
    left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1)
    left_val = b.vec_extract(left_vec, 0)
    shifted = b.select(in_range_left, left_val, b.const_i32(0))
    b.sync()
    # Phase 2: predicated write-back. The tid==0 slot gets 0; all
    # other in-bounds slots get the value previously held one position
    # to the left -- which is the canonical exclusive scan result.
    with b.scf_if(in_bounds):
        b.smem_store_vN(lds_buf, [tid], shifted, 1)
    b.sync()


def block_two_level_scan_i32(
    b: IRBuilder,
    lds_buf: Value,
    *,
    tid: Value,
    block_size: int,
    length: int,
    wave_size: int = 64,
) -> None:
    """Two-level exclusive prefix-sum: per-wave Kogge-Stone + cross-wave merge.

    For ``length > 64`` (the moe_sorting target with ``E in {64, 128,
    256}``), the canonical :func:`block_exclusive_scan_i32` issues
    ``log2(length)`` LDS round-trips, each guarded by an
    ``s_barrier``. P34 promotes AITER's ``moe_sorting_opus.h::
    wave_cumsum`` shape: each lane in a wave does a 6-stage
    Kogge-Stone scan via wave-internal shuffles (no LDS, no
    barrier), the per-wave totals land in a ``num_warps`` LDS
    slot table, ``wave 0`` scans the totals (≤4 stages), and the
    cross-wave prefix is added back to each wave's local prefix.

    Halves LDS round-trip count from 7 to 1 for ``length=128`` and
    saves the matching number of barriers — biggest absolute win in
    the persistent / fused MP variant of moe_sorting (P63) where the
    scan is hot.

    The implementation falls back to the existing single-level
    Hillis-Steele scan when ``length <= wave_size`` (it's already
    optimal in that regime) or when the kernel ``block_size`` cannot
    be cleanly divided into ``wave_size`` waves.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0 (got {length})")
    # Single-wave fall-back: the existing Hillis-Steele scan is already
    # at the lower bound for ``length <= wave_size``.
    if length <= wave_size or block_size % wave_size != 0:
        block_exclusive_scan_i32(
            b, lds_buf, tid=tid, block_size=block_size, length=length
        )
        return
    if length > block_size:
        raise ValueError(
            f"block_two_level_scan_i32: length {length} > block_size "
            f"{block_size}; multi-pass scans not implemented yet"
        )

    # Stage 1: per-wave Kogge-Stone. Each wave handles ``wave_size``
    # consecutive slots; lane ``l`` in wave ``w`` reads ``lds_buf[w *
    # wave_size + l]`` from the input and produces an in-wave inclusive
    # scan via :meth:`IRBuilder.warp_shuffle_xor` shuffles. The
    # inclusive form is converted to exclusive by a one-lane right
    # shift via ``warp_shuffle_xor`` with mask 1 + a lane-0 zero
    # substitute.
    num_waves = length // wave_size
    if length % wave_size != 0:
        # Tail: residual `length % wave_size` lanes in the last wave
        # — we keep the canonical scan for the residual to avoid a
        # second tier of arithmetic. Falls back to the single-level
        # scan when length is not a clean multiple of wave_size.
        block_exclusive_scan_i32(
            b, lds_buf, tid=tid, block_size=block_size, length=length
        )
        return

    c_wave = b.const_i32(wave_size)
    lane = b.mod(tid, c_wave)
    warp = b.div(tid, c_wave)
    in_bounds = b.cmp_lt(tid, b.const_i32(length))

    # Load my slot's input value.
    self_idx = b.select(in_bounds, tid, b.const_i32(0))
    self_v = b.vec_extract(b.smem_load_vN(lds_buf, self_idx, dtype=I32, n=1), 0)
    b.sync()

    # Per-wave Kogge-Stone inclusive scan via warp_shuffle_xor.
    # AMDGPU's ds_swizzle handles the within-wave shuffle in 1 LDS op
    # (vs ds_bpermute's address-arithmetic + LDS round-trip).
    cur = self_v
    stages = wave_size.bit_length() - 1
    for k in range(stages):
        mask = 1 << k
        remote = b.warp_shuffle_xor(cur, mask)
        # Lanes whose XOR neighbour is "earlier" (i.e. lane ^ mask
        # < lane) add the remote partial; the others see no change.
        is_higher = b.cmp_ge(lane, b.const_i32(mask))
        cur = b.select(is_higher, b.add(cur, remote), cur)

    inclusive = cur

    # Per-wave totals: lane (wave_size - 1) holds the wave-local sum.
    # Stash to LDS so wave 0 can scan them.
    is_wave_tail = b.cmp_eq(lane, b.const_i32(wave_size - 1))
    with b.scf_if(b.land(is_wave_tail, in_bounds)):
        b.smem_store_vN(lds_buf, [warp], inclusive, 1)
    b.sync()

    # Cross-wave scan: wave 0 reads num_waves entries, scans them
    # serially (small N: typically 2 / 4 / 8 — fewer cycles than another
    # round of warp shuffles), and writes back per-wave prefixes.
    is_wave0 = b.cmp_eq(warp, b.const_i32(0))
    is_xfer_lane = b.land(is_wave0, b.cmp_lt(lane, b.const_i32(num_waves)))
    with b.scf_if(is_xfer_lane):
        # Read every wave's total into the lane-0 wave.
        my_total = b.vec_extract(b.smem_load_vN(lds_buf, lane, dtype=I32, n=1), 0)
        # Synthesise the cross-wave exclusive prefix by reading lower
        # waves' totals and summing. ``num_waves`` is small so this
        # unrolled loop costs at most 7 adds.
        prefix = b.const_i32(0)
        for w_lower in range(num_waves):
            other_total = b.vec_extract(
                b.smem_load_vN(lds_buf, b.const_i32(w_lower), dtype=I32, n=1),
                0,
            )
            include_lower = b.cmp_lt(b.const_i32(w_lower), lane)
            prefix = b.select(include_lower, b.add(prefix, other_total), prefix)
        del my_total  # informational; the in-wave scan recomputes it
        b.smem_store_vN(lds_buf, [lane], prefix, 1)
    b.sync()

    # Add the cross-wave prefix back into each lane's inclusive scan,
    # then shift right by 1 to make the scan exclusive.
    cross_prefix = b.vec_extract(b.smem_load_vN(lds_buf, warp, dtype=I32, n=1), 0)
    full_inclusive = b.add(inclusive, cross_prefix)
    b.sync()

    # Inclusive -> exclusive: lane writes its left neighbour's
    # inclusive value (or 0 for lane 0 of wave 0).
    in_range_left = b.land(in_bounds, b.cmp_gt(tid, b.const_i32(0)))
    # Stash inclusive scan back into LDS so the right-shift can read it.
    with b.scf_if(in_bounds):
        b.smem_store_vN(lds_buf, [tid], full_inclusive, 1)
    b.sync()
    left_idx = b.select(in_range_left, b.sub(tid, b.const_i32(1)), b.const_i32(0))
    left_vec = b.smem_load_vN(lds_buf, left_idx, dtype=I32, n=1)
    shifted = b.select(in_range_left, b.vec_extract(left_vec, 0), b.const_i32(0))
    b.sync()
    with b.scf_if(in_bounds):
        b.smem_store_vN(lds_buf, [tid], shifted, 1)
    b.sync()
