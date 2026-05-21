# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cooperative block-wide scan + histogram helpers.

CK Tile's MoE-sort and similar bucket-style kernels share three
patterns that aren't covered by the existing
:func:`ck_dsl.helpers.block_lds_reduce`:

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
