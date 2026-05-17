# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MoE sorting kernels (CK Tile ``13_moe_sorting`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/13_moe_sorting``. Given
the topk-softmax router output ``(topk_ids, topk_weights)`` of shape
``(tokens, topk)``, the kernel rearranges the per-token routing so
every expert receives its assigned tokens in a contiguous block. This
is the standard prerequisite for the per-expert batched GEMMs in a
fused MoE forward.

Pipeline (three kernel launches):

  1. ``moe_sort_histogram``  : count how many tokens each expert
     receives. Reads ``topk_ids[t, k]`` for every ``(t, k)`` pair and
     ``atomic_add(Hist[expert_id], 1)``.
  2. ``moe_sort_scan``       : exclusive prefix sum over ``Hist`` to
     turn per-expert counts into per-expert offsets in the sorted
     output. Single-block kernel; the helper
     :func:`ck_dsl.helpers.block_exclusive_scan_i32` does the work.
  3. ``moe_sort_scatter``    : per ``(t, k)`` pair, claim the next
     slot in ``expert_id``'s bucket via ``atomic_add(Counter[expert_id],
     1)``, then write the token id + weight into that slot. The
     offsets from step 2 turn the local bucket index into the global
     output index.

After the three launches, the outputs are:

* ``Offsets``       : ``(experts,)`` i32 — start of each expert's run.
* ``Counts``        : ``(experts,)`` i32 — number of (token, topk) pairs.
  This is the saved histogram after the scan (so consumers can use it
  either as offsets or as counts).
* ``SortedTokenIds``: ``(tokens * topk,)`` i32 — flat list of token
  ids, expert-major.
* ``SortedTopkIds`` : ``(tokens * topk,)`` i32 — matching topk slot
  (which of the K experts this is for the token; useful for the
  ``y[t,:] += w_k * out[bucket_idx,:]`` reduce later).
* ``SortedWeights`` : ``(tokens * topk,)`` f32 — matching softmax
  weights (passed through from the router).

What we cover today:

* Index dtype : ``i32`` (matches CK Tile's only-supported case).
* Weight dtype : ``f32`` (ditto).
* No persistent variant in v1 — the three-launch path is the natural
  fit for the spec-driven IR builder; the persistent-kernel fused
  variant is a v2 follow-on shared with :mod:`ck_dsl.helpers.persistent`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.scan import (
    block_exclusive_scan_i32,
    lds_zero_i32,
)
from ..helpers.spec import (
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
)


@dataclass(frozen=True)
class MoeSortingSpec:
    """One concrete MoE-sorting configuration.

    ``tokens`` and ``topk`` and ``experts`` are compile-time constants
    so the per-kernel IR can statically size the histogram and the
    grid; the runtime args are the buffer pointers + the shapes (for
    ABI compatibility with the CK Tile reference).
    """

    tokens: int
    topk: int
    experts: int
    block_size: int = 256
    name: str = "ck_dsl_moe_sorting"

    @property
    def total_pairs(self) -> int:
        return self.tokens * self.topk

    def kernel_name(self, phase: str) -> str:
        return kernel_name_join(
            self.name,
            phase,
            f"T{self.tokens}",
            f"K{self.topk}",
            f"E{self.experts}",
            f"b{self.block_size}",
        )


def is_valid_spec(spec: MoeSortingSpec) -> Tuple[bool, str]:
    if spec.tokens <= 0 or spec.topk <= 0 or spec.experts <= 0:
        return False, (
            f"tokens / topk / experts must be > 0 "
            f"(got {spec.tokens}, {spec.topk}, {spec.experts})"
        )
    if spec.experts > 1024:
        return False, f"experts {spec.experts} > 1024 (LDS scan cap)"
    if spec.block_size not in (64, 128, 256, 512, 1024):
        return False, f"block_size {spec.block_size} not in {{64..1024}}"
    if spec.experts > spec.block_size:
        # The single-block scan requires every expert id to map to one
        # lane (length <= block_size). Multi-pass scan would lift this
        # cap but isn't shipped yet.
        return False, (
            f"experts ({spec.experts}) > block_size ({spec.block_size}); "
            "pick a larger block_size or wait for multi-pass scan"
        )
    return True, "ok"


# ---------------------------------------------------------------------
# Kernel 1: histogram
# ---------------------------------------------------------------------


def build_moe_sort_histogram(spec: MoeSortingSpec) -> KernelDef:
    """``Hist[expert_id] += 1`` for every ``(t, k)`` in ``TopkIds``.

    Kernel signature::

        (TopkIds: ptr<i32, global>,   # (tokens, topk) flat
         Hist: ptr<i32, global>,      # (experts,) pre-cleared to 0
         num_pairs: i32,              # = tokens * topk
         num_experts: i32)

    Grid: ``(ceil(num_pairs / block_size), 1, 1)``. Each thread takes
    one (t, k) pair and does one global atomic_add.

    The caller is responsible for clearing ``Hist`` to 0 before the
    launch (we don't fuse the zero pass because CK Tile's reference
    exposes ``clear_workspace_inside_api=true/false`` as a knob).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid moe_sorting spec: {why}")

    b = IRBuilder(spec.kernel_name("hist"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    TopkIds = b.param(
        "TopkIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Hist = b.param("Hist", PtrType(I32, "global"), align=4)
    num_pairs = b.param("num_pairs", I32)
    num_experts = b.param("num_experts", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    pair_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    in_bounds = b.cmp_lt(pair_idx, num_pairs)
    with b.scf_if(in_bounds):
        eid = b.global_load_i32(TopkIds, pair_idx)
        # Guard the expert id against [0, num_experts). Mal-formed
        # router output (an out-of-range id) silently drops the
        # contribution rather than corrupting an unrelated counter.
        valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)), b.cmp_lt(eid, num_experts))
        with b.scf_if(valid_e):
            b.global_atomic_add(Hist, eid, b.const_i32(1))

    return b.kernel


# ---------------------------------------------------------------------
# Kernel 2: exclusive scan (single block)
# ---------------------------------------------------------------------


def build_moe_sort_scan(spec: MoeSortingSpec) -> KernelDef:
    """Exclusive prefix sum over ``Hist[0..experts)`` written to
    ``Offsets[0..experts)``; also copies the unchanged counts to
    ``Counts`` (so consumers can use either offsets or counts).

    Kernel signature::

        (Hist: ptr<i32, global>,        # (experts,) histogram from phase 1
         Offsets: ptr<i32, global>,     # (experts,) exclusive prefix sum out
         Counts: ptr<i32, global>,      # (experts,) copy of Hist (post-scan source)
         num_experts: i32)

    Grid: ``(1, 1, 1)`` — single block of size ``block_size``. The
    helper :func:`block_exclusive_scan_i32` does the in-place scan in
    LDS; we copy in from ``Hist`` and copy out to ``Offsets`` /
    ``Counts`` around it.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid moe_sorting spec: {why}")

    BS = spec.block_size
    E = spec.experts

    b = IRBuilder(spec.kernel_name("scan"))
    b.kernel.attrs["max_workgroup_size"] = BS

    Hist = b.param("Hist", PtrType(I32, "global"), noalias=True, readonly=True, align=4)
    Offsets = b.param("Offsets", PtrType(I32, "global"), writeonly=True, align=4)
    Counts = b.param("Counts", PtrType(I32, "global"), writeonly=True, align=4)
    _ = b.param("num_experts", I32)  # noqa: F841 - matches CK Tile ABI

    tid = b.thread_id_x()
    c_E = b.const_i32(E)
    in_bounds = b.cmp_lt(tid, c_E)

    lds = b.smem_alloc(I32, [E], name_hint="lds_scan")

    # 1) Copy Hist -> LDS (and into Counts unchanged).
    with b.scf_if(in_bounds):
        v = b.global_load_i32(Hist, tid)
        b.smem_store_vN(lds, [tid], v, 1)
        b.global_store(Counts, tid, v, align=4)
    b.sync()

    # 2) In-place exclusive scan in LDS.
    block_exclusive_scan_i32(b, lds, tid=tid, block_size=BS, length=E)

    # 3) Copy LDS -> Offsets.
    with b.scf_if(in_bounds):
        v = b.vec_extract(b.smem_load_vN(lds, tid, dtype=I32, n=1), 0)
        b.global_store(Offsets, tid, v, align=4)

    return b.kernel


# ---------------------------------------------------------------------
# Kernel 3: scatter
# ---------------------------------------------------------------------


def build_moe_sort_scatter(spec: MoeSortingSpec) -> KernelDef:
    """Scatter each ``(t, k)`` pair into ``expert_id``'s bucket.

    Per pair::

        eid          = TopkIds[t * topk + k]
        local_off    = atomic_add(Counter[eid], 1)
        global_off   = Offsets[eid] + local_off
        SortedTokenIds[global_off] = t
        SortedTopkIds[global_off]  = k
        SortedWeights[global_off]  = TopkWeights[t * topk + k]

    Kernel signature::

        (TopkIds: ptr<i32, global>,           # (tokens, topk)
         TopkWeights: ptr<f32, global>,       # (tokens, topk)
         Offsets: ptr<i32, global>,           # (experts,) exclusive scan output
         Counter: ptr<i32, global>,           # (experts,) per-expert next-free, cleared
         SortedTokenIds: ptr<i32, global>,    # (tokens * topk,)
         SortedTopkIds: ptr<i32, global>,     # (tokens * topk,)
         SortedWeights: ptr<f32, global>,     # (tokens * topk,)
         tokens: i32, topk: i32, num_experts: i32)

    Grid: ``(ceil(num_pairs / block_size), 1, 1)``.

    The caller is responsible for clearing ``Counter`` (size
    ``experts`` i32) to 0 before the launch; the histogram pass's
    ``Hist`` buffer is a fine reuse target since it's exhausted by
    that point.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid moe_sorting spec: {why}")

    b = IRBuilder(spec.kernel_name("scatter"))
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    TopkIds = b.param(
        "TopkIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    TopkWeights = b.param(
        "TopkWeights",
        PtrType(F32, "global"),
        noalias=True,
        readonly=True,
        align=4,
    )
    Offsets = b.param(
        "Offsets", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Counter = b.param("Counter", PtrType(I32, "global"), align=4)
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    SortedTopkIds = b.param(
        "SortedTopkIds", PtrType(I32, "global"), writeonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), writeonly=True, align=4
    )
    tokens = b.param("tokens", I32)  # noqa: F841 - ABI
    topk = b.param("topk", I32)
    num_experts = b.param("num_experts", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    pair_idx = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    # Decode (t, k) from the flat pair index. ``topk`` is the inner
    # dim (so ``pair_idx = t * topk + k``).
    t_idx = b.div(pair_idx, topk)
    k_idx = b.mod(pair_idx, topk)

    num_pairs = b.mul(tokens, topk)
    in_bounds = b.cmp_lt(pair_idx, num_pairs)

    with b.scf_if(in_bounds):
        eid = b.global_load_i32(TopkIds, pair_idx)
        valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)), b.cmp_lt(eid, num_experts))
        with b.scf_if(valid_e):
            local_off = b.global_atomic_add(Counter, eid, b.const_i32(1))
            base = b.global_load_i32(Offsets, eid)
            global_off = b.add(base, local_off)

            w = b.global_load_f32(TopkWeights, pair_idx)

            b.global_store(SortedTokenIds, global_off, t_idx, align=4)
            b.global_store(SortedTopkIds, global_off, k_idx, align=4)
            b.global_store(SortedWeights, global_off, w, align=4)

    return b.kernel


# ---------------------------------------------------------------------
# Launch helpers (host-side glue)
# ---------------------------------------------------------------------


def moe_sort_histogram_grid(spec: MoeSortingSpec) -> Tuple[int, int, int]:
    """Grid for phase 1: one CTA per ``block_size``-wide slab of pairs."""
    return ceil_div_grid((spec.total_pairs, spec.block_size))


def moe_sort_scan_grid(spec: MoeSortingSpec) -> Tuple[int, int, int]:
    """Grid for phase 2: single block (the scan fits in one CTA)."""
    return (1, 1, 1)


def moe_sort_scatter_grid(spec: MoeSortingSpec) -> Tuple[int, int, int]:
    """Grid for phase 3: same as phase 1."""
    return ceil_div_grid((spec.total_pairs, spec.block_size))


def moe_sort_histogram_signature(spec: MoeSortingSpec):
    return (
        SignatureBuilder()
        .ptr("TopkIds", "i32")
        .ptr("Hist", "i32")
        .scalar("num_pairs", "i32")
        .scalar("num_experts", "i32")
        .build()
    )


def moe_sort_scan_signature(spec: MoeSortingSpec):
    return (
        SignatureBuilder()
        .ptr("Hist", "i32")
        .ptr("Offsets", "i32")
        .ptr("Counts", "i32")
        .scalar("num_experts", "i32")
        .build()
    )


def moe_sort_scatter_signature(spec: MoeSortingSpec):
    return (
        SignatureBuilder()
        .ptr("TopkIds", "i32")
        .ptr("TopkWeights", "f32")
        .ptr("Offsets", "i32")
        .ptr("Counter", "i32")
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedTopkIds", "i32")
        .ptr("SortedWeights", "f32")
        .scalar("tokens", "i32")
        .scalar("topk", "i32")
        .scalar("num_experts", "i32")
        .build()
    )


def moe_sorting_workspace_bytes(spec: MoeSortingSpec) -> int:
    """Return the GPU workspace size required to run the full MoE sort.

    The workspace holds the histogram (``experts`` i32) reused across
    phase 1 (write) and phase 3 (per-expert next-free counter). The
    caller pre-clears it to zero before phase 1; phase 2 reads it; the
    caller re-clears it between phases 2 and 3.
    """
    return 4 * spec.experts


# Convenience: the underlying ``lds_zero_i32`` helper is re-exported
# for callers building custom MoE pipelines that want to zero their
# own LDS counters from the spec layer (rather than reaching into
# ``ck_dsl.helpers`` directly).
__all__ = [
    "MoeSortingSpec",
    "build_moe_sort_histogram",
    "build_moe_sort_scan",
    "build_moe_sort_scatter",
    "is_valid_spec",
    "lds_zero_i32",
    "moe_sort_histogram_grid",
    "moe_sort_histogram_signature",
    "moe_sort_scan_grid",
    "moe_sort_scan_signature",
    "moe_sort_scatter_grid",
    "moe_sort_scatter_signature",
    "moe_sorting_workspace_bytes",
]
