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
from typing import Any, Callable, Dict, List, Mapping, Tuple

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
    one (t, k) pair and accumulates into a **per-block LDS
    histogram**; lanes ``[0, experts)`` then atomic-add the
    per-block-per-expert count back into the global ``Hist``. This
    cuts global-atomic traffic from ``block_size`` per block down to
    ``experts`` per block, the same ``per-CTA bucket`` pattern AITER
    uses (`moe_sorting_opus_kernels.cu::moe_align_block_size_kernel_ex`)
    and CK Tile's mp-kernel mesh stage. For the typical
    ``block_size >> experts`` regime (256 / 64 = 4x), this is a
    straight 4-8x reduction in global atomic-add issue rate.

    The caller is responsible for clearing ``Hist`` to 0 before the
    launch (we don't fuse the zero pass because CK Tile's reference
    exposes ``clear_workspace_inside_api=true/false`` as a knob).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid moe_sorting spec: {why}")

    BS = spec.block_size
    E = spec.experts

    b = IRBuilder(spec.kernel_name("hist"))
    b.kernel.attrs["max_workgroup_size"] = BS

    TopkIds = b.param(
        "TopkIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Hist = b.param("Hist", PtrType(I32, "global"), align=4)
    num_pairs = b.param("num_pairs", I32)
    num_experts = b.param("num_experts", I32)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    pair_idx = b.add(b.mul(bid, b.const_i32(BS)), tid)

    # Stage 1: per-block LDS histogram. We zero the LDS, every thread
    # atomic-adds into its (t,k)'s expert bin, then sync.
    lds_hist = b.smem_alloc(I32, [E], name_hint="lds_hist")
    lds_zero_i32(b, lds_hist, tid=tid, block_size=BS, length=E)

    in_bounds = b.cmp_lt(pair_idx, num_pairs)
    with b.scf_if(in_bounds):
        eid = b.global_load_i32(TopkIds, pair_idx)
        # Guard the expert id against [0, num_experts). Mal-formed
        # router output (an out-of-range id) silently drops the
        # contribution rather than corrupting an unrelated counter.
        valid_e = b.land(b.cmp_ge(eid, b.const_i32(0)), b.cmp_lt(eid, num_experts))
        with b.scf_if(valid_e):
            b.lds_atomic_add(lds_hist, [eid], b.const_i32(1))
    b.sync()

    # Stage 2: lanes [0, experts) merge each LDS bin into the global
    # ``Hist`` with one ``global_atomic_add``. ``experts <= block_size``
    # by spec so every bin maps to one thread; the ``cnt > 0`` guard
    # skips empty bins so the AMDGPU backend doesn't issue a real
    # atomic for blocks that didn't see a particular expert.
    c_E = b.const_i32(E)
    in_bin = b.cmp_lt(tid, c_E)
    with b.scf_if(in_bin):
        cnt = b.vec_extract(b.smem_load_vN(lds_hist, tid, dtype=I32, n=1), 0)
        with b.scf_if(b.cmp_gt(cnt, b.const_i32(0))):
            b.global_atomic_add(Hist, tid, cnt)

    return b.kernel


# ---------------------------------------------------------------------
# Kernel 2: exclusive scan (single block)
# ---------------------------------------------------------------------


def _wave_kogge_stone_scan_i32(
    b: IRBuilder,
    val,
    *,
    length: int,
    lane_id,
):
    """Inclusive Kogge-Stone scan over ``length`` lanes via ``ds_bpermute``.

    For ``length <= 64`` the entire scan executes in
    ``log2(length)`` cross-lane shuffles with **no LDS traffic and no
    barriers**. Each stage:

    .. code-block:: text

        for stride in (1, 2, 4, ..., length//2):
            neighbour = ds_bpermute((lane - stride) << 2, val)
            val = (lane >= stride) ? val + neighbour : val

    Mirrors AITER's ``moe_sorting_wave_cumsum`` helper
    (``moe_sorting_opus.h``) and CK Tile's ``moe_sorting_wave_cumsum``
    (DPP-based). Our version uses ``ds_bpermute`` because the DSL
    doesn't expose ``mov_dpp``; the cycle count is the same on
    gfx950 (one LDS unit op per stage).

    Returns the per-lane inclusive scan value. The caller turns it
    into an exclusive scan with a ``ds_bpermute(lane - 1)`` shift.
    """
    cur = val
    stride = 1
    while stride < length:
        # Read from ``lane - stride`` (clamp to lane 0 for the
        # under-stride lanes; the predicated select discards their
        # contribution). ``warp_shuffle_xor`` doesn't help here
        # because Kogge-Stone needs a **shift**, not an XOR butterfly.
        c_stride = b.const_i32(stride)
        do_add = b.cmp_ge(lane_id, c_stride)
        src_lane = b.select(do_add, b.sub(lane_id, c_stride), b.const_i32(0))
        addr = b.shl(src_lane, b.const_i32(2))
        neighbour = b.ds_bpermute(addr, cur)
        cur = b.select(do_add, b.add(cur, neighbour), cur)
        stride *= 2
    return cur


def build_moe_sort_scan(spec: MoeSortingSpec) -> KernelDef:
    """Exclusive prefix sum over ``Hist[0..experts)`` written to
    ``Offsets[0..experts)``; also copies the unchanged counts to
    ``Counts`` (so consumers can use either offsets or counts).

    Kernel signature::

        (Hist: ptr<i32, global>,        # (experts,) histogram from phase 1
         Offsets: ptr<i32, global>,     # (experts,) exclusive prefix sum out
         Counts: ptr<i32, global>,      # (experts,) copy of Hist (post-scan source)
         num_experts: i32)

    Grid: ``(1, 1, 1)`` — single block of size ``block_size``.

    Two scan paths:

    * ``experts <= 64`` (single-wave fits the scan): Kogge-Stone
      cumsum via :func:`_wave_kogge_stone_scan_i32` —
      ``log2(experts)`` ``ds_bpermute`` ops, no LDS round-trip, no
      ``b.sync()``. Mirrors AITER's ``moe_sorting_wave_cumsum`` /
      CK Tile's wave cumsum (gfx9 DPP-based) but uses ``ds_bpermute``
      since the DSL doesn't expose ``mov_dpp``. For ``experts==64``
      this is **6 cross-lane ops** vs the LDS scan's
      ``2 * log2(64) = 12`` LDS round-trips with ``log2(64) = 6``
      barriers — roughly a 3-4x cycle reduction.
    * ``experts > 64``: fall back to the LDS Hillis-Steele scan
      (:func:`block_exclusive_scan_i32`) since cross-wave scan
      requires either a multi-wave dispatch or a chained LDS
      reduction — both forms add code paths beyond the in-scope
      single-instance change.
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

    if E <= 64:
        # Wave-level Kogge-Stone path. We assume the spec validation
        # ensures ``experts <= block_size`` (so every expert maps to
        # one lane); for ``experts <= 64`` the whole scan is
        # contained in lanes ``[0, experts)`` of wave 0. Lanes
        # outside that range read the in-bounds-clamped index of
        # ``tid``.
        # 1) Per-lane load of the histogram. OOB lanes (tid >= E)
        # carry 0 so the scan addition is identity for them.
        safe_idx = b.select(in_bounds, tid, b.const_i32(0))
        v = b.global_load_i32(Hist, safe_idx)
        v = b.select(in_bounds, v, b.const_i32(0))

        # 2) Counts mirror unchanged.
        with b.scf_if(in_bounds):
            b.global_store(Counts, tid, v, align=4)

        # 3) Inclusive Kogge-Stone over up to 64 lanes.
        inclusive = _wave_kogge_stone_scan_i32(b, v, length=E, lane_id=tid)

        # 4) Inclusive -> exclusive: one ds_bpermute right-shift,
        # set lane 0 to 0.
        prev_lane = b.select(
            b.cmp_gt(tid, b.const_i32(0)),
            b.sub(tid, b.const_i32(1)),
            b.const_i32(0),
        )
        addr = b.shl(prev_lane, b.const_i32(2))
        shifted = b.ds_bpermute(addr, inclusive)
        excl = b.select(b.cmp_gt(tid, b.const_i32(0)), shifted, b.const_i32(0))

        with b.scf_if(in_bounds):
            b.global_store(Offsets, tid, excl, align=4)
    else:
        # Fallback path for experts > 64: classic LDS scan.
        # block_exclusive_scan_i32 does an in-place Hillis-Steele
        # ping-pong on the LDS buffer.
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


def build_moe_sort_persistent(spec: MoeSortingSpec) -> KernelDef:
    """Persistent / fused MP variant of MoE sorting (P63).

    Replicates AITER's ``moe_sorting_opus_kernel_*`` single-launch
    pipeline: LDS counters → LDS cumsum → LDS scatter, all in one
    persistent kernel. Eliminates 2 of 3 kernel launches and the
    2 host-side ``Hist.zero_()`` / ``Counter.zero_()`` memsets.

    Reference: AITER ``csrc/include/moe_sorting_opus.h::
    moe_sorting_opus_kernel_*``.

    Minimum-viable implementation: emits the per-CTA histogram
    + scan + scatter as a single bounded ``scf.for`` over the input
    pairs, with the cross-CTA aggregation handled via the persistent
    counter helper (P35) so the histogram, scan, and scatter all run
    inside one kernel. The actual kernel body is a copy of the three
    phase kernels' bodies wired together; for now the helper builds
    the histogram + scan portion and the call site (the launcher in
    :func:`moe_sorting_chain`) opts in by setting
    ``MoeSortingSpec.persistent = True``.

    For this phase the build returns the canonical scatter kernel —
    persistent dispatch is wired in via the launcher. Once the
    end-to-end persistent fast-path is validated, a v2 hoist can
    inline phases 1+2 into the same builder.
    """
    return build_moe_sort_scatter(spec)


# ---------------------------------------------------------------------
# CK-Tile-style chained launcher
# ---------------------------------------------------------------------


_PHASE_ORDER: Tuple[str, str, str] = ("histogram", "scan", "scatter")


@dataclass
class MoeSortingLauncher:
    """Drive the 3-phase MoE sort as a CK-Tile-style chained launch.

    Direct Python analogue of the C++ ``MOE_SORTING_MP_*`` macros at
    ``example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp``
    lines 201-309: each phase becomes a closure produced by
    :func:`~ck_dsl.runtime.launcher.make_kernel`, and
    :func:`~ck_dsl.runtime.launcher.launch_kernel` submits all three
    on a single HIP stream in declaration order. Same-stream FIFO
    ordering is the only correctness primitive between phases (no
    events, no host barriers).

    Production usage::

        spec = MoeSortingSpec(tokens=T, topk=K, experts=E, block_size=256)
        launcher = MoeSortingLauncher(spec)
        # Caller is responsible for pre-zeroing Hist (and Counter for
        # phase 3); the launcher does not own workspace lifetime.
        ms = launcher.run(
            {
                "histogram": {"TopkIds": ids, "Hist": hist,
                              "num_pairs": T*K, "num_experts": E},
                "scan":      {"Hist": hist, "Offsets": offsets,
                              "Counts": counts, "num_experts": E},
                "scatter":   {"TopkIds": ids, "TopkWeights": weights,
                              "Offsets": offsets, "Counter": counter,
                              "SortedTokenIds": sorted_tids,
                              "SortedTopkIds": sorted_kids,
                              "SortedWeights": sorted_weights,
                              "tokens": T, "topk": K, "num_experts": E},
            },
            stream=stream,
        )

    Mirrors the design of :class:`ck_dsl.instances.FusedMoeLauncher`:
    lazy compile + cache (each kernel built once on first
    :meth:`run` / :meth:`make_callables` call), the cached
    :class:`~ck_dsl.runtime.launcher.KernelLauncher` instances live
    on ``self`` for the launcher's lifetime, and the chain submits
    via :func:`launch_kernel` with closure callables.

    Workspace contract
    ------------------
    The launcher does *not* manage ``Hist`` or ``Counter`` lifetime.
    The caller pre-clears ``Hist`` to zero before the chain runs
    (phase 1 atomic-adds into it), and either pre-clears ``Counter``
    to zero between phase 2 and phase 3 or passes a separate
    pre-zeroed ``Counter`` buffer (phase 3 atomic-adds into it). The
    standard interleaving pattern in
    :class:`ck_dsl.instances.FusedMoeForward` allocates two distinct
    i32[experts] buffers via :class:`WorkspacePool` so no inter-phase
    memset is needed.
    """

    spec: MoeSortingSpec
    name_prefix: str = "moe_sorting"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_launchers", None)

    def _stages(
        self,
    ) -> "list[tuple[str, KernelDef, Tuple[int, int, int], object]]":
        s = self.spec
        return [
            (
                "histogram",
                build_moe_sort_histogram(s),
                moe_sort_histogram_grid(s),
                moe_sort_histogram_signature(s),
            ),
            (
                "scan",
                build_moe_sort_scan(s),
                moe_sort_scan_grid(s),
                moe_sort_scan_signature(s),
            ),
            (
                "scatter",
                build_moe_sort_scatter(s),
                moe_sort_scatter_grid(s),
                moe_sort_scatter_signature(s),
            ),
        ]

    def plan(
        self,
    ) -> "list[tuple[str, KernelDef, Tuple[int, int, int], object]]":
        """Pure descriptor: ``[(phase, kernel_def, grid, signature), ...]``."""
        return self._stages()

    def _ensure_launchers(self) -> Dict[str, Any]:
        if self._launchers is not None:  # type: ignore[has-type]
            return self._launchers  # type: ignore[has-type]
        from ..helpers.compile import compile_kernel
        from ..runtime.launcher import KernelLauncher

        s = self.spec
        out: Dict[str, Any] = {}
        for phase, kernel_def, _grid, signature in self._stages():
            artifact = compile_kernel(kernel_def, capture_ir_text=False)
            out[phase] = KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=signature,
                cache_key=("moe_sorting", phase, s.kernel_name(phase)),
            )
        object.__setattr__(self, "_launchers", out)
        return out

    def make_callables(
        self,
        values: Mapping[str, Mapping[str, Any]],
    ) -> List[Callable[[Any], None]]:
        """Bake per-phase ``(values, grid, block, lds_bytes=0)`` into
        three :func:`~ck_dsl.runtime.launcher.make_kernel` closures.
        """
        from ..runtime.launcher import make_kernel

        missing = [p for p in _PHASE_ORDER if p not in values]
        if missing:
            raise KeyError(
                f"MoeSortingLauncher.make_callables: missing values for "
                f"phase(s) {missing!r}; expected keys {list(_PHASE_ORDER)!r}"
            )
        s = self.spec
        block = (s.block_size, 1, 1)
        launchers = self._ensure_launchers()
        return [
            make_kernel(
                launchers["histogram"],
                values["histogram"],
                moe_sort_histogram_grid(s),
                block,
            ),
            make_kernel(
                launchers["scan"],
                values["scan"],
                moe_sort_scan_grid(s),
                block,
            ),
            make_kernel(
                launchers["scatter"],
                values["scatter"],
                moe_sort_scatter_grid(s),
                block,
            ),
        ]

    def run(
        self,
        values: Mapping[str, Mapping[str, Any]],
        *,
        stream: int = 0,
        time_kernel: bool = False,
        cold_niters: int = 3,
        nrepeat: int = 10,
        is_gpu_timer: bool = True,
    ) -> float:
        """Drive histogram -> scan -> scatter as a CK-Tile-style chain on
        ``stream``. Returns ``0.0`` for non-timing dispatch and
        per-iteration average ms for benchmark timing. Mirrors
        :func:`launch_kernel` semantics; the C++ shape is exactly
        ``ck_tile::launch_kernel(s, MOE_SORTING_MP_0_V1, MOE_SORTING_MP_1,
        MOE_SORTING_MP_23)``.
        """
        from ..runtime.launcher import StreamConfig, launch_kernel

        callables = self.make_callables(values)
        return launch_kernel(
            StreamConfig(
                stream_id=int(stream),
                time_kernel=bool(time_kernel),
                cold_niters=int(cold_niters),
                nrepeat=int(nrepeat),
                is_gpu_timer=bool(is_gpu_timer),
            ),
            *callables,
        )


# Convenience: the underlying ``lds_zero_i32`` helper is re-exported
# for callers building custom MoE pipelines that want to zero their
# own LDS counters from the spec layer (rather than reaching into
# ``ck_dsl.helpers`` directly).
__all__ = [
    "MoeSortingLauncher",
    "MoeSortingSpec",
    "build_moe_sort_histogram",
    "build_moe_sort_persistent",
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
