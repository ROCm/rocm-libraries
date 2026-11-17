# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The implicit-GEMM convolution AOT grid: which kernels the cache builds, and
how one of them is built.

A :class:`BuildJob` is one kernel: a :class:`KernelIdentity` plus the spec
kwargs and capability axes that produce it. The generators below walk the
grid (``CACHE_*``) and yield jobs; :func:`job_for_identity` inverts them for one
identity and :func:`kernel_for_job` turns a job into kernel IR. Nothing here
compiles, caches or launches -- that is :mod:`benchmarks.common.kernel_sweep`
-- so the dispatcher (:mod:`dispatch.grouped_convolution`) can build any
registered instance without reaching above its layer.

What the implicit-GEMM cache sweeps
-----------------------------------
One binary per point of the grid below that passes its direction's spec
validator (``--compile-all`` prints the per-direction breakdown of what
survived). Values are the ``CACHE_*`` constants of this module; the dtype is
fp16 or bf16 (one cache pass per dtype, A/B/D all of that dtype).

Base tile geometry, every direction (wgrad adds to it, see below)::

    tile_m, tile_n   16, 32, 64, 128, 256
    tile_k           16, 32, 64
    warp_m, warp_n   1, 2, 4, 8   (warp_m * warp_tile must divide tile_m, same for n)
    warp_tile_m/n    16, 32       (square; warp_tile_k = the widest MMA atom
                                   for the dtype, which must divide tile_k)
    accumulator      (tile_m/warp_m) * (tile_n/warp_n) / wave_size <= 256
                     f32 registers per lane (CACHE_MAX_ACC_REGS; above that
                     the kernel spills and is the slowest to compile)
    epilogue         default, cshuffle
    pipeline         mem, compv3, compv4, wavelet (wavelet: gfx1250 WMMA only;
                     "basic" is never cached -- it emits mem's code)

Per direction, on top of that:

* **fwd** -- vector widths: a = b in {1, 2, 4, 8} (X and W run along cpg),
  c in {1, 2, 4, 8} (Y along kpg). K loop: plain, ``unroll_k`` (double-
  buffered 2x loop), and ``async_dma`` (direct-to-LDS; built once per
  geometry, under pipeline "mem", since it ignores the pipeline). Plus
  group-merged depthwise kernels (``group_merge`` in 4..64, cshuffle, plain
  and ``unroll_k`` loops, load widths derived by the builder, on the tile
  subset the ``CACHE_FWD_GROUP_MERGE_*`` constants allow), served only to
  depthwise problems whose merged GEMM-N fits the tile.
* **wgrad** -- tile_m/tile_n also take 144 and 192, and warp_m/warp_n also
  take 3 and 6 (``CACHE_WGRAD_EXTRA_*``): wgrad's GEMM N is Z*Y*X*cpg, which
  carries a factor of 9 for a 3x3 filter, and the base ladder steps by 2x, so
  it has nothing to offer between 128 and 256. Vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (X and dW along cpg); with the default epilogue (all
  two-stage kernels) c = 1 and b in {1, 2, 4, 8}. pipeline "wavelet" is not cached (wgrad
  builds it as mem). Both the non-split kernel (``split_k`` = 1) and the
  split-K kernel (``split_k`` > 1, recorded as 2; the degree is a kernarg and
  the benchmark sweeps it at launch) are cached, the latter as atomic or
  two-stage (f32 scratch + Stage-2 reduce, ``ws_replicas`` = 8). Plus
  group-merged depthwise kernels (``group_merge`` in 2..64, split-K
  two-stage only, load widths derived by the builder), served only to
  depthwise problems whose merged GEMM fits the tile. ``lds_k_outer`` and
  ``prefetch_addr`` follow ``WgradConvSpec.default_lds_k_outer`` /
  ``default_prefetch_addr`` per geometry.
  Opt-in (``streamk_modes``; ``--compile-all --streamk``): stream-K
  kernels, default epilogue only, ``split_k`` = 1, c = 1. At the widest
  vector width W of the dtype (a = W; b in {W, 1}): one per launch mode
  (``dp_sk``, ``persistent``) x reduction (``linear``, ``workspace`` with
  ``ws_replicas`` = 8, and ``atomic`` where dW is fp32). Fully scalar
  (a = b = 1), on tiles with tile_n >= 128 only: ``dp_sk`` x (``tree``,
  ``workspace``), for large-filter depthwise layers. The CTA pool is a
  kernarg, so it is not swept here.
* **dgrad** -- vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (W and dX along cpg); with the default epilogue c = 1 and b in
  {1, 2, 4, 8}. One tilde binary per configuration serves every stride and
  dilation: the tilde decomposition is computed on the host and passed in
  the sub-GEMM record buffer, up to 1024 sub-GEMMs (stride 32x32). Next to
  it, on wave64 MFMA targets outside the wavelet pipeline, a stride-1 /
  dilation-1 binary of the same configuration (``CACHE_DGRAD_STRIDE1``),
  which rebuilds its single sub-GEMM record from the kernargs instead of
  searching and loading the buffer, stores through the unit-stride
  epilogue, and runs the tap-outer K loop (``tap_outer_k``, in place of
  ``uniform_k_decode``; served to problems whose kpg is a multiple of
  tile_k). A stride-1 binary with the flat K loop is not built: it serves
  the same problems as the tilde binary of its configuration with the same
  K loop. On gfx950 a capped sub-grid adds the dY halo mode of the tap-outer
  loop (``dy_halo`` = 2): the three halo tiles (``CACHE_DGRAD_HALO_TILES``)
  x square filters 3/5/7 (baked, ``filter_h/w``) x the widest output row
  served (``halo_max_wo`` in ``CACHE_DGRAD_HALO_MAX_WO``) x the
  ``dy_halo_setprio`` / ``dy_halo_kouter_pad`` pairs of the tile
  (``_dgrad_halo_knobs``), K-outer only (b = c in {2, 4, 8}, a = 8). The B
  tile takes the layout the JIT sweep
  picks (``DgradConvSpec.default_lds_k_outer``): K-outer wherever the arch,
  dtype, atom and pipeline allow it and the B load is wider than one element,
  M-outer otherwise. ``uniform_k_decode`` follows
  ``DgradConvSpec.default_uniform_k_decode``.

Every direction builds the grouped kernel only: it serves groups == 1 as well
(``CACHE_GROUPED``), so there is no separate ungrouped binary.

Not swept (spec defaults): fwd ``lds_k_outer`` (off), wgrad ``async_dma``,
``chiplet_swizzle``, ``lds_layout``, ``waves_per_eu``, accumulator epilogues,
``cshuffle_no_alias``, the pointwise fast path (1x1 problems run the general
kernels) and 3-D convolution. The direct-conv cache sweeps its own grid: see
:mod:`benchmarks.common.direct_kernel_sweep`.

Two more are not swept but are not left at the spec default either -- both are
pure functions of the geometry, so they are derived rather than searched.
wgrad ``lds_k_outer`` comes from
:meth:`~kernels.common.conv_implicit_gemm_wgrad.WgradConvSpec.default_lds_k_outer`,
and ``lds_k_pad`` from
:meth:`~kernels.common.conv_implicit_gemm_wgrad.WgradConvSpec.default_lds_k_pad`,
which supplies the bank-conflict-free row stride for the M-outer tile and
stands aside (returning ``None``) wherever ``default_lds_k_outer`` chose the
K-outer one, since that layout takes its stride from the builder.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from kernels.conv.kernel_identity import KernelIdentity, current_llvm_flavor


# ---------------------------------------------------------------------------
# Implicit-GEMM AOT configuration grid
#
# These axes match the JIT sweep in benchmark_implicit_gemm_conv.py (minus the
# gfx1250-only extremes). An AOT cache is compiled once, so every axis value
# that can produce the fastest kernel on the target must be present --
# a value omitted here is permanently missing from --run-from-cache regardless
# of what the benchmark found.
#
# GFX1250-only extensions (tile=512, warp=16) are omitted: they require a
# minimum register file that gfx950 and gfx942 do not have, so those kernels
# fail spec validation and are silently skipped on those targets. The grid
# below does build for gfx1250 (WMMA: "mem" and "wavelet" pipelines, group-
# merged depthwise kernels included); only those extensions are left out.
# ---------------------------------------------------------------------------

CACHE_TILE_MN: Tuple[int, ...] = (16, 32, 64, 128, 256)
CACHE_TILE_K: Tuple[int, ...] = (16, 32, 64)
CACHE_WARP_MN: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_WARP_TILE_MN: Tuple[int, ...] = (16, 32)

# ---- wgrad-only geometry extension ----------------------------------------
# wgrad is the one direction whose GEMM dims are filter-shaped rather than
# tensor-shaped: N = Z*Y*X*cpg and M = kpg. For a 3x3 filter N carries a
# factor of 9, which the power-of-two ladder has no way to represent, and the
# ladder's steps are 2x apart -- so between 128 and 256 it offers nothing at
# all. 144 and 192 fill that gap with the multiples of 16 that a 9*cpg N
# actually lands on.
#
# Both failure modes this fixes were measured, and they are not the same one:
#
#  * cpg = 48 (N = 432): no power of two divides it, so every base-grid
#    kernel runs a partial tile on the N edge and the waves on it do
#    full-tile work for a fraction of a tile of output.
#  * cpg = 128 (N = 1152): 128 divides it exactly and the edge is clean, yet
#    192 still wins -- 6 fatter tiles beat 9 thinner ones on grid shape and
#    per-workgroup arithmetic intensity, which divisibility says nothing
#    about.
#
# So do not read these as "edge tiles for indivisible shapes". The ladder is
# simply too coarse for this direction, and clean division is one of several
# reasons a 1.5x step can win.
#
# Warps 3 and 6 come with them rather than separately: a 144 tile is only
# reachable with warp_m in (1, 3) at warp_tile 16 (144 % (2*16) != 0), so the
# odd splits are what makes the new edges usable at all. Adding the warps
# without the tiles changes nothing -- no power-of-two tile divides by 3.
#
# fwd and dgrad keep the base grid: their GEMM dims come from N*Ho*Wo and C,
# which have no factor of 9 to chase, and widening all three directions costs
# several times the disk for no measured gain.
CACHE_WGRAD_EXTRA_TILE_MN: Tuple[int, ...] = (144, 192)
CACHE_WGRAD_EXTRA_WARP_MN: Tuple[int, ...] = (3, 6)
# Cap on the f32 accumulator registers each lane holds:
# (tile_m / warp_m) * (tile_n / warp_n) / wave_size. Above 256 the
# accumulators no longer fit the 256 AGPRs MFMA accumulates into, so the
# kernel spills -- it is not a contender, and those are the slowest kernels in
# the grid to compile (several seconds each, against a fraction of a second
# for the rest). Not a validity rule: the JIT sweep still builds them.
CACHE_MAX_ACC_REGS = 256
CACHE_VECS: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_PIPELINES: Tuple[str, ...] = ("mem", "compv3", "compv4", "wavelet", "basic")
CACHE_EPILOGUES: Tuple[str, ...] = ("default", "cshuffle")

# Pipeline names that emit exactly the code of another one, so the AOT cache
# skips them (the JIT sweep shares CACHE_PIPELINES and still walks them).
#
# * "basic" in every direction: its K loop is the shared scf.for_iter body of
#   "mem" and its schedule policy is mem's (no hints, no setprio), so the
#   binary differs only in its name.
# * "wavelet" in wgrad: the wgrad builder has no load/math wave split, and the
#   wavelet schedule policy is mem's too. (fwd/dgrad have a real wavelet
#   kernel, gfx1250 WMMA only; their validators reject it elsewhere.)
_AOT_ALIAS_PIPELINES: Dict[str, Tuple[str, ...]] = {
    "fwd": ("basic",),
    "wgrad": ("basic", "wavelet"),
    "dgrad": ("basic",),
}

# ---- capability axes ------------------------------------------------------
# These are NOT tuning knobs: they change which problems a binary can serve.
# Grouped convolution takes a different code path in every direction (the
# contraction index only spans one group). An ungrouped binary gives wrong
# numbers -- not an error -- on a grouped problem, but the grouped binary
# computes groups == 1 correctly too (group index 0, cpg == C), so the cache
# builds the grouped one only and offers it to both; KernelCache keeps
# honouring ungrouped entries of older caches for ungrouped problems only.
#
# Implicit-GEMM dgrad reads its stride and dilation from the tilde sub-GEMM
# records at run time, so one tilde binary (stride/dilation recorded as 0,
# runtime) serves every stride and dilation within max_sub_gemms. A stride-1
# binary is a second capability: at stride 1 the kernel rebuilds its single
# sub-GEMM record from the kernargs instead of searching and loading the
# record buffer (DgradConvSpec.folds_sub_gemm_record), and with dilation 1 it
# also takes the unit-stride epilogue. Its identity records stride 1 and
# dilation 1, so the cache offers it to stride-1, dilation-1 problems only;
# the tilde binary of the same configuration still serves those too. The
# stride-1 capability is built with the tap-outer K loop only (see
# _dgrad_tap_outer_variants).
CACHE_GROUPED: Tuple[bool, ...] = (True,)
# Whether the stride-1 dgrad binaries are built next to the tilde ones.
CACHE_DGRAD_STRIDE1: tuple[bool, ...] = (False, True)
# dY halo binaries (DgradConvSpec.dy_halo): a capped sub-grid, not the
# geometry product. The tiles are the halo tiles of the gfx950 dgrad pick, as
# (tile_m, tile_n, tile_k, warp_m, warp_n, warp tile): 64x64 on 2x2 waves and
# 128x64 / 256x64 on 4x1 waves, 32x32 atoms. The filter is baked (the taps
# are unrolled), square 3, 5 and 7 (stride 1 and Ho == Hi fix the pad). The
# staged halo is tile_m + (Y-1)*halo_max_wo + (X-1) LDS rows, so a binary
# serves Wo <= halo_max_wo and a wider bucket costs LDS on a narrower image;
# that costs time only where the extra rows cross an LDS occupancy step. Powers
# of two keep the over-size under 2x, and 48 / 96 bound it at 1.5x between
# the wider ones, where the steps lie closest together. The loads are K-outer
# (B = D width > 1); A loads 8 wide, which every halo problem admits (kpg is a
# multiple of tile_k). dy_halo = 2 (the double-buffered B tile) only.
CACHE_DGRAD_HALO_TILES: tuple[tuple[int, ...], ...] = (
    (64, 64, 64, 2, 2, 32),
    (128, 64, 64, 4, 1, 32),
    (256, 64, 64, 4, 1, 32),
)
CACHE_DGRAD_HALO_FILTERS: tuple[int, ...] = (3, 5, 7)
CACHE_DGRAD_HALO_MAX_WO: tuple[int, ...] = (8, 16, 32, 48, 64, 96, 128)
CACHE_DGRAD_HALO_VEC_BC: tuple[int, ...] = (2, 4, 8)
CACHE_DGRAD_HALO_VEC_A = 8
CACHE_DGRAD_HALO_MODES: tuple[int, ...] = (2,)
# Sub-GEMM records an implicit-GEMM dgrad binary dispatches over (its tilde
# decomposition has stride_h * stride_w of them at most): 32x32, so the
# patch-embedding strides 14 and 16 of vision transformers fit. The CTA's
# search runs only as many steps as a problem's records need, so the bound
# costs the common stride-1 / stride-2 problems nothing.
CACHE_DGRAD_MAX_SUB_GEMMS = 1024
# The patch-embedding dgrad (DgradConvSpec.patch): one GEMM over the filter's
# (y, x, c) for stride >= filter and no padding. Built on these ungrouped
# cshuffle geometries only -- (tile_m, tile_n, tile_k, warp_m, warp_n, warp
# tile, pipeline), the ones that led a sweep of ViT/SigLIP/ConvNeXt stems --
# at every B/D width, since the X*C run of a stem is often odd or 2 mod 4.
CACHE_DGRAD_PATCH_GEOMETRIES: frozenset = frozenset(
    (tm, tn, tk, wm, wn, wt, pipe)
    for (tm, tn, tk, wm, wn, wt), pipes in (
        ((128, 256, 64, 1, 4, 16), ("compv3", "compv4")),
        ((128, 256, 64, 2, 4, 16), ("mem", "compv3", "compv4")),
        ((128, 256, 64, 2, 2, 16), ("compv3", "compv4")),
        ((256, 128, 64, 2, 2, 16), ("compv3", "compv4")),
        ((256, 128, 64, 4, 1, 16), ("compv3", "compv4")),
        ((256, 128, 32, 4, 2, 16), ("mem", "compv3")),
        ((128, 128, 64, 2, 2, 16), ("mem", "compv3", "compv4")),
        ((128, 128, 64, 2, 4, 16), ("mem",)),
        ((128, 128, 64, 1, 4, 32), ("mem",)),
        ((128, 128, 64, 2, 2, 32), ("mem", "compv3", "compv4")),
        ((128, 64, 64, 4, 1, 32), ("mem", "compv3", "compv4")),
        ((128, 64, 64, 4, 2, 32), ("compv4",)),
        ((64, 64, 64, 2, 2, 16), ("mem", "compv3", "compv4")),
    )
    for pipe in pipes
)
# wgrad group merging (WgradConvSpec.group_merge): Gm depthwise groups share
# one workgroup. The degree is baked in and the binary serves depthwise
# problems only (groups divisible by Gm, Y*X*Gm within the tile). Mirrors
# _GROUP_MERGE_DEGREES of the wgrad instance.
CACHE_WGRAD_GROUP_MERGES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
# fwd group merging (ImplicitGemmConvSpec.group_merge): the same idea on the
# forward GEMM -- Gm depthwise groups fold onto GEMM-N and GEMM-K. Baked in;
# the binary serves depthwise problems with groups divisible by Gm and
# Gm <= tile_n. _GROUP_MERGE_DEGREES of the fwd instance minus Gm = 2, which
# was never the best merged degree on the depthwise shapes the merged path was
# tuned on. Depthwise problems whose group count is even but not divisible by
# 4 therefore get no merged binary and run the unmerged kernels.
CACHE_FWD_GROUP_MERGES: Tuple[int, ...] = (4, 8, 16, 32, 64)
# The merged fwd kernels are built on a subset of the shared tile geometry.
# The merged GEMM-N is exactly Gm <= 64, so a wider tile only adds idle
# N-lanes. tile_k = 16, warp_n = 4 and tile_m = 16 were never needed for the
# best merged kernel on the depthwise shapes the merged path was tuned on.
# Together these drop most of the merged binaries; unroll_k stays whole.
CACHE_FWD_GROUP_MERGE_MAX_TILE_N = 64
CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_K: Tuple[int, ...] = (16,)
CACHE_FWD_GROUP_MERGE_EXCLUDED_WARP_N: Tuple[int, ...] = (4,)
CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_M: Tuple[int, ...] = (16,)

# ---- wgrad split-K axis ---------------------------------------------------
# The split degree is never compiled in: it is passed as kernargs
# (``ks_count`` / ``ks``), so the one split-K binary -- any spec split_k > 1
# builds the same kernel -- handles every degree > 1 at launch time. The cache
# records it as split_k=2 ("splits"); the benchmark sweeps the real degrees.
#
# split_k=1 (direct store, no atomics) is a structurally different kernel
# (different epilogue, dW is writeonly, vec_c can be > 1), so it is a separate
# set of binaries. The JIT sweep builds it for every non-depthwise problem and
# it wins where the grid already fills the device without splitting K, so the
# cache carries it too. Group-merged (depthwise) kernels skip it, as the JIT
# sweep does: one workgroup per merged group cannot fill the device.
CACHE_WGRAD_SPLIT_KS: Tuple[int, ...] = (1, 2)

# ---- wgrad stream-K axis (opt-in) -------------------------------------------
# Stream-K (WgradConvSpec.streamk) bakes its launch mode and its partial-sum
# reduction into the kernel; the CTA pool -- and with it the whole iteration
# partition -- is a kernarg, so one binary per (mode, reduction) serves every
# shape and pool. It is off unless a caller asks for it (--streamk): it adds
# a default-epilogue kernel per (mode, reduction) to every wgrad geometry,
# which more than doubles the wgrad grid. "atomic" accumulates into dW with
# f32 atomics and the validator keeps it to fp32 dW only.
#
# Only a slice of the (mode, reduction) x vector-width grid is cached (see
# _wgrad_streamk_builds). Stream-K pays off where the shape loads at the
# widest vector width, so the requested variants are built there, minus
# "tree" ("linear" is the other deterministic fixup). Narrower widths are left
# to the split-K kernels, which are cached at every width and serve those
# shapes as well -- except for one case: large-filter depthwise layers, whose
# per-group reduction is long and whose GEMM is wide. They load scalar, and
# stream-K wins there, so a scalar dp_sk slice is kept on the wide tiles.
CACHE_WGRAD_STREAMK_MODES: Tuple[str, ...] = ("dp_sk", "persistent")
CACHE_WGRAD_STREAMK_REDUCTIONS: Tuple[str, ...] = (
    "linear",
    "tree",
    "workspace",
    "atomic",
)
# Bytes per stream-K operand load at the widest vector width: 8 elements for
# fp16/bf16, 4 for fp32.
CACHE_WGRAD_STREAMK_LOAD_BYTES = 16
# Reductions not cached at the widest vector widths.
CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS: Tuple[str, ...] = ("tree",)
# The scalar (vector width 1) slice for large-filter depthwise layers.
CACHE_WGRAD_STREAMK_SCALAR_MODES: Tuple[str, ...] = ("dp_sk",)
CACHE_WGRAD_STREAMK_SCALAR_REDUCTIONS: Tuple[str, ...] = ("tree", "workspace")
CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N = 128

# Private aliases for internal use -- the generators below refer to these.
_TILE_MN = CACHE_TILE_MN
_TILE_K = CACHE_TILE_K
_WARP_MN = CACHE_WARP_MN
_WGRAD_EXTRA_TILE_MN = CACHE_WGRAD_EXTRA_TILE_MN
_WGRAD_EXTRA_WARP_MN = CACHE_WGRAD_EXTRA_WARP_MN
_WARP_TILE_MN = CACHE_WARP_TILE_MN
_MAX_ACC_REGS = CACHE_MAX_ACC_REGS
_VECS = CACHE_VECS
_PIPELINES = CACHE_PIPELINES
_EPILOGUES = CACHE_EPILOGUES
_GROUPED = CACHE_GROUPED
_WGRAD_GROUP_MERGES = CACHE_WGRAD_GROUP_MERGES
_FWD_GROUP_MERGES = CACHE_FWD_GROUP_MERGES
_WGRAD_SPLIT_KS = CACHE_WGRAD_SPLIT_KS


@dataclass(frozen=True)
class BuildJob:
    """One kernel to build: an identity plus the spec kwargs that produce it."""

    identity: KernelIdentity
    direction: str
    spec_kwargs: dict
    # Capability axes this job is built for. They pick the probe problem
    # (see _probe_problem); they are not spec kwargs.
    caps: dict


def _dtype_triple(dtype: str) -> Tuple[str, str, str]:
    return dtype, dtype, dtype


def _grid_axes(direction: Optional[str]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(tile_mn, warp_mn)`` for a direction, sorted so order is stable.

    ``None`` means the base grid, which is what a caller that walks an
    explicit geometry list already holds.
    """
    if direction != "wgrad":
        return _TILE_MN, _WARP_MN
    return (
        tuple(sorted(set(_TILE_MN) | set(_WGRAD_EXTRA_TILE_MN))),
        tuple(sorted(set(_WARP_MN) | set(_WGRAD_EXTRA_WARP_MN))),
    )


def _geometries(
    target,
    mma_family: str,
    da: str,
    db: str,
    direction: Optional[str] = None,
) -> Iterator[tuple]:
    """The tile/warp/atom geometries a direction's generator walks.

    Yields ``(tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue,
    atom)`` for each combination whose warp tiling fits the block tile, whose
    K tile is a multiple of the selected MMA atom, and whose per-lane
    accumulator stays within :data:`CACHE_MAX_ACC_REGS`. Shared by the
    generators and :func:`count_jobs`, so the progress total cannot drift from
    what the generators actually yield.

    ``direction`` selects the grid: wgrad gets the filter-shaped tile and warp
    extensions (see :data:`CACHE_WGRAD_EXTRA_TILE_MN`), everything else the
    base grid. It must be threaded through wherever a geometry list is built
    for a direction, or the chunk bounds stop matching the list the worker
    walks.
    """
    tile_mn, warp_mn = _grid_axes(direction)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
    ) in itertools.product(
        tile_mn,
        tile_mn,
        _TILE_K,
        warp_mn,
        warp_mn,
        _WARP_TILE_MN,
        _PIPELINES,
        _EPILOGUES,
    ):
        if warp_m * wt > tile_m or warp_n * wt > tile_n:
            continue
        if tile_m % (warp_m * wt) or tile_n % (warp_n * wt):
            continue
        acc_regs = (tile_m // warp_m) * (tile_n // warp_n) // target.wave_size
        if acc_regs > _MAX_ACC_REGS:
            continue
        atom = target.mma.select_largest_k(
            family=mma_family, a_dtype=da, b_dtype=db, c_dtype="fp32", m=wt, n=wt
        )
        if atom is None or tile_k % atom.k:
            continue
        yield tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom


def _aot_pipeline(direction: str, pipeline: str) -> bool:
    """Whether the AOT cache builds ``pipeline`` for ``direction``."""
    return pipeline not in _AOT_ALIAS_PIPELINES[direction]


def _fwd_k_loops(pipeline: str) -> List[Tuple[bool, bool]]:
    """``(unroll_k, async_dma)`` K-loop drivers emitted per forward geometry.

    async_dma replaces the K-loop driver entirely and ignores the pipeline
    string, so it is only emitted once per geometry instead of as identical
    binaries under different pipeline labels.
    """
    k_loops = [(False, False), (True, False)]
    if pipeline == "mem":
        k_loops.append((False, True))
    return k_loops


def _wgrad_two_stages(split_k: int) -> Tuple[bool, ...]:
    """Two-stage variants for a split-K degree.

    Two-stage applies whenever the reduction is split (split_k > 1). It is
    what reaches problems the packed 16-bit atomic cannot address (an odd dW
    row, e.g. a 3-channel stem conv).
    """
    return (False, True) if split_k > 1 else (False,)


def _layout_caps() -> List[dict]:
    """The grouped capability variants fwd and wgrad build."""
    return [dict(grouped=g) for g in _GROUPED]


def _dgrad_caps() -> List[dict]:
    """dgrad's capability variants: grouped x (tilde, stride 1 / dilation 1)."""
    return [
        (
            {"grouped": g, "stride": 1, "dilation": 1}
            if s1
            else {"grouped": g, "tilde": True}
        )
        for s1 in CACHE_DGRAD_STRIDE1
        for g in _GROUPED
    ]


def _dgrad_tap_outer_variants(
    caps: dict, wave_size: int, pipeline: str
) -> tuple[bool, ...]:
    """``tap_outer_k`` values built for one dgrad configuration.

    The tap-outer K loop (DgradConvSpec.tap_outer_k) is a stride-1,
    dilation-1 loop of the wave64 MFMA kernel, so only the stride-1
    capability gets it, and not under the wavelet pipeline (WMMA, its own K
    loop). The probe's kpg is a multiple of every cached tile_k; the cache
    checks kpg % tile_k on the problem.

    The stride-1 capability is built with that loop only, so it yields
    nothing where the loop does not apply: a stride-1 binary with the flat K
    loop only folds the sub-GEMM record, and the tilde binary of the same
    configuration (same K loop, record searched and loaded) serves the same
    problems at no measurable cost, so it would double the stride-1 slice of
    the catalog for nothing. The fold itself stays: the tap-outer (and dY
    halo) binaries are built on it.
    """
    stride1 = caps.get("stride") == 1 and caps.get("dilation") == 1
    if not stride1:
        return (False,)
    if wave_size == 64 and pipeline != "wavelet":
        return (True,)
    return ()


def _dgrad_halo_knobs(warp_n: int) -> tuple[tuple[int, int], ...]:
    """``(dy_halo_setprio, dy_halo_kouter_pad)`` pairs built per halo tile.

    The 4x1-wave tiles carry the wave priority and the wider K-outer B row
    pad as options (each with and without); the 2x2 tile neither.
    """
    if warp_n == 1:
        return tuple((prio, kpad) for prio in (0, 1) for kpad in (0, 32))
    return ((0, 0),)


def _dgrad_halo_jobs_for(
    arch: str, geometry: tuple, wave_size: int, db: str
) -> list[tuple[int, int, int, int, int, int, int]]:
    """The dY halo variants of one dgrad geometry, as ``(dy_halo, vec_bc,
    filter, halo_max_wo, setprio, kpad, vec_c)``; empty off the halo grid.

    Shared by :func:`_dgrad_jobs` and :func:`count_jobs`.
    """
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, _ = geometry
    if (tile_m, tile_n, tile_k, warp_m, warp_n, wt) not in CACHE_DGRAD_HALO_TILES:
        return []
    if pipeline != "mem" or wave_size != 64:
        return []
    out = []
    for vec_bc in CACHE_DGRAD_HALO_VEC_BC:
        if not DgradConvSpec.default_lds_k_outer(
            arch=arch,
            dtype_b=db,
            warp_tile_n=wt,
            cpg=vec_bc,
            wave_size=wave_size,
            pipeline=pipeline,
        ):
            continue
        vec_c = 1 if epilogue == "default" else vec_bc
        for mode, filt, max_wo, (prio, kpad) in itertools.product(
            CACHE_DGRAD_HALO_MODES,
            CACHE_DGRAD_HALO_FILTERS,
            CACHE_DGRAD_HALO_MAX_WO,
            _dgrad_halo_knobs(warp_n),
        ):
            out.append((mode, vec_bc, filt, max_wo, prio, kpad, vec_c))
    return out


def _is_patch_geometry(geometry: tuple) -> bool:
    """Is ``geometry`` (a :func:`_geometries` entry) one the patch dgrad builds?

    MFMA only: the patch store is not implemented for WMMA.
    """
    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom = geometry
    return (
        atom.family == "mma"
        and epilogue == "cshuffle"
        and (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline)
        in CACHE_DGRAD_PATCH_GEOMETRIES
    )


def _job_flavor(llvm_flavor: Optional[str]) -> str:
    """The LLVM flavor recorded in every identity of one enumeration.

    Resolved once per walk instead of per identity: ``KernelIdentity`` would
    otherwise call :func:`current_llvm_flavor` from its ``default_factory`` for
    each of millions of candidates, and that lookup globs the filesystem for
    the comgr library every time -- it was most of the enumeration's runtime.
    The parallel enumeration resolves it in the parent and passes it down, so
    every worker records the same flavor the parent would.
    """
    return llvm_flavor if llvm_flavor is not None else current_llvm_flavor()


def _fwd_jobs(
    arch: str,
    dtype: str,
    wave_size: int,
    mma_family: str,
    target,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
) -> Iterator[BuildJob]:
    """Every forward implicit-GEMM variant worth caching for this arch/dtype.

    ``geometries`` restricts the walk to a slice of :func:`_geometries` (the
    parallel enumeration hands each worker one); ``llvm_flavor`` pins the
    identity's flavor (see :func:`_job_flavor`).
    """
    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "fwd")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("fwd", pipeline):
            continue
        # fwd's A (X) and B (W) are both contiguous along cpg, so they share
        # one width; D (Y) runs along kpg.
        for vec_ab, vec_c in itertools.product(_VECS, _VECS):
            for (unroll_k, async_dma), caps in itertools.product(
                _fwd_k_loops(pipeline), _layout_caps()
            ):
                grouped = caps["grouped"]
                cfg = dict(
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    warp_m=warp_m,
                    warp_n=warp_n,
                    warp_tile_m=wt,
                    warp_tile_n=wt,
                    warp_tile_k=atom.k,
                    pipeline=pipeline,
                    epilogue=epilogue,
                    wave_size=wave_size,
                    vector_size_a=vec_ab,
                    vector_size_b=vec_ab,
                    vector_size_c=vec_c,
                    unroll_k=unroll_k,
                    async_dma=async_dma,
                )
                yield BuildJob(
                    identity=KernelIdentity(
                        arch=arch,
                        direction="fwd",
                        algorithm="implicit_gemm",
                        llvm_flavor=flavor,
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        **_async_chunks(cfg, (da, db, dd), caps, arch),
                        **cfg,
                    ),
                    direction="fwd",
                    spec_kwargs=cfg,
                    caps=caps,
                )
        yield from _fwd_merged_jobs(
            arch,
            da,
            db,
            dd,
            flavor,
            wave_size,
            (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom),
        )


def _fwd_merged_k_loops() -> Tuple[bool, ...]:
    """``unroll_k`` values built for a group-merged forward geometry.

    async_dma is excluded: its chunk-granular predicate cannot carry the
    per-element diagonal mask of the merged B load, so the gate rejects it.
    """
    return (False, True)


def _fwd_merged_geometry(geometry: tuple) -> bool:
    """Whether group-merged forward kernels are built for ``geometry``.

    Only the cshuffle epilogue: the merged store runs Gm channels wide, which
    the default (scalar-store) epilogue rejects. On top of that the tile is
    restricted per the ``CACHE_FWD_GROUP_MERGE_*`` constants.
    """
    tile_m, tile_n, tile_k, _warp_m, warp_n, _wt, _pipeline, epilogue, _atom = geometry
    return (
        epilogue == "cshuffle"
        and tile_n <= CACHE_FWD_GROUP_MERGE_MAX_TILE_N
        and tile_k not in CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_K
        and warp_n not in CACHE_FWD_GROUP_MERGE_EXCLUDED_WARP_N
        and tile_m not in CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_M
    )


def _fwd_merged_jobs(
    arch: str,
    da: str,
    db: str,
    dd: str,
    flavor: str,
    wave_size: int,
    geometry: tuple,
) -> Iterator[BuildJob]:
    """Group-merged depthwise forward kernels for one geometry.

    Nothing is built for a geometry :func:`_fwd_merged_geometry` excludes.
    The vector widths are left to
    the builder: it derives them from the Gm-wide merged channel run -- exactly
    Gm for every problem the binary serves (cpg = kpg = 1) -- while an explicit
    width is validated against the per-group run of 1 and could only be 1.
    """
    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom = geometry
    if not _fwd_merged_geometry(geometry):
        return
    for gm, unroll_k in itertools.product(_FWD_GROUP_MERGES, _fwd_merged_k_loops()):
        cfg = dict(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=wt,
            warp_tile_n=wt,
            warp_tile_k=atom.k,
            pipeline=pipeline,
            epilogue=epilogue,
            wave_size=wave_size,
            unroll_k=unroll_k,
            group_merge=gm,
        )
        yield BuildJob(
            identity=KernelIdentity(
                arch=arch,
                direction="fwd",
                algorithm="implicit_gemm",
                llvm_flavor=flavor,
                dtype_a=da,
                dtype_b=db,
                dtype_d=dd,
                grouped=True,
                # 0 = derived by the builder (from the Gm merged run).
                vector_size_a=0,
                vector_size_b=0,
                vector_size_c=0,
                **cfg,
            ),
            direction="fwd",
            spec_kwargs=cfg,
            caps=dict(grouped=True, group_merge=gm),
        )


def _async_chunks(cfg: dict, dtypes, caps: dict, arch: str) -> dict:
    """The fwd async loaders' chunk widths, for the identity.

    They are chosen from the build-time cpg -- the probe problem's -- so they
    are a capability of the binary and have to be recorded. Taken from the
    builder's own loader construction so the two cannot disagree. An invalid
    spec records 0; the job is dropped by the validity filter anyway.

    ``arch`` matters because the width is also capped by what the target's
    DRAM->LDS DMA can move per lane, so the same ``cfg`` records different
    chunk widths on gfx942 and gfx950.
    """
    if not cfg.get("async_dma"):
        return {}
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm import (
        ImplicitGemmConvSpec,
        async_tile_loaders,
    )

    da, db, dd = dtypes
    try:
        spec = ImplicitGemmConvSpec(
            problem=_probe_problem("fwd", caps),
            data=ConvDataSpec(dtype_a=da, dtype_b=db, dtype_d=dd),
            **cfg,
        )
        a_loader, b_loader = async_tile_loaders(spec, arch)
    except ValueError:
        return {}
    return dict(
        async_chunk_a=a_loader.elems_per_chunk,
        async_chunk_b=b_loader.elems_per_chunk,
    )


def _wgrad_streamk_variants(
    epilogue: str,
    streamk_modes: Sequence[str],
    streamk_reductions: Sequence[str],
) -> List[Tuple[str, str]]:
    """``(mode, reduction)`` stream-K variants built per wgrad geometry.

    Stream-K runs its epilogue once per tile inside the stream-K tile loop and
    only the default epilogue supports that, so cshuffle geometries get none.
    """
    if epilogue != "default":
        return []
    return list(itertools.product(streamk_modes, streamk_reductions))


def _wgrad_streamk_builds(
    epilogue: str,
    tile_n: int,
    dtype: str,
    streamk_modes: Sequence[str],
    streamk_reductions: Sequence[str],
) -> List[Tuple[str, str, int, int]]:
    """``(mode, reduction, vector_size_a, vector_size_b)`` built per geometry.

    Of the requested variants: every one but the
    ``CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS`` at the widest width W of
    the dtype on both operands, and at W on dY with a scalar X for the
    narrow-channel layers (C = 3 stems) that cannot load X wide; plus the
    ``CACHE_WGRAD_STREAMK_SCALAR_*`` slice, fully scalar, on tiles at least
    ``CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N`` wide. Every other combination
    falls to the split-K kernels.
    """
    variants = _wgrad_streamk_variants(epilogue, streamk_modes, streamk_reductions)
    w = CACHE_WGRAD_STREAMK_LOAD_BYTES // {"fp16": 2, "bf16": 2, "fp32": 4}[dtype]
    builds = [
        (mode, reduction, vec_a, vec_b)
        for mode, reduction in variants
        if reduction not in CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS
        for vec_a, vec_b in ((w, w), (w, 1))
    ]
    if tile_n >= CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N:
        builds += [
            (mode, reduction, 1, 1)
            for mode, reduction in variants
            if mode in CACHE_WGRAD_STREAMK_SCALAR_MODES
            and reduction in CACHE_WGRAD_STREAMK_SCALAR_REDUCTIONS
        ]
    return builds


def _wgrad_jobs(
    arch,
    dtype,
    wave_size,
    mma_family,
    target,
    split_ks,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> Iterator[BuildJob]:
    from kernels.common.conv_implicit_gemm_wgrad import (
        _DEFAULT_WS_REPLICAS,
        WgradConvSpec,
    )

    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "wgrad")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("wgrad", pipeline):
            continue
        # The K-outer LDS tile, picked per geometry by the predicate the JIT
        # sweep and dispatch use. It is a pure function of arch/dtype/atom, so
        # a shape-generic binary can carry it; leaving it at the spec default
        # (M-outer) caches only the slower layout on gfx950.
        lds_k_outer = WgradConvSpec.default_lds_k_outer(
            arch=arch,
            dtype_a=da,
            dtype_b=db,
            warp_tile_m=wt,
            warp_tile_n=wt,
            wave_size=wave_size,
        )
        # The row stride is the LDS bank-conflict lever on the M-outer tile.
        # Left unset the spec falls back to a flat pad of 8, which is not
        # conflict-free; derive it instead. This has to be told which layout
        # the geometry above settled on: a K-outer tile takes its stride from
        # the builder and rejects an explicit pad, so the derivation returns
        # None there -- as it does wherever else it does not apply (see
        # default_lds_k_pad). The key is then omitted so the spec and the
        # identity both keep their defaults.
        _k_pad = WgradConvSpec.default_lds_k_pad(
            arch=arch,
            tile_k=tile_k,
            dtype_a=da,
            dtype_b=db,
            lds_k_outer=lds_k_outer,
            wave_size=wave_size,
        )
        _pad_cfg = {} if _k_pad is None else {"lds_k_pad": _k_pad}
        # wgrad's B (X) and D (dW) are both contiguous along cpg, so they
        # share one width; A (dY) runs along kpg. The default epilogue (and so
        # every two-stage kernel) stores scalar, so there D is 1 and B takes
        # its own width -- tied to D it could only ever load scalar.
        for vec_a, vec_bc, split_k in itertools.product(_VECS, _VECS, split_ks):
            vec_c = 1 if epilogue == "default" else vec_bc
            for two_stage, caps in itertools.product(
                _wgrad_two_stages(split_k), _layout_caps()
            ):
                grouped = caps["grouped"]
                cfg = dict(
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    warp_m=warp_m,
                    warp_n=warp_n,
                    warp_tile_m=wt,
                    warp_tile_n=wt,
                    warp_tile_k=atom.k,
                    pipeline=pipeline,
                    epilogue=epilogue,
                    wave_size=wave_size,
                    vector_size_a=vec_a,
                    vector_size_b=vec_bc,
                    vector_size_c=vec_c,
                    split_k=split_k,
                    two_stage=two_stage,
                    lds_k_outer=lds_k_outer,
                    **_pad_cfg,
                )
                if two_stage:
                    # Pinned in the spec and the identity alike, so the host
                    # sizes the scratch and Stage 2 from what was compiled.
                    cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
                cfg["prefetch_addr"] = _wgrad_prefetch_addr(arch, cfg)
                yield BuildJob(
                    identity=KernelIdentity(
                        arch=arch,
                        direction="wgrad",
                        algorithm="implicit_gemm",
                        llvm_flavor=flavor,
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        **cfg,
                    ),
                    direction="wgrad",
                    spec_kwargs=cfg,
                    caps=caps,
                )
        # Stream-K kernels (opt-in). Default epilogue only, so the store is
        # scalar and B takes its own width; split_k stays 1 (stream-K owns the
        # K partition). The workspace reduction reuses the two-stage scratch,
        # so its replica count is pinned exactly as for two-stage.
        for (mode, reduction, vec_a, vec_b), caps in itertools.product(
            _wgrad_streamk_builds(
                epilogue, tile_n, dtype, streamk_modes, streamk_reductions
            ),
            _layout_caps(),
        ):
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=vec_a,
                vector_size_b=vec_b,
                vector_size_c=1,
                streamk=mode,
                streamk_reduction=reduction,
                lds_k_outer=lds_k_outer,
                **_pad_cfg,
            )
            if reduction == "workspace":
                cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="wgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=caps["grouped"],
                    **cfg,
                ),
                direction="wgrad",
                spec_kwargs=cfg,
                caps=caps,
            )
        # Group-merged depthwise kernels. Merged split-K must take the
        # two-stage path (the packed atomic cannot drop off-diagonal pairs).
        # The vector widths are left to the builder: it derives them from the
        # Gm-wide merged channel run -- exactly Gm for every problem the binary
        # serves (cpg = kpg = 1) -- while an explicit width is validated
        # against the per-group run of 1 and could only be 1.
        for gm, split_k in itertools.product(
            _WGRAD_GROUP_MERGES, [sk for sk in split_ks if sk > 1]
        ):
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                split_k=split_k,
                two_stage=split_k > 1,
                group_merge=gm,
                lds_k_outer=lds_k_outer,
                **_pad_cfg,
            )
            if split_k > 1:
                cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
            cfg["prefetch_addr"] = _wgrad_prefetch_addr(arch, cfg)
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="wgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=True,
                    # 0 = derived by the builder (from the Gm merged run).
                    vector_size_a=0,
                    vector_size_b=0,
                    vector_size_c=0,
                    **cfg,
                ),
                direction="wgrad",
                spec_kwargs=cfg,
                caps=dict(grouped=True, group_merge=gm),
            )


def _wgrad_prefetch_addr(arch: str, cfg: dict) -> bool:
    """``prefetch_addr`` for a wgrad job, from the policy the JIT sweep uses.

    A pure function of the binary's own configuration, so a shape-generic
    kernel can carry it; the identity records it through ``cfg``.
    """
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    return WgradConvSpec.default_prefetch_addr(
        arch=arch,
        tile_m=cfg["tile_m"],
        tile_n=cfg["tile_n"],
        tile_k=cfg["tile_k"],
        lds_k_outer=cfg.get("lds_k_outer", False),
        wave_size=cfg["wave_size"],
        async_dma=cfg.get("async_dma", False),
        unroll_k=cfg.get("unroll_k", False),
    )


def _dgrad_jobs(
    arch,
    dtype,
    wave_size,
    mma_family,
    target,
    max_sub_gemms,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
) -> Iterator[BuildJob]:
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "dgrad")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("dgrad", pipeline):
            continue
        # The tilde binary takes stride and dilation at run time (the tilde
        # records); its identity records them as 0. The stride-1 binary
        # records the stride and dilation it serves.
        # dgrad's B (W, KYXC) and D (dX) are both contiguous along cpg, so
        # they share one width; A (dY) runs along kpg. The default epilogue
        # stores scalar, so there D is 1 and B takes its own width.
        for vec_a, vec_bc, caps in itertools.product(_VECS, _VECS, _dgrad_caps()):
            vec_c = 1 if epilogue == "default" else vec_bc
            # The JIT sweep's B-tile layout. Its predicate keys on the parity
            # of cpg, which is runtime here; the B load width stands in for
            # it: a width > 1 only serves problems whose cpg it divides (an
            # even channel run), and width 1 is the odd-cpg kernel, where
            # K-outer has no scatter to remove.
            lds_k_outer = DgradConvSpec.default_lds_k_outer(
                arch=arch,
                dtype_b=db,
                warp_tile_n=wt,
                cpg=vec_bc,
                wave_size=wave_size,
                pipeline=pipeline,
            )
            grouped = caps["grouped"]
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=vec_a,
                vector_size_b=vec_bc,
                vector_size_c=vec_c,
                lds_k_outer=lds_k_outer,
                max_sub_gemms=max_sub_gemms,
                # A pure function of the binary's own configuration (the
                # kernel picks between its two K loops from the kernargs), so
                # a shape-generic binary can carry it. The policy the JIT
                # sweep uses; the identity records it through cfg.
                uniform_k_decode=DgradConvSpec.default_uniform_k_decode(
                    arch=arch, wave_size=wave_size, pipeline=pipeline
                ),
            )
            # The stride-1 binary is the tap-outer K loop, which replaces the
            # uniform_k_decode loop and serves only problems whose kpg is a
            # multiple of tile_k (supports_problem).
            for tapk in _dgrad_tap_outer_variants(caps, wave_size, pipeline):
                if tapk:
                    cfg = {**cfg, "uniform_k_decode": False, "tap_outer_k": True}
                yield BuildJob(
                    identity=KernelIdentity(
                        arch=arch,
                        direction="dgrad",
                        algorithm="implicit_gemm",
                        llvm_flavor=flavor,
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        stride_h=caps.get("stride", 0),
                        stride_w=caps.get("stride", 0),
                        dilation_h=caps.get("dilation", 0),
                        dilation_w=caps.get("dilation", 0),
                        **cfg,
                    ),
                    direction="dgrad",
                    spec_kwargs=cfg,
                    caps=caps,
                )

        # The dY halo mode of the tap-outer loop (stride 1 / dilation 1, the
        # filter and the widest Wo baked), on the halo tiles only.
        geometry = (
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            wt,
            pipeline,
            epilogue,
            atom,
        )
        for mode, vec_bc, filt, max_wo, prio, kpad, vec_c in _dgrad_halo_jobs_for(
            arch, geometry, wave_size, db
        ):
            caps = {
                "grouped": True,
                "stride": 1,
                "dilation": 1,
                "halo_filter": filt,
                "halo_max_wo": max_wo,
            }
            cfg = {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "warp_m": warp_m,
                "warp_n": warp_n,
                "warp_tile_m": wt,
                "warp_tile_n": wt,
                "warp_tile_k": atom.k,
                "pipeline": pipeline,
                "epilogue": epilogue,
                "wave_size": wave_size,
                "vector_size_a": CACHE_DGRAD_HALO_VEC_A,
                "vector_size_b": vec_bc,
                "vector_size_c": vec_c,
                "lds_k_outer": True,
                "max_sub_gemms": max_sub_gemms,
                "uniform_k_decode": False,
                "tap_outer_k": True,
                "dy_halo": mode,
                "dy_halo_setprio": prio,
                "dy_halo_kouter_pad": kpad,
                "halo_max_wo": max_wo,
            }
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="dgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=True,
                    stride_h=1,
                    stride_w=1,
                    dilation_h=1,
                    dilation_w=1,
                    filter_h=filt,
                    filter_w=filt,
                    **cfg,
                ),
                direction="dgrad",
                spec_kwargs=cfg,
                caps=caps,
            )

        # The patch-embedding GEMM (ungrouped, on the tilde record path) on its
        # geometries only, at every B/D width; A (dY) runs along K.
        if not _is_patch_geometry(
            (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom)
        ):
            continue
        caps = dict(grouped=False, tilde=True, patch=True)
        for vec_bc in _VECS:
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=8,
                vector_size_b=vec_bc,
                vector_size_c=vec_bc,
                lds_k_outer=DgradConvSpec.default_lds_k_outer(
                    arch=arch,
                    dtype_b=db,
                    warp_tile_n=wt,
                    cpg=vec_bc,
                    wave_size=wave_size,
                    pipeline=pipeline,
                ),
                max_sub_gemms=max_sub_gemms,
                uniform_k_decode=DgradConvSpec.default_uniform_k_decode(
                    arch=arch, wave_size=wave_size, pipeline=pipeline
                ),
                patch=True,
            )
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="dgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=False,
                    **cfg,
                ),
                direction="dgrad",
                spec_kwargs=cfg,
                caps=caps,
            )


def job_for_identity(identity: KernelIdentity) -> BuildJob:
    """The job ``--compile-all`` builds ``identity`` from.

    The inverse of the generators above, obtained by running them: the
    identity's geometry is fed to its direction's generator (with the
    identity's own split-K degree, sub-GEMM cap and LLVM flavor) and the job
    whose identity is equal is returned. So a rebuilt job is exactly the one
    the cache build made, never a reconstruction that could drift from it.

    Raises ``LookupError`` for an identity the generators of this checkout do
    not produce (a different grid, or a field they never set).
    """
    from rocke.core.arch import ArchTarget

    if identity.algorithm != "implicit_gemm" or identity.direction not in (
        "fwd",
        "wgrad",
        "dgrad",
    ):
        raise LookupError(
            f"not an implicit-GEMM conv identity: {identity.direction}/"
            f"{identity.algorithm}"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        raise LookupError("the AOT grid builds A, B and D in one dtype")
    if identity.warp_tile_m != identity.warp_tile_n:
        raise LookupError("the AOT grid builds square warp tiles only")
    target = ArchTarget.from_gfx(identity.arch)
    if target.wave_size != identity.wave_size:
        raise LookupError(
            f"{identity.arch} runs wave{target.wave_size}, the identity records "
            f"wave{identity.wave_size}"
        )
    dtype = identity.dtype_a
    wt = identity.warp_tile_m
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    atom = target.mma.select_largest_k(
        family=mma_family, a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=wt, n=wt
    )
    if atom is None or atom.k != identity.warp_tile_k:
        raise LookupError(
            f"no {wt}x{wt}x{identity.warp_tile_k} {dtype} MMA atom on {identity.arch}"
        )
    geometry = (
        identity.tile_m,
        identity.tile_n,
        identity.tile_k,
        identity.warp_m,
        identity.warp_n,
        wt,
        identity.pipeline,
        identity.epilogue,
        atom,
    )
    want = identity.stable_hash()
    streamk = (
        {
            "streamk_modes": (identity.streamk,),
            "streamk_reductions": (identity.streamk_reduction,),
        }
        if identity.direction == "wgrad" and identity.streamk != "off"
        else {}
    )
    for job in _direction_jobs(
        identity.direction,
        identity.arch,
        dtype,
        target,
        (identity.split_k,),
        identity.max_sub_gemms,
        geometries=[geometry],
        llvm_flavor=identity.llvm_flavor,
        **streamk,
    ):
        if job.identity.stable_hash() == want:
            return job
    raise LookupError(
        f"{identity.label()} is not produced by this checkout's "
        f"{identity.direction} generator (another grid, or a field it never sets)"
    )


# Forward LDS budget per arch for the AOT grid, below the arch's capacity.
# gfx1250 has 320 KB of LDS per workgroup; its catalog value was a 160 KB
# placeholder until the WMMA direct dgrad. The forward configurations between
# the two (256x256 wavelet/cshuffle tiles) have not been run with --verify on
# gfx1250 yet, so its forward grid stays at the old budget until they have.
_AOT_FWD_LDS_BUDGET = {"gfx1250": 160 * 1024}

# Configurations that validate but are left out of the AOT grid, per
# (arch, direction): each entry is matched against the job's spec_kwargs.
_AOT_EXCLUDED = {
    # Fails --verify on the C=3 stem dgrad rows: the pre-existing
    # vector_size_a=8 C=3 implicit-GEMM dgrad bug. The 320 KB gfx1250 LDS
    # admits this tile; the bug's other configurations predate it.
    ("gfx1250", "dgrad"): (
        dict(
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=1,
            warp_n=8,
            pipeline="wavelet",
            epilogue="cshuffle",
            vector_size_a=8,
            vector_size_b=1,
            vector_size_c=1,
        ),
    ),
}


def _aot_excluded(job: BuildJob, arch: str) -> bool:
    return any(
        all(job.spec_kwargs.get(k) == v for k, v in entry.items())
        for entry in _AOT_EXCLUDED.get((arch, job.direction), ())
    )


def _spec_is_valid(job: BuildJob, arch: str, dtype: str) -> bool:
    """Would this configuration build at all?

    The grid is deliberately over-generated, and the per-direction validators
    reject most of it on LDS budget, fragment widths, atomic pairing rules and
    so on. Running that check here costs microseconds; discovering it inside
    ``compile_kernel`` costs an LLVM invocation, so the prefilter is the
    difference between a cache build that finishes and one that does not.

    It also applies the grid's own limits on top of spec validity
    (:data:`_AOT_FWD_LDS_BUDGET`, :data:`_AOT_EXCLUDED`).
    """
    if _aot_excluded(job, arch):
        return False
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
    problem = _probe_problem(job.direction, job.caps)
    try:
        if job.direction == "fwd":
            from kernels.common.conv_implicit_gemm import (
                ImplicitGemmConvSpec,
                implicit_gemm_lds_bytes,
                is_valid_spec,
            )

            spec = ImplicitGemmConvSpec(problem=problem, data=data, **job.spec_kwargs)
            spec.validate()
            budget = _AOT_FWD_LDS_BUDGET.get(arch)
            if budget is not None and implicit_gemm_lds_bytes(spec)[0] > budget:
                return False
            return is_valid_spec(spec, arch=arch)[0]
        if job.direction == "wgrad":
            from kernels.common.conv_implicit_gemm_wgrad import (
                WgradConvSpec,
                is_valid_wgrad_spec,
            )

            spec = WgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
            spec.validate()
            return is_valid_wgrad_spec(spec, arch=arch)[0]
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        spec = DgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        spec.validate()
        return is_valid_dgrad_spec(spec, arch=arch)[0]
    except Exception:  # noqa: BLE001 - an invalid combination, not an error
        return False


def count_jobs(
    *,
    dtype: str,
    target,
    directions: Sequence[str],
    split_ks: Sequence[int] = CACHE_WGRAD_SPLIT_KS,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> int:
    """Raw candidate count ``kernel_sweep.enumerate_jobs`` walks, before
    dedup/validation.

    Cheap: it runs only the geometry filter and multiplies out the inner axes,
    without building a single identity -- which is where enumeration spends
    its time. Used as the denominator of the enumeration progress.

    Counted per direction rather than once over a shared grid: wgrad walks a
    wider one (see :func:`_grid_axes`).
    """
    da, db, _ = _dtype_triple(dtype)
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    vecs = len(_VECS) * len(_VECS)
    layouts = len(_layout_caps())
    merged = sum(1 for sk in split_ks if sk > 1) * len(_WGRAD_GROUP_MERGES)
    wave_size = target.wave_size
    total = 0
    for direction in directions:
        if direction not in _AOT_ALIAS_PIPELINES:
            continue
        for geo in _geometries(target, mma_family, da, db, direction):
            pipeline = geo[6]
            if not _aot_pipeline(direction, pipeline):
                continue
            if direction == "fwd":
                total += vecs * len(_fwd_k_loops(pipeline)) * layouts
                if _fwd_merged_geometry(geo):
                    total += len(_FWD_GROUP_MERGES) * len(_fwd_merged_k_loops())
            elif direction == "wgrad":
                total += (
                    vecs * sum(len(_wgrad_two_stages(sk)) for sk in split_ks) * layouts
                )
                total += merged
                total += layouts * len(
                    _wgrad_streamk_builds(
                        geo[7], geo[1], dtype, streamk_modes, streamk_reductions
                    )
                )
            elif direction == "dgrad":
                total += vecs * sum(
                    len(_dgrad_tap_outer_variants(caps, wave_size, pipeline))
                    for caps in _dgrad_caps()
                )
                total += len(_dgrad_halo_jobs_for(target.gfx, geo, wave_size, db))
                if _is_patch_geometry(geo):
                    total += len(_VECS)
    return total


def _direction_jobs(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    *,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
    **kw,
) -> Iterator[BuildJob]:
    wave_size = target.wave_size
    mma_family = "wmma" if wave_size == 32 else "mma"
    if direction == "fwd":
        return _fwd_jobs(arch, dtype, wave_size, mma_family, target, **kw)
    if direction == "wgrad":
        return _wgrad_jobs(
            arch,
            dtype,
            wave_size,
            mma_family,
            target,
            split_ks,
            streamk_modes=streamk_modes,
            streamk_reductions=streamk_reductions,
            **kw,
        )
    if direction == "dgrad":
        return _dgrad_jobs(
            arch, dtype, wave_size, mma_family, target, max_sub_gemms, **kw
        )
    raise ValueError(f"unknown direction {direction!r}")


def _geometry_list(target, dtype: str, direction: Optional[str] = None) -> List[tuple]:
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    da, db, _ = _dtype_triple(dtype)
    return list(_geometries(target, mma_family, da, db, direction))


# ---------------------------------------------------------------------
# building one job
# ---------------------------------------------------------------------
#
# A build needs a ConvProblem because the spec dataclasses still carry one for
# validation and grid sizing. The *emitted IR does not depend on it* -- that is
# what test_conv_abi.py's shape-invariance cases assert -- so any problem
# that satisfies the spec's own validity rules produces the cacheable binary.
# We use a small canonical one.


def _probe_problem(direction: str, caps: Optional[dict] = None):
    """A canonical problem to build against.

    The emitted IR no longer depends on the extents, so any shape produces the
    same binary -- but it *does* depend on the capability axes, so the probe
    has to match the ones this job is being built for. Getting that wrong is
    not a build error: it silently produces a binary that the cache then
    offers for shapes it cannot compute.
    """
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    caps = caps or {}
    if caps.get("patch"):
        # The patch dgrad's record is the patch embedding's own.
        return ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=2, X=2, sH=2, sW=2)
    groups = 2 if caps.get("grouped") else 1
    # The tilde dgrad kernel is the strided one: any stride > 1 selects its
    # code path, and the binary then serves every stride at run time.
    stride = 2 if caps.get("tilde") else int(caps.get("stride", 1))
    dilation = int(caps.get("dilation", 1))
    # Keep the dilated filter inside the image so Ho/Wo stay positive.
    extent = 16 + 2 * (dilation - 1)
    channels = 64
    if caps.get("halo_filter"):
        # A dY halo binary bakes its filter and is sized for halo_max_wo: the
        # probe is a same-size output (pad (F-1)/2) that wide.
        f = int(caps["halo_filter"])
        return ConvProblem(
            N=1,
            Hi=max(f, 8),
            Wi=int(caps["halo_max_wo"]),
            C=channels * 2,
            K=channels * 2,
            Y=f,
            X=f,
            pH=(f - 1) // 2,
            pW=(f - 1) // 2,
            groups=2,
        )
    if caps.get("group_merge", 1) > 1:
        # Merging is depthwise only: cpg = kpg = 1, and 64 groups divide by
        # every merge degree.
        groups, channels = 64, 1
    return ConvProblem(
        N=1,
        Hi=extent,
        Wi=extent,
        C=channels * groups,
        K=channels * groups,
        Y=3,
        X=3,
        sH=stride,
        sW=stride,
        dH=dilation,
        dW=dilation,
        pH=dilation,
        pW=dilation,
        groups=groups,
    )


def kernel_for_job(job: BuildJob, arch: str, dtype: str):
    """Build one job's kernel IR. Returns ``(kernel, meta)``; nothing compiled.

    Raises on an invalid configuration; the caller counts those as skipped
    rather than failed -- the grid is deliberately over-generated and most
    rejections are ordinary spec-validity rules.
    """
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
    problem = _probe_problem(job.direction, job.caps)

    if job.direction == "fwd":
        from kernels.common.conv_implicit_gemm import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        spec = ImplicitGemmConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv(spec, arch=arch)
    elif job.direction == "wgrad":
        from kernels.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            build_implicit_gemm_conv_wgrad,
        )

        spec = WgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    elif job.direction == "dgrad":
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
        )

        spec = DgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv_dgrad(spec, arch=arch)
    else:
        raise ValueError(f"unknown direction {job.direction!r}")

    meta = {
        "spec_kernel_name": spec.kernel_name(),
        # launch_block_size only exists where a pipeline appends extra waves
        # (wavelet); wgrad has no such pipeline and exposes block_size alone.
        "block_size": getattr(spec, "launch_block_size", spec.block_size),
    }
    return kernel, meta
