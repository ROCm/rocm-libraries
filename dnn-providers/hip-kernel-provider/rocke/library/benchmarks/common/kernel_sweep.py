# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two AOT benchmark modes: build the cache, then run any shape from it.

``--compile-all`` walks the configuration grid for an arch/dtype, builds and
compiles every variant that validates, and stores the HSACOs in an
:class:`~benchmarks.common.kernel_cache.KernelCache`. No GPU is needed and no
shape is involved --
that is what makes the artifacts reusable.

``--run-from-cache`` takes a concrete problem, asks the cache which of its
kernels can run it, and benchmarks those. Nothing is compiled.

The split exists because AOT kernels are shape-generic: the expensive step
(compilation) no longer depends on the problem, so it can be done once and
amortised over every shape the cache is later asked about.

What the implicit-GEMM cache sweeps
-----------------------------------
One binary per point of the grid below that passes its direction's spec
validator (``--compile-all`` prints the per-direction breakdown of what
survived). Values are the ``CACHE_*`` constants of this module; the dtype is
fp16 or bf16 (one cache pass per dtype, A/B/D all of that dtype).

Shared tile geometry, every direction::

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
  geometry, under pipeline "mem", since it ignores the pipeline).
* **wgrad** -- vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (X and dW along cpg); with the default epilogue (all
  two-stage kernels) c = 1 and b in {1, 2, 4, 8}. pipeline "wavelet" is not cached (wgrad
  builds it as mem). Only the split-K kernel is cached (``split_k`` > 1,
  recorded as 2; the degree is a kernarg and the benchmark sweeps it at
  launch), as atomic or two-stage (f32 scratch + Stage-2 reduce,
  ``ws_replicas`` = 8). Plus group-merged depthwise kernels
  (``group_merge`` in 2..64, two-stage, load widths derived by the builder),
  served only to depthwise problems whose merged GEMM fits the tile.
* **dgrad** -- vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (W and dX along cpg); with the default epilogue c = 1 and b in
  {1, 2, 4, 8}. stride and dilation are baked into the
  tilde decomposition, so they are capabilities: stride in {1, 2} x dilation
  in {1, 2}; up to 64 sub-GEMMs.

Every direction builds the grouped kernel only: it serves groups == 1 as well
(``CACHE_GROUPED``), so there is no separate ungrouped binary.

Not swept (spec defaults): ``lds_k_outer`` (off, so wgrad ``async_dma``, which
needs it, is never built), ``chiplet_swizzle``, ``lds_k_pad`` / ``lds_layout``,
``waves_per_eu``, accumulator epilogues, ``cshuffle_no_alias``, the wgrad
non-split (``split_k`` = 1) kernel, the pointwise fast path (1x1 problems run
the general kernels) and 3-D convolution. The direct-conv cache sweeps its own
grid: see :mod:`benchmarks.common.direct_kernel_sweep`.
"""

from __future__ import annotations

import gc
import itertools
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)  # noqa: F401 (Tuple used in annotations)

from benchmarks.common.kernel_cache import (
    KernelCache,
    KernelIdentity,
    comgr_input_key,
    current_comgr_id,
    current_emitter_digest,
    current_llvm_flavor,
)

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
# fail spec validation and are silently skipped on those targets. They can be
# added when a gfx1250 cache path is implemented.
# ---------------------------------------------------------------------------

CACHE_TILE_MN: Tuple[int, ...] = (16, 32, 64, 128, 256)
CACHE_TILE_K: Tuple[int, ...] = (16, 32, 64)
CACHE_WARP_MN: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_WARP_TILE_MN: Tuple[int, ...] = (16, 32)
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
# Implicit-GEMM dgrad folds the stride and dilation into its tilde
# decomposition, so those need a binary per value.
CACHE_GROUPED: Tuple[bool, ...] = (True,)
CACHE_DGRAD_STRIDES: Tuple[int, ...] = (1, 2)
CACHE_DGRAD_DILATIONS: Tuple[int, ...] = (1, 2)
# wgrad group merging (WgradConvSpec.group_merge): Gm depthwise groups share
# one workgroup. The degree is baked in and the binary serves depthwise
# problems only (groups divisible by Gm, Y*X*Gm within the tile). Mirrors
# _GROUP_MERGE_DEGREES of the wgrad instance.
CACHE_WGRAD_GROUP_MERGES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)

# ---- wgrad split-K axis ---------------------------------------------------
# The split degree is never compiled in: it is passed as kernargs
# (``ks_count`` / ``ks``), so the one split-K binary -- any spec split_k > 1
# builds the same kernel -- handles every degree > 1 at launch time. The cache
# records it as split_k=2 ("splits"); the benchmark sweeps the real degrees.
#
# split_k=1 (direct store, no atomics) is a structurally different kernel
# (different epilogue, dW is writeonly, vec_c can be > 1) so it remains a
# separate set of binaries and is NOT included here.  Enumerate split_k=1
# separately if you need it; the wgrad benchmark's JIT sweep still covers it.
CACHE_WGRAD_SPLIT_KS: Tuple[int, ...] = (2,)

# Private aliases for internal use -- the generators below refer to these.
_TILE_MN = CACHE_TILE_MN
_TILE_K = CACHE_TILE_K
_WARP_MN = CACHE_WARP_MN
_WARP_TILE_MN = CACHE_WARP_TILE_MN
_MAX_ACC_REGS = CACHE_MAX_ACC_REGS
_VECS = CACHE_VECS
_PIPELINES = CACHE_PIPELINES
_EPILOGUES = CACHE_EPILOGUES
_GROUPED = CACHE_GROUPED
_WGRAD_GROUP_MERGES = CACHE_WGRAD_GROUP_MERGES
_DGRAD_STRIDES = CACHE_DGRAD_STRIDES
_DGRAD_DILATIONS = CACHE_DGRAD_DILATIONS
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


def _geometries(target, mma_family: str, da: str, db: str) -> Iterator[tuple]:
    """The tile/warp/atom geometries every direction's generator walks.

    Yields ``(tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue,
    atom)`` for each combination whose warp tiling fits the block tile, whose
    K tile is a multiple of the selected MMA atom, and whose per-lane
    accumulator stays within :data:`CACHE_MAX_ACC_REGS`. Shared by the
    generators and :func:`count_jobs`, so the progress total cannot drift from
    what the generators actually yield.
    """
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
        _TILE_MN,
        _TILE_MN,
        _TILE_K,
        _WARP_MN,
        _WARP_MN,
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
    """dgrad's capability variants: grouped x stride x dilation."""
    return [
        dict(grouped=g, stride=s, dilation=d)
        for s, d, g in itertools.product(_DGRAD_STRIDES, _DGRAD_DILATIONS, _GROUPED)
    ]


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
        _geometries(target, mma_family, da, db) if geometries is None else geometries
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
                        **_async_chunks(cfg, (da, db, dd), caps),
                        **cfg,
                    ),
                    direction="fwd",
                    spec_kwargs=cfg,
                    caps=caps,
                )


def _async_chunks(cfg: dict, dtypes, caps: dict) -> dict:
    """The fwd async loaders' chunk widths, for the identity.

    They are chosen from the build-time cpg -- the probe problem's -- so they
    are a capability of the binary and have to be recorded. Taken from the
    builder's own loader construction so the two cannot disagree. An invalid
    spec records 0; the job is dropped by the validity filter anyway.
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
        a_loader, b_loader = async_tile_loaders(spec)
    except ValueError:
        return {}
    return dict(
        async_chunk_a=a_loader.elems_per_chunk,
        async_chunk_b=b_loader.elems_per_chunk,
    )


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
) -> Iterator[BuildJob]:
    from kernels.common.conv_implicit_gemm_wgrad import _DEFAULT_WS_REPLICAS

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
        _geometries(target, mma_family, da, db) if geometries is None else geometries
    ):
        if not _aot_pipeline("wgrad", pipeline):
            continue
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
                )
                if two_stage:
                    # Pinned in the spec and the identity alike, so the host
                    # sizes the scratch and Stage 2 from what was compiled.
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
                        grouped=grouped,
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
        for gm, split_k in itertools.product(_WGRAD_GROUP_MERGES, split_ks):
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
            )
            if split_k > 1:
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
        _geometries(target, mma_family, da, db) if geometries is None else geometries
    ):
        if not _aot_pipeline("dgrad", pipeline):
            continue
        # dgrad folds the stride and dilation into its tilde decomposition,
        # so those are capabilities here, not launch parameters.
        # dgrad's B (W, KYXC) and D (dX) are both contiguous along cpg, so
        # they share one width; A (dY) runs along kpg. The default epilogue
        # stores scalar, so there D is 1 and B takes its own width.
        for vec_a, vec_bc, caps in itertools.product(_VECS, _VECS, _dgrad_caps()):
            vec_c = 1 if epilogue == "default" else vec_bc
            grouped, stride, dilation = (
                caps["grouped"],
                caps["stride"],
                caps["dilation"],
            )
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
                max_sub_gemms=max_sub_gemms,
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
                    grouped=grouped,
                    stride_h=stride,
                    stride_w=stride,
                    dilation_h=dilation,
                    dilation_w=dilation,
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
    for job in _direction_jobs(
        identity.direction,
        identity.arch,
        dtype,
        target,
        (identity.split_k,),
        identity.max_sub_gemms,
        geometries=[geometry],
        llvm_flavor=identity.llvm_flavor,
    ):
        if job.identity.stable_hash() == want:
            return job
    raise LookupError(
        f"{identity.label()} is not produced by this checkout's "
        f"{identity.direction} generator (another grid, or a field it never sets)"
    )


def _spec_is_valid(job: BuildJob, arch: str, dtype: str) -> bool:
    """Would this configuration build at all?

    The grid is deliberately over-generated, and the per-direction validators
    reject most of it on LDS budget, fragment widths, atomic pairing rules and
    so on. Running that check here costs microseconds; discovering it inside
    ``compile_kernel`` costs an LLVM invocation, so the prefilter is the
    difference between a cache build that finishes and one that does not.
    """
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
    problem = _probe_problem(job.direction, job.caps)
    try:
        if job.direction == "fwd":
            from kernels.common.conv_implicit_gemm import (
                ImplicitGemmConvSpec,
                is_valid_spec,
            )

            spec = ImplicitGemmConvSpec(problem=problem, data=data, **job.spec_kwargs)
            spec.validate()
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
) -> int:
    """Raw candidate count :func:`enumerate_jobs` walks, before dedup/validation.

    Cheap: it runs only the geometry filter and multiplies out the inner axes,
    without building a single identity -- which is where enumeration spends
    its time. Used as the denominator of the enumeration progress.
    """
    da, db, _ = _dtype_triple(dtype)
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    vecs = len(_VECS) * len(_VECS)
    layouts = len(_layout_caps())
    merged = len(split_ks) * len(_WGRAD_GROUP_MERGES)
    dgrad_caps = len(_dgrad_caps())
    total = 0
    for geo in _geometries(target, mma_family, da, db):
        pipeline = geo[6]
        if "fwd" in directions and _aot_pipeline("fwd", pipeline):
            total += vecs * len(_fwd_k_loops(pipeline)) * layouts
        if "wgrad" in directions and _aot_pipeline("wgrad", pipeline):
            total += vecs * sum(len(_wgrad_two_stages(sk)) for sk in split_ks) * layouts
            total += merged
        if "dgrad" in directions and _aot_pipeline("dgrad", pipeline):
            total += vecs * dgrad_caps
    return total


def _direction_jobs(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    **kw,
) -> Iterator[BuildJob]:
    wave_size = target.wave_size
    mma_family = "wmma" if wave_size == 32 else "mma"
    if direction == "fwd":
        return _fwd_jobs(arch, dtype, wave_size, mma_family, target, **kw)
    if direction == "wgrad":
        return _wgrad_jobs(arch, dtype, wave_size, mma_family, target, split_ks, **kw)
    if direction == "dgrad":
        return _dgrad_jobs(
            arch, dtype, wave_size, mma_family, target, max_sub_gemms, **kw
        )
    raise ValueError(f"unknown direction {direction!r}")


def _geometry_list(target, dtype: str) -> List[tuple]:
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    da, db, _ = _dtype_triple(dtype)
    return list(_geometries(target, mma_family, da, db))


def _enumerate_chunk(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    validate: bool,
    llvm_flavor: str,
    geometries: Sequence[tuple],
) -> Tuple[int, List[Tuple[str, BuildJob]]]:
    """Walk one direction over ``geometries``.

    Returns the raw candidate count and the ``(key, job)`` pairs that are
    valid and first-seen within the chunk, in generator order. Duplicates
    across chunks are resolved by the caller, which merges the chunks in order
    and so keeps the same first occurrence a single serial walk would.
    """
    out: List[Tuple[str, BuildJob]] = []
    seen = set()
    n_raw = 0
    for job in _direction_jobs(
        direction,
        arch,
        dtype,
        target,
        split_ks,
        max_sub_gemms,
        geometries=geometries,
        llvm_flavor=llvm_flavor,
    ):
        n_raw += 1
        key = job.identity.stable_hash()
        if key in seen:
            continue
        # An identity fixes the spec kwargs and capabilities the validator
        # reads, so a repeat would get the same verdict: skip it either way.
        seen.add(key)
        if validate and not _spec_is_valid(job, arch, dtype):
            continue
        out.append((key, job))
    return n_raw, out


# Per-process geometry lists, so a worker running many chunks rebuilds the
# grid once rather than once per chunk.
_WORKER_GEOMETRIES: Dict[Tuple[str, str], List[tuple]] = {}


def _enumerate_worker(payload):
    (direction, arch, dtype, split_ks, max_sub_gemms, validate, flavor, lo, hi) = (
        payload
    )
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    geos = _WORKER_GEOMETRIES.get((arch, dtype))
    if geos is None:
        geos = _WORKER_GEOMETRIES[(arch, dtype)] = _geometry_list(target, dtype)
    return _enumerate_chunk(
        direction,
        arch,
        dtype,
        target,
        split_ks,
        max_sub_gemms,
        validate,
        flavor,
        geos[lo:hi],
    )


def enumerate_jobs(
    *,
    arch: str,
    dtype: str,
    target,
    directions: Sequence[str],
    split_ks: Sequence[int] = CACHE_WGRAD_SPLIT_KS,
    max_sub_gemms: int = 64,
    validate: bool = True,
    jobs: int = 1,
    log=None,
    log_every_s: float = 5.0,
) -> List[BuildJob]:
    """All buildable jobs for the requested directions, deduped by identity.

    With ``validate=True`` (the default) each candidate is run through its
    direction's spec validator first, so the returned list is what will
    actually compile rather than the raw cross product.

    The raw cross product runs to millions of candidates. It is split into
    chunks of tile geometries and, with ``jobs > 1``, the chunks are walked in
    a process pool -- the work is pure-Python CPU, so threads would serialise
    on the GIL. The result does not depend on ``jobs``: chunks are merged in
    order, so the list (including its order) is the one a single serial walk
    produces. With ``log`` set a progress line is emitted at most every
    ``log_every_s`` seconds.
    """
    for direction in directions:
        if direction not in ("fwd", "wgrad", "dgrad"):
            raise ValueError(f"unknown direction {direction!r}")

    flavor = current_llvm_flavor()
    geos = _geometry_list(target, dtype)
    jobs = max(1, int(jobs))
    # Enough chunks per direction to keep every worker busy despite uneven
    # chunk costs, and to give the serial walk regular progress points.
    n_chunks = max(1, min(len(geos), jobs * 8 if jobs > 1 else 64))
    bounds = [
        (len(geos) * i // n_chunks, len(geos) * (i + 1) // n_chunks)
        for i in range(n_chunks)
    ]
    payloads = [
        (direction, arch, dtype, tuple(split_ks), max_sub_gemms, validate, flavor)
        + bound
        for direction in directions
        for bound in bounds
    ]

    total = 0
    if log is not None:
        total = count_jobs(
            dtype=dtype, target=target, directions=directions, split_ks=split_ks
        )
        log(f"  {total} candidates to check with {jobs} process(es)")

    n_raw = n_valid = 0
    started = last_log = time.perf_counter()
    results: List[Optional[List[Tuple[str, BuildJob]]]] = [None] * len(payloads)

    def _done(idx: int, chunk_raw: int, chunk_out) -> None:
        nonlocal n_raw, n_valid, last_log
        results[idx] = chunk_out
        n_raw += chunk_raw
        n_valid += len(chunk_out)
        if log is None:
            return
        now = time.perf_counter()
        if now - last_log >= log_every_s:
            last_log = now
            pct = 100.0 * n_raw / total if total else 0.0
            log(
                f"  enumerating: {n_raw}/{total} candidates checked "
                f"({pct:.1f}%), ~{n_valid} valid ({now - started:.0f}s)"
            )

    if jobs <= 1:
        # In-process, with the caller's target and the module's own grid.
        for idx, payload in enumerate(payloads):
            direction, lo, hi = payload[0], payload[-2], payload[-1]
            _done(
                idx,
                *_enumerate_chunk(
                    direction,
                    arch,
                    dtype,
                    target,
                    split_ks,
                    max_sub_gemms,
                    validate,
                    flavor,
                    geos[lo:hi],
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_enumerate_worker, payload): idx
                for idx, payload in enumerate(payloads)
            }
            for fut in as_completed(futures):
                _done(futures[fut], *fut.result())

    # Merge each direction's chunks in order, keeping the first occurrence.
    seen = set()
    per_direction: Dict[str, List[BuildJob]] = {d: [] for d in directions}
    for payload, chunk_out in zip(payloads, results):
        bucket = per_direction[payload[0]]
        for key, job in chunk_out:
            if key in seen:
                continue
            seen.add(key)
            bucket.append(job)

    # Round-robin across the requested directions rather than draining one
    # before starting the next. The full grid is far larger than any single
    # cache build, so callers routinely truncate it with --limit; taking them
    # in order would make a truncated build contain only forward kernels and
    # silently leave the backward directions unserved.
    out: List[BuildJob] = []
    streams = [iter(per_direction[d]) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            job = next(stream, None)
            if job is not None:
                out.append(job)
                still_running.append(stream)
        streams = still_running

    if log is not None:
        log(
            f"  enumerated {n_raw} candidates -> {len(out)} unique valid "
            f"({time.perf_counter() - started:.0f}s)"
        )
    return out


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
    groups = 2 if caps.get("grouped") else 1
    stride = int(caps.get("stride", 1))
    dilation = int(caps.get("dilation", 1))
    # Keep the dilated filter inside the image so Ho/Wo stay positive.
    extent = 16 + 2 * (dilation - 1)
    channels = 64
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


def build_kernel(job: BuildJob, arch: str, dtype: str):
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


def _emit(build, job: BuildJob, arch: str, dtype: str):
    """Build and lower one job: ``(ComgrInput, content key, meta)``."""
    from rocke.helpers.compile import lower_kernel_for_comgr

    kernel, meta = build(job, arch, dtype)
    comgr_input = lower_kernel_for_comgr(kernel, arch=arch)
    return comgr_input, comgr_input_key(comgr_input), meta


def _emit_worker(payload):
    """Phase 1 worker: the job's binary content key, without compiling.

    Returns ``(job, key, meta, error)``. The lowered IR itself is not sent
    back -- it is large and most keys are already compiled; phase 2 re-emits
    the few it needs.
    """
    build, job, arch, dtype = payload
    try:
        _, key, meta = _emit(build, job, arch, dtype)
        return job, key, meta, None
    except Exception as exc:  # noqa: BLE001 - reported per job, never fatal
        return job, None, None, f"{type(exc).__name__}: {exc}"


def _compile_worker(payload):
    """Phase 2 worker: compile one binary. Returns ``(key, hsaco, name, error)``."""
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    build, job, arch, dtype, key = payload
    try:
        comgr_input, got, _ = _emit(build, job, arch, dtype)
        if got != key:
            # The emitter is deterministic; a different key here means the
            # sources changed between the two phases.
            raise RuntimeError("kernel IR changed between emit and compile")
        hsaco, _ = build_hsaco_from_llvm_ir(
            comgr_input.llvm_text,
            isa=comgr_input.isa,
            options=list(comgr_input.options),
        )
        return key, hsaco, comgr_input.kernel_name, None
    except Exception as exc:  # noqa: BLE001 - reported per binary, never fatal
        return key, None, None, f"{type(exc).__name__}: {exc}"


# A main compile pool that keeps breaking is not a single bad kernel.
_MAX_POOL_RESTARTS = 20

_WORKER_DIED = "worker process died (out of memory, or a crash in the compiler)"


# Each single-worker pool holds a handful of pipes; the slot count is capped so
# the reruns fit the process's open-file limit (often 1024 in containers).
_FDS_PER_POOL = 16
_MAX_ISOLATED_SLOTS = 16


def _isolated_slots(jobs: int) -> int:
    try:
        import resource  # POSIX only

        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if soft == resource.RLIM_INFINITY:
            soft = 1 << 16
    except (ImportError, OSError, ValueError):
        soft = 1024
    by_fds = max(1, (soft // 2) // _FDS_PER_POOL)
    return max(1, min(jobs, _MAX_ISOLATED_SLOTS, by_fds))


def _died_result(kind: str, payload):
    if kind == "emit":
        return (payload[1], None, None, _WORKER_DIED)
    return (payload[-1], None, None, _WORKER_DIED)


def _run_isolated(suspects, jobs: int):
    """Run each ``(kind, payload)`` with nothing else in its worker.

    A few single-worker pools ("slots") take the suspects one at a time, so a
    worker that dies takes down exactly one job. That job gets the worker's
    usual result tuple with :data:`_WORKER_DIED` as its error -- reported like
    any other failure instead of ending the run -- and only its slot gets a
    fresh pool. Yields ``(kind, payload, result)``.
    """
    pending = list(suspects)
    pending.reverse()
    slots = [ProcessPoolExecutor(max_workers=1) for _ in range(_isolated_slots(jobs))]
    running: Dict = {}  # future -> (slot index, kind, payload)

    def submit(i: int) -> None:
        kind, payload = pending.pop()
        fn = _emit_worker if kind == "emit" else _compile_worker
        running[slots[i].submit(fn, payload)] = (i, kind, payload)

    try:
        for i in range(len(slots)):
            if pending:
                submit(i)
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in done:
                i, kind, payload = running.pop(fut)
                try:
                    result = fut.result()
                except BrokenProcessPool:
                    result = _died_result(kind, payload)
                    slots[i].shutdown(wait=True)
                    slots[i] = ProcessPoolExecutor(max_workers=1)
                yield kind, payload, result
                if pending:
                    submit(i)
    finally:
        for pool in slots:
            pool.shutdown(wait=True)


def _pool_map(fn, payloads, jobs: int, on_result) -> None:
    if jobs <= 1:
        for payload in payloads:
            on_result(fn(payload))
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fn, p) for p in payloads]
        for fut in as_completed(futures):
            on_result(fut.result())


def compile_all(
    *,
    cache: KernelCache,
    arch: str,
    dtype: str,
    target,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
) -> int:
    """Populate ``cache`` with every implicit-GEMM variant for ``arch``/``dtype``.

    Compilation is embarrassingly parallel and dominated by LLVM, so it fans
    out over processes; ``jobs`` defaults to one but the caller normally passes
    ``os.cpu_count()``.
    """
    log(
        f"AOT compile-all: enumerating variants for {arch}/{dtype} "
        f"({', '.join(directions)}) -- this walks the full tuning grid before "
        f"compilation starts"
    )
    return compile_jobs(
        cache=cache,
        all_jobs=enumerate_jobs(
            arch=arch,
            dtype=dtype,
            target=target,
            directions=directions,
            jobs=jobs,
            log=log,
        ),
        build=build_kernel,
        arch=arch,
        dtype=dtype,
        directions=directions,
        jobs=jobs,
        limit=limit,
        log=log,
    )


# (label, how to read it off an identity) for describe_jobs. Axes that take a
# single value within a direction are not printed.
_BREAKDOWN_AXES = (
    ("variant", lambda i: i.algorithm),
    ("pipeline", lambda i: i.pipeline),
    ("epilogue", lambda i: i.epilogue),
    (
        "k-loop",
        lambda i: "async_dma" if i.async_dma else "unroll_k" if i.unroll_k else "plain",
    ),
    ("grouped", lambda i: "yes" if i.grouped else "no"),
    (
        "split-K",
        lambda i: ("two-stage" if i.two_stage else "atomic") if i.split_k > 1 else "no",
    ),
    ("group_merge", lambda i: i.group_merge),
    ("stride x dilation", lambda i: f"{i.stride_h}x{i.dilation_h}"),
    ("filter", lambda i: f"{i.filter_h}x{i.filter_w}"),
    ("pad", lambda i: i.pad_h),
    ("cpg/kpg", lambda i: f"{i.cpg}/{i.kpg}"),
    ("atom", lambda i: f"{i.warp_tile_m}x{i.warp_tile_n}x{i.warp_tile_k}"),
    ("vec a/b/c", lambda i: f"{i.vector_size_a}/{i.vector_size_b}/{i.vector_size_c}"),
)


def describe_jobs(jobs: Sequence[BuildJob], log=print) -> None:
    """Log what a job list covers: per direction, the kernel count and how it
    splits along every swept axis (axes with a single value are omitted)."""
    by_direction: Dict[str, List[KernelIdentity]] = {}
    for job in jobs:
        by_direction.setdefault(job.identity.direction, []).append(job.identity)
    for direction, idents in by_direction.items():
        tiles = {(i.tile_m, i.tile_n, i.tile_k) for i in idents}
        warps = {(i.warp_m, i.warp_n) for i in idents}
        log(f"  {direction}: {len(idents)} kernels")
        if tiles != {(0, 0, 0)}:
            log(
                f"    geometry: {len(tiles)} block tiles (MxNxK) x "
                f"{len(warps)} warp layouts"
            )
        for label, get in _BREAKDOWN_AXES:
            counts: Dict[object, int] = {}
            for ident in idents:
                key = get(ident)
                counts[key] = counts.get(key, 0) + 1
            if len(counts) > 1:
                log(
                    f"    {label}: "
                    + ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
                )


def compile_jobs(
    *,
    cache: KernelCache,
    all_jobs: Sequence[BuildJob],
    build,
    arch: str,
    dtype: str,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    log_every_s: float = 5.0,
) -> int:
    """Bring the cache up to date for ``all_jobs``; compile only what changed.

    Shared by every kernel family: the family only decides which jobs exist
    and how one's kernel is built (``build(job, arch, dtype) -> (kernel,
    meta)``). Three steps:

    1. **Up to date.** An entry built from this process's emitter sources and
       COMGR, whose binary is present, is skipped outright -- after a rebuild
       with nothing changed, nothing below runs.
    2. **Emit** (cheap). Every remaining job is built and lowered to LLVM IR
       and keyed by :func:`comgr_input_key`, without compiling. A job whose
       key already has a binary is linked to it on the spot.
    3. **Compile** (expensive). The first job to emit a key with no binary
       compiles it, in the same process pool as the emits, as soon as its
       emit is done; every job with that key is linked when it lands. So
       after an emitter change only the kernels whose code changed are
       recompiled, identities that emit the same code share one compile, and
       an interrupted run keeps every entry it finished.

    A worker that dies outright (OOM kill, native crash in COMGR) breaks the
    pool: its in-flight jobs are rerun one per process, a job that kills its
    worker again is reported as failed, and the run goes on.

    ``limit`` caps how many jobs go through steps 2-3 (smoke tests).
    """
    digest = current_emitter_digest()
    comgr_id = current_comgr_id()
    # Only the directions these jobs live in: a shared cache can hold ~10^6
    # entries of other families, and every one parsed here is resident in the
    # process the compile workers fork from.
    index = cache.index({j.identity.direction for j in all_jobs})

    def _up_to_date(job) -> bool:
        meta = index.get(job.identity.stable_hash())
        return (
            meta is not None
            and meta.get("emitter_digest") == digest
            and meta.get("comgr_id") == comgr_id
        )

    pending = [j for j in all_jobs if not _up_to_date(j)]
    up_to_date = len(all_jobs) - len(pending)
    to_check = f"{len(pending)} to check"
    if limit is not None and len(pending) > limit:
        # The limit counts this call's work only -- each dtype gets its own
        # --limit -- and up-to-date entries never count against it.
        to_check = f"{limit} of {len(pending)} out-of-date to check (--limit {limit})"
        pending = pending[:limit]

    describe_jobs(all_jobs, log)
    log(
        f"AOT compile-all: {len(all_jobs)} variants for {arch}/{dtype} "
        f"({', '.join(directions)}); {up_to_date} up to date, "
        f"{to_check} with {jobs} job(s)"
    )
    if not pending:
        log("AOT compile-all done: nothing to do")
        return 0

    started = last_log = time.perf_counter()
    state = dict(emitted=0, rejected=0, compiled=0, failed=0, linked=0, reused=0)
    members: Dict[str, List[Tuple[BuildJob, dict]]] = {}
    in_flight: set = set()

    def _link(key: str, job: BuildJob, meta: dict) -> None:
        cache.link(
            job.identity, key, dict(meta, emitter_digest=digest, comgr_id=comgr_id)
        )
        state["linked"] += 1

    def _progress(force: bool = False) -> None:
        nonlocal last_log
        now = time.perf_counter()
        if not force and now - last_log < log_every_s:
            return
        last_log = now
        log(
            f"  emitted {state['emitted']}/{len(pending)} "
            f"({100.0 * state['emitted'] / len(pending):.1f}%), "
            f"{len(members)} distinct binaries: {state['reused']} already "
            f"compiled, {state['compiled']} compiled, {len(in_flight)} compiling "
            f"({now - started:.0f}s)"
        )

    def on_emitted(result, submit_compile) -> None:
        job, key, meta, err = result
        state["emitted"] += 1
        if err is not None:
            state["rejected"] += 1
            if state["rejected"] <= 10:
                log(f"  [skip] {job.identity.label()}: {err}")
        else:
            first = key not in members
            members.setdefault(key, []).append((job, meta))
            if cache.has_blob(key):
                # Already compiled (by an earlier run, or earlier in this
                # one): link straight away.
                if first and key not in in_flight:
                    state["reused"] += 1
                _link(key, job, meta)
            elif key not in in_flight:
                # The first job to emit a missing binary compiles it; any
                # later job with the same key is linked when it lands.
                in_flight.add(key)
                submit_compile((build, job, arch, dtype, key))
        _progress()

    def on_compiled(result) -> None:
        key, hsaco, kernel_name, err = result
        in_flight.discard(key)
        if err is not None:
            state["failed"] += 1
            if state["failed"] <= 10:
                job = members[key][0][0]
                log(f"  [fail] {job.identity.label()}: {err}")
            return
        cache.put_blob(key, hsaco, kernel_name)
        state["compiled"] += 1
        # Entries are written as each binary lands, so an interrupted run
        # keeps everything it finished.
        for job, meta in members[key]:
            _link(key, job, meta)
        _progress()

    emit_payloads = [(build, j, arch, dtype) for j in pending]
    if jobs <= 1:
        for payload in emit_payloads:
            on_emitted(_emit_worker(payload), lambda p: on_compiled(_compile_worker(p)))
    else:
        # One pool for both steps: a binary is compiled as soon as the first
        # job emitting it is done, so a few slow-to-emit kernels never hold
        # the compile workers idle.
        #
        # Emits are fed through a bounded window rather than submitted up
        # front: the executor's queue is FIFO, so a compile submitted behind
        # hundreds of thousands of queued emits would not start until every
        # emit ran (and submitting them all takes seconds by itself). With
        # the window a compile waits behind at most ~2*jobs emits.
        window = 2 * jobs
        # Forked workers share this process's heap copy-on-write, but a
        # garbage collection in a worker writes to every tracked object's
        # header and so copies every page holding one: with a large cache
        # index resident, each of the 64 workers grew a private copy of it
        # (GBs apiece) and the run was OOM-killed. Frozen objects are left
        # alone by the collector.
        gc.freeze()
        emit_iter = iter(emit_payloads)
        # Work handed over from a pool that broke (see below).
        carry: List[Tuple[str, tuple]] = []
        restarts = 0
        while True:
            broken: List[Tuple[str, tuple]] = []
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                futures: Dict = {}

                def submit(kind: str, payload) -> None:
                    fn = _emit_worker if kind == "emit" else _compile_worker
                    futures[pool.submit(fn, payload)] = (kind, payload)

                def submit_compile(payload) -> None:
                    submit("compile", payload)

                def refill() -> None:
                    while len(futures) < window:
                        payload = next(emit_iter, None)
                        if payload is None:
                            return
                        submit("emit", payload)

                for kind, payload in carry:
                    submit(kind, payload)
                carry = []
                refill()
                try:
                    while futures:
                        # The timeout keeps the progress line coming while only
                        # a few slow kernels are left and nothing finishes for
                        # minutes.
                        ready, _ = wait(
                            list(futures),
                            timeout=log_every_s,
                            return_when=FIRST_COMPLETED,
                        )
                        _progress()
                        for fut in ready:
                            kind, payload = futures[fut]
                            result = fut.result()
                            del futures[fut]
                            if kind == "emit":
                                on_emitted(result, submit_compile)
                            else:
                                on_compiled(result)
                        refill()
                except BrokenProcessPool:
                    # A worker died outright -- killed by the OOM killer, or a
                    # native crash inside the compiler -- and took the whole
                    # pool with it. Every job still in flight is a suspect.
                    broken = list(futures.values())
                    futures.clear()
            if not broken:
                break
            restarts += 1
            if restarts > _MAX_POOL_RESTARTS:
                raise RuntimeError(
                    f"compile workers died {restarts} times; giving up. A worker "
                    f"dying repeatedly usually means memory pressure -- rerun "
                    f"with a lower --jobs (was {jobs})."
                )
            log(
                f"  [warn] a worker process died (out of memory, or a crash in "
                f"the compiler); rerunning its {len(broken)} in-flight job(s) "
                f"one per process to find the culprit. Lower --jobs if this "
                f"repeats."
            )
            # Rerun the suspects one per single-worker pool, so a job that
            # kills its worker again breaks only its own pool and is reported.
            # Compiles they trigger go to the next main pool.
            for kind, payload, result in _run_isolated(broken, jobs):
                if kind == "emit":
                    on_emitted(result, lambda p: carry.append(("compile", p)))
                else:
                    on_compiled(result)
            _progress()

    _progress(force=True)
    log(
        f"AOT compile-all done: {state['compiled']} compiled, "
        f"{state['linked']} entries updated, {up_to_date} up to date, "
        f"{state['rejected']} rejected, {state['failed']} failed "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return 0


# ---------------------------------------------------------------------
# running a shape out of the cache
# ---------------------------------------------------------------------


def _launch_values_for(direction, problem, identity, ptrs, sizes, extras):
    from kernels.common.conv_args import ConvArgs

    tm, tn, tk = identity.tile_m, identity.tile_n, identity.tile_k
    if direction == "fwd":
        return ConvArgs.from_problem(problem, tile_m=tm, tile_n=tn).to_launch_values(
            *ptrs, *sizes
        )
    if direction == "wgrad":
        return ConvArgs.from_problem(
            problem, direction="wgrad", tile_m=tm, tile_n=tn, tile_k=tk
        ).to_launch_values(
            *ptrs,
            *sizes,
            split_k=extras.get("split_k", 1),
            ws_ptr=extras.get("ws_ptr"),
            ws_bytes=extras.get("ws_bytes"),
        )
    return ConvArgs.from_problem(
        problem, direction="dgrad", tile_m=tm, tile_n=tn
    ).to_launch_values(
        *ptrs,
        *sizes,
        sub_gemm_buf=extras["sub_gemm_buf"],
        num_sub_gemms=extras["num_sub_gemms"],
    )


def describe_cache(
    cache: KernelCache, log=print, directions: Optional[Sequence[str]] = None
) -> int:
    """Print what is in the cache, grouped by direction.

    ``directions`` limits the listing to the directions a run will use: every
    entry is a metadata file to parse, and a full implicit-GEMM cache holds
    hundreds of thousands of them -- reading all of them costs tens of seconds
    before a run that needs a few hundred.
    """
    by_direction: Dict[str, int] = {}
    entries = (
        cache.list_all()
        if directions is None
        else [e for d in directions for e in cache.list_all(d)]
    )
    for identity, _ in entries:
        by_direction[identity.direction] = by_direction.get(identity.direction, 0) + 1
    if not by_direction:
        if directions is None:
            log("AOT cache is empty.")
        else:
            log(f"AOT cache has no {', '.join(directions)} kernels.")
        return 2
    stale_by_direction = {}
    for direction, count in sorted(by_direction.items()):
        stale = cache.stale_entries(direction)
        stale_by_direction[direction] = stale
        note = f" ({stale} from older emitter sources)" if stale else ""
        log(f"  {direction:8s} {count} kernels{note}")
    stale_dirs = [d for d, n in stale_by_direction.items() if n]
    if stale_dirs:
        log(
            f"  [warn] {sum(stale_by_direction.values())} entries "
            f"({', '.join(stale_dirs)}) were built from different emitter sources "
            f"than this checkout -- built before a code change and not rebuilt "
            f"since. They still run; rerun --compile-all for those directions to "
            f"refresh them (only kernels whose code changed are recompiled)"
        )
    # What a rebuild from the names below reproduces against: the binary is a
    # function of these as well as of the kernel's identity.
    log(
        f"  this checkout: emitter {current_emitter_digest()[:12]}, "
        f"LLVM {current_llvm_flavor()}, comgr {current_comgr_id()}"
    )
    log(
        "  kernel names are complete identities; rebuild one with: "
        "python -m benchmarks.common.reproduce_kernel <name> --cache <cache dir>"
    )
    return 0
