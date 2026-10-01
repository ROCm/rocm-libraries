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
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)  # noqa: F401 (Tuple used in annotations)

from benchmarks.common.kernel_cache import KernelCache, KernelIdentity

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
CACHE_VECS: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_PIPELINES: Tuple[str, ...] = ("mem", "compv3", "compv4", "wavelet", "basic")
CACHE_EPILOGUES: Tuple[str, ...] = ("default", "cshuffle")

# ---- capability axes ------------------------------------------------------
# These are NOT tuning knobs: they change which problems a binary can serve.
# Grouped convolution takes a different code path in every direction (the
# contraction index only spans one group), and implicit-GEMM dgrad folds the
# stride and dilation into its tilde decomposition. A binary built one way
# produces wrong numbers -- not an error -- on the other, so each value needs
# its own binary and its own identity.
CACHE_GROUPED: Tuple[bool, ...] = (False, True)
CACHE_DGRAD_STRIDES: Tuple[int, ...] = (1, 2)
CACHE_DGRAD_DILATIONS: Tuple[int, ...] = (1, 2)

# Private aliases for internal use -- the generators below refer to these.
_TILE_MN = CACHE_TILE_MN
_TILE_K = CACHE_TILE_K
_WARP_MN = CACHE_WARP_MN
_WARP_TILE_MN = CACHE_WARP_TILE_MN
_VECS = CACHE_VECS
_PIPELINES = CACHE_PIPELINES
_EPILOGUES = CACHE_EPILOGUES
_GROUPED = CACHE_GROUPED
_DGRAD_STRIDES = CACHE_DGRAD_STRIDES
_DGRAD_DILATIONS = CACHE_DGRAD_DILATIONS


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


def _fwd_jobs(
    arch: str, dtype: str, wave_size: int, mma_family: str, target
) -> Iterator[BuildJob]:
    """Every forward implicit-GEMM variant worth caching for this arch/dtype."""
    da, db, dd = _dtype_triple(dtype)
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
        atom = target.mma.select_largest_k(
            family=mma_family, a_dtype=da, b_dtype=db, c_dtype="fp32", m=wt, n=wt
        )
        if atom is None or tile_k % atom.k:
            continue
        for vec_ab, vec_c in itertools.product(_VECS, _VECS):
            # async_dma replaces the K-loop driver entirely and ignores the
            # pipeline string, so only emit it once per geometry instead of
            # four identical binaries under different pipeline labels.
            k_loops = [(False, False), (True, False)]
            if pipeline == "mem":
                k_loops.append((False, True))
            for (unroll_k, async_dma), grouped in itertools.product(k_loops, _GROUPED):
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
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        **_async_chunks(cfg, (da, db, dd), dict(grouped=grouped)),
                        **cfg,
                    ),
                    direction="fwd",
                    spec_kwargs=cfg,
                    caps=dict(grouped=grouped),
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
    arch, dtype, wave_size, mma_family, target, split_ks
) -> Iterator[BuildJob]:
    da, db, dd = _dtype_triple(dtype)
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
        atom = target.mma.select_largest_k(
            family=mma_family, a_dtype=da, b_dtype=db, c_dtype="fp32", m=wt, n=wt
        )
        if atom is None or tile_k % atom.k:
            continue
        for vec_ab, vec_c, split_k in itertools.product(_VECS, _VECS, split_ks):
            two_stages = (False, True) if split_k > 1 else (False,)
            for two_stage, grouped in itertools.product(two_stages, _GROUPED):
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
                    split_k=split_k,
                    two_stage=two_stage,
                )
                yield BuildJob(
                    identity=KernelIdentity(
                        arch=arch,
                        direction="wgrad",
                        algorithm="implicit_gemm",
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        **cfg,
                    ),
                    direction="wgrad",
                    spec_kwargs=cfg,
                    caps=dict(grouped=grouped),
                )


def _dgrad_jobs(
    arch, dtype, wave_size, mma_family, target, max_sub_gemms
) -> Iterator[BuildJob]:
    da, db, dd = _dtype_triple(dtype)
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
        atom = target.mma.select_largest_k(
            family=mma_family, a_dtype=da, b_dtype=db, c_dtype="fp32", m=wt, n=wt
        )
        if atom is None or tile_k % atom.k:
            continue
        # dgrad folds the stride and dilation into its tilde decomposition,
        # so those are capabilities here, not launch parameters.
        for vec_ab, vec_c, stride, dilation, grouped in itertools.product(
            _VECS, _VECS, _DGRAD_STRIDES, _DGRAD_DILATIONS, _GROUPED
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
                vector_size_a=vec_ab,
                vector_size_b=vec_ab,
                vector_size_c=vec_c,
                max_sub_gemms=max_sub_gemms,
            )
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="dgrad",
                    algorithm="implicit_gemm",
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
                caps=dict(grouped=grouped, stride=stride, dilation=dilation),
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


def enumerate_jobs(
    *,
    arch: str,
    dtype: str,
    target,
    directions: Sequence[str],
    split_ks: Sequence[int] = (1, 2, 4, 8),
    max_sub_gemms: int = 64,
    validate: bool = True,
) -> List[BuildJob]:
    """All buildable jobs for the requested directions, deduped by identity.

    With ``validate=True`` (the default) each candidate is run through its
    direction's spec validator first, so the returned list is what will
    actually compile rather than the raw cross product.
    """
    wave_size = target.wave_size
    mma_family = "wmma" if wave_size == 32 else "mma"
    gens = {
        "fwd": lambda: _fwd_jobs(arch, dtype, wave_size, mma_family, target),
        "wgrad": lambda: _wgrad_jobs(
            arch, dtype, wave_size, mma_family, target, split_ks
        ),
        "dgrad": lambda: _dgrad_jobs(
            arch, dtype, wave_size, mma_family, target, max_sub_gemms
        ),
    }
    for direction in directions:
        if direction not in gens:
            raise ValueError(f"unknown direction {direction!r}")

    # Round-robin across the requested directions rather than draining one
    # before starting the next. The full grid is far larger than any single
    # cache build, so callers routinely truncate it with --params-limit; taking
    # them in order would make a truncated build contain only forward kernels
    # and silently leave the backward directions unserved.
    seen: Dict[str, BuildJob] = {}
    streams = [iter(gens[d]()) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            for job in stream:
                key = job.identity.stable_hash()
                if key in seen:
                    continue
                if validate and not _spec_is_valid(job, arch, dtype):
                    continue
                seen[key] = job
                still_running.append(stream)
                break
        streams = still_running
    return list(seen.values())


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
    return ConvProblem(
        N=1,
        Hi=extent,
        Wi=extent,
        C=64 * groups,
        K=64 * groups,
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


def build_and_compile(job: BuildJob, arch: str, dtype: str):
    """Build + compile one job. Returns ``(hsaco, kernel_name, meta)``.

    Raises on an invalid configuration; the caller counts those as skipped
    rather than failed -- the grid is deliberately over-generated and most
    rejections are ordinary spec-validity rules.
    """
    from rocke import compile_kernel
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

    artifact = compile_kernel(kernel, arch=arch)
    meta = {
        "kernel_name": artifact.kernel_name,
        "spec_kernel_name": spec.kernel_name(),
        # launch_block_size only exists where a pipeline appends extra waves
        # (wavelet); wgrad has no such pipeline and exposes block_size alone.
        "block_size": getattr(spec, "launch_block_size", spec.block_size),
        "timings": artifact.timings,
    }
    return artifact.hsaco, artifact.kernel_name, meta


def _worker(payload):
    """ProcessPoolExecutor entry point: rebuild the job and compile it.

    ``build`` is a module-level function so the payload pickles; it returns
    ``(hsaco, kernel_name, meta)`` like :func:`build_and_compile`.
    """
    build, job, arch, dtype = payload
    try:
        hsaco, kernel_name, meta = build(job, arch, dtype)
        return job, hsaco, meta, None
    except Exception as exc:  # noqa: BLE001 - reported per job, never fatal
        return job, None, None, f"{type(exc).__name__}: {exc}"


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
    return compile_jobs(
        cache=cache,
        all_jobs=enumerate_jobs(
            arch=arch, dtype=dtype, target=target, directions=directions
        ),
        build=build_and_compile,
        arch=arch,
        dtype=dtype,
        directions=directions,
        jobs=jobs,
        limit=limit,
        log=log,
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
) -> int:
    """Compile ``all_jobs`` with ``build`` and store what is not cached yet.

    Shared by every kernel family: the family only decides which jobs exist
    and how one is built; skipping cached entries, the ``limit`` cut and the
    process fan-out are the same for all of them.
    """
    pending = [j for j in all_jobs if not cache.has(j.identity)]
    cached = len(all_jobs) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    log(
        f"AOT compile-all: {len(all_jobs)} variants for {arch}/{dtype} "
        f"({', '.join(directions)}); {cached} already cached, "
        f"{len(pending)} to build with {jobs} job(s)"
    )

    built = failed = 0
    started = time.perf_counter()
    payloads = [(build, j, arch, dtype) for j in pending]

    def _record(job, hsaco, meta, err):
        nonlocal built, failed
        if err is not None:
            failed += 1
            if failed <= 10:
                log(f"  [skip] {job.identity.short_label()}: {err}")
            return
        cache.put(job.identity, hsaco, meta)
        built += 1
        if built % 50 == 0:
            log(f"  ... {built} built")

    if jobs <= 1:
        for payload in payloads:
            _record(*_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_worker, p) for p in payloads]
            for fut in as_completed(futures):
                _record(*fut.result())

    elapsed = time.perf_counter() - started
    log(
        f"AOT compile-all done: {built} built, {cached} already cached, "
        f"{failed} rejected ({elapsed:.1f}s)"
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


def describe_cache(cache: KernelCache, log=print) -> int:
    """Print what is in the cache, grouped by direction."""
    by_direction: Dict[str, int] = {}
    for identity, _ in cache.list_all():
        by_direction[identity.direction] = by_direction.get(identity.direction, 0) + 1
    if not by_direction:
        log("AOT cache is empty.")
        return 2
    for direction, count in sorted(by_direction.items()):
        log(f"  {direction:8s} {count} kernels")
    return 0
