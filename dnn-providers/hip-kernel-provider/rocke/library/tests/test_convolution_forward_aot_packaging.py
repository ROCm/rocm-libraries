# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT forward instances (hipkernel:Gfx950ConvFwdAot).

An instance is packaged from a rocKE ``KernelIdentity``; these tests hold the adapter to
the sweep it comes from. Implicit GEMM: the same kernel IR as ``kernel_sweep.build_kernel``
and the same served-problem predicate as ``KernelCache.supports_problem``. Direct, every
forward algorithm of the direct sweep: the same probe and IR as
``direct_kernel_sweep.build_direct_job``, the served problems of ``supports_problem`` plus
the per-shape re-validation ``plan_for`` runs, and ``plan_for``'s launch.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.common import direct_kernel_sweep, kernel_sweep
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_forward import (
    AOT_DIRECT_ALGORITHMS,
    DIRECT_VARIANT_GROUPED,
    DIRECT_VARIANT_NONGROUPED,
    KERNEL_FAMILY_DIRECT_DEPTHWISE,
    Gfx950ConvFwdAotSpec,
    Gfx950ConvFwdSpec,
    aot_direct_fields_from_knobs,
    aot_direct_instance_fields,
    build_gfx950_conv_fwd_aot,
    gfx950_conv_fwd_aot_problem_for_request,
    gfx950_conv_fwd_aot_runtime_candidates,
    gfx950_conv_fwd_aot_serves,
    supports_gfx950_conv_fwd_aot,
    supports_gfx950_conv_fwd_aot_problem,
)
from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common.conv_abi import conv_direct_args_signature
from kernels.common.conv_args import ConvArgs
from kernels.common.conv_direct_grouped import DirectConvProblem
from rocke.core.arch import ArchTarget
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_SEED = 2026
_SAMPLE_PER_K_LOOP = 3


def _spec(identity: KernelIdentity) -> Gfx950ConvFwdAotSpec:
    return Gfx950ConvFwdAotSpec(
        dtype=identity.dtype_a,
        tile_m=identity.tile_m,
        tile_n=identity.tile_n,
        tile_k=identity.tile_k,
        warp_m=identity.warp_m,
        warp_n=identity.warp_n,
        warp_tile_m=identity.warp_tile_m,
        warp_tile_n=identity.warp_tile_n,
        warp_tile_k=identity.warp_tile_k,
        wave_size=identity.wave_size,
        pipeline=identity.pipeline,
        epilogue=identity.epilogue,
        vector_size_a=identity.vector_size_a,
        vector_size_b=identity.vector_size_b,
        vector_size_c=identity.vector_size_c,
        unroll_k=int(identity.unroll_k),
        async_dma=int(identity.async_dma),
        async_chunk_a=identity.async_chunk_a,
        async_chunk_b=identity.async_chunk_b,
        grouped=int(identity.grouped),
        instance_id=identity.stable_hash(),
    )


@pytest.fixture(scope="module")
def sweep_jobs():
    """A seeded sample of the AOT sweep's valid gfx950 forward jobs, per dtype and K loop."""
    target = ArchTarget.from_gfx("gfx950")
    rng = random.Random(_SEED)
    sample = []
    for dtype in ("fp16", "bf16"):
        jobs = kernel_sweep._direction_jobs("fwd", "gfx950", dtype, target, (2,), 64)
        by_loop: dict = {}
        for job in jobs:
            key = (job.identity.unroll_k, job.identity.async_dma)
            by_loop.setdefault(key, []).append(job)
        for key in sorted(by_loop):
            candidates = by_loop[key]
            rng.shuffle(candidates)
            picked = [
                job
                for job in itertools.islice(
                    (
                        j
                        for j in candidates
                        if kernel_sweep._spec_is_valid(j, "gfx950", dtype)
                    ),
                    _SAMPLE_PER_K_LOOP,
                )
            ]
            assert len(picked) == _SAMPLE_PER_K_LOOP, (dtype, key)
            sample += [(dtype, job) for job in picked]
    return sample


def _knob_class(identity: KernelIdentity) -> tuple:
    """The compile-time knob class of an implicit-GEMM identity: everything but the
    tile and warp geometry."""
    k_loop = (
        "async_dma"
        if identity.async_dma
        else "unroll" if identity.unroll_k else "plain"
    )
    return (
        identity.pipeline,
        identity.epilogue,
        k_loop,
        identity.vector_size_a,
        identity.vector_size_b,
        identity.vector_size_c,
        identity.async_chunk_a,
        identity.async_chunk_b,
    )


@pytest.fixture(scope="module")
def knob_class_jobs():
    """One valid job of every knob class of the AOT sweep's gfx950 forward grid, per
    dtype, drawn by a seeded shuffle: every pipeline, epilogue, K loop (with each
    async chunk pair) and load/store width combination the shipped catalog holds."""
    target = ArchTarget.from_gfx("gfx950")
    rng = random.Random(_SEED + 2)
    sample = []
    for dtype in ("fp16", "bf16"):
        by_class: dict = {}
        for job in kernel_sweep._direction_jobs(
            "fwd", "gfx950", dtype, target, (2,), 64
        ):
            if kernel_sweep._spec_is_valid(job, "gfx950", dtype):
                by_class.setdefault(_knob_class(job.identity), []).append(job)
        for key in sorted(by_class, key=repr):
            sample.append((dtype, rng.choice(by_class[key])))
    return sample


def test_the_knob_class_sample_covers_every_knob_value(knob_class_jobs):
    classes = {_knob_class(job.identity) for _, job in knob_class_jobs}
    assert {c[0] for c in classes} == {"mem", "compv3", "compv4"}
    assert {c[1] for c in classes} == {"default", "cshuffle"}
    assert {c[2] for c in classes} == {"plain", "unroll", "async_dma"}
    for position in (3, 4, 5):
        assert {c[position] for c in classes} == {1, 2, 4, 8}
    assert {(c[1], c[5]) for c in classes} >= {("default", 1), ("cshuffle", 8)}
    assert len({c[6:] for c in classes if c[2] == "async_dma"}) > 1


def test_every_knob_class_is_packageable_with_the_sweeps_ir(knob_class_jobs):
    """The packaged kernel is the sweep's, byte for byte up to its symbol, for one
    instance of every knob class in each dtype."""
    for dtype, job in knob_class_jobs:
        spec = _spec(job.identity)
        assert supports_gfx950_conv_fwd_aot(spec) == (True, "ok"), job.identity.label()
        ours = build_gfx950_conv_fwd_aot(spec)
        theirs, _ = kernel_sweep.build_kernel(job, "gfx950", dtype)
        ours_ir = _lower_kernel_to_llvm_python(
            ours, arch="gfx950", llvm_flavor="llvm22"
        )
        theirs_ir = _lower_kernel_to_llvm_python(
            theirs, arch="gfx950", llvm_flavor="llvm22"
        )
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), job.identity.label()


def test_every_sampled_sweep_identity_is_packageable(sweep_jobs):
    for _, job in sweep_jobs:
        assert supports_gfx950_conv_fwd_aot(_spec(job.identity)) == (True, "ok")


def test_packaged_ir_is_the_sweep_kernel_up_to_its_symbol(sweep_jobs):
    for dtype, job in sweep_jobs:
        spec = _spec(job.identity)
        ours = build_gfx950_conv_fwd_aot(spec)
        theirs, _ = kernel_sweep.build_kernel(job, "gfx950", dtype)
        assert ours.name.startswith(spec.kernel_symbol_prefix())
        ours_ir = _lower_kernel_to_llvm_python(
            ours, arch="gfx950", llvm_flavor="llvm22"
        )
        theirs_ir = _lower_kernel_to_llvm_python(
            theirs, arch="gfx950", llvm_flavor="llvm22"
        )
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), job.identity.label()


def _random_problems(count: int, dtype: str):
    rng = random.Random(_SEED + 1)
    problems = []
    while len(problems) < count:
        groups = rng.choice([1, 1, 1, 2, 3, 4, 8, 16, 32])
        cpg = rng.choice([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64])
        kpg = rng.choice([1, 2, 3, 4, 5, 8, 12, 16, 20, 32, 64])
        filt = rng.choice([1, 3, 5, 7])
        problem = Gfx950ConvFwdSpec(
            N=rng.choice([1, 2, 4, 64]),
            Hi=rng.choice([7, 14, 28, 56, 224]),
            Wi=rng.choice([7, 14, 28, 56, 224]),
            C=groups * cpg,
            K=groups * kpg,
            Y=filt,
            X=filt,
            sH=rng.choice([1, 2]),
            sW=rng.choice([1, 2]),
            pH=filt // 2,
            pW=filt // 2,
            groups=groups,
            dtype=dtype,
        )
        problems.append(problem)
    return problems


def test_served_problems_agree_with_kernel_cache_supports_problem(sweep_jobs):
    """Every (instance, problem) verdict matches the AOT cache's own predicate.

    supports_problem leaves out the bounds ConvArgs checks at launch (the 24-bit
    reduction extent) and the packaged contract's own problem checks, so those
    problems are excluded rather than compared.
    """
    cache = KernelCache.__new__(KernelCache)
    compared = served = 0
    for dtype, job in sweep_jobs:
        identity = replace(job.identity, llvm_flavor=kernel_sweep.current_llvm_flavor())
        spec = _spec(job.identity)
        for problem in _random_problems(60, dtype):
            ours, reason = gfx950_conv_fwd_aot_serves(spec, problem)
            if not ours and ("2**23" in reason or "32-bit" in reason):
                continue
            theirs, why = cache.supports_problem(identity, problem.to_problem())
            assert ours == theirs, (job.identity.label(), problem, reason, why)
            compared += 1
            served += ours
    # Both verdicts occur, so the agreement is not vacuous.
    assert compared > 1000 and 0 < served < compared


def test_launch_values_and_grid_are_conv_args_for_the_runtime_problem(sweep_jobs):
    _, job = sweep_jobs[0]
    spec = _spec(job.identity)
    for problem in _random_problems(20, spec.dtype):
        if not gfx950_conv_fwd_aot_serves(spec, problem)[0]:
            continue
        args = ConvArgs.from_problem(
            problem.to_problem(), tile_m=spec.tile_m, tile_n=spec.tile_n
        )
        assert spec.grid(problem) == args.grid()
        assert spec.launch_values(problem, 0, 0, 0, 1, 2, 3) == args.to_launch_values(
            0, 0, 0, 1, 2, 3
        )


def test_one_instance_serves_dense_grouped_and_grouped_pointwise_problems():
    target = ArchTarget.from_gfx("gfx950")
    job = next(
        j
        for j in kernel_sweep._direction_jobs("fwd", "gfx950", "fp16", target, (2,), 64)
        if (
            j.identity.tile_m,
            j.identity.tile_n,
            j.identity.vector_size_a,
            j.identity.vector_size_c,
        )
        == (64, 64, 1, 1)
        and not (j.identity.unroll_k or j.identity.async_dma)
        and kernel_sweep._spec_is_valid(j, "gfx950", "fp16")
    )
    spec = _spec(job.identity)
    for problem in (
        Gfx950ConvFwdSpec(N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1),
        Gfx950ConvFwdSpec(
            N=2, Hi=11, Wi=11, C=12, K=24, Y=3, X=3, pH=1, pW=1, groups=4
        ),
        Gfx950ConvFwdSpec(
            N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1, groups=32
        ),
        # The exact-shape contract declines grouped pointwise; an instance was not
        # built as pointwise, so it serves it.
        Gfx950ConvFwdSpec(N=2, Hi=8, Wi=8, C=32, K=32, Y=1, X=1, groups=4),
    ):
        assert gfx950_conv_fwd_aot_serves(spec, problem) == (True, "ok"), problem


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"dtype": "fp32"}, "fp16 or bf16"),
        ({"layout": "NCHW"}, "NHWC"),
        ({"wave_size": 32}, "wave_size"),
        ({"pipeline": "wavelet"}, "pipelines"),
        ({"epilogue": "none"}, "epilogue"),
        ({"vector_size_c": 3}, "vector_size_c"),
        ({"unroll_k": 2}, "unroll_k"),
        ({"unroll_k": 1, "async_dma": 1}, "separate K-loop"),
        ({"async_chunk_a": 8}, "0 without async_dma"),
        ({"instance_id": "xyz"}, "stable_hash"),
    ],
)
def test_out_of_contract_instances_are_declined(sweep_jobs, change, reason):
    plain = next(
        _spec(job.identity)
        for _, job in sweep_jobs
        if not (job.identity.unroll_k or job.identity.async_dma)
    )
    ok, why = supports_gfx950_conv_fwd_aot(replace(plain, **change))
    assert not ok and reason in why, why


def test_recorded_async_chunks_must_be_the_builders(sweep_jobs):
    spec = next(_spec(job.identity) for _, job in sweep_jobs if job.identity.async_dma)
    ok, why = supports_gfx950_conv_fwd_aot(
        replace(spec, async_chunk_a=spec.async_chunk_a * 2)
    )
    assert not ok and "async chunk widths" in why


def test_profile_dispatch_resolves_the_problem_including_grouped_pointwise():
    request = ConvGroupedRequest(
        N=2, Hi=8, Wi=8, C=32, K=32, Y=1, X=1, G=4, arch="gfx950", dtype="fp16"
    )
    problem = gfx950_conv_fwd_aot_problem_for_request(request)
    assert (problem.groups, problem.Y, problem.X) == (4, 1, 1)
    assert supports_gfx950_conv_fwd_aot_problem(problem) == (True, "ok")
    with pytest.raises(ValueError, match="2D"):
        gfx950_conv_fwd_aot_problem_for_request(replace(request, Di=4, Z=1))


# ---- Runtime selection: which serving instances run ----

#: (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups). Odd and even C/groups and
#: K/groups, C/groups of 2 mod 4, grouped (odd group counts too), depthwise, pointwise
#: and grouped pointwise, strided, dilated and large filters, and two output extents
#: past 65535 small M tiles, the second past every tile but the largest, so the grid
#: bound decides which width combinations remain.
_SELECTION_GRID = (
    (2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1),
    (2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (1, 3, 5, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 10, 6, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (2, 32, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    (2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4),
    (1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3),
    (2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    (2, 64, 128, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3),
    (2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1),
    (1, 4, 12, 31, 31, 11, 11, 4, 4, 5, 5, 1, 1, 1),
    (32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (36, 2, 2, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1),
)


@pytest.fixture(scope="module")
def full_sweep():
    """Every valid gfx950 forward identity of the AOT sweep, per dtype, in rocKE's order."""
    target = ArchTarget.from_gfx("gfx950")
    jobs = max(1, min(os.cpu_count() or 1, 32))
    return {
        dtype: [
            job.identity
            for job in kernel_sweep.enumerate_jobs(
                arch="gfx950",
                dtype=dtype,
                target=target,
                directions=("fwd",),
                jobs=jobs,
            )
        ]
        for dtype in ("fp16", "bf16")
    }


def test_runtime_candidates_are_rockes_runtime_selection(full_sweep):
    """Over the whole sweep, the instances selected for a problem are exactly the ones
    rocKE's --run-from-cache benchmark runs by default: KernelCache.compatible, then
    _widest_vector_sizes. Both functions are rocKE's own, called on the identities."""
    from benchmarks.common.benchmark_implicit_gemm_conv import _widest_vector_sizes

    narrowed = kept_default = 0
    for dtype, identities in full_sweep.items():
        cache = KernelCache.__new__(KernelCache)
        cache._entries = lambda direction, ids=identities: [(i, None, {}) for i in ids]
        specs = [_spec(identity) for identity in identities]
        for values in _SELECTION_GRID:
            n, c, k, hi, wi, y, x, sh, sw, ph, pw, dh, dw, groups = values
            problem = Gfx950ConvFwdSpec(
                N=n, C=c, K=k, Hi=hi, Wi=wi, Y=y, X=x, sH=sh, sW=sw, pH=ph, pW=pw,
                dH=dh, dW=dw, groups=groups, dtype=dtype,
            )  # fmt: skip
            compatible = [
                entry
                for entry in cache.compatible(problem.to_problem(), direction="fwd")
                if entry[0].dtype_a == dtype
            ]
            theirs = {
                ident.stable_hash() for ident, *_ in _widest_vector_sizes(compatible)
            }
            ours = gfx950_conv_fwd_aot_runtime_candidates(problem, specs)
            assert [s.instance_id for s in ours] == [
                s.instance_id for s in specs if s.instance_id in theirs
            ], (dtype, values)
            assert len(ours) == len(theirs), (dtype, values)
            narrowed += len(ours) < len(compatible)
            kept_default += any(s.epilogue == "default" for s in ours)
    # The policy narrows wherever some width is wider than 1, and keeps the scalar-store
    # epilogue where nothing stores wider.
    assert narrowed > len(_SELECTION_GRID) and kept_default > 0


def test_runtime_candidates_keep_every_serving_direct_instance(direct_jobs):
    specs = [_direct_spec(job.identity) for _, job in direct_jobs]
    problem = Gfx950ConvFwdSpec(
        N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1, groups=32
    )
    serving = [s for s in specs if gfx950_conv_fwd_aot_serves(s, problem)[0]]
    assert serving and gfx950_conv_fwd_aot_runtime_candidates(problem, specs) == serving


# ---- Direct instances ----

#: Direct identities per (dtype, algorithm) in the sweep's forward grid.
_DIRECT_COUNTS = {
    "direct_depthwise": 54,
    "direct_depthwise_spatial": 30,
    "direct_depthwise_tiled": 1536,
    "direct_depthwise_col": 540,
    "direct_grouped": 80,
    "direct_nongrouped": 3058,
}
#: Identities per (dtype, algorithm) the IR and serving tests sample, spread evenly
#: over the sweep's order (capabilities outermost, then knobs). std and spatial are
#: checked whole. ``check_direct_ir.py``-style runs of every identity are out of tree.
_DIRECT_SAMPLE = 10


def _direct_spec(identity: KernelIdentity) -> Gfx950ConvFwdAotSpec:
    """The instance make_conv_aot_config.py packages for a direct identity."""
    return replace(_spec(identity), **aot_direct_instance_fields(identity))


@pytest.fixture(scope="module")
def direct_jobs():
    """Every forward job of the direct sweep that the AOT engine packages, both
    dtypes, in the sweep's order."""
    target = ArchTarget.from_gfx("gfx950")
    return [
        (dtype, job)
        for dtype in ("fp16", "bf16")
        for job in direct_kernel_sweep.enumerate_direct_jobs(
            arch="gfx950", target=target, directions=("fwd",), dtype=dtype
        )
        if job.identity.algorithm in AOT_DIRECT_ALGORITHMS
    ]


def _sampled(direct_jobs, count: int = _DIRECT_SAMPLE):
    """Whole std and spatial; ``count`` evenly spread jobs of every other algorithm."""
    by_cell: dict = {}
    for dtype, job in direct_jobs:
        by_cell.setdefault((dtype, job.identity.algorithm), []).append(job)
    picked = []
    for (dtype, algorithm), jobs in by_cell.items():
        if algorithm in ("direct_depthwise", "direct_depthwise_spatial"):
            picked += [(dtype, job) for job in jobs]
        else:
            step = len(jobs) / count
            picked += [(dtype, jobs[int(i * step)]) for i in range(count)]
    return picked


def test_every_direct_sweep_identity_is_packageable_on_the_sweeps_probe(direct_jobs):
    counts: dict = {}
    for dtype, job in direct_jobs:
        counts[dtype, job.identity.algorithm] = (
            counts.get((dtype, job.identity.algorithm), 0) + 1
        )
        spec = _direct_spec(job.identity)
        assert supports_gfx950_conv_fwd_aot(spec) == (True, "ok"), job.identity.label()
        assert spec.direct_variant == AOT_DIRECT_ALGORITHMS[job.identity.algorithm]
        caps = direct_kernel_sweep.DirectCaps(**job.caps)
        assert spec.probe_problem() == direct_kernel_sweep.probe_problem(
            caps, dtype, job.identity.algorithm
        )
        assert spec.direct_knobs() == json.loads(job.identity.knobs)
    assert counts == {
        (dtype, algorithm): count
        for dtype in ("fp16", "bf16")
        for algorithm, count in _DIRECT_COUNTS.items()
    }


def test_packaged_direct_ir_is_the_sweep_kernel_up_to_its_symbol(direct_jobs):
    sample = _sampled(direct_jobs)
    assert {job.identity.algorithm for _, job in sample} == set(_DIRECT_COUNTS)
    for dtype, job in sample:
        spec = _direct_spec(job.identity)
        ours = build_gfx950_conv_fwd_aot(spec)
        theirs, _ = direct_kernel_sweep.build_direct_job(job, "gfx950", dtype)
        assert ours.name.startswith(spec.kernel_symbol_prefix())
        assert ours.name == spec.kernel_name()
        ours_ir = _lower_kernel_to_llvm_python(
            ours, arch="gfx950", llvm_flavor="llvm22"
        )
        theirs_ir = _lower_kernel_to_llvm_python(
            theirs, arch="gfx950", llvm_flavor="llvm22"
        )
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), job.identity.label()


def _depthwise_problems(dtype: str):
    """Depthwise-heavy problems over the sweep's filters and strides, with padding,
    dilation and channel-multiplier variations the direct kernels must decline."""
    rng = random.Random(_SEED + 2)
    problems = []
    for _ in range(300):
        groups = rng.choice([1, 2, 5, 8, 16, 32, 48, 63, 64, 96, 256])
        cpg, kpg = rng.choice([(1, 1)] * 6 + [(1, 2), (2, 2)])
        filt = rng.choice([3, 5, 7, 9, 11])
        stride = rng.choice([1, 2])
        dilation = rng.choice([1] * 5 + [2])
        pad = rng.choice([filt // 2] * 4 + [0, filt // 2 + 1])
        problems.append(
            Gfx950ConvFwdSpec(
                N=rng.choice([1, 2, 3]),
                Hi=rng.choice([7, 12, 17, 31, 56]),
                Wi=rng.choice([7, 12, 17, 31, 56]),
                C=groups * cpg,
                K=groups * kpg,
                Y=filt,
                X=filt,
                sH=stride,
                sW=stride,
                pH=pad,
                pW=pad,
                dH=dilation,
                dW=dilation,
                groups=groups,
                dtype=dtype,
            )
        )
    return problems


def _instance_problems(spec: Gfx950ConvFwdAotSpec, count: int = 24):
    """Problems around one direct instance: mostly its own compiled filter, stride,
    padding, dilation and channel counts over varied batches, images and group counts
    (tails, images smaller than a block, too small for the filter), and a share with
    one capability changed, which it must decline. Only problems inside the graph
    contract are kept."""
    rng = random.Random(f"{_SEED}-{spec.instance_id}-{spec.dtype}")
    variant = spec.direct_variant
    problems = []
    while len(problems) < count:
        filt, stride, pad = spec.filter_h, spec.stride_h, spec.pad_h
        dil_h, dil_w = spec.dilation_h, spec.dilation_w
        stride_w, pad_w = stride, pad
        change = rng.choice([None] * 7 + ["filter", "stride", "pad", "dil", "axes"])
        if change == "filter":
            filt = rng.choice([1, 3, 5, 7, 13]) if filt != 3 else 5
        elif change == "stride":
            stride = stride_w = 3 - stride if stride in (1, 2) else 1
        elif change == "pad":
            pad = pad_w = rng.choice([p for p in (0, 1, 2, filt // 2 + 1) if p != pad])
        elif change == "dil":
            dil_h = dil_w = 2 if dil_h == 1 else 1
        elif change == "axes":
            stride_w = stride + 1
        if variant == DIRECT_VARIANT_NONGROUPED:
            groups = rng.choice([1] * 6 + [2])
            c = rng.choice([8, 16, 24, 32, 48, 64, 96, 128, 256])
            k = rng.choice([16, 24, 32, 48, 64, 96, 128, 160, 256])
            c, k = c * groups, k * groups
            hi = rng.choice([3, 8, 13, 17, 28, 33])
            wi = rng.choice([14, 16, 28, 30, 32, 48, 56, 64, 96, 100, 192])
        else:
            if variant == DIRECT_VARIANT_GROUPED:
                cpg, kpg = rng.choice(
                    [(spec.cpg, spec.kpg)] * 6
                    + [(spec.cpg, 2 * spec.kpg), (4 + spec.cpg, 4 + spec.kpg)]
                )
                groups = rng.choice([1, 2, 3, 4, 6, 8, 16, 24, 32, 48, 64, 128])
            else:
                cpg, kpg = rng.choice([(1, 1)] * 8 + [(1, 2), (2, 2)])
                groups = rng.choice([1, 2, 5, 8, 16, 32, 48, 63, 64, 65, 96, 128, 300])
            c, k = groups * cpg, groups * kpg
            hi = rng.choice([1, 2, 3, 5, 7, 8, 12, 17, 28, 31, 33, 56])
            wi = rng.choice([1, 2, 3, 5, 7, 8, 12, 17, 28, 31, 33, 56])
        problem = Gfx950ConvFwdSpec(
            N=rng.choice([1, 2, 3]),
            Hi=hi,
            Wi=wi,
            C=c,
            K=k,
            Y=filt,
            X=filt,
            sH=stride,
            sW=stride_w,
            pH=pad,
            pW=pad_w,
            dH=dil_h,
            dW=dil_w,
            groups=groups,
            dtype=spec.dtype,
        )
        if supports_gfx950_conv_fwd_aot_problem(problem)[0]:
            problems.append(problem)
    return problems


def _rocke_direct_serves(identity: KernelIdentity, problem: Gfx950ConvFwdSpec):
    """What rocKE's direct benchmark offers ``identity`` for (``direct_plans``): the
    cache's capability match, the non-grouped family only for one group, then
    ``plan_for``'s re-validation of the spec on the real problem."""
    if problem.dtype != identity.dtype_a:
        return False, "dtype"
    direct = DirectConvProblem(
        N=problem.N,
        H=problem.Hi,
        W=problem.Wi,
        groups=problem.groups,
        cpg=problem.C // problem.groups,
        kpg=problem.K // problem.groups,
        KH=problem.Y,
        KW=problem.X,
        PAD=problem.pH,
        stride=problem.sH,
        dtype=problem.dtype,
        dil_h=problem.dH,
        dil_w=problem.dW,
    )
    cache = KernelCache.__new__(KernelCache)
    ok, why = cache.supports_problem(identity, direct)
    if not ok:
        return False, why
    if identity.algorithm == "direct_nongrouped" and direct.groups != 1:
        return False, "direct_plans offers the non-grouped family one group only"
    plan, why = direct_kernel_sweep.plan_for(
        cache, identity, Path("unused.hsaco"), {"kernel_name": "k"}, direct, "gfx950"
    )
    return plan is not None, why


def _known_difference(spec, problem, ours: bool, reason: str, theirs: bool) -> bool:
    """Where the packaged guard is knowingly stricter than rocKE's offer: rocKE's
    spatial validator admits 64 groups (one column per wave), which its own benchmark
    never runs and the direct guard declines, and rocKE's direct problem has one
    stride and one padding, so it cannot state a graph whose axes differ."""
    if problem.sH != problem.sW or problem.pH != problem.pW:
        assert not ours, problem
        return True
    if spec.direct_variant == "spatial" and theirs and not ours:
        assert problem.groups == 64 and "groups < 64" in reason, reason
        return True
    return False


def test_direct_served_problems_agree_with_the_direct_benchmarks_offer(direct_jobs):
    """Every (direct instance, problem) verdict matches what rocKE would launch."""
    compared: dict = {}
    served: dict = {}
    for dtype, job in _sampled(direct_jobs):
        identity = replace(job.identity, llvm_flavor=kernel_sweep.current_llvm_flavor())
        spec = _direct_spec(job.identity)
        problems = _instance_problems(spec)
        if spec.direct_variant in ("std", "spatial"):
            problems += _depthwise_problems(dtype)[::15]
        for problem in problems:
            ours, reason = gfx950_conv_fwd_aot_serves(spec, problem)
            theirs, why = _rocke_direct_serves(identity, problem)
            if _known_difference(spec, problem, ours, reason, theirs):
                continue
            assert ours == theirs, (job.identity.label(), problem, reason, why)
            compared[spec.direct_variant] = compared.get(spec.direct_variant, 0) + 1
            served[spec.direct_variant] = served.get(spec.direct_variant, 0) + ours
    # Both verdicts occur for every variant, so the agreement is not vacuous.
    for variant, count in compared.items():
        assert count > 300 and 0 < served[variant] < count, (variant, count)
    assert set(compared) == set(AOT_DIRECT_ALGORITHMS.values())


def test_direct_launch_is_rockes_for_the_runtime_problem(direct_jobs):
    for _, job in _sampled(direct_jobs, 4)[::3]:
        spec = _direct_spec(job.identity)
        problems = [
            p
            for p in _instance_problems(spec, 48)
            if gfx950_conv_fwd_aot_serves(spec, p)[0]
        ]
        assert problems, job.identity.label()
        for problem in problems:
            assert spec.launch_signature() == conv_direct_args_signature(spec.dtype)
            direct = DirectConvProblem(
                N=problem.N,
                H=problem.Hi,
                W=problem.Wi,
                groups=problem.groups,
                cpg=problem.C // problem.groups,
                kpg=problem.K // problem.groups,
                KH=problem.Y,
                KW=problem.X,
                PAD=problem.pH,
                stride=problem.sH,
                dtype=problem.dtype,
                dil_h=problem.dH,
                dil_w=problem.dW,
            )
            plan, why = direct_kernel_sweep.plan_for(
                KernelCache.__new__(KernelCache),
                replace(job.identity, llvm_flavor=kernel_sweep.current_llvm_flavor()),
                Path("unused.hsaco"),
                {"kernel_name": "k"},
                direct,
                "gfx950",
            )
            assert plan is not None, why
            (step,) = plan.steps
            assert (spec.grid(problem), spec.block()) == (step.grid, step.block)
            assert spec.launch_values(
                problem, 0, 0, 0, 1, 2, 3
            ) == step.conv_args.to_launch_values(0, 0, 0, 1, 2, 3)


def test_one_direct_instance_serves_other_images_batches_and_group_counts(direct_jobs):
    spatial = next(
        _direct_spec(job.identity)
        for _, job in direct_jobs
        if job.identity.algorithm == "direct_depthwise_spatial"
        and (job.identity.filter_h, job.identity.stride_h) == (3, 1)
    )
    std = replace(spatial, direct_variant="std", block_w=4)
    for groups, n, h, w in (
        (8, 1, 64, 64),
        (5, 3, 9, 21),
        (32, 2, 14, 14),
        (63, 1, 7, 7),
    ):
        problem = Gfx950ConvFwdSpec(
            N=n, Hi=h, Wi=w, C=groups, K=groups, Y=3, X=3, pH=1, pW=1, groups=groups
        )
        assert gfx950_conv_fwd_aot_serves(spatial, problem) == (True, "ok"), problem
        assert gfx950_conv_fwd_aot_serves(std, problem) == (True, "ok"), problem
    wide = Gfx950ConvFwdSpec(
        N=2, Hi=14, Wi=14, C=96, K=96, Y=3, X=3, pH=1, pW=1, groups=96
    )
    assert gfx950_conv_fwd_aot_serves(std, wide) == (True, "ok")
    assert not gfx950_conv_fwd_aot_serves(spatial, wide)[0]
    # The filter, stride and padding are compiled in.
    for change in ({"Y": 5, "X": 5, "pH": 2, "pW": 2}, {"sH": 2, "sW": 2}):
        ok, why = gfx950_conv_fwd_aot_serves(std, replace(wide, **change))
        assert not ok and "compiled for" in why, why


def _first(direct_jobs, algorithm: str, **caps) -> Gfx950ConvFwdAotSpec:
    return next(
        _direct_spec(job.identity)
        for _, job in direct_jobs
        if job.identity.algorithm == algorithm
        and all(getattr(job.identity, k) == v for k, v in caps.items())
    )


def test_col_serves_valid_padding_and_bounds_its_rows_by_the_output(direct_jobs):
    valid = _first(direct_jobs, "direct_depthwise_col", filter_h=5, pad_h=0)
    same = _first(direct_jobs, "direct_depthwise_col", filter_h=5, pad_h=2)
    problem = Gfx950ConvFwdSpec(
        N=2, Hi=13, Wi=17, C=24, K=24, Y=5, X=5, pH=0, pW=0, groups=24
    )
    assert gfx950_conv_fwd_aot_serves(valid, problem) == (True, "ok")
    ok, why = gfx950_conv_fwd_aot_serves(same, problem)
    assert not ok and "padding" in why
    ok, _ = gfx950_conv_fwd_aot_serves(same, replace(problem, pH=2, pW=2))
    assert ok
    # A block wider than the output is declined by rocKE's validator.
    wide = next(
        _direct_spec(job.identity)
        for _, job in direct_jobs
        if job.identity.algorithm == "direct_depthwise_col"
        and (job.identity.filter_h, job.identity.pad_h, job.identity.stride_h)
        == (5, 0, 1)
        and json.loads(job.identity.knobs)["block_w"] == 4
    )
    ok, why = gfx950_conv_fwd_aot_serves(wide, replace(problem, Wi=6))
    assert not ok and "block_w 4 > Wo 2" in why, why


def test_tiled_serves_large_filters_the_row_stream_cannot(direct_jobs):
    tiled = _first(direct_jobs, "direct_depthwise_tiled", filter_h=31, stride_h=2)
    problem = Gfx950ConvFwdSpec(
        N=1, Hi=40, Wi=37, C=96, K=96, Y=31, X=31, sH=2, sW=2, pH=15, pW=15, groups=96
    )
    assert gfx950_conv_fwd_aot_serves(tiled, problem) == (True, "ok")
    assert not any(
        gfx950_conv_fwd_aot_serves(_direct_spec(job.identity), problem)[0]
        for _, job in direct_jobs
        if job.identity.algorithm in ("direct_depthwise", "direct_depthwise_spatial")
        and job.identity.dtype_a == "fp16"
    )


@pytest.mark.parametrize("cpg", [4, 8, 16, 32])
def test_grouped_serves_its_compiled_channel_counts_only(direct_jobs, cpg):
    spec = _first(direct_jobs, "direct_grouped", cpg=cpg)
    problem = Gfx950ConvFwdSpec(
        N=2,
        Hi=9,
        Wi=19,
        C=16 * cpg,
        K=16 * cpg,
        Y=3,
        X=3,
        pH=1,
        pW=1,
        groups=16,
    )
    assert gfx950_conv_fwd_aot_serves(spec, problem) == (True, "ok")
    for change in ({"K": 32 * cpg}, {"C": 32 * cpg, "K": 32 * cpg}):
        ok, why = gfx950_conv_fwd_aot_serves(spec, replace(problem, **change))
        assert not ok and "cpg, kpg" in why, why
    ok, why = gfx950_conv_fwd_aot_serves(
        replace(spec, block_groups=32), replace(problem)
    )
    assert not ok and "not divisible by block_groups" in why, why


def test_nongrouped_serves_dense_runtime_channels_and_dilation(direct_jobs):
    spec = _first(
        direct_jobs, "direct_nongrouped", dilation_h=3, dilation_w=1, stride_h=2
    )
    assert (spec.tile_w, spec.atom) == (32, "32x32x16")
    # Wo = 96, which a 32-wide tile divides; C and K are kernel arguments.
    problem = Gfx950ConvFwdSpec(
        N=1,
        Hi=20,
        Wi=191,
        C=64,
        K=96,
        Y=3,
        X=3,
        sH=2,
        sW=2,
        pH=1,
        pW=1,
        dH=3,
        dW=1,
        dtype=spec.dtype,
    )
    for c, k in ((64, 96), (16, 32), (256, 64)):
        ok, why = gfx950_conv_fwd_aot_serves(spec, replace(problem, C=c, K=k))
        assert ok, (c, k, why)
    # Grouped graphs, unaligned channel counts and another dilation are declined.
    for change, reason in (
        ({"C": 128, "K": 192, "groups": 2}, "groups == 1"),
        ({"C": 60}, "C 60"),
        ({"dH": 1}, "dilation"),
    ):
        ok, why = gfx950_conv_fwd_aot_serves(spec, replace(problem, **change))
        assert not ok and reason in why, (change, why)
    # A sweep for Wo = 96 tries the 32- and 96-wide tiles of the 32-wide atom only.
    wider = next(
        s
        for s in (
            _direct_spec(job.identity)
            for _, job in direct_jobs
            if job.identity.algorithm == "direct_nongrouped"
        )
        if (s.dilation_h, s.stride_h, s.tile_w, s.atom) == (3, 2, 64, "32x32x16")
    )
    ok, why = gfx950_conv_fwd_aot_serves(wider, replace(problem, dtype=wider.dtype))
    assert not ok and "does not fit Wo=96" in why, why


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"tile_m": 64}, "tile_m"),
        ({"pipeline": "none"}, "pipeline"),
        ({"vector_size_a": 1}, "vector_size_a"),
        ({"grouped": 1}, "grouped"),
        ({"direct_variant": "none"}, "direct_variant"),
        ({"kernel_family": 2}, "kernel_family"),
        ({"stride_w": 2}, "one stride"),
        ({"pad_w": 0}, "one stride and one padding"),
        ({"filter_w": 5}, "square"),
        ({"dilation_h": 2, "dilation_w": 2}, "no dilation"),
        ({"block_waves": 0}, "block_waves"),
        ({"block_waves": 17}, "block_waves"),
        ({"pad_h": 2, "pad_w": 2}, "same' padding"),
        ({"filter_h": 4, "filter_w": 4}, "odd filter"),
        ({"block_h": 4}, "does not use block_h"),
        ({"atom": "32x32x16"}, "does not use atom"),
        ({"iglp": 0}, "does not use iglp"),
    ],
)
def test_out_of_contract_direct_instances_are_declined(direct_jobs, change, reason):
    spec = _first(direct_jobs, "direct_depthwise")
    ok, why = supports_gfx950_conv_fwd_aot(replace(spec, **change))
    assert not ok and reason in why, why


@pytest.mark.parametrize(
    ("algorithm", "change", "reason"),
    [
        ("direct_depthwise_tiled", {"block_h": 0}, "block_h must be a positive"),
        ("direct_depthwise_tiled", {"unroll_rows": 2}, "unroll_rows must be 0 or 1"),
        ("direct_depthwise_tiled", {"block_q": 16}, "does not use block_q"),
        ("direct_depthwise_tiled", {"dilation_h": 2}, "no dilation"),
        ("direct_depthwise_col", {"block_w": 0}, "block_w must be a positive"),
        ("direct_depthwise_col", {"unroll_rows": 1}, "does not use unroll_rows"),
        ("direct_depthwise_col", {"block_waves": 17}, "block_waves"),
        ("direct_grouped", {"block_h": 8}, "row tile block_h=16"),
        ("direct_grouped", {"cpg": 0}, "cpg must be a positive"),
        ("direct_grouped", {"block_w": 4}, "does not use block_w"),
        ("direct_grouped", {"stride_h": 2, "stride_w": 2}, "stride > 1"),
        ("direct_grouped", {"block_q": 24}, "multiple of 16"),
        ("direct_nongrouped", {"atom": "8x8x8"}, "atom must be one of"),
        ("direct_nongrouped", {"tile_k_out": 48}, "multiple of"),
        ("direct_nongrouped", {"waves_per_eu": -1}, "waves_per_eu"),
        ("direct_nongrouped", {"iglp": -2}, "iglp"),
        ("direct_nongrouped", {"block_waves": 1}, "does not use block_waves"),
        ("direct_nongrouped", {"cpg": 16}, "does not use cpg"),
        ("direct_nongrouped", {"stride_h": 3, "stride_w": 3}, "stride 3"),
    ],
)
def test_out_of_contract_variant_knobs_are_declined(
    direct_jobs, algorithm, change, reason
):
    spec = _first(direct_jobs, algorithm)
    assert supports_gfx950_conv_fwd_aot(spec) == (True, "ok")
    ok, why = supports_gfx950_conv_fwd_aot(replace(spec, **change))
    assert not ok and reason in why, why


def test_direct_block_width_follows_the_variant(direct_jobs):
    spatial = _first(direct_jobs, "direct_depthwise_spatial")
    ok, why = supports_gfx950_conv_fwd_aot(replace(spatial, block_w=4))
    assert not ok and "derives block_w" in why
    ok, why = supports_gfx950_conv_fwd_aot(replace(spatial, direct_variant="std"))
    assert not ok and "block_w must be a positive" in why


def test_direct_knobs_round_trip_rockes_spelling():
    # Booleans are descriptor ints.
    assert aot_direct_fields_from_knobs(
        "tiled", {"block_w": 4, "block_h": 1, "block_waves": 1, "unroll_rows": True}
    ) == {"block_w": 4, "block_h": 1, "block_waves": 1, "unroll_rows": 1}
    with pytest.raises(ValueError, match="bool"):
        aot_direct_fields_from_knobs(
            "tiled", {"block_w": 4, "block_h": 1, "block_waves": 1, "unroll_rows": 1}
        )
    with pytest.raises(ValueError, match="are not"):
        aot_direct_fields_from_knobs("tiled", {"block_w": 4, "block_waves": 1})
    # None is a sentinel, and tile_k is carried as tile_k_out.
    assert aot_direct_fields_from_knobs(
        "nongrouped",
        {
            "tile_h": 8,
            "tile_w": 32,
            "tile_k": 64,
            "ck": 16,
            "waves_m": 2,
            "waves_n": 2,
            "atom": "32x32x16",
            "swizzle_wgm": 8,
            "iglp": None,
            "waves_per_eu": None,
        },
    ) == {
        "tile_h": 8,
        "tile_w": 32,
        "tile_k_out": 64,
        "ck": 16,
        "waves_m": 2,
        "waves_n": 2,
        "atom": "32x32x16",
        "swizzle_wgm": 8,
        "iglp": -1,
        "waves_per_eu": 0,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"kernel_family": KERNEL_FAMILY_DIRECT_DEPTHWISE},
        {"direct_variant": "std"},
        {"filter_h": 3},
        {"dilation_w": 1},
        {"block_waves": 1},
        {"block_h": 4},
        {"cpg": 4},
        {"atom": "32x32x16"},
        {"iglp": 0},
    ],
)
def test_implicit_instances_leave_every_direct_field_at_its_default(sweep_jobs, change):
    plain = next(
        _spec(job.identity)
        for _, job in sweep_jobs
        if not (job.identity.unroll_k or job.identity.async_dma)
    )
    ok, why = supports_gfx950_conv_fwd_aot(replace(plain, **change))
    assert not ok, why
