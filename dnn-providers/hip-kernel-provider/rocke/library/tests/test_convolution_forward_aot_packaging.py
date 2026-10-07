# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT forward instances (hipkernel:Gfx950ConvFwdAot).

An instance is packaged from a rocKE ``KernelIdentity``; these tests hold the adapter to
the AOT sweep it comes from: the same kernel IR as ``kernel_sweep.build_kernel``, and the
same served-problem predicate as ``KernelCache.supports_problem``.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

import pytest
from benchmarks.common import kernel_sweep
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_forward import (
    Gfx950ConvFwdAotSpec,
    Gfx950ConvFwdSpec,
    build_gfx950_conv_fwd_aot,
    gfx950_conv_fwd_aot_problem_for_request,
    gfx950_conv_fwd_aot_serves,
    supports_gfx950_conv_fwd_aot,
    supports_gfx950_conv_fwd_aot_problem,
)
from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.common.conv_args import ConvArgs
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
