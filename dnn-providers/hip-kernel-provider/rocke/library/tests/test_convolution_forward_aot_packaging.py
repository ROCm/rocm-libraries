# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT forward instances (hipkernel:Gfx950ConvFwdAot).

An instance is packaged from a rocKE ``KernelIdentity``; these tests hold the adapter to
the sweep it comes from. Implicit GEMM: the same kernel IR as ``kernel_sweep.build_kernel``
and the same served-problem predicate as ``KernelCache.supports_problem``. Direct
depthwise: the same probe and IR as ``direct_kernel_sweep.build_direct_job``, and the
served problems of ``supports_problem`` plus the per-shape validator ``plan_for`` runs.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import replace

import pytest
from benchmarks.common import direct_kernel_sweep, kernel_sweep
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_forward import (
    KERNEL_FAMILY_DIRECT_DEPTHWISE,
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
from kernels.common.conv_direct_grouped import DirectConvProblem, direct_launch_geometry
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


# ---- Direct depthwise instances ----

#: The direct-sweep algorithms the packaged direct launch runs, and their variant.
_DIRECT_VARIANTS = {"direct_depthwise": "std", "direct_depthwise_spatial": "spatial"}


def _direct_spec(identity: KernelIdentity) -> Gfx950ConvFwdAotSpec:
    """The instance make_conv_aot_config.py packages for a direct identity."""
    knobs = json.loads(identity.knobs)
    return replace(
        _spec(identity),
        kernel_family=KERNEL_FAMILY_DIRECT_DEPTHWISE,
        direct_variant=_DIRECT_VARIANTS[identity.algorithm],
        filter_h=identity.filter_h,
        filter_w=identity.filter_w,
        stride_h=identity.stride_h,
        stride_w=identity.stride_w,
        pad_h=identity.pad_h,
        pad_w=identity.pad_w,
        dilation_h=identity.dilation_h,
        dilation_w=identity.dilation_w,
        block_w=knobs.get("block_w", 0),
        block_waves=knobs["block_waves"],
    )


@pytest.fixture(scope="module")
def direct_jobs():
    """Every forward std and spatial depthwise job of the direct sweep, both dtypes."""
    jobs = []
    for dtype in ("fp16", "bf16"):
        for algorithm in _DIRECT_VARIANTS:
            direction, caps_list = direct_kernel_sweep.DIRECT_CAPABILITIES[algorithm]
            for caps in caps_list:
                _, cell = direct_kernel_sweep._caps_jobs(
                    "gfx950", 64, algorithm, direction, caps, dtype
                )
                jobs += [(dtype, job) for _, job in cell]
    return jobs


def test_every_direct_sweep_identity_is_packageable_on_the_sweeps_probe(direct_jobs):
    # The native direct launch runs every std and spatial depthwise kernel the sweep builds.
    assert len(direct_jobs) == 2 * (54 + 30)
    for dtype, job in direct_jobs:
        spec = _direct_spec(job.identity)
        assert supports_gfx950_conv_fwd_aot(spec) == (True, "ok"), job.identity.label()
        caps = direct_kernel_sweep.DirectCaps(**job.caps)
        assert spec.probe_problem() == direct_kernel_sweep.probe_problem(
            caps, dtype, job.identity.algorithm
        )


def test_packaged_direct_ir_is_the_sweep_kernel_up_to_its_symbol(direct_jobs):
    for dtype, job in direct_jobs:
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


def _rocke_direct_serves(identity: KernelIdentity, problem: Gfx950ConvFwdSpec):
    """What rocKE's direct benchmark offers ``identity`` for: the cache's capability
    match, then ``plan_for``'s re-validation of the spec on the real problem."""
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
    if problem.dtype != identity.dtype_a:
        return False, "dtype"
    cache = KernelCache.__new__(KernelCache)
    ok, why = cache.supports_problem(identity, direct)
    if not ok:
        return False, why
    spec = direct_kernel_sweep.make_spec(
        identity.algorithm, direct, json.loads(identity.knobs)
    )
    return direct_kernel_sweep.validate_spec(identity.algorithm, spec, "gfx950")


def test_direct_served_problems_agree_with_the_direct_benchmarks_offer(direct_jobs):
    """Every (direct instance, problem) verdict matches what rocKE would launch.

    The packaged guard is stricter in one place: rocKE's spatial validator admits 64
    groups (one column per wave), which its own benchmark never runs, and the direct
    guard declines it.
    """
    compared = served = 0
    for dtype, job in direct_jobs:
        identity = replace(job.identity, llvm_flavor=kernel_sweep.current_llvm_flavor())
        spec = _direct_spec(job.identity)
        for problem in _depthwise_problems(dtype)[::3]:
            ours, reason = gfx950_conv_fwd_aot_serves(spec, problem)
            theirs, why = _rocke_direct_serves(identity, problem)
            if theirs and not ours and spec.direct_variant == "spatial":
                assert problem.groups == 64 and "groups < 64" in reason, reason
                continue
            assert ours == theirs, (job.identity.label(), problem, reason, why)
            compared += 1
            served += ours
    assert compared > 10000 and 0 < served < compared


def test_direct_launch_is_rockes_for_the_runtime_problem(direct_jobs):
    for _, job in direct_jobs[:: len(direct_jobs) // 12]:
        spec = _direct_spec(job.identity)
        assert (
            spec.launch_signature()
            == spec.runtime_spec(_depthwise_problems(spec.dtype)[0]).launch_signature()
        )
        for problem in _depthwise_problems(spec.dtype):
            if not gfx950_conv_fwd_aot_serves(spec, problem)[0]:
                continue
            direct = problem.to_direct_problem()
            rocke = direct_kernel_sweep.make_spec(
                job.identity.algorithm, direct, json.loads(job.identity.knobs)
            )
            assert (spec.grid(problem), spec.block()) == direct_launch_geometry(rocke)
            assert spec.launch_values(
                problem, 0, 0, 0, 1, 2, 3
            ) == ConvArgs.from_problem(direct).to_launch_values(0, 0, 0, 1, 2, 3)


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
    ],
)
def test_out_of_contract_direct_instances_are_declined(direct_jobs, change, reason):
    spec = next(
        _direct_spec(job.identity)
        for _, job in direct_jobs
        if job.identity.algorithm == "direct_depthwise"
    )
    ok, why = supports_gfx950_conv_fwd_aot(replace(spec, **change))
    assert not ok and reason in why, why


def test_direct_block_width_follows_the_variant(direct_jobs):
    spatial = next(
        _direct_spec(job.identity)
        for _, job in direct_jobs
        if job.identity.algorithm == "direct_depthwise_spatial"
    )
    ok, why = supports_gfx950_conv_fwd_aot(replace(spatial, block_w=4))
    assert not ok and "derives block_w" in why
    ok, why = supports_gfx950_conv_fwd_aot(replace(spatial, direct_variant="std"))
    assert not ok and "block_w must be a positive" in why


@pytest.mark.parametrize(
    "change",
    [
        {"kernel_family": KERNEL_FAMILY_DIRECT_DEPTHWISE},
        {"direct_variant": "std"},
        {"filter_h": 3},
        {"dilation_w": 1},
        {"block_waves": 1},
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
