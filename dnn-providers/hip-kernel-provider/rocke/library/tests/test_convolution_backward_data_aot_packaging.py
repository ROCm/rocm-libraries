# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT backward-data instances (hipkernel:Gfx950ConvDgradAot).

An instance is packaged from a rocKE dgrad ``KernelIdentity``; these tests hold the
adapter (``builders/common/convolution_backward_data.py``) to the sweeps it comes from:

* the packaged IR is the sweep kernel's, byte for byte up to the symbol, for one
  instance of every implicit-GEMM knob class and every direct capability row,
  including the MFMA pipeline's weight-transform helpers;
* the served predicate is ``KernelCache.supports_problem`` (plus, for the direct
  kernels, ``plan_for``'s per-shape re-validation) over a problem grid with strided,
  dilated, patch-embedding, grouped, depthwise and MFMA shapes, except that an MFMA
  instance using the reorganize helper (``waves_k > 1`` or ``runtime_k_loop``) is
  refused for ``groups > 1``, which rocKE offers but computes wrongly;
* the launch mirrors are rocKE's own launch code: the benchmark's sub-GEMM table, grid
  and kernel arguments for implicit GEMM, and ``direct_kernel_sweep.plan_for`` /
  ``launch_values`` for the direct and MFMA pipelines.
"""

from __future__ import annotations

import ctypes
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_sweep
from benchmarks.common.benchmark_implicit_gemm_conv import _extras_and_grid
from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from builders.common.convolution_backward_data import (
    AOT_DGRAD_DIRECT_ALGORITHMS,
    COALESCED_GROUPS_REASON,
    DIRECT_VARIANT_DEPTHWISE,
    DIRECT_VARIANT_MFMA,
    HELPER_VARIANT_REORGANIZE,
    HELPER_VARIANT_TRANSPOSE,
    KERNEL_FAMILY_DGRAD_HELPER,
    Gfx950ConvDgradAotSpec,
    Gfx950ConvDgradProblem,
    aot_dgrad_direct_instance_fields,
    build_gfx950_conv_dgrad_aot,
    gfx950_conv_dgrad_aot_helper_fields,
    gfx950_conv_dgrad_aot_mfma_launch_values,
    gfx950_conv_dgrad_aot_mfma_plan,
    gfx950_conv_dgrad_aot_problem_for_request,
    gfx950_conv_dgrad_aot_runtime_candidates,
    gfx950_conv_dgrad_aot_serves,
    sub_gemm_buffer_for,
    supports_gfx950_conv_dgrad_aot,
    supports_gfx950_conv_dgrad_aot_problem,
)
from dispatch.grouped_convolution import ConvGroupedRequest
from rocke.core.arch import ArchTarget
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_SEED = 2027
_ARCH = "gfx950"
_DTYPES = ("fp16", "bf16")


# ---------------------------------------------------------------------------
# identity -> instance
# ---------------------------------------------------------------------------


def _spec(identity: KernelIdentity) -> Gfx950ConvDgradAotSpec:
    """The packaged instance of a dgrad sweep identity (what the IG converter makes)."""
    if identity.is_direct:
        return Gfx950ConvDgradAotSpec(
            dtype=identity.dtype_a,
            tile_m=0,
            tile_n=0,
            tile_k=0,
            warp_m=0,
            warp_n=0,
            warp_tile_m=0,
            warp_tile_n=0,
            warp_tile_k=0,
            wave_size=identity.wave_size,
            pipeline=identity.pipeline,
            epilogue=identity.epilogue,
            vector_size_a=0,
            vector_size_b=0,
            vector_size_c=0,
            lds_k_outer=0,
            max_sub_gemms=0,
            grouped=0,
            instance_id=identity.stable_hash(),
            **aot_dgrad_direct_instance_fields(identity),
        )
    return Gfx950ConvDgradAotSpec(
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
        lds_k_outer=int(identity.lds_k_outer),
        max_sub_gemms=identity.max_sub_gemms,
        grouped=int(identity.grouped),
        instance_id=identity.stable_hash(),
    )


def _knob_class(identity: KernelIdentity) -> tuple:
    """Everything of an implicit-GEMM dgrad identity but its tile and warp geometry."""
    return (
        identity.pipeline,
        identity.epilogue,
        identity.vector_size_a,
        identity.vector_size_b,
        identity.vector_size_c,
        identity.lds_k_outer,
        identity.warp_tile_m,
    )


@pytest.fixture(scope="module")
def implicit_jobs():
    """One valid job of every knob class of the dgrad AOT grid, per dtype (seeded)."""
    target = ArchTarget.from_gfx(_ARCH)
    rng = random.Random(_SEED)
    sample = []
    for dtype in _DTYPES:
        by_class: dict = {}
        for job in kernel_sweep._direction_jobs(
            "dgrad",
            _ARCH,
            dtype,
            target,
            (1,),
            kernel_sweep.CACHE_DGRAD_MAX_SUB_GEMMS,
        ):
            by_class.setdefault(_knob_class(job.identity), []).append(job)
        for key in sorted(by_class, key=repr):
            candidates = by_class[key]
            rng.shuffle(candidates)
            valid = next(
                (j for j in candidates if kernel_sweep._spec_is_valid(j, _ARCH, dtype)),
                None,
            )
            if valid is not None:
                sample.append((dtype, valid))
    return sample


@pytest.fixture(scope="module")
def direct_jobs():
    """Every dgrad direct-sweep job of a packaged algorithm, both dtypes."""
    target = ArchTarget.from_gfx(_ARCH)
    out = []
    for dtype in _DTYPES:
        for job in dks.enumerate_direct_jobs(
            arch=_ARCH, target=target, directions=("dgrad",), dtype=dtype, jobs=8
        ):
            if job.identity.algorithm in AOT_DGRAD_DIRECT_ALGORITHMS:
                out.append((dtype, job))
    return out


@pytest.fixture(scope="module")
def direct_sample(direct_jobs):
    """Per dtype: one depthwise job per capability row, every MFMA job, every helper."""
    rng = random.Random(_SEED + 1)
    by_row: dict = {}
    sample = []
    for dtype, job in direct_jobs:
        ident = job.identity
        if ident.algorithm == "direct_depthwise_dgrad":
            key = (dtype, ident.filter_h, ident.stride_h, ident.pad_h)
            by_row.setdefault(key, []).append((dtype, job))
        else:
            sample.append((dtype, job))
    for key in sorted(by_row):
        sample.append(rng.choice(by_row[key]))
    return sample


def _assert_same_ir(ours, theirs, label):
    ours_ir = _lower_kernel_to_llvm_python(ours, arch=_ARCH, llvm_flavor="llvm22")
    theirs_ir = _lower_kernel_to_llvm_python(theirs, arch=_ARCH, llvm_flavor="llvm22")
    assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
        theirs.name, "@K"
    ), label


# ---------------------------------------------------------------------------
# Build: the packaged kernel is the sweep's
# ---------------------------------------------------------------------------


def test_the_implicit_sample_covers_every_knob_value(implicit_jobs):
    classes = {_knob_class(job.identity) for _, job in implicit_jobs}
    assert {c[0] for c in classes} == {"mem", "compv3", "compv4"}
    assert {c[1] for c in classes} == {"default", "cshuffle"}
    for position in (2, 3):
        assert {c[position] for c in classes} == {1, 2, 4, 8}
    assert {c[5] for c in classes} == {False, True}
    assert {c[6] for c in classes} == {16, 32}


def test_the_probes_are_the_sweeps():
    spec = _spec(
        KernelIdentity(
            arch=_ARCH,
            direction="dgrad",
            algorithm="implicit_gemm",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_d="fp16",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=8,
            grouped=True,
            lds_k_outer=True,
            max_sub_gemms=1024,
        )
    )
    (caps,) = kernel_sweep._dgrad_caps()
    assert spec.probe_problem() == kernel_sweep._probe_problem("dgrad", caps)
    from builders.common import convolution_backward_data as adapter

    assert adapter._DENSE_MFMA_CHANNELS == dks._DENSE_MFMA_DGRAD_CHANNELS


def test_every_implicit_knob_class_is_packageable_with_the_sweeps_ir(implicit_jobs):
    for dtype, job in implicit_jobs:
        spec = _spec(job.identity)
        label = job.identity.label()
        assert supports_gfx950_conv_dgrad_aot(spec) == (True, "ok"), label
        ours = build_gfx950_conv_dgrad_aot(spec)
        theirs, _ = kernel_sweep.build_kernel(job, _ARCH, dtype)
        assert ours.name == spec.kernel_name(), label
        assert ours.name.startswith(spec.kernel_symbol_prefix()), label
        _assert_same_ir(ours, theirs, label)


def test_every_direct_row_and_helper_is_packageable_with_the_sweeps_ir(direct_sample):
    algorithms = set()
    for dtype, job in direct_sample:
        ident = job.identity
        spec = _spec(ident)
        label = ident.label()
        assert supports_gfx950_conv_dgrad_aot(spec) == (True, "ok"), label
        # The identity round-trips through the instance's own fields.
        assert spec.algorithm == ident.algorithm
        assert spec.direct_knobs() == (json.loads(ident.knobs) if ident.knobs else {})
        assert dks.direct_job_for_identity(ident) == job, label
        ours = build_gfx950_conv_dgrad_aot(spec)
        theirs, _ = dks.build_direct_job(job, _ARCH, dtype)
        assert ours.name == spec.kernel_name(), label
        assert ours.name.startswith(spec.kernel_symbol_prefix()), label
        _assert_same_ir(ours, theirs, label)
        algorithms.add(ident.algorithm)
    assert algorithms == set(AOT_DGRAD_DIRECT_ALGORITHMS)


def test_helper_symbols_are_unique_per_instance(direct_jobs):
    helpers = [_spec(job.identity) for _, job in direct_jobs]
    helpers = [s for s in helpers if s.is_helper]
    names = {build_gfx950_conv_dgrad_aot(s).name for s in helpers}
    assert len(names) == len(helpers) == len({s.instance_id for s in helpers})


def test_an_mfma_instance_names_the_helpers_its_pipeline_pulls_in(direct_jobs):
    """gfx950_conv_dgrad_aot_helper_fields names exactly the helper identities the
    direct sweep enumerates next to each MFMA entry, by their KMD field values."""
    helpers = {}
    for _, job in direct_jobs:
        spec = _spec(job.identity)
        if spec.is_helper:
            key = (
                spec.dtype,
                spec.direct_variant,
                spec.filter_h,
                spec.cpg,
                spec.kpg,
                spec.fold_k32,
            )
            assert key not in helpers, job.identity.label()
            helpers[key] = spec
    seen = set()
    for _, job in direct_jobs:
        spec = _spec(job.identity)
        if not spec.is_mfma:
            continue
        wanted = gfx950_conv_dgrad_aot_helper_fields(spec)
        knobs = json.loads(job.identity.knobs)
        assert [h["direct_variant"] for h in wanted] == (
            [HELPER_VARIANT_TRANSPOSE, HELPER_VARIANT_REORGANIZE]
            if dks._uses_coalesced_weights(knobs)
            else [HELPER_VARIANT_TRANSPOSE]
        )
        for fields_ in wanted:
            helper = helpers[
                (
                    fields_["dtype"],
                    fields_["direct_variant"],
                    fields_["filter_h"],
                    fields_["cpg"],
                    fields_["kpg"],
                    fields_["fold_k32"],
                )
            ]
            for name, value in fields_.items():
                assert getattr(helper, name) == value, (name, job.identity.label())
            assert helper.kernel_family == KERNEL_FAMILY_DGRAD_HELPER
            seen.add(helper.instance_id)
    assert seen == {s.instance_id for s in helpers.values()}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"pipeline": "basic"}, "pipelines"),
        ({"epilogue": "none"}, "epilogue"),
        ({"vector_size_a": 3}, "vector_size_a"),
        ({"lds_k_outer": 2}, "lds_k_outer"),
        ({"max_sub_gemms": 0}, "max_sub_gemms"),
        ({"wave_size": 32}, "wave_size"),
        ({"instance_id": "x" * 40}, "instance_id"),
        ({"dtype": "fp32"}, "fp16 or bf16"),
        ({"block_w": 4}, "block_w"),
        ({"kernel_family": 3}, "kernel_family"),
    ],
)
def test_an_implicit_instance_outside_the_contract_is_refused(
    implicit_jobs, change, reason
):
    spec = replace(_spec(implicit_jobs[0][1].identity), **change)
    ok, why = supports_gfx950_conv_dgrad_aot(spec)
    assert not ok and reason in why, why


@pytest.mark.parametrize(
    ("variant", "change", "reason"),
    [
        (DIRECT_VARIANT_DEPTHWISE, {"block_q": 16}, "block_q"),
        (DIRECT_VARIANT_DEPTHWISE, {"cpg": 2, "kpg": 2}, "cpg == kpg == 1"),
        (DIRECT_VARIANT_DEPTHWISE, {"tile_m": 64}, "tile_m"),
        (DIRECT_VARIANT_MFMA, {"waves_k": 5}, "waves_k"),
        (DIRECT_VARIANT_MFMA, {"dilation_h": 2}, "dilation"),
        (DIRECT_VARIANT_MFMA, {"stride_h": 1, "stride_w": 2}, "one stride"),
        (
            HELPER_VARIANT_TRANSPOSE,
            {"stride_h": 1, "stride_w": 1},
            "stride and padding",
        ),
        (
            HELPER_VARIANT_TRANSPOSE,
            {"direct_variant": DIRECT_VARIANT_MFMA},
            "direct_variant",
        ),
    ],
)
def test_a_direct_instance_outside_the_contract_is_refused(
    direct_jobs, variant, change, reason
):
    spec = next(
        s
        for s in (_spec(job.identity) for _, job in direct_jobs)
        if s.direct_variant == variant
    )
    ok, why = supports_gfx950_conv_dgrad_aot(replace(spec, **change))
    assert not ok and reason in why, why


# ---------------------------------------------------------------------------
# Served problems: rocKE's own offer
# ---------------------------------------------------------------------------


def _p(N, Hi, Wi, C, K, Y, X, s=(1, 1), p=(0, 0), d=(1, 1), groups=1, dtype="fp16"):
    return Gfx950ConvDgradProblem(
        N=N,
        Hi=Hi,
        Wi=Wi,
        C=C,
        K=K,
        Y=Y,
        X=X,
        sH=s[0],
        sW=s[1],
        pH=p[0],
        pW=p[1],
        dH=d[0],
        dW=d[1],
        groups=groups,
        dtype=dtype,
    )


#: Shapes every family is decided on: dense, strided, dilated, patch embedding, a
#: decomposition too large for the record table, pointwise, odd and narrow channel
#: counts, grouped, depthwise (with and without a direct row), and the MFMA rows.
_PROBLEMS = [
    _p(2, 14, 14, 64, 64, 3, 3, p=(1, 1)),
    _p(2, 15, 15, 32, 48, 3, 3, s=(2, 2), p=(1, 1)),
    _p(1, 17, 13, 16, 24, 3, 3, s=(3, 2), p=(2, 1), d=(2, 1)),
    _p(1, 20, 20, 24, 40, 3, 3, p=(2, 2), d=(2, 2)),
    _p(2, 64, 64, 8, 64, 16, 16, s=(16, 16)),
    _p(1, 66, 66, 8, 16, 33, 33, s=(33, 33)),
    _p(2, 28, 28, 64, 128, 1, 1, s=(2, 2)),
    _p(2, 14, 14, 128, 64, 1, 1),
    _p(1, 9, 9, 12, 20, 3, 3, p=(1, 1)),
    _p(1, 9, 9, 6, 10, 3, 3, p=(1, 1)),
    _p(1, 10, 10, 3, 64, 7, 7, s=(2, 2), p=(3, 3)),
    _p(2, 14, 14, 64, 128, 3, 3, p=(1, 1), groups=4),
    _p(1, 15, 15, 96, 96, 3, 3, s=(2, 2), p=(1, 1), groups=3),
    _p(2, 28, 28, 64, 64, 3, 3, p=(1, 1), groups=64),
    _p(2, 28, 28, 64, 64, 3, 3, s=(2, 2), p=(1, 1), groups=64),
    _p(1, 14, 14, 96, 96, 7, 7, p=(3, 3), groups=96),
    _p(1, 15, 15, 32, 32, 3, 3, s=(3, 3), p=(1, 1), groups=32),
    _p(1, 14, 14, 32, 32, 3, 3, groups=32),
    _p(1, 14, 14, 32, 32, 3, 3, p=(1, 1), groups=8),
    _p(1, 14, 14, 128, 128, 3, 3, p=(1, 1), groups=16),
    _p(1, 14, 14, 64, 64, 3, 3, p=(1, 1), groups=4),
    _p(1, 14, 14, 24, 24, 3, 3, p=(1, 1)),
    _p(1, 14, 14, 48, 192, 3, 3, p=(1, 1)),
    _p(1, 14, 14, 4, 320, 3, 3, p=(1, 1)),
    _p(1, 14, 14, 12, 12, 3, 3, p=(1, 1), groups=3),
    _p(1, 14, 14, 64, 64, 3, 3, p=(1, 1), groups=2),
    # The dense MFMA rows' channels per group with groups > 1: rocKE offers the
    # instances using the reorganize helper, which the adapter refuses.
    _p(1, 14, 14, 96, 384, 3, 3, p=(1, 1), groups=2),
    _p(2, 9, 9, 16, 1280, 3, 3, p=(1, 1), groups=4),
]
_PROBLEMS += [replace(p, dtype="bf16") for p in _PROBLEMS[::3]]


def _rocke_offer(identity: KernelIdentity, problem: Gfx950ConvDgradProblem):
    """``KernelCache.supports_problem``, then for direct kernels ``plan_for``'s
    re-validation (its MFMA knob rule and the spec validator on the real shape)."""
    if identity.dtype_a != problem.dtype:
        return False
    cache = KernelCache(Path("/nonexistent"), _ARCH)
    if not identity.is_direct:
        return cache.supports_problem(identity, problem.to_problem())[0]
    direct = problem.to_direct_problem()
    if problem.sH != problem.sW or problem.pH != problem.pW:
        return False
    if not cache.supports_problem(identity, direct)[0]:
        return False
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if identity.algorithm == "direct_grouped_dgrad_mfma" and dks._mfma_knobs_reason(
        direct, knobs
    ):
        return False
    spec = dks.make_spec(identity.algorithm, direct, knobs)
    return dks.validate_spec(identity.algorithm, spec, _ARCH)[0]


def _uses_reorganize(spec: Gfx950ConvDgradAotSpec) -> bool:
    return spec.is_mfma and (spec.waves_k > 1 or bool(spec.runtime_k_loop))


def test_the_served_predicate_is_rockes_offer(implicit_jobs, direct_sample):
    """rocKE's offer, except the grouped problems of the MFMA instances that use the
    reorganize helper (rocKE ``build_direct_reorganize_weights`` mis-addresses
    groups > 1), which the adapter refuses."""
    served = 0
    refused_grouped_reorganize = 0
    families = set()
    for _, job in implicit_jobs + direct_sample:
        ident = job.identity
        if ident.direction == "direct_dgrad_helper":
            continue
        spec = _spec(ident)
        for problem in _PROBLEMS:
            ours, why = gfx950_conv_dgrad_aot_serves(spec, problem)
            expected = _rocke_offer(ident, problem)
            if expected and _uses_reorganize(spec) and problem.groups != 1:
                assert not ours and why == COALESCED_GROUPS_REASON, (
                    ident.label(),
                    problem,
                    why,
                )
                refused_grouped_reorganize += 1
                continue
            assert ours == expected, (ident.label(), problem, why)
            if ours:
                served += 1
                families.add(spec.direct_variant)
    assert families == {"none", DIRECT_VARIANT_DEPTHWISE, DIRECT_VARIANT_MFMA}
    assert served > 100
    assert refused_grouped_reorganize > 0


def test_an_mfma_instance_using_the_reorganize_helper_serves_groups_1_only(
    direct_jobs,
):
    """Every reorganizing MFMA row is refused at groups 2 with the reason, served at
    groups 1; a row without the reorganize step keeps serving groups > 1."""
    reorganizing = 0
    for _, job in direct_jobs:
        spec = _spec(job.identity)
        if not spec.is_mfma:
            continue
        # Smallest groups > 1 the row's block_groups divides.
        groups = 2 * spec.block_groups if spec.block_groups > 1 else 2
        grouped = _p(
            1,
            14,
            14,
            spec.cpg * groups,
            spec.kpg * groups,
            3,
            3,
            p=(1, 1),
            groups=groups,
            dtype=spec.dtype,
        )
        ok, why = gfx950_conv_dgrad_aot_serves(spec, grouped)
        if _uses_reorganize(spec):
            reorganizing += 1
            assert not ok and why == COALESCED_GROUPS_REASON, (spec, why)
            dense = replace(grouped, C=spec.cpg, K=spec.kpg, groups=1)
            ok, why = gfx950_conv_dgrad_aot_serves(spec, dense)
            assert ok, (spec, why)
        else:
            assert why != COALESCED_GROUPS_REASON, (spec, why)
    assert reorganizing > 0


def test_a_helper_serves_nothing(direct_jobs):
    for _, job in direct_jobs:
        spec = _spec(job.identity)
        if spec.is_helper:
            for problem in _PROBLEMS:
                assert not gfx950_conv_dgrad_aot_serves(spec, problem)[0]


def test_every_listed_problem_class_is_decided_as_expected(implicit_jobs, direct_jobs):
    """Coverage the catalog gaps predict: implicit GEMM needs a width of the sample to
    divide the channels and refuses depthwise and record tables above 1024; the
    direct rows serve their exact capabilities only."""
    specs = [_spec(job.identity) for _, job in implicit_jobs] + [
        _spec(job.identity) for _, job in direct_jobs
    ]

    def families(problem):
        return {
            s.direct_variant
            for s in gfx950_conv_dgrad_aot_runtime_candidates(problem, specs)
        }

    assert families(_PROBLEMS[0]) == {"none"}
    assert families(_PROBLEMS[5]) == set()  # 1089 sub-GEMMs
    assert families(_PROBLEMS[13]) == {DIRECT_VARIANT_DEPTHWISE}
    assert families(_PROBLEMS[15]) == {DIRECT_VARIANT_DEPTHWISE}
    assert families(_PROBLEMS[16]) == set()  # depthwise stride 3: no row
    assert families(_PROBLEMS[18]) == {"none", DIRECT_VARIANT_MFMA}
    assert families(_PROBLEMS[22]) == {"none", DIRECT_VARIANT_MFMA}
    assert families(_PROBLEMS[23]) == {"none", DIRECT_VARIANT_MFMA}


def test_problem_for_request_is_the_dgrad_graph_contract():
    request = ConvGroupedRequest(
        N=2,
        C=64,
        K=32,
        Hi=15,
        Wi=15,
        Y=3,
        X=3,
        arch=_ARCH,
        G=2,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        direction="dgrad",
    )
    problem = gfx950_conv_dgrad_aot_problem_for_request(request)
    assert (problem.Hi, problem.Ho, problem.cpg, problem.kpg) == (15, 8, 32, 16)
    assert supports_gfx950_conv_dgrad_aot_problem(problem) == (True, "ok")
    with pytest.raises(ValueError, match="backward-data"):
        gfx950_conv_dgrad_aot_problem_for_request(replace(request, direction="fwd"))
    assert not supports_gfx950_conv_dgrad_aot_problem(replace(problem, groups=3))[0]
    assert not supports_gfx950_conv_dgrad_aot_problem(replace(problem, dtype="fp32"))[0]


# ---------------------------------------------------------------------------
# Launch: rocKE's own launch code
# ---------------------------------------------------------------------------


class _RecordingRuntime:
    """The two runtime calls the benchmark's dgrad launch prep makes."""

    def __init__(self):
        self.next = 0x1000
        self.copied = {}

    def alloc(self, nbytes):
        self.next += 0x10000
        return self.next

    def memcpy_h2d(self, dst, src, nbytes):
        self.copied[dst] = ctypes.string_at(src, nbytes)


def _i32(values):
    import numpy as np

    return np.array(values, dtype=np.int32).tobytes()


def test_the_implicit_launch_is_the_benchmarks(implicit_jobs):
    ptrs = (0x100000, 0x200000, 0x300000)
    checked = 0
    for _, job in implicit_jobs:
        ident = job.identity
        spec = _spec(ident)
        for problem in _PROBLEMS:
            if not gfx950_conv_dgrad_aot_serves(spec, problem)[0]:
                continue
            rt = _RecordingRuntime()
            conv = problem.to_problem()
            extras, grid = _extras_and_grid("dgrad", conv, ident, rt, problem.dtype)
            sizes = problem.tensor_bytes()
            theirs = kernel_sweep._launch_values_for(
                "dgrad", conv, ident, ptrs, sizes, extras
            )
            table = spec.sub_gemm_buffer(problem)
            assert _i32(table) == rt.copied[extras["sub_gemm_buf"]]
            assert table == sub_gemm_buffer_for(
                problem, spec.tile_m, spec.tile_n, spec.tile_k
            )
            assert spec.workspace_bytes(problem) == len(table) * 4
            assert spec.grid(problem) == grid
            assert spec.block() == (ident.warp_m * ident.warp_n * 64, 1, 1)
            ours = spec.launch_values(
                problem, *ptrs, *sizes, sub_gemm_buf=extras["sub_gemm_buf"]
            )
            assert ours == theirs
            assert [a["name"] for a in spec.launch_signature()] == list(ours)
            assert len(ours) == 44
            assert spec.needs_zeroed_dx(problem) == (problem.sH > 1 or problem.sW > 1)
            checked += 1
    assert checked > 50


class _HelperCache:
    """The two cache calls ``plan_for`` makes for the MFMA helpers."""

    def get(self, identity):
        return Path("x.hsaco"), {"kernel_name": identity.algorithm}

    def hsaco_path(self, identity):
        return Path("x.hsaco")


def test_the_direct_launch_is_plan_fors(direct_sample):
    ptrs = {"dy": 0x100000, "w": 0x200000, "dx": 0x300000}
    checked = set()
    for _, job in direct_sample:
        ident = job.identity
        if ident.direction == "direct_dgrad_helper":
            continue
        spec = _spec(ident)
        for problem in _PROBLEMS:
            if not gfx950_conv_dgrad_aot_serves(spec, problem)[0]:
                continue
            plan, why = dks.plan_for(
                _HelperCache(),
                ident,
                Path("x.hsaco"),
                {"kernel_name": "k"},
                problem.to_direct_problem(),
                _ARCH,
            )
            assert plan is not None, why
            dy, w, dx = problem.tensor_bytes()
            sizes = {"dy": dy, "w": w, "dx": dx}
            if spec.is_depthwise:
                (step,) = plan.steps
                assert spec.grid(problem) == step.grid
                assert spec.block() == step.block
                assert spec.launch_signature() == step.signature
                assert spec.launch_values(
                    problem, ptrs["dy"], ptrs["w"], ptrs["dx"], dy, w, dx
                ) == dks.launch_values(step, ptrs, sizes)
                assert len(step.signature) == 20
                assert spec.workspace_bytes(problem) == 0
                assert not spec.needs_zeroed_dx(problem)
                checked.add(DIRECT_VARIANT_DEPTHWISE)
                continue
            ours = gfx950_conv_dgrad_aot_mfma_plan(spec, problem)
            assert len(ours["steps"]) == len(plan.steps)
            assert {k: v[1] for k, v in ours["workspaces"].items()} == plan.workspaces
            all_ptrs = dict(ptrs)
            all_sizes = dict(sizes)
            for name, (offset, size) in ours["workspaces"].items():
                assert offset % 256 == 0
                all_ptrs[name] = 0x400000 + offset
                all_sizes[name] = size
            assert ours["workspace_bytes"] == max(
                off + size for off, size in ours["workspaces"].values()
            )
            for mine, theirs in zip(ours["steps"], plan.steps):
                assert mine["grid"] == theirs.grid
                assert mine["block"] == theirs.block
                assert mine["buffers"] == theirs.buffers
                assert mine["signature"] == theirs.signature
                assert gfx950_conv_dgrad_aot_mfma_launch_values(
                    spec, problem, mine, all_ptrs, all_sizes
                ) == dks.launch_values(theirs, all_ptrs, all_sizes)
            assert spec.grid(problem) == plan.steps[-1].grid
            assert spec.block() == plan.steps[-1].block
            checked.add(DIRECT_VARIANT_MFMA)
    assert checked == {DIRECT_VARIANT_DEPTHWISE, DIRECT_VARIANT_MFMA}


def test_the_sub_gemm_table_is_records_then_magic_entries():
    """22 i32 per record, tile counts in fields 1 and 2, then 8 i32 of magic pairs
    per record; the flat grid is the last record's block_end."""
    problem = _p(1, 17, 13, 16, 24, 3, 3, s=(3, 2), p=(2, 1), d=(2, 1))
    table = sub_gemm_buffer_for(problem, 64, 32, 32)
    from kernels.common.conv_implicit_gemm_dgrad import (
        compute_tilde,
        enumerate_sub_gemms,
    )

    conv = problem.to_problem()
    subs = enumerate_sub_gemms(conv, compute_tilde(conv), 64, 32, tile_k=32)
    assert len(table) == 30 * len(subs) > 30
    for index, sg in enumerate(subs):
        record = table[22 * index : 22 * index + 22]
        assert record[0] == sg.block_start
        assert record[1] == -(-sg.gemm_m // 64)
        assert record[2] == -(-sg.gemm_n // 32)
        assert record[3:] == sg.pack_i32()[3:]
