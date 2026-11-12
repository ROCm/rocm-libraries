# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT backward-data instances (hipkernel:Gfx950ConvDgradAot
and hipkernel:Gfx1250ConvDgradAot).

An instance is packaged from a rocKE dgrad ``KernelIdentity``; these tests hold the
adapter (``builders/common/convolution_backward_data.py``) to the sweeps it comes from,
over each engine's shipped catalog (the dispatcher's registered dgrad instances of its
architecture -- ``dispatch/grouped_convolution_instances/<arch>_dgrad.json`` -- and the
weight-transform helpers gfx950's MFMA rows pull in):

* the packaged IR is the sweep kernel's, byte for byte up to the symbol, for one
  instance of every pipeline and implicit-GEMM form;
* the launch plan (``conv_dgrad_aot_plan``) serves exactly rocKE's offer --
  ``identity_supports_problem`` on the problem the implicit-GEMM sweep launches (folded,
  batch-split), ``plan_for``'s re-validation for the direct pipelines -- except that an
  MFMA instance using the reorganize helper refuses ``groups > 1``, which rocKE offers
  but computes wrongly;
* the plan is rocKE's own launch: the implicit-GEMM benchmark's record tables, grids,
  batch slices and kernel arguments, and ``direct_kernel_sweep.plan_for``'s steps and
  ``launch_values`` for the direct pipelines;
* dX is zero-filled exactly when the records leave some of it unwritten.
"""

from __future__ import annotations

import ctypes
import json
import random
import tracemalloc
from dataclasses import fields, replace
from pathlib import Path

import pytest
from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_sweep
from benchmarks.common.benchmark_implicit_gemm_conv import (
    _batch_chunks,
    _extras_and_grid,
    _launch_problem,
)
from builders.common.convolution_backward_data import (
    ABI_DIRECT_DGRAD,
    ABI_DIRECT_FWD,
    ABI_IMPLICIT,
    AOT_DGRAD_DIRECT_ALGORITHMS,
    BUFFER_DX,
    BUFFER_DY,
    BUFFER_SUB_GEMM,
    BUFFER_W,
    DIRECT_VARIANT_DEPTHWISE,
    DIRECT_VARIANT_MFMA,
    DIRECT_VARIANT_WMMA,
    KERNEL_FAMILY_DGRAD_HELPER,
    KERNEL_SELF,
    SPEC_CLASSES,
    TARGETS,
    ConvDgradAotProblem,
    ConvDgradAotSpec,
    _covered,
    aot_dgrad_direct_fields_from_knobs,
    aot_dgrad_direct_instance_fields,
    build_gfx950_conv_dgrad_aot,
    build_gfx1250_conv_dgrad_aot,
    conv_dgrad_aot_launch_problem,
    conv_dgrad_aot_plan,
    conv_dgrad_aot_problem_for_request,
    conv_dgrad_aot_step_signature,
    gfx950_conv_dgrad_aot_helper_fields,
    supports_conv_dgrad_aot,
    supports_conv_dgrad_aot_problem,
)
from dispatch.grouped_convolution import (
    ConvGroupedRequest,
    registered_conv_grouped_instances,
)
from kernels.conv import aot_jobs, direct_aot_jobs
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_variants import direct_spec_takes_target_wave, make_direct_spec
from kernels.conv.kernel_identity import (
    KernelIdentity,
    dgrad_sub_gemms,
    identity_supports_problem,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

#: Each engine's typed build entry point.
_BUILDERS = {
    "gfx950": build_gfx950_conv_dgrad_aot,
    "gfx1250": build_gfx1250_conv_dgrad_aot,
}

# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


def _spec(identity: KernelIdentity) -> ConvDgradAotSpec:
    """The packaged instance of a dgrad sweep identity (what the IG converter makes)."""
    spec_class = SPEC_CLASSES[identity.arch]
    if identity.is_direct:
        return spec_class(
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
    carried = (
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "wave_size",
        "pipeline",
        "epilogue",
        "vector_size_a",
        "vector_size_b",
        "vector_size_c",
        "max_sub_gemms",
        "dy_halo",
        "dy_halo_setprio",
        "dy_halo_kouter_pad",
        "halo_max_wo",
        "filter_h",
        "filter_w",
        "stride_h",
        "stride_w",
        "dilation_h",
        "dilation_w",
    )
    flags = ("lds_k_outer", "grouped", "patch", "tap_outer_k", "uniform_k_decode")
    return spec_class(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        **{name: getattr(identity, name) for name in carried},
        **{name: int(getattr(identity, name)) for name in flags},
    )


def _with_helpers(identities):
    """The identities plus the weight-transform helpers each MFMA row pulls in, once."""
    out = {ident.stable_hash(): ident for ident in identities}
    for ident in identities:
        if ident.algorithm not in dks._MFMA_DGRAD:
            continue
        knobs = json.loads(ident.knobs)
        caps = direct_aot_jobs._caps_of(ident)
        for helper, helper_knobs, _, _ in dks.direct_dgrad_weight_transforms(
            knobs, ident.algorithm
        ):
            helper_identity = replace(
                direct_aot_jobs._helper_identity(
                    ident.arch,
                    ident.wave_size,
                    helper,
                    caps,
                    helper_knobs,
                    ident.dtype_a,
                ),
                llvm_flavor=ident.llvm_flavor,
            )
            out.setdefault(helper_identity.stable_hash(), helper_identity)
    return list(out.values())


@pytest.fixture(scope="module", params=sorted(TARGETS))
def arch(request) -> str:
    """The engine under test, by architecture."""
    return request.param


@pytest.fixture(scope="module")
def catalog(arch):
    """``[(identity, spec)]`` of ``arch``'s shipped catalog, helpers included."""
    identities = _with_helpers(
        [
            spec.identity
            for spec in registered_conv_grouped_instances(arch, direction="dgrad")
        ]
    )
    return [(ident, _spec(ident)) for ident in identities]


def _served_variants(arch: str) -> set:
    """The ``direct_variant`` values of ``arch``'s selectable instances."""
    return {
        AOT_DGRAD_DIRECT_ALGORITHMS[algorithm][1]
        for algorithm in TARGETS[arch].direct_algorithms
        if AOT_DGRAD_DIRECT_ALGORITHMS[algorithm][0] != KERNEL_FAMILY_DGRAD_HELPER
    }


def _form(identity: KernelIdentity) -> tuple:
    """The class of a catalog entry its emission is decided by."""
    if identity.is_direct:
        knobs = json.loads(identity.knobs) if identity.knobs else {}
        return (identity.algorithm, bool(knobs.get("mfma")), identity.dtype_a)
    return (
        "implicit_gemm",
        identity.pipeline,
        identity.epilogue,
        identity.dy_halo,
        identity.patch,
        identity.tap_outer_k,
        identity.uniform_k_decode,
        identity.grouped,
    )


def test_the_catalog_converts_and_reproduces_its_identities(arch, catalog):
    # Every selectable instance plus (gfx950) their weight-transform helpers.
    registered = registered_conv_grouped_instances(arch, direction="dgrad")
    helpers = sum(1 for _, spec in catalog if spec.is_helper)
    assert len(catalog) == len(registered) + helpers
    assert len(catalog) > {"gfx950": 400, "gfx1250": 200}[arch]
    assert (helpers > 0) == (arch == "gfx950")
    algorithms = set()
    for ident, spec in catalog:
        assert spec.ARCH == arch
        assert supports_conv_dgrad_aot(spec) == (True, "ok"), ident.label()
        assert ident.llvm_flavor == TARGETS[arch].llvm_flavor, ident.label()
        identity = spec.to_identity()
        if spec.is_direct:
            # The knobs JSON may spell a knob at its default or leave it out; every field
            # else agrees, and both spellings make the same rocKE spec.
            assert replace(identity, knobs="") == replace(
                ident, knobs=""
            ), ident.label()
            if not spec.is_helper:
                probe = spec.probe_problem()
                assert make_direct_spec(
                    ident.algorithm, probe, json.loads(identity.knobs)
                ) == make_direct_spec(ident.algorithm, probe, json.loads(ident.knobs))
        else:
            assert identity.stable_hash() == spec.instance_id, ident.label()
        algorithms.add(ident.algorithm)
    assert algorithms == set(TARGETS[arch].direct_algorithms) | {"implicit_gemm"}


def test_every_form_is_packageable_with_the_sweeps_ir(arch, catalog, monkeypatch):
    """One instance of each emission form, built by the adapter and by the sweep's
    ``--compile-all`` job, lowers to the same IR up to the symbol."""
    # Lower as the catalog was swept: the builder refuses an older process flavor.
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", TARGETS[arch].llvm_flavor)
    seen = set()
    for ident, spec in catalog:
        form = _form(ident)
        if form in seen:
            continue
        seen.add(form)
        ours = _BUILDERS[arch](spec)
        if ident.is_direct:
            job = direct_aot_jobs.direct_job_for_identity(ident)
            theirs = direct_aot_jobs.direct_kernel_for_job(job, arch, ident.dtype_a)
        else:
            job = job_for_identity(ident)
            theirs = kernel_for_job(job, arch, ident.dtype_a)
        theirs = theirs[0] if isinstance(theirs, tuple) else theirs
        flavor = TARGETS[arch].llvm_flavor
        ours_ir = _lower_kernel_to_llvm_python(ours, arch=arch, llvm_flavor=flavor)
        theirs_ir = _lower_kernel_to_llvm_python(theirs, arch=arch, llvm_flavor=flavor)
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), ident.label()
    # Every direct algorithm and implicit-GEMM form of the catalog (gfx950 has 35,
    # gfx1250 16).
    assert len(seen) > {"gfx950": 30, "gfx1250": 15}[arch]


def test_an_mfma_instance_names_the_helpers_its_pipeline_pulls_in(arch, catalog):
    # Only gfx950 has an MFMA pipeline.
    assert any(s.is_mfma for _, s in catalog) == (arch == "gfx950")
    helpers = {
        (s.dtype, s.direct_variant, s.filter_h, s.cpg, s.kpg, s.fold_k32): s
        for _, s in catalog
        if s.is_helper
    }
    for _, spec in catalog:
        if not spec.is_mfma:
            continue
        for fields_ in gfx950_conv_dgrad_aot_helper_fields(spec):
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
                assert getattr(helper, name) == value, name
            assert helper.kernel_family == KERNEL_FAMILY_DGRAD_HELPER


def test_packaging_refuses_an_older_flavor(arch, catalog, monkeypatch):
    _, spec = next((i, s) for i, s in catalog if not s.is_direct)
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    with pytest.raises(ValueError, match="older COMGR"):
        _BUILDERS[arch](spec)


def test_each_builder_has_the_support_predicate_hkp_pack_resolves(arch, catalog):
    """hkp_pack validates a spec with ``supports_`` + its builder's name, whose
    first parameter takes the builder's spec class (rocke_compile)."""
    import typing

    from builders.common import convolution_backward_data as adapter

    builder = _BUILDERS[arch].__name__
    predicate = getattr(adapter, "supports_" + builder.removeprefix("build_"))
    spec_class = SPEC_CLASSES[arch]
    hints = typing.get_type_hints(predicate)
    assert hints["spec"] is spec_class
    assert typing.get_type_hints(_BUILDERS[arch])["spec"] is spec_class
    for _, spec in catalog[:20]:
        assert predicate(spec, arch=arch) == (True, "ok")


def test_an_engine_builds_only_its_own_architectures_specs(catalog, monkeypatch):
    _, spec = next((i, s) for i, s in catalog if not s.is_direct)
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", TARGETS[spec.ARCH].llvm_flavor)
    other = next(a for a in TARGETS if a != spec.ARCH)
    ok, why = supports_conv_dgrad_aot(spec, arch=other)
    assert not ok and f"requires arch={spec.ARCH}" in why
    with pytest.raises(ValueError, match=f"requires arch={spec.ARCH}"):
        _BUILDERS[spec.ARCH](spec, arch=other)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _p(N, Hi, Wi, C, K, Y, X, s=(1, 1), p=(0, 0), d=(1, 1), groups=1, dtype="fp16"):
    return ConvDgradAotProblem(
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


#: Shapes every family is decided on: dense, strided, dilated, folded and patch
#: embeddings, pointwise, narrow channels, grouped at the direct rows' channel counts,
#: depthwise at the direct rows' filters, the MFMA rows, halo widths either side of the
#: binaries' bounds, and a batch past the 32-bit ABI.
_PROBLEMS = [
    _p(2, 14, 14, 64, 64, 3, 3, p=(1, 1)),
    _p(2, 15, 15, 32, 48, 3, 3, s=(2, 2), p=(1, 1)),
    _p(1, 17, 13, 16, 24, 3, 3, s=(3, 2), p=(2, 1), d=(2, 1)),
    _p(1, 20, 20, 24, 40, 3, 3, p=(2, 2), d=(2, 2)),
    _p(2, 64, 64, 3, 64, 16, 16, s=(16, 16)),
    _p(2, 64, 64, 3, 64, 4, 4, s=(4, 4)),
    _p(2, 32, 32, 128, 256, 2, 2, s=(2, 2)),
    _p(2, 28, 28, 64, 128, 1, 1, s=(2, 2)),
    _p(2, 14, 14, 128, 64, 1, 1),
    _p(1, 10, 10, 3, 64, 7, 7, s=(2, 2), p=(3, 3)),
    _p(2, 28, 28, 64, 128, 3, 3, p=(1, 1), groups=4),
    _p(2, 28, 28, 128, 128, 3, 3, p=(1, 1), groups=8),
    _p(2, 32, 32, 256, 256, 3, 3, p=(1, 1), groups=64),
    _p(2, 28, 28, 768, 768, 3, 3, p=(1, 1), groups=32),
    _p(4, 56, 56, 24, 24, 3, 3, p=(1, 1)),
    _p(2, 28, 28, 64, 64, 3, 3, p=(1, 1), groups=64),
    _p(2, 28, 28, 64, 64, 3, 3, s=(2, 2), p=(1, 1), groups=64),
    _p(2, 14, 14, 96, 96, 7, 7, p=(3, 3), groups=96),
    _p(2, 14, 14, 96, 96, 5, 5, p=(2, 2), groups=96),
    _p(1, 32, 32, 4, 320, 3, 3, p=(1, 1)),
    _p(1, 32, 32, 12, 320, 3, 3, p=(1, 1)),
    _p(1, 32, 32, 8, 640, 3, 3, p=(1, 1), groups=2),
    _p(2, 32, 32, 64, 64, 3, 3, p=(1, 1)),
    _p(2, 16, 96, 64, 64, 3, 3, p=(1, 1)),
    _p(2, 16, 97, 64, 64, 3, 3, p=(1, 1)),
    _p(1, 4, 25600, 128, 64, 1, 11, p=(0, 5)),
    _p(16, 768, 768, 256, 256, 3, 3, p=(1, 1), dtype="bf16"),
]
_PROBLEMS += [replace(p, dtype="bf16") for p in _PROBLEMS[:-1:2]]


def _rocke_offer(identity: KernelIdentity, problem: ConvDgradAotProblem) -> bool:
    """rocKE's own offer: the implicit-GEMM sweep's (``identity_supports_problem`` on
    each batch slice of the problem it launches), or ``direct_plans``' (the capability
    match, then ``plan_for``'s re-validation)."""
    if identity.dtype_a != problem.dtype:
        return False
    if not identity.is_direct:
        launched = _launch_problem("dgrad", problem.to_problem())
        try:
            chunks = _batch_chunks("dgrad", launched, problem.dtype)
        except ValueError:
            return False
        return all(
            identity_supports_problem(
                identity, replace(launched, N=n), llvm_flavor=identity.llvm_flavor
            )[0]
            for _, n in chunks
        )
    if problem.sH != problem.sW or problem.pH != problem.pW:
        return False
    direct = problem.to_direct_problem()
    if not identity_supports_problem(
        identity, direct, llvm_flavor=identity.llvm_flavor
    )[0]:
        return False
    plan, _ = dks.plan_for(
        _HelperCache(),
        identity,
        Path("x.hsaco"),
        _entry_meta(identity),
        direct,
        identity.arch,
    )
    return plan is not None


def _entry_meta(identity: KernelIdentity) -> dict:
    """The meta a fresh ``--compile-all`` entry of ``identity`` carries: a
    wave-parametric build records the wave size it was built at, the identity's."""
    meta = {"kernel_name": "k"}
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if direct_spec_takes_target_wave(identity.algorithm, knobs):
        meta["spec_wave_size"] = identity.wave_size
    return meta


class _HelperCache:
    """The cache calls ``plan_for`` makes for the MFMA helpers."""

    def get(self, identity):
        return Path("x.hsaco"), {"kernel_name": identity.algorithm}

    def entry_meta(self, identity):
        return {"kernel_name": identity.algorithm}

    def hsaco_path(self, identity):
        return Path("x.hsaco")


def test_the_plan_serves_rockes_offer(arch, catalog):
    """rocKE's offer: a helper serves nothing, every other instance matches it exactly.

    The coalesced-weights MFMA instances (waves_k > 1 or runtime_k_loop) used to be
    refused on every grouped problem, for a build_direct_reorganize_weights bug fixed
    upstream (commit aa9ff78) -- confirmed fixed by running the grouped_cpg4_kpg320
    shape at groups=2 and groups=4 against the CPU reference on
    gfx950 (benchmark_grouped_conv.py --direction dgrad --algorithm
    direct_dgrad_mfma_rtc,direct_grouped_dgrad_mfma --verify): every instance passed.
    """
    served = {}
    for ident, spec in catalog:
        for problem in _PROBLEMS:
            plan, why = conv_dgrad_aot_plan(spec, problem)
            if spec.is_helper:
                assert plan is None
                continue
            expected = _rocke_offer(ident, problem)
            assert (plan is not None) == expected, (ident.label(), problem, why)
            if plan is not None:
                served[spec.direct_variant] = served.get(spec.direct_variant, 0) + 1
    assert set(served) == {"none"} | _served_variants(arch), served


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


def test_the_implicit_plan_is_the_benchmarks_launch(catalog):
    """Per batch slice of the folded problem: the benchmark's record table, grid and
    kernel arguments, at its view of dY and dX."""
    checked = set()
    for ident, spec in catalog:
        if spec.is_direct:
            continue
        block = aot_jobs.spec_launch_block_size(
            aot_jobs.spec_for_job(job_for_identity(ident), ident.dtype_a)
        )
        for problem in _PROBLEMS:
            plan, _ = conv_dgrad_aot_plan(spec, problem)
            if plan is None:
                continue
            launched = _launch_problem("dgrad", problem.to_problem())
            assert plan.launched.to_problem() == launched
            assert conv_dgrad_aot_launch_problem(problem) == plan.launched
            chunks = _batch_chunks("dgrad", launched, problem.dtype)
            assert len(plan.steps) == len(chunks)
            dy_image, _, dx_image = plan.launched.tensor_bytes(1)
            table_offset = 0
            for step, (n0, n) in zip(plan.steps, chunks):
                chunk = replace(launched, N=n)
                rt = _RecordingRuntime()
                extras, grid = _extras_and_grid(
                    "dgrad", chunk, ident, rt, problem.dtype
                )
                assert step.kernel == KERNEL_SELF and step.abi == ABI_IMPLICIT
                assert step.buffers == (BUFFER_DY, BUFFER_W, BUFFER_DX)
                assert step.grid == grid
                assert step.block == block
                assert _i32(step.table) == rt.copied[extras["sub_gemm_buf"]]
                assert step.table_offset == table_offset
                table_offset += 4 * len(step.table)
                sizes = replace(plan.launched, N=n).tensor_bytes()
                theirs = kernel_sweep._launch_values_for(
                    "dgrad", chunk, ident, (0, 0, 0), sizes, extras
                )
                names = [
                    a["name"]
                    for a in conv_dgrad_aot_step_signature(ABI_IMPLICIT, spec.dtype)
                ]
                assert names[6:] == [n for n, _ in step.values[:-1]] + [
                    "sub_gemm_buf",
                    "num_sub_gemms",
                ]
                assert dict(step.values) == {
                    k: v for k, v in theirs.items() if k in dict(step.values)
                }
                if len(chunks) > 1:
                    assert step.views == (
                        (n0 * dy_image, n * dy_image),
                        None,
                        (n0 * dx_image, n * dx_image),
                    )
                    checked.add("batch")
                else:
                    assert step.views == (None, None, None)
            assert plan.workspaces == ((BUFFER_SUB_GEMM, 0, table_offset),)
            if plan.launched != problem:
                checked.add("folded")
            checked.add("patch" if spec.patch else "halo" if spec.dy_halo else "tilde")
    # The forms the catalog ships: gfx1250's has no patch or dY-halo kernel.
    forms = {
        "patch" if s.patch else "halo" if s.dy_halo else "tilde" for _, s in catalog
    }
    assert checked == {"batch", "folded"} | forms


def test_the_direct_plan_is_plan_fors(arch, catalog):
    ptrs = {"dy": 0x100000, "w": 0x200000, "dx": 0x300000}
    checked = set()
    for ident, spec in catalog:
        if not spec.is_direct or spec.is_helper:
            continue
        for problem in _PROBLEMS:
            ours, _ = conv_dgrad_aot_plan(spec, problem)
            if ours is None:
                continue
            theirs, why = dks.plan_for(
                _HelperCache(),
                ident,
                Path("x.hsaco"),
                _entry_meta(ident),
                problem.to_direct_problem(),
                arch,
            )
            assert theirs is not None, why
            assert not ours.zero_dx
            assert {
                name: size for name, _, size in ours.workspaces
            } == theirs.workspaces
            all_ptrs = dict(ptrs)
            all_sizes = dict(zip(("dy", "w", "dx"), problem.tensor_bytes()))
            for name, offset, size in ours.workspaces:
                assert offset % 256 == 0
                all_ptrs[name] = 0x400000 + offset
                all_sizes[name] = size
            assert len(ours.steps) == len(theirs.steps)
            for mine, step in zip(ours.steps, theirs.steps):
                assert mine.grid == step.grid
                assert (mine.block, 1, 1) == step.block
                assert mine.buffers == step.buffers
                signature = conv_dgrad_aot_step_signature(mine.abi, spec.dtype)
                assert signature == step.signature
                values = dks.launch_values(step, all_ptrs, all_sizes)
                first = 4 if mine.buffers[1] is None else 6
                assert [a["name"] for a in signature[first:]] == [
                    n for n, _ in mine.values
                ]
                assert dict(mine.values) == {k: values[k] for k, _ in mine.values}
            if spec.is_mfma:
                assert ours.steps[-1].abi == ABI_DIRECT_FWD
            else:
                assert len(ours.steps) == 1
                assert ours.steps[0].abi in (ABI_DIRECT_DGRAD, ABI_DIRECT_FWD)
            checked.add(spec.direct_variant)
    assert checked == _served_variants(arch)


def _written(problem: ConvDgradAotProblem, sub_gemms, patch: bool) -> set:
    """Every dX pixel some record stores, pixel by pixel."""
    p = problem.to_problem()
    out = set()
    for sg in sub_gemms:
        for th in range(sg.h_tilde_slice):
            for tw in range(sg.w_tilde_slice):
                h0 = sg.d_h_offset + sg.d_h_stride * th
                w0 = sg.d_w_offset + sg.d_w_stride * tw
                for dy in range(p.Y if patch else 1):
                    for dx in range(p.X if patch else 1):
                        out.add((h0 + dy, w0 + dx))
    return out


@pytest.mark.parametrize(
    "problem",
    [
        _p(1, 9, 9, 8, 8, 3, 3, p=(1, 1)),
        _p(1, 9, 9, 8, 8, 3, 3, s=(2, 2), p=(1, 1)),
        _p(1, 14, 14, 8, 8, 3, 3, s=(2, 2), p=(1, 1)),
        _p(1, 15, 15, 8, 8, 2, 2, s=(2, 2)),
        _p(1, 14, 14, 8, 8, 1, 1, s=(2, 2)),
        _p(1, 14, 14, 8, 8, 1, 1, s=(1, 2)),
        _p(1, 16, 16, 8, 8, 3, 3, s=(4, 4), p=(1, 1)),
        _p(1, 17, 13, 16, 24, 3, 3, s=(3, 2), p=(2, 1), d=(2, 1)),
        _p(1, 20, 20, 24, 40, 3, 3, p=(2, 2), d=(2, 2)),
        _p(1, 31, 29, 16, 32, 7, 7, s=(4, 4), p=(3, 3)),
    ],
)
def test_dx_is_zero_filled_exactly_where_the_records_leave_it_unwritten(
    problem, catalog
):
    # A runtime-geometry tilde kernel, the narrowest so it serves every problem here.
    spec = min(
        (
            s
            for _, s in catalog
            if not s.is_direct
            and not s.patch
            and not s.dy_halo
            and not s.tap_outer_k
            and s.filter_h == 0
            and s.dtype == problem.dtype
        ),
        key=lambda s: (s.tile_m, s.tile_n, s.instance_id),
    )
    plan, why = conv_dgrad_aot_plan(spec, problem)
    assert plan is not None, why
    p = plan.launched.to_problem()
    written = _written(plan.launched, dgrad_sub_gemms(spec.to_identity(), p), False)
    full = all((h, w) in written for h in range(p.Hi) for w in range(p.Wi))
    assert plan.zero_dx == (not full)


def test_a_patch_plan_zero_fills_only_between_patches(arch, catalog):
    patches = [s for _, s in catalog if s.patch and s.dtype == "fp16"]
    # gfx1250 serves patch embeddings with its direct WMMA patch kernel instead.
    assert bool(patches) == (arch == "gfx950")
    if not patches:
        return
    for problem, gaps in (
        (_p(2, 64, 64, 3, 64, 4, 4, s=(4, 4)), False),
        (_p(2, 30, 30, 64, 64, 4, 4, s=(5, 5)), True),
    ):
        # The first patch instance that serves the problem (C=3 needs a narrow dX vector).
        spec, plan = next(
            (spec, plan)
            for spec in patches
            for plan in [conv_dgrad_aot_plan(spec, problem)[0]]
            if plan is not None
        )
        p = plan.launched.to_problem()
        written = _written(plan.launched, dgrad_sub_gemms(spec.to_identity(), p), True)
        full = all((h, w) in written for h in range(p.Hi) for w in range(p.Wi))
        assert full != gaps
        assert plan.zero_dx == gaps


def _covered_by_marking(extent: int, runs) -> bool:
    """``_covered`` by marking every position: the definition it must match."""
    hit = bytearray(extent)
    for offset, stride, count, width in runs:
        for t in range(count):
            start = offset + stride * t
            for pos in range(max(start, 0), min(start + width, extent)):
                hit[pos] = 1
    return all(hit)


def test_coverage_is_decided_as_marking_every_position_would():
    rng = random.Random(0)
    covered = 0
    for _ in range(20000):
        extent = rng.randint(0, 40)
        runs = [
            (
                rng.randint(-6, 12),
                rng.randint(-4, 7),
                rng.randint(-1, 12),
                rng.randint(-1, 6),
            )
            for _ in range(rng.randint(0, 4))
        ]
        expected = _covered_by_marking(extent, runs)
        assert _covered(extent, runs) == expected, (extent, runs)
        covered += expected
    assert 0 < covered < 20000


def test_coverage_costs_nothing_per_position(catalog):
    """A long, narrow problem (Wi = 1, Hi up to what one image admits) is decided
    without a buffer of Hi: the check is arithmetic over the records."""
    extent = 2**31 - 1
    tracemalloc.start()
    try:
        assert _covered(extent, [(r, 3, extent // 3 + 1, 1) for r in range(3)])
        assert not _covered(extent, [(r, 3, extent // 3 + 1, 1) for r in range(2)])
        assert not _covered(extent, [(0, 1, extent - 1, 1)])
        spec = next(
            s
            for _, s in catalog
            if not s.is_direct and not s.patch and s.filter_h == 0 and s.dtype == "bf16"
        )
        plan, why = conv_dgrad_aot_plan(
            spec, _p(1, 2**22, 1, 64, 64, 3, 1, p=(1, 0), dtype="bf16")
        )
        assert plan is not None, why
        assert not plan.zero_dx
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2**20


def test_the_digest_is_fnv1a_over_the_serialized_plan(catalog):
    plan = next(
        plan
        for _, spec in catalog
        if not spec.is_direct
        for plan in [conv_dgrad_aot_plan(spec, _PROBLEMS[0])[0]]
        if plan is not None
    )
    values = plan.serialize()
    h = 0xCBF29CE484222325
    for value in values:
        for byte in (value & ((1 << 64) - 1)).to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert plan.digest() == h
    assert values[:3] == [int(plan.zero_dx), plan.workspace_bytes, 1]


def test_problem_for_request_is_the_dgrad_graph_contract(arch):
    request = ConvGroupedRequest(
        N=2,
        C=64,
        K=32,
        Hi=15,
        Wi=15,
        Y=3,
        X=3,
        arch=arch,
        G=2,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        direction="dgrad",
    )
    problem = conv_dgrad_aot_problem_for_request(request)
    assert (problem.Hi, problem.Ho, problem.cpg, problem.kpg) == (15, 8, 32, 16)
    assert supports_conv_dgrad_aot_problem(problem, arch=arch) == (True, "ok")
    with pytest.raises(ValueError, match="backward-data"):
        conv_dgrad_aot_problem_for_request(replace(request, direction="fwd"))
    assert not supports_conv_dgrad_aot_problem(replace(problem, groups=3), arch=arch)[0]
    assert not supports_conv_dgrad_aot_problem(
        replace(problem, dtype="fp32"), arch=arch
    )[0]
    # A batch past the 32-bit ABI is in the contract (it runs as slices); one image
    # past it is not.
    big = _p(16, 768, 768, 256, 256, 3, 3, p=(1, 1))
    assert supports_conv_dgrad_aot_problem(big, arch=arch) == (True, "ok")
    assert not supports_conv_dgrad_aot_problem(
        _p(1, 8192, 8192, 32, 32, 3, 3, p=(1, 1)), arch=arch
    )[0]
    with pytest.raises(ValueError, match="no packaged dgrad engine"):
        conv_dgrad_aot_problem_for_request(replace(request, arch="gfx942"))


# ---------------------------------------------------------------------------
# Per-arch refusals
# ---------------------------------------------------------------------------


#: Dilations a variant without dilation must refuse: each axis alone and both.
_DILATIONS = (
    ("h", {"dilation_h": 2}),
    ("w", {"dilation_w": 2}),
    ("hw", {"dilation_h": 2, "dilation_w": 2}),
)


def _guard_cases(arch: str, catalog) -> list:
    """``(label, spec, arch passed, refusal text)`` per per-arch guard of
    :func:`supports_conv_dgrad_aot`: a packaged spec of ``arch`` with one field (or
    one pair) replaced by what the other engine, or no engine, accepts."""
    target = TARGETS[arch]
    other = TARGETS[next(a for a in TARGETS if a != arch)]
    specs = [s for _, s in catalog]

    def first(predicate):
        return next((s for s in specs if predicate(s)), None)

    implicit = first(lambda s: not s.is_direct)
    depthwise = first(lambda s: s.direct_variant == DIRECT_VARIANT_DEPTHWISE)
    wmma = first(lambda s: s.direct_variant == DIRECT_VARIANT_WMMA)
    wmma_undilated = first(
        lambda s: s.direct_variant in ("dgrad_smallc_wmma", "dgrad_patch_wmma")
    )
    helper = first(lambda s: s.is_helper)
    runtime_channels = first(lambda s: s.is_runtime_channels)
    wave = f"wave_size must be {target.wave_size} on {arch}"
    foreign_pipeline = next(p for p in other.pipelines if p not in target.pipelines)
    foreign_variant = {"gfx950": DIRECT_VARIANT_WMMA, "gfx1250": DIRECT_VARIANT_MFMA}

    class _NoEngineSpec(SPEC_CLASSES[arch]):
        ARCH = "gfx942"

    stray = _NoEngineSpec(
        **{f.name: getattr(implicit, f.name) for f in fields(implicit)}
    )
    # A block knob of this engine that its depthwise variant does not take.
    used = set(depthwise.DIRECT_KNOBS[depthwise.direct_variant].values())
    unused = next(
        n for n, v in depthwise.KNOB_DEFAULTS.items() if n not in used and v == 0
    )
    implicit_knob = "waves_c" if "waves_c" in implicit.KNOB_DEFAULTS else "block_waves"
    cases = [
        ("not a spec", object(), None, "expected one of"),
        ("spec of no engine", stray, None, "expected one of"),
        ("another arch", implicit, other.arch, f"requires arch={arch}"),
        (
            "implicit wave size",
            replace(implicit, wave_size=other.wave_size),
            None,
            wave,
        ),
        (
            "implicit pipeline",
            replace(implicit, pipeline=foreign_pipeline),
            None,
            f"supports pipelines {target.pipelines} on {arch}",
        ),
        (
            "implicit block knob",
            replace(implicit, **{implicit_knob: 1}),
            None,
            f"must leave {implicit_knob} at 0",
        ),
        (
            "direct wave size",
            replace(depthwise, wave_size=other.wave_size),
            None,
            wave,
        ),
        (
            "direct variant",
            replace(depthwise, direct_variant=foreign_variant[arch]),
            None,
            "direct_variant must be one of",
        ),
        ("one runtime channel count", replace(depthwise, cpg=0), None, "both runtime"),
        (
            "runtime channels",
            replace(depthwise, cpg=0, kpg=0),
            None,
            "only the runtime-channel pipeline",
        ),
        *(
            (
                f"direct dilation {axes}",
                replace(depthwise, **dilated),
                None,
                "have no dilation",
            )
            for axes, dilated in _DILATIONS
        ),
        (
            "unused knob",
            replace(depthwise, **{unused: 1}),
            None,
            f"does not use {unused}",
        ),
        (
            "knob type",
            replace(depthwise, **{unused: "1"}),
            None,
            f"{unused} must be a int",
        ),
    ]
    if runtime_channels is not None:
        cases.append(
            (
                "baked channels of a runtime-channel pipeline",
                replace(runtime_channels, cpg=8, kpg=8),
                None,
                "and it takes them",
            )
        )
    if helper is not None:
        cases.append(
            (
                "helper wave size",
                replace(helper, wave_size=other.wave_size),
                None,
                wave,
            )
        )
    if wmma is not None:
        cases += [
            *(
                (
                    f"WMMA dilation {axis}",
                    replace(wmma, **{axis: 0}),
                    None,
                    "dilation_h/dilation_w must be positive",
                )
                for axis in ("dilation_h", "dilation_w")
            ),
            # The native matcher pairs a dense WMMA binary's channel counts.
            *(
                (
                    f"WMMA one runtime channel count {pair}",
                    replace(wmma, **pair),
                    None,
                    "both runtime",
                )
                for pair in ({"cpg": 0, "kpg": 8}, {"cpg": 8, "kpg": 0})
            ),
            ("WMMA string knob", replace(wmma, sched=0), None, "sched must be a str"),
            (
                "WMMA bool knob",
                replace(wmma, wave_epilogue=2),
                None,
                "wave_epilogue must be 0 or 1",
            ),
            (
                "WMMA int knob",
                replace(wmma, waves_c=-1),
                None,
                "waves_c must be a nonnegative",
            ),
        ]
    if wmma_undilated is not None:
        cases.append(
            (
                "runtime input channels of a WMMA stem",
                replace(wmma_undilated, cpg=0),
                None,
                "bakes its input channels",
            )
        )
        cases += [
            (
                f"dilation {axes} of an undilated WMMA variant",
                replace(wmma_undilated, **dilated),
                None,
                "have no dilation",
            )
            for axes, dilated in _DILATIONS
        ]
    return cases


def test_each_per_arch_guard_refuses_with_its_own_text(arch, catalog):
    """Every per-arch refusal of :func:`supports_conv_dgrad_aot`, of the identity
    conversion and of the problem predicate, each by its own text: a packaged spec
    of ``arch`` with one field replaced is refused for that field."""
    wrong = []
    for label, spec, on, text in _guard_cases(arch, catalog):
        ok, why = supports_conv_dgrad_aot(spec, arch=on)
        if ok or text not in why:
            wrong.append((label, ok, why, text))
    # The identity side: an arch with no engine, the other engine's algorithm, a
    # string knob of the wrong type.
    direct = next(i for i, s in catalog if s.is_direct and not s.is_helper)
    foreign = next(
        a
        for a in TARGETS[next(x for x in TARGETS if x != arch)].direct_algorithms
        if a not in TARGETS[arch].direct_algorithms
    )
    identity_cases = [
        (replace(direct, arch="gfx942"), "no packaged dgrad engine for arch"),
        (replace(direct, algorithm=foreign), f"no packaged dgrad launch on {arch}"),
    ]
    wmma = next((i for i, _ in catalog if i.algorithm == "direct_dgrad_wmma"), None)
    if wmma is not None:
        knobs = {**json.loads(wmma.knobs), "sched": 1}
        identity_cases.append(
            (replace(wmma, knobs=json.dumps(knobs)), "must be a nonempty string")
        )
    for identity, text in identity_cases:
        try:
            aot_dgrad_direct_instance_fields(identity)
        except ValueError as exc:
            if text not in str(exc):
                wrong.append((identity.label(), False, str(exc), text))
        else:
            wrong.append((identity.label(), True, "ok", text))
    # A variant this engine's knob table lacks.
    variant = AOT_DGRAD_DIRECT_ALGORITHMS[foreign][1]
    with pytest.raises(ValueError, match="no packaged dgrad launch takes variant"):
        aot_dgrad_direct_fields_from_knobs(variant, {}, SPEC_CLASSES[arch].DIRECT_KNOBS)
    problem = _p(2, 14, 14, 64, 64, 3, 3, p=(1, 1))
    assert supports_conv_dgrad_aot_problem(problem, arch=arch) == (True, "ok")
    ok, why = supports_conv_dgrad_aot_problem(problem, arch="gfx942")
    if ok or "no packaged dgrad engine" not in why:
        wrong.append(("problem of no engine", ok, why, "no packaged dgrad engine"))
    assert not wrong, wrong
