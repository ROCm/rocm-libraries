# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT backward-data instances (hipkernel:Gfx950ConvDgradAot).

An instance is packaged from a rocKE dgrad ``KernelIdentity``; these tests hold the
adapter (``builders/common/convolution_backward_data.py``) to the sweeps it comes from,
over the shipped catalog (``instances_dgrad.txt`` and the weight-transform helpers its
MFMA rows pull in):

* the packaged IR is the sweep kernel's, byte for byte up to the symbol, for one
  instance of every pipeline and implicit-GEMM form;
* the launch plan (``gfx950_conv_dgrad_aot_plan``) serves exactly rocKE's offer --
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
from dataclasses import replace
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
    COALESCED_GROUPS_REASON,
    KERNEL_FAMILY_DGRAD_HELPER,
    KERNEL_SELF,
    PACKAGED_LLVM_FLAVOR,
    Gfx950ConvDgradAotSpec,
    Gfx950ConvDgradProblem,
    aot_dgrad_direct_instance_fields,
    build_gfx950_conv_dgrad_aot,
    gfx950_conv_dgrad_aot_helper_fields,
    gfx950_conv_dgrad_aot_launch_problem,
    gfx950_conv_dgrad_aot_plan,
    gfx950_conv_dgrad_aot_problem_for_request,
    gfx950_conv_dgrad_aot_step_signature,
    supports_gfx950_conv_dgrad_aot,
    supports_gfx950_conv_dgrad_aot_problem,
)
from dispatch.grouped_convolution import ConvGroupedRequest
from kernels.conv import direct_aot_jobs
from kernels.conv.aot_jobs import job_for_identity, kernel_for_job
from kernels.conv.direct_variants import make_direct_spec
from kernels.conv.kernel_identity import (
    KernelIdentity,
    dgrad_sub_gemms,
    identity_supports_problem,
)
from rocke.assets import platform_root
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCH = "gfx950"

# ---------------------------------------------------------------------------
# The catalog
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
    return Gfx950ConvDgradAotSpec(
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


@pytest.fixture(scope="module")
def catalog():
    """``[(identity, spec)]`` of the shipped catalog, helpers included."""
    labels = (platform_root().parent / "instances_dgrad.txt").read_text().split()
    identities = _with_helpers([KernelIdentity.from_label(label) for label in labels])
    return [(ident, _spec(ident)) for ident in identities]


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


def test_the_catalog_converts_and_reproduces_its_identities(catalog):
    assert len(catalog) > 600
    algorithms = set()
    for ident, spec in catalog:
        assert supports_gfx950_conv_dgrad_aot(spec) == (True, "ok"), ident.label()
        assert ident.llvm_flavor == PACKAGED_LLVM_FLAVOR, ident.label()
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
    assert algorithms >= set(AOT_DGRAD_DIRECT_ALGORITHMS) | {"implicit_gemm"}


def test_every_form_is_packageable_with_the_sweeps_ir(catalog):
    """One instance of each emission form, built by the adapter and by the sweep's
    ``--compile-all`` job, lowers to the same IR up to the symbol."""
    seen = set()
    for ident, spec in catalog:
        form = _form(ident)
        if form in seen:
            continue
        seen.add(form)
        ours = build_gfx950_conv_dgrad_aot(spec)
        if ident.is_direct:
            job = direct_aot_jobs.direct_job_for_identity(ident)
            theirs = direct_aot_jobs.direct_kernel_for_job(job, _ARCH, ident.dtype_a)
        else:
            job = job_for_identity(ident)
            theirs = kernel_for_job(job, _ARCH, ident.dtype_a)
        theirs = theirs[0] if isinstance(theirs, tuple) else theirs
        ours_ir = _lower_kernel_to_llvm_python(ours, arch=_ARCH, llvm_flavor="llvm22")
        theirs_ir = _lower_kernel_to_llvm_python(
            theirs, arch=_ARCH, llvm_flavor="llvm22"
        )
        assert ours_ir.replace(ours.name, "@K") == theirs_ir.replace(
            theirs.name, "@K"
        ), ident.label()
    assert len(seen) > 20


def test_an_mfma_instance_names_the_helpers_its_pipeline_pulls_in(catalog):
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


def test_packaging_refuses_an_older_flavor(catalog, monkeypatch):
    _, spec = next((i, s) for i, s in catalog if not s.is_direct)
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    with pytest.raises(ValueError, match="older COMGR"):
        build_gfx950_conv_dgrad_aot(spec)


# ---------------------------------------------------------------------------
# Problems
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


def _rocke_offer(identity: KernelIdentity, problem: Gfx950ConvDgradProblem) -> bool:
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
        _HelperCache(), identity, Path("x.hsaco"), {"kernel_name": "k"}, direct, _ARCH
    )
    return plan is not None


def _uses_reorganize(spec: Gfx950ConvDgradAotSpec) -> bool:
    return spec.is_mfma and (spec.waves_k > 1 or bool(spec.runtime_k_loop))


class _HelperCache:
    """The two cache calls ``plan_for`` makes for the MFMA helpers."""

    def get(self, identity):
        return Path("x.hsaco"), {"kernel_name": identity.algorithm}

    def hsaco_path(self, identity):
        return Path("x.hsaco")


def test_the_plan_serves_rockes_offer(catalog):
    """rocKE's offer, except the grouped problems of the MFMA instances that use the
    reorganize helper (rocKE ``build_direct_reorganize_weights`` mis-addresses
    groups > 1), which the adapter refuses; a helper serves nothing."""
    served = {}
    refused_grouped_reorganize = 0
    for ident, spec in catalog:
        for problem in _PROBLEMS:
            plan, why = gfx950_conv_dgrad_aot_plan(spec, problem)
            if spec.is_helper:
                assert plan is None
                continue
            expected = _rocke_offer(ident, problem)
            if expected and _uses_reorganize(spec) and problem.groups != 1:
                assert plan is None and why == COALESCED_GROUPS_REASON, (
                    ident.label(),
                    problem,
                )
                refused_grouped_reorganize += 1
                continue
            assert (plan is not None) == expected, (ident.label(), problem, why)
            if plan is not None:
                served[spec.direct_variant] = served.get(spec.direct_variant, 0) + 1
    assert set(served) == {"none"} | {
        variant
        for family, variant in AOT_DGRAD_DIRECT_ALGORITHMS.values()
        if family != KERNEL_FAMILY_DGRAD_HELPER
    }, served
    assert refused_grouped_reorganize > 0


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
        for problem in _PROBLEMS:
            plan, _ = gfx950_conv_dgrad_aot_plan(spec, problem)
            if plan is None:
                continue
            launched = _launch_problem("dgrad", problem.to_problem())
            assert plan.launched.to_problem() == launched
            assert gfx950_conv_dgrad_aot_launch_problem(problem) == plan.launched
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
                assert step.block == ident.warp_m * ident.warp_n * 64
                assert _i32(step.table) == rt.copied[extras["sub_gemm_buf"]]
                assert step.table_offset == table_offset
                table_offset += 4 * len(step.table)
                sizes = replace(plan.launched, N=n).tensor_bytes()
                theirs = kernel_sweep._launch_values_for(
                    "dgrad", chunk, ident, (0, 0, 0), sizes, extras
                )
                names = [
                    a["name"]
                    for a in gfx950_conv_dgrad_aot_step_signature(
                        ABI_IMPLICIT, spec.dtype
                    )
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
    assert checked == {"batch", "folded", "patch", "halo", "tilde"}


def test_the_direct_plan_is_plan_fors(catalog):
    ptrs = {"dy": 0x100000, "w": 0x200000, "dx": 0x300000}
    checked = set()
    for ident, spec in catalog:
        if not spec.is_direct or spec.is_helper:
            continue
        for problem in _PROBLEMS:
            ours, _ = gfx950_conv_dgrad_aot_plan(spec, problem)
            if ours is None:
                continue
            theirs, why = dks.plan_for(
                _HelperCache(),
                ident,
                Path("x.hsaco"),
                {"kernel_name": "k"},
                problem.to_direct_problem(),
                _ARCH,
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
                signature = gfx950_conv_dgrad_aot_step_signature(mine.abi, spec.dtype)
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
    assert checked == {
        variant
        for family, variant in AOT_DGRAD_DIRECT_ALGORITHMS.values()
        if family != KERNEL_FAMILY_DGRAD_HELPER
    }


def _written(problem: Gfx950ConvDgradProblem, sub_gemms, patch: bool) -> set:
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
    _, spec = next(
        (i, s)
        for i, s in catalog
        if not s.is_direct
        and not s.patch
        and not s.dy_halo
        and not s.tap_outer_k
        and s.filter_h == 0
        and s.tile_m <= 64
    )
    plan, why = gfx950_conv_dgrad_aot_plan(spec, problem)
    assert plan is not None, why
    p = plan.launched.to_problem()
    written = _written(plan.launched, dgrad_sub_gemms(spec.to_identity(), p), False)
    full = all((h, w) in written for h in range(p.Hi) for w in range(p.Wi))
    assert plan.zero_dx == (not full)


def test_a_patch_plan_zero_fills_only_between_patches(catalog):
    patches = [s for _, s in catalog if s.patch and s.dtype == "fp16"]
    for problem, gaps in (
        (_p(2, 64, 64, 3, 64, 4, 4, s=(4, 4)), False),
        (_p(2, 30, 30, 64, 64, 4, 4, s=(5, 5)), True),
    ):
        # The first patch instance that serves the problem (C=3 needs a narrow dX vector).
        spec, plan = next(
            (spec, plan)
            for spec in patches
            for plan in [gfx950_conv_dgrad_aot_plan(spec, problem)[0]]
            if plan is not None
        )
        p = plan.launched.to_problem()
        written = _written(plan.launched, dgrad_sub_gemms(spec.to_identity(), p), True)
        full = all((h, w) in written for h in range(p.Hi) for w in range(p.Wi))
        assert full != gaps
        assert plan.zero_dx == gaps


def test_the_digest_is_fnv1a_over_the_serialized_plan(catalog):
    plan = next(
        plan
        for _, spec in catalog
        if not spec.is_direct
        for plan in [gfx950_conv_dgrad_aot_plan(spec, _PROBLEMS[0])[0]]
        if plan is not None
    )
    values = plan.serialize()
    h = 0xCBF29CE484222325
    for value in values:
        for byte in (value & ((1 << 64) - 1)).to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert plan.digest() == h
    assert values[:3] == [int(plan.zero_dx), plan.workspace_bytes, 1]


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
    # A batch past the 32-bit ABI is in the contract (it runs as slices); one image
    # past it is not.
    big = _p(16, 768, 768, 256, 256, 3, 3, p=(1, 1))
    assert supports_gfx950_conv_dgrad_aot_problem(big) == (True, "ok")
    assert not supports_gfx950_conv_dgrad_aot_problem(
        _p(1, 8192, 8192, 32, 32, 3, 3, p=(1, 1))
    )[0]
