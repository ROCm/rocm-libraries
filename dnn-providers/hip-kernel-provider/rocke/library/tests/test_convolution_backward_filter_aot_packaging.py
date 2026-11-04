# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests for the shape-generic AOT backward-weight instances
(hipkernel:Gfx950ConvWgradAot).

An instance is packaged from a rocKE wgrad ``KernelIdentity``; these tests hold
the adapter (``builders/common/convolution_backward_filter.py``) to the catalog
it comes from (the dispatcher's registered gfx950 wgrad instances --
``dispatch/grouped_convolution_instances/gfx950_wgrad.json``):

* every catalog instance converts to a valid ``Gfx950ConvWgradAotSpec`` that
  ``supports_gfx950_conv_wgrad_aot`` accepts;
* the launch plan (``gfx950_conv_wgrad_aot_plan``) serves exactly what Phase 0's
  dispatcher (``dispatch.grouped_convolution``) admits and plans, for a
  representative shape per family, including the resolved split-K degree for
  implicit-GEMM instances (cross-checked directly against
  ``dispatch.grouped_convolution._wgrad_split_k`` / ``resolve_conv_split_k``);
* a two-stage implicit-GEMM instance is a valid spec but is refused at plan
  time with a clear reason, never silently mis-planned;
* ``serialize()``/``digest()`` round-trip and are stable.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from dispatch.grouped_convolution import ConvGroupedRequest, registered_conv_grouped_instances
from kernels.common.conv_split_k import resolve_conv_split_k
from kernels.conv.kernel_identity import KernelIdentity

from builders.common.convolution_backward_filter import (
    AOT_WGRAD_DIRECT_ALGORITHMS,
    PACKAGED_LLVM_FLAVOR,
    Gfx950ConvWgradAotSpec,
    Gfx950ConvWgradProblem,
    aot_wgrad_direct_instance_fields,
    gfx950_conv_wgrad_aot_plan,
    gfx950_conv_wgrad_aot_problem_for_request,
    gfx950_conv_wgrad_aot_serves,
    gfx950_conv_wgrad_aot_split_k,
    supports_gfx950_conv_wgrad_aot,
    supports_gfx950_conv_wgrad_aot_problem,
)
import dispatch.grouped_convolution as gc

_ARCH = "gfx950"

# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

_IMPLICIT_CARRIED = (
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
    "group_merge",
    "ws_replicas",
    "split_k",
    "lds_k_pad",
    "lds_layout",
)
_IMPLICIT_FLAGS = (
    "lds_k_outer",
    "grouped",
    "prefetch_addr",
    "two_stage",
    "async_dma",
    "unroll_k",
)


def _spec(identity: KernelIdentity) -> Gfx950ConvWgradAotSpec:
    """The packaged instance of a wgrad catalog identity (what the IG
    converter makes), mirroring the dgrad packaging test's ``_spec``."""
    if identity.is_direct:
        return Gfx950ConvWgradAotSpec(
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
            grouped=0,
            instance_id=identity.stable_hash(),
            **aot_wgrad_direct_instance_fields(identity),
        )
    return Gfx950ConvWgradAotSpec(
        dtype=identity.dtype_a,
        instance_id=identity.stable_hash(),
        **{name: getattr(identity, name) for name in _IMPLICIT_CARRIED},
        **{name: int(getattr(identity, name)) for name in _IMPLICIT_FLAGS},
    )


@pytest.fixture(scope="module")
def catalog():
    """``[(identity, spec)]`` of the shipped wgrad catalog."""
    identities = [
        spec.identity
        for spec in registered_conv_grouped_instances(_ARCH, direction="wgrad")
    ]
    return [(ident, _spec(ident)) for ident in identities]


def test_wgrad_catalog_has_446_instances_across_all_8_families(catalog):
    assert len(catalog) == 446
    algorithms = {ident.algorithm for ident, _ in catalog}
    assert algorithms == {"implicit_gemm"} | set(AOT_WGRAD_DIRECT_ALGORITHMS) - {
        "direct_wgrad_fold"
    }


def test_the_catalog_converts_and_reproduces_its_identities(catalog):
    for ident, spec in catalog:
        assert supports_gfx950_conv_wgrad_aot(spec) == (True, "ok"), (
            ident.label(),
            supports_gfx950_conv_wgrad_aot(spec),
        )
        assert ident.llvm_flavor == PACKAGED_LLVM_FLAVOR, ident.label()
        identity = spec.to_identity()
        if spec.is_direct:
            assert replace(identity, knobs="") == replace(ident, knobs=""), ident.label()
        else:
            assert identity.stable_hash() == spec.instance_id, ident.label()
            assert identity == ident, ident.label()


def test_two_stage_instances_are_valid_specs(catalog):
    """A two-stage implicit-GEMM instance is exactly as packageable as any
    other -- it is only the *plan* that refuses it (see
    ``test_two_stage_instances_are_refused_at_plan_time``)."""
    two_stage = [spec for _, spec in catalog if spec.two_stage]
    assert len(two_stage) == 220
    for spec in two_stage:
        assert not spec.is_direct
        ok, why = supports_gfx950_conv_wgrad_aot(spec)
        assert ok, why


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------


def _p(N, Hi, Wi, C, K, Y, X, s=(1, 1), p=(0, 0), groups=1, dtype="fp16"):
    return Gfx950ConvWgradProblem(
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
        groups=groups,
        dtype=dtype,
    )


#: Mirrors Phase 0's ``_WGRAD_SHAPES`` (``tests/dispatch/test_grouped_conv_dispatch.py``):
#: a shape reaching each of the 8 registered families.
_SHAPES = {
    "dense3x3": _p(2, 28, 28, 64, 64, 3, 3, p=(1, 1)),
    "dense_stride2": _p(2, 28, 28, 64, 128, 3, 3, s=(2, 2), p=(1, 1)),
    "grouped_cpg4": _p(2, 28, 28, 512, 512, 3, 3, p=(1, 1), groups=128),
    "depthwise3x3": _p(4, 56, 56, 64, 64, 3, 3, p=(1, 1), groups=64),
    "stem_c3_11x11": _p(4, 64, 64, 3, 3, 11, 11, p=(5, 5), groups=3),
    "gather_c24": _p(4, 56, 56, 24, 128, 3, 3, p=(1, 1)),
    "grouped_mfma_cpg4": _p(4, 56, 56, 16, 16, 3, 3, p=(1, 1), groups=4),
}
_SHAPES_BF16 = {f"{name}_bf16": replace(p, dtype="bf16") for name, p in _SHAPES.items()}


def _req(problem: Gfx950ConvWgradProblem) -> ConvGroupedRequest:
    return ConvGroupedRequest(
        N=problem.N,
        C=problem.C,
        K=problem.K,
        Hi=problem.Hi,
        Wi=problem.Wi,
        Y=problem.Y,
        X=problem.X,
        arch=_ARCH,
        G=problem.groups,
        stride_h=problem.sH,
        stride_w=problem.sW,
        pad_h=problem.pH,
        pad_w=problem.pW,
        dtype=problem.dtype,
        direction="wgrad",
    )


@pytest.mark.parametrize("shape", sorted({**_SHAPES, **_SHAPES_BF16}))
def test_the_plan_serves_exactly_rockes_offer(catalog, shape):
    """Phase 0's own offer (``_admits``) and plan (``launch_plan_for``) are the
    ground truth: a two-stage instance is refused here even where Phase 0
    would admit it (``_admits`` refuses two-stage before reaching the shape
    checks at all, so this is never exercised the other way), everything
    else matches exactly."""
    problem = {**_SHAPES, **_SHAPES_BF16}[shape]
    req = _req(problem)
    served_families = set()
    for ident, spec in catalog:
        if ident.dtype_a != problem.dtype:
            continue
        admits, why_admits = gc._admits(ident, req)
        plan, why_plan = gfx950_conv_wgrad_aot_plan(spec, problem)
        if spec.two_stage:
            assert plan is None, ident.label()
            continue
        assert (plan is not None) == admits, (ident.label(), why_admits, why_plan)
        if plan is not None:
            served_families.add(ident.algorithm)
    # Not every family necessarily reaches every shape; the parametrized set
    # together must still reach all 8.


def test_every_family_is_reached_by_the_parametrized_shapes(catalog):
    served = set()
    for name, problem in {**_SHAPES, **_SHAPES_BF16}.items():
        for ident, spec in catalog:
            if ident.dtype_a != problem.dtype or spec.two_stage:
                continue
            plan, _ = gfx950_conv_wgrad_aot_plan(spec, problem)
            if plan is not None:
                served.add(ident.algorithm)
    assert served == {
        "implicit_gemm",
        "direct_wgrad",
        "direct_wgrad_tiled",
        "direct_wgrad_dw",
        "direct_wgrad_dw_cols",
        "direct_wgrad_dw_mfma",
        "direct_wgrad_gather",
        "direct_wgrad_grouped_mfma",
    }


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_implicit_gemm_split_k_matches_resolve_conv_split_k(catalog, shape):
    """The split-K degree the plan resolves is exactly
    ``kernels.common.conv_split_k.resolve_conv_split_k`` called with Phase 0's
    own arguments (``dispatch.grouped_convolution._wgrad_split_k``), cross-checked
    both directly and through the dispatcher's own ``launch_plan_for``.

    Not every shape is reached by an implicit-GEMM instance (``group_merge``
    is depthwise-only, and the catalog's merged rows only reach a handful of
    group counts), so this checks "every admitted instance agrees", not
    "every shape is admitted" -- the latter is
    ``test_every_family_is_reached_by_the_parametrized_shapes``.
    """
    problem = _SHAPES[shape]
    req = _req(problem)
    checked = 0
    for ident, spec in catalog:
        if ident.algorithm != "implicit_gemm" or spec.two_stage:
            continue
        if ident.dtype_a != problem.dtype:
            continue
        admits, _ = gc._admits(ident, req)
        if not admits:
            continue
        if ident.split_k == 1:
            # A binary baked split_k == 1 was built with no atomic-accumulate
            # epilogue at all (WgradConvSpec's _needs_atomic), so it is held
            # to degree 1 regardless of what the resolver would otherwise
            # pick for this shape -- mirrors
            # dispatch.grouped_convolution._wgrad_split_k's same gate.
            want = 1
        else:
            want = resolve_conv_split_k(
                "wgrad",
                problem.to_problem(),
                arch=ident.arch,
                dtype=ident.dtype_d,
                tile_m=ident.tile_m,
                tile_n=ident.tile_n,
                tile_k=ident.tile_k,
                block_size=ident.warp_m * ident.warp_n * ident.wave_size,
                group_merge=ident.group_merge,
                ws_replicas=ident.ws_replicas,
                reduction="atomic",
                epilogue=ident.epilogue,
                allow_unsplit=True,
            ).split_k
        got = gfx950_conv_wgrad_aot_split_k(spec, problem)
        assert got == want, ident.label()
        plan, why = gfx950_conv_wgrad_aot_plan(spec, problem)
        assert plan is not None, (ident.label(), why)
        assert plan.split_k == want
        theirs_plan = gc.launch_plan_for(req, next(s for s in registered_conv_grouped_instances(_ARCH, direction="wgrad") if s.identity == ident))
        assert theirs_plan.split_k == want
        checked += 1
    # dense3x3 / dense_stride2 reach implicit-GEMM instances; the grouped and
    # depthwise shapes above only reach direct families (see the docstring).
    if shape in ("dense3x3", "dense_stride2"):
        assert checked > 0


def test_direct_plans_match_the_dispatchers_grid_block_and_buffers(catalog):
    """The direct-family plan (grid, block, buffers) matches
    ``dispatch.grouped_convolution.launch_plan_for`` exactly, for every shape
    and dtype reaching a direct instance."""
    checked = 0
    for name, problem in {**_SHAPES, **_SHAPES_BF16}.items():
        req = _req(problem)
        for inst_spec in registered_conv_grouped_instances(_ARCH, direction="wgrad"):
            ident = inst_spec.identity
            if not ident.is_direct or ident.dtype_a != problem.dtype:
                continue
            admits, _ = gc._admits(ident, req)
            spec = _spec(ident)
            plan, why = gfx950_conv_wgrad_aot_plan(spec, problem)
            if not admits:
                assert plan is None, (ident.label(), why)
                continue
            assert plan is not None, (ident.label(), why)
            theirs = gc.launch_plan_for(req, inst_spec)
            assert len(plan.steps) == len(theirs.steps), ident.label()
            for mine, step in zip(plan.steps, theirs.steps):
                assert mine.grid == step.grid, (ident.label(), mine.grid, step.grid)
                assert (mine.block, 1, 1) == step.block, ident.label()
                assert mine.buffers == step.buffers, ident.label()
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# Two-stage refusal
# ---------------------------------------------------------------------------


def test_two_stage_instances_are_refused_at_plan_time(catalog):
    """Every two-stage instance is refused with a clear reason, for every
    representative shape -- never silently mis-planned."""
    refused = 0
    for name, problem in {**_SHAPES, **_SHAPES_BF16}.items():
        for ident, spec in catalog:
            if not spec.two_stage or ident.dtype_a != problem.dtype:
                continue
            plan, why = gfx950_conv_wgrad_aot_plan(spec, problem)
            assert plan is None
            assert "two-stage" in why
            ok, why2 = gfx950_conv_wgrad_aot_serves(spec, problem)
            assert not ok
            assert why2 == why
            refused += 1
    assert refused > 0


# ---------------------------------------------------------------------------
# serialize() / digest()
# ---------------------------------------------------------------------------


def test_the_digest_is_fnv1a_over_the_serialized_plan(catalog):
    plan = None
    for ident, spec in catalog:
        if spec.is_direct or spec.two_stage:
            continue
        plan, why = gfx950_conv_wgrad_aot_plan(spec, _SHAPES["dense3x3"])
        if plan is not None:
            break
    assert plan is not None
    values = plan.serialize()
    h = 0xCBF29CE484222325
    for value in values:
        for byte in (value & ((1 << 64) - 1)).to_bytes(8, "little"):
            h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert plan.digest() == h


def test_serialize_and_digest_round_trip_and_are_stable(catalog):
    """Same spec + problem -> same digest across two independent calls, for
    an implicit-GEMM instance, a plain atomic direct instance and a folded
    direct instance."""
    checked = set()
    for ident, spec in catalog:
        if spec.two_stage:
            continue
        for name, problem in _SHAPES.items():
            if ident.dtype_a != problem.dtype:
                continue
            plan_a, why = gfx950_conv_wgrad_aot_plan(spec, problem)
            if plan_a is None:
                continue
            plan_b, _ = gfx950_conv_wgrad_aot_plan(spec, problem)
            assert plan_a.serialize() == plan_b.serialize(), ident.label()
            assert plan_a.digest() == plan_b.digest(), ident.label()
            checked.add(spec.is_direct)
        if len(checked) == 2:
            break
    assert checked == {True, False}


# ---------------------------------------------------------------------------
# Problem contract
# ---------------------------------------------------------------------------


def test_problem_for_request_is_the_wgrad_graph_contract():
    request = ConvGroupedRequest(
        N=2,
        C=64,
        K=32,
        Hi=28,
        Wi=28,
        Y=3,
        X=3,
        arch=_ARCH,
        G=2,
        pad_h=1,
        pad_w=1,
        direction="wgrad",
    )
    problem = gfx950_conv_wgrad_aot_problem_for_request(request)
    assert (problem.Hi, problem.Ho, problem.cpg, problem.kpg) == (28, 28, 32, 16)
    assert supports_gfx950_conv_wgrad_aot_problem(problem) == (True, "ok")
    with pytest.raises(ValueError, match="backward-weight"):
        gfx950_conv_wgrad_aot_problem_for_request(replace(request, direction="dgrad"))
    assert not supports_gfx950_conv_wgrad_aot_problem(replace(problem, groups=3))[0]
    assert not supports_gfx950_conv_wgrad_aot_problem(replace(problem, dtype="fp32"))[0]
