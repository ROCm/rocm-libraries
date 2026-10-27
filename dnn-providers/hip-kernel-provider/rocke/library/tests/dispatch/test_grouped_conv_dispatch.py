# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The grouped-convolution dispatcher registers its instance lists and nothing else.

``dispatch.grouped_convolution`` registers one candidate per record of
``dispatch/grouped_convolution_instances/*.json`` -- shape-generic AOT
binaries, named by their lossless labels, forward and dgrad. These tests pin
down that:

* each registry is exactly its lists, in list order, and every instance has
  the AOT build job ``--compile-all`` makes its binary from;
* the list files are canonical and their loader refuses a broken one;
* nothing outside the lists (another arch, another direction) dispatches;
* the dispatcher admits a shape exactly when the AOT cache's run path would
  launch that binary on it, with the same kernels, grids, blocks, signatures,
  buffers, kernarg values and scratch -- the binaries were built and verified
  through that path;
* pins and the priority order behave.

CPU only: nothing is compiled or launched.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_sweep as ks
from benchmarks.common.grouped_conv import instance_list
from dispatch import grouped_convolution as gc
from kernels.conv.direct_variants import DIRECT_WAVE_PARAMETRIC
from kernels.conv.kernel_identity import KernelIdentity
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.dispatch.core import PinRefused

_ARCH = "gfx950"
_INSTANCE_DIR = Path(gc.__file__).resolve().with_name("grouped_convolution_instances")
_LIST = _INSTANCE_DIR / "gfx950_fwd.json"
_LISTS = {key: _INSTANCE_DIR / name for key, name in gc._INSTANCE_LISTS.items()}


def _listed_names(path: Path = _LIST) -> list[str]:
    return [i.label() for i in gc.load_instance_list(path).identities]


@pytest.fixture(autouse=True)
def _instance_flavor(monkeypatch):
    """Lower the flavor the instances were named for.

    Dispatch does not need it (a request carries its flavor); the AOT cache
    and job builders the tests compare against check the process's own.
    """
    flavors = {
        s.identity.llvm_flavor
        for direction in ("fwd", "dgrad")
        for s in gc.registered_conv_grouped_instances(direction=direction)
    }
    assert len(flavors) == 1
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavors.pop())


def _req(**kw) -> gc.ConvGroupedRequest:
    base = dict(N=2, C=64, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1, arch=_ARCH)
    base.update(kw)
    return gc.ConvGroupedRequest(**base)


# Shapes that reach every registered forward family: dense (implicit GEMM and
# direct_nongrouped), grouped cpg=16 and cpg=4 (direct_grouped), depthwise
# 3x3 / 7x7 / 11x11 / stride 2 (the depthwise variants), the dilated
# non-grouped instances, odd channels (scalar-vector implicit GEMM) and bf16.
_SHAPES = {
    "dense3x3": dict(),
    "dense1x1_bf16": dict(Y=1, X=1, pad_h=0, pad_w=0, C=256, K=128, dtype="bf16"),
    "dense_stride2": dict(stride_h=2, stride_w=2, K=128),
    "grouped_cpg16": dict(C=1024, K=1024, G=64, dtype="bf16"),
    "grouped_cpg4": dict(C=512, K=512, G=128),
    "depthwise3x3": dict(C=144, K=144, G=144),
    "depthwise7x7": dict(C=96, K=96, G=96, Y=7, X=7, pad_h=3, pad_w=3),
    "depthwise11x11": dict(C=256, K=256, G=256, Y=11, X=11, pad_h=5, pad_w=5),
    # The spatial depthwise kernel holds every channel in one wavefront.
    "depthwise11x11_g32": dict(C=32, K=32, G=32, Y=11, X=11, pad_h=5, pad_w=5),
    "depthwise_s2": dict(C=144, K=144, G=144, stride_h=2, stride_w=2),
    "dilated_h3": dict(Hi=32, Wi=32, C=128, K=128, dilation_h=3),
    "stem_c3": dict(
        C=3, K=64, Y=7, X=7, Hi=64, Wi=64, stride_h=2, stride_w=2, pad_h=3, pad_w=3
    ),
}

# Shapes that reach every registered dgrad family: implicit GEMM with one
# tilde sub-GEMM and with several (stride 2, dilation 2), the depthwise dgrad
# filters and strides, and every channel pair the MFMA dgrad instances bake.
# The 2x2/s2 downsample and the 16x16/s16 stem run width-folded (fold_width).
_DGRAD_SHAPES = {
    "dense3x3": dict(),
    "dense_stride2": dict(stride_h=2, stride_w=2, K=128),
    "dense2x2_stride2_folds": dict(Y=2, X=2, pad_h=0, pad_w=0, stride_h=2, stride_w=2),
    "patch16_stem_folds_bf16": dict(
        N=2,
        C=3,
        K=768,
        Hi=224,
        Wi=224,
        Y=16,
        X=16,
        pad_h=0,
        pad_w=0,
        stride_h=16,
        stride_w=16,
        dtype="bf16",
    ),
    "dense_dilated2_bf16": dict(
        Hi=32, Wi=32, dilation_h=2, dilation_w=2, pad_h=2, pad_w=2, dtype="bf16"
    ),
    "grouped_cpg4": dict(C=512, K=512, G=128),
    "grouped_cpg8": dict(C=512, K=512, G=64),
    "grouped_cpg16_bf16": dict(C=1024, K=1024, G=64, dtype="bf16"),
    "grouped_cpg24": dict(C=384, K=384, G=16),
    "grouped_cpg48_kpg192": dict(C=384, K=1536, G=8),
    "grouped_cpg4_kpg320": dict(C=64, K=5120, G=16),
    "depthwise3x3": dict(C=144, K=144, G=144),
    "depthwise3x3_s2": dict(C=144, K=144, G=144, stride_h=2, stride_w=2),
    "depthwise7x7": dict(C=96, K=96, G=96, Y=7, X=7, pad_h=3, pad_w=3),
}

_CASES = [
    pytest.param(arch, direction, shape, id=f"{arch}-{direction}-{shape}")
    for arch, direction in sorted(gc._INSTANCE_LISTS)
    for shape in sorted(_SHAPES if direction == "fwd" else _DGRAD_SHAPES)
]


def _case_req(arch: str, direction: str, shape: str) -> gc.ConvGroupedRequest:
    shapes = _SHAPES if direction == "fwd" else _DGRAD_SHAPES
    return _req(arch=arch, direction=direction, **shapes[shape])


# ---------------------------------------------------------------------------
# What is registered
# ---------------------------------------------------------------------------


def test_registry_is_exactly_the_instance_lists_in_order():
    for (arch, direction), path in _LISTS.items():
        names = _listed_names(path)
        got = gc.registered_conv_grouped_instances(arch, direction=direction)
        assert [s.name for s in got] == names
    for direction in ("fwd", "dgrad"):
        for candidate in gc.conv_grouped_candidates(direction):
            assert candidate.spec_id == candidate.name
            assert len(candidate.capability.arches) == 1


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_every_instance_is_a_canonical_name(direction):
    arches = {arch for arch, d in gc._INSTANCE_LISTS if d == direction}
    for spec in gc.registered_conv_grouped_instances(direction=direction):
        assert KernelIdentity.from_label(spec.name) == spec.identity
        assert spec.identity.label() == spec.name
        assert spec.identity.arch in arches
        assert spec.direction == direction
        assert spec.identity.direction == (
            f"direct_{direction}" if spec.is_direct else direction
        )
        assert spec.algorithm == spec.identity.algorithm


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_every_instance_has_the_aot_build_job_compile_all_uses(direction):
    """``--compile-all`` builds each instance (and the helpers it launches) from
    its family's job; a name the generators cannot reproduce would be an
    instance with no binary."""
    for spec in gc.registered_conv_grouped_instances(direction=direction):
        for ident in (*spec.helpers, spec.identity):
            if ident.is_direct:
                job = dks.direct_job_for_identity(ident)
            else:
                job = ks.job_for_identity(ident)
            assert job.identity == ident


def test_execution_registries_only_hold_buildable_candidates():
    """The sweep registries refuse a candidate without a ``build``, as
    attention's do, and every registered instance has one."""
    for registry in (gc.CONV_FWD_EXECUTION_REGISTRY, gc.CONV_DGRAD_EXECUTION_REGISTRY):
        assert registry.require_build
        assert registry.candidates()
        assert all(c.build is not None for c in registry.candidates())


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_every_swept_family_builds_the_kernel_compile_all_builds(direction):
    """``result.build()`` of a sweep result is the IR of the cached binary: the
    kernel ``--compile-all`` builds from the identity's job. One result per
    family the shapes reach."""
    shapes = _SHAPES if direction == "fwd" else _DGRAD_SHAPES
    picked = {}
    for kw in shapes.values():
        req = _req(direction=direction, **kw)
        for result in gc.iter_dispatch_conv_grouped_all(req):
            picked.setdefault(result.spec.algorithm, result)
    assert len(picked) > 1
    for result in picked.values():
        ident = result.spec.identity
        if ident.is_direct:
            want = dks.direct_kernel_for_job(
                dks.direct_job_for_identity(ident), _ARCH, ident.dtype_a
            )[0]
        else:
            want = ks.kernel_for_job(ks.job_for_identity(ident), _ARCH, ident.dtype_a)[
                0
            ]
        assert result.build().name == want.name, result.spec.name


def test_an_instance_builds_only_for_its_own_arch():
    result = gc.dispatch_conv_grouped(_req())
    with pytest.raises(ValueError, match="gfx950 instance, not gfx942"):
        result.candidate.built(result.spec, "gfx942")


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(arch="gfx90a"), "arch 'gfx90a'"),
        (dict(arch="gfx942"), "arch 'gfx942'"),
        (dict(arch="gfx1250"), "arch 'gfx1250'"),
        (dict(direction="wgrad"), "no 'wgrad' instances"),
        (dict(dtype="fp32"), "fp16 or bf16 only"),
        (dict(C=63, G=2), "divisible by G"),
        (dict(Di=8, Z=3, stride_d=1, pad_d=1, dilation_d=1), "3-D capability"),
        (
            dict(direction="dgrad", Di=8, Z=3, stride_d=1, pad_d=1, dilation_d=1),
            "3-D capability",
        ),
    ],
)
def test_nothing_outside_the_lists_dispatches(kw, match):
    fields = dict(N=2, C=64, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1, G=1)
    fields.update(kw)
    fields.setdefault("arch", _ARCH)
    req = gc.ConvGroupedRequest(**fields)
    with pytest.raises(ValueError, match=match):
        gc.dispatch_conv_grouped(req)
    assert not gc.dispatch_conv_grouped_all(req)


def test_only_wgrad_has_no_candidates():
    assert gc.conv_grouped_candidates("wgrad") == ()
    assert gc.registered_conv_grouped_instances(direction="wgrad") == ()
    assert gc.conv_grouped_candidates("dgrad")


def test_dispatch_stays_in_the_requested_direction():
    for direction in ("fwd", "dgrad"):
        results = gc.dispatch_conv_grouped_all(_req(direction=direction))
        assert results
        assert {r.spec.direction for r in results} == {direction}
        assert {r.kernel_id.op for r in results} == {f"conv_{direction}"}


def test_another_llvm_flavor_admits_nothing():
    """Pin a real, differing flavor via the request -- not the environment.

    An unrecognized env value (e.g. "llvm-other") now raises immediately
    (see test_unrecognized_llvm_flavor_env_raises), so it can't be used here
    to force a mismatch. Picking a real flavor other than the instances' own
    guarantees the mismatch deterministically.
    """
    instance_flavor = next(
        iter(
            {
                s.identity.llvm_flavor
                for s in gc.registered_conv_grouped_instances(direction="fwd")
            }
        )
    )
    other_flavor = next(f for f in LLVM_FLAVORS if f != instance_flavor)
    with pytest.raises(ValueError, match="LLVM flavor"):
        gc.dispatch_conv_grouped(_req(llvm_flavor=other_flavor))


def test_default_llvm_flavor_is_the_registered_one():
    flavors = {
        s.identity.llvm_flavor
        for direction in ("fwd", "dgrad")
        for s in gc.registered_conv_grouped_instances(direction=direction)
    }
    assert flavors == {gc.DEFAULT_LLVM_FLAVOR}
    assert _req().llvm_flavor == gc.DEFAULT_LLVM_FLAVOR


def test_dispatch_does_not_read_the_process_flavor(monkeypatch):
    """Selection is a pure function of the request: ROCKE_LLVM_FLAVOR naming
    another flavor (or COMGR detecting one) changes nothing."""
    picks = {}
    for flavor in LLVM_FLAVORS:
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        picks[flavor] = [
            r.spec.name for r in gc.dispatch_conv_grouped_all(_req(direction="dgrad"))
        ] + [r.spec.name for r in gc.dispatch_conv_grouped_all(_req())]
    assert len({tuple(v) for v in picks.values()}) == 1
    assert next(iter(picks.values()))


def test_unknown_llvm_flavor_request_is_rejected():
    with pytest.raises(ValueError, match="unknown llvm_flavor"):
        gc.dispatch_conv_grouped(_req(llvm_flavor="llvm-other"))


def test_unrecognized_llvm_flavor_env_raises(monkeypatch):
    """A typo'd ROCKE_LLVM_FLAVOR must fail loudly, not fall back to auto-detect."""
    from rocke.core.lower_llvm import _detect_llvm_flavor

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm-other")
    with pytest.raises(ValueError, match="llvm-other"):
        _detect_llvm_flavor()


# ---------------------------------------------------------------------------
# Admission and the launch contract agree with the AOT cache's run path
# ---------------------------------------------------------------------------


def _implicit_problem(req):
    from kernels.common.conv_implicit_gemm import ConvProblem

    return ConvProblem(
        N=req.N,
        Hi=req.Hi,
        Wi=req.Wi,
        C=req.C,
        K=req.K,
        Y=req.Y,
        X=req.X,
        sH=req.stride_h,
        sW=req.stride_w,
        pH=req.pad_h,
        pW=req.pad_w,
        dH=req.dilation_h,
        dW=req.dilation_w,
        groups=req.G,
    )


def _direct_problem(req):
    from kernels.common.conv_direct_grouped import DirectConvProblem

    return DirectConvProblem(
        N=req.N,
        H=req.Hi,
        W=req.Wi,
        groups=req.G,
        cpg=req.C // req.G,
        kpg=req.K // req.G,
        KH=req.Y,
        KW=req.X,
        PAD=req.pad_h,
        stride=req.stride_h,
        dtype=req.dtype,
        dil_h=req.dilation_h,
        dil_w=req.dilation_w,
    )


class _HelperCache:
    """Stands in for a kernel cache holding every MFMA dgrad helper; records
    which helper identities ``plan_for`` looks up."""

    def __init__(self):
        self.looked_up: list[KernelIdentity] = []

    def get(self, identity):
        self.looked_up.append(identity)
        return b"", {"kernel_name": "helper"}

    def hsaco_path(self, identity):
        return Path("unused.hsaco")


def _direct_plan(spec, req):
    cache = _HelperCache()
    # The meta a fresh --compile-all entry carries: a wave-parametric build
    # records the wave size it was built at, the identity's.
    meta = {"kernel_name": "k"}
    if spec.algorithm in DIRECT_WAVE_PARAMETRIC:
        meta["spec_wave_size"] = spec.identity.wave_size
    plan, why = dks.plan_for(
        cache,
        spec.identity,
        Path("unused.hsaco"),
        meta,
        _direct_problem(req),
        req.arch,
    )
    return plan, why, cache.looked_up


def _cache_side_admits(spec, req) -> bool:
    """Would the AOT run path launch ``spec``'s binary on ``req``?

    Built the way the implicit-GEMM and direct benchmarks build their
    problems, and asked through ``KernelCache.supports_problem`` and
    ``direct_kernel_sweep.plan_for``.
    """
    from benchmarks.common.kernel_cache import KernelCache

    ident = spec.identity
    if ident.dtype_a != req.dtype:
        return False
    if not ident.is_direct:
        from benchmarks.common.benchmark_implicit_gemm_conv import _launch_problem

        problem = _launch_problem(spec.direction, _implicit_problem(req))
        return KernelCache.supports_problem(ident, problem)[0]
    if req.stride_h != req.stride_w or req.pad_h != req.pad_w:
        return False
    if ident.algorithm == "direct_nongrouped" and req.G != 1:
        return False
    if not KernelCache.supports_problem(ident, _direct_problem(req))[0]:
        return False
    return _direct_plan(spec, req)[0] is not None


@pytest.mark.parametrize("arch, direction, shape", _CASES)
def test_admission_matches_the_aot_run_path(arch, direction, shape):
    req = _case_req(arch, direction, shape)
    admitted = [r.spec.name for r in gc.dispatch_conv_grouped_all(req)]
    expected = [
        s.name
        for s in gc.registered_conv_grouped_instances(arch, direction=direction)
        if _cache_side_admits(s, req)
    ]
    assert admitted == expected
    assert admitted, f"{arch} {shape}: no registered instance runs it"


@pytest.mark.parametrize("direction", ["fwd", "dgrad"])
def test_the_shapes_reach_every_registered_family(direction):
    shapes = _SHAPES if direction == "fwd" else _DGRAD_SHAPES
    families = {
        s.algorithm
        for s in gc.registered_conv_grouped_instances(_ARCH, direction=direction)
    }
    reached = {
        r.spec.algorithm
        for kw in shapes.values()
        for r in gc.dispatch_conv_grouped_all(_req(direction=direction, **kw))
    }
    assert reached == families


class _UploadRecorder:
    """Stands in for the HIP runtime: records the one table a dgrad run
    uploads."""

    PTR = 0x9000

    def __init__(self):
        self.table: tuple[int, ...] = ()

    def alloc(self, nbytes):
        return self.PTR

    def memcpy_h2d(self, dst, src, nbytes):
        import numpy as np

        raw = ctypes.string_at(src, nbytes)
        self.table = tuple(int(v) for v in np.frombuffer(raw, dtype=np.int32))


def _cache_side_plan(spec, req, ptrs, sizes):
    """``(steps, workspaces, table)`` the AOT run path launches, one step being
    ``(identity, grid, block, signature names, buffers, values)``."""
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _extras_and_grid,
        _launch_problem,
        _signature_for_identity,
    )

    ident = spec.identity
    direction = spec.direction
    if not ident.is_direct:
        problem = _launch_problem(direction, _implicit_problem(req))
        rt = _UploadRecorder()
        extras, grid = _extras_and_grid(direction, problem, ident, rt, ident.dtype_a)
        buffers = gc._OPERANDS[direction]
        values = ks._launch_values_for(
            direction,
            problem,
            ident,
            tuple(ptrs[b] for b in buffers),
            tuple(sizes[b] for b in buffers),
            extras,
        )
        step = (
            ident,
            tuple(grid),
            (ident.warp_m * ident.warp_n * ident.wave_size, 1, 1),
            [a["name"] for a in _signature_for_identity(ident, ident.dtype_a)],
            buffers,
            values,
        )
        return [step], {}, rt.table
    plan, why, helpers = _direct_plan(spec, req)
    assert plan is not None, why
    steps = [
        (
            identity,
            tuple(step.grid),
            tuple(step.block),
            [a["name"] for a in step.signature],
            step.buffers,
            dks.launch_values(step, ptrs, sizes),
        )
        for identity, step in zip((*helpers, ident), plan.steps, strict=True)
    ]
    return steps, dict(plan.workspaces), ()


_PTRS = {
    name: 0x1000 * (i + 1)
    for i, name in enumerate(("x", "w", "y", "dy", "dx", "ws_t", "ws_coa"))
}
_PTRS["sub_gemms"] = _UploadRecorder.PTR
_SIZES = {name: 100 + i for i, name in enumerate(_PTRS)}


@pytest.mark.parametrize("arch, direction, shape", _CASES)
def test_launch_plan_matches_the_aot_run_path(arch, direction, shape):
    for result in gc.dispatch_conv_grouped_all(_case_req(arch, direction, shape)):
        name = result.spec.name
        plan = gc.launch_plan_for(result.request, result.spec)
        got = [
            (
                s.identity,
                s.grid,
                s.block,
                [a["name"] for a in s.signature],
                s.buffers,
                plan.values(s, _PTRS, _SIZES),
            )
            for s in plan.steps
        ]
        want, workspaces, table = _cache_side_plan(
            result.spec, result.request, _PTRS, _SIZES
        )
        assert got == want, name
        assert dict(plan.workspaces) == workspaces, name
        assert plan.sub_gemm_table == table, name
        # The dispatch result describes the instance's own kernel: the last step.
        own = plan.steps[-1]
        assert own.identity == result.spec.identity, name
        assert (result.grid, result.block) == (own.grid, own.block), name
        assert result.signature == own.signature, name
        for s in plan.steps:
            assert set(plan.values(s, _PTRS, _SIZES)) == {
                a["name"] for a in s.signature
            }, name


# The runtime-channel MFMA dgrad (direct_dgrad_mfma_rtc) runs its own pair of
# weight transforms, which take the per-group channel counts as kernargs. No
# list registers one yet, so the parametrized test above never meets it.
_RTC_DGRAD = (
    "dgrad-direct_dgrad_mfma_rtc-fp16-2d-f3x3-p1x1-s1x1-d1x1-c0k0-block_groups=1-"
    "block_h=8-block_q=16-fold_k32=true-k_atoms=10-m_tiles=1-persistent_grid=false-"
    "runtime_k_loop=false-waves_k=2-waves_q=1-wave64-gfx950-llvm22"
)


def test_launch_plan_of_the_runtime_channel_mfma_dgrad():
    identity = KernelIdentity.from_label(_RTC_DGRAD)
    spec = gc.ConvInstanceSpec(
        name=_RTC_DGRAD,
        identity=identity,
        block=gc._instance_block(identity),
        helpers=gc._instance_helpers(identity),
    )
    req = _case_req("gfx950", "dgrad", "grouped_cpg4_kpg320")
    ok, why = gc._admits(identity, req)
    assert ok, why
    assert [h.algorithm for h in spec.helpers] == [
        "direct_dgrad_transpose_rtc",
        "direct_dgrad_reorganize_rtc",
    ]
    plan = gc.launch_plan_for(req, spec)
    got = [
        (
            s.identity,
            s.grid,
            s.block,
            [a["name"] for a in s.signature],
            s.buffers,
            plan.values(s, _PTRS, _SIZES),
        )
        for s in plan.steps
    ]
    want, workspaces, _ = _cache_side_plan(spec, req, _PTRS, _SIZES)
    assert got == want
    assert dict(plan.workspaces) == workspaces


@pytest.mark.parametrize(
    "shape", [s for s in _DGRAD_SHAPES if s.endswith(("_folds", "_folds_bf16"))]
)
def test_the_folding_shapes_fold(shape):
    from kernels.common.conv_implicit_gemm_dgrad import fold_width

    p = _implicit_problem(_case_req(_ARCH, "dgrad", shape))
    assert fold_width(p) != p


# The patch-embedding dgrad (DgradConvSpec.patch) runs one record over the
# filter's (y, x, c). No list registers one yet, so the parametrized tests
# never meet it; the AOT run path must hand its binary that record too, not
# the Y*X tilde records of C columns each.
_PATCH_DGRAD = (
    "dgrad-implicit_gemm-bf16-2d-t128x256x64-w2x4-a16x16x32-v8x8x8-mem-"
    "cshuffle-kouter-ukd-patch-msg1024-wave64-gfx950-llvm22"
)


def test_launch_plan_of_the_patch_dgrad():
    from kernels.common.conv_implicit_gemm_dgrad import (
        fold_width,
        pack_sub_gemm_buffer,
        patch_sub_gemms,
    )

    identity = KernelIdentity.from_label(_PATCH_DGRAD)
    spec = gc.ConvInstanceSpec(
        name=_PATCH_DGRAD,
        identity=identity,
        block=gc._instance_block(identity),
    )
    req = _case_req("gfx950", "dgrad", "patch16_stem_folds_bf16")
    ok, why = gc._admits(identity, req)
    assert ok, why
    assert _cache_side_admits(spec, req)
    plan = gc.launch_plan_for(req, spec)
    (step,) = plan.steps
    got = (
        step.identity,
        step.grid,
        step.block,
        [a["name"] for a in step.signature],
        step.buffers,
        plan.values(step, _PTRS, _SIZES),
    )
    want, workspaces, table = _cache_side_plan(spec, req, _PTRS, _SIZES)
    assert [got] == want
    assert dict(plan.workspaces) == workspaces
    assert plan.sub_gemm_table == table
    # One record of all Y*X*C = 768 columns, on the folded problem.
    p = fold_width(_implicit_problem(req))
    (sg,) = patch_sub_gemms(p, identity.tile_m, identity.tile_n, identity.tile_k)
    assert sg.gemm_n == req.Y * req.X * req.C
    assert plan.num_sub_gemms == 1
    assert table == tuple(pack_sub_gemm_buffer([sg], identity.tile_m, identity.tile_n))


# ---------------------------------------------------------------------------
# Selection and pins
# ---------------------------------------------------------------------------


def test_default_pick_is_the_first_admitted_instance_in_list_order():
    order = {name: i for i, name in enumerate(_listed_names())}
    for kw in _SHAPES.values():
        req = _req(**kw)
        admitted = [r.spec.name for r in gc.dispatch_conv_grouped_all(req)]
        assert admitted == sorted(admitted, key=order.__getitem__)
        assert gc.dispatch_conv_grouped(req).spec.name == admitted[0]
        assert [s.name for s in gc.conv_grouped_sweep_space(req)] == admitted


@pytest.mark.parametrize(
    "shape",
    [s for s in _SHAPES if s.startswith("depthwise")],
)
def test_depthwise_shapes_default_to_the_direct_depthwise_kernel(shape):
    """A depthwise-shaped request must not fall back to a generic IGEMM tile
    just because it happens to sort earlier in the list -- direct_depthwise_tiled
    candidates are admitted only for genuinely depthwise shapes (cpg=kpg=1), so
    they are listed ahead of implicit_gemm so they win whenever both are admitted.
    """
    req = _req(**_SHAPES[shape])
    assert gc.dispatch_conv_grouped(req).spec.algorithm == "direct_depthwise_tiled"


def test_spec_id_pins_one_instance():
    req = _req(**_SHAPES["depthwise7x7"])
    admitted = [r.spec.name for r in gc.dispatch_conv_grouped_all(req)]
    last = admitted[-1]
    pinned = gc.dispatch_conv_grouped(_req(**_SHAPES["depthwise7x7"], spec_id=last))
    assert pinned.spec.name == last
    assert pinned.kernel_id.spec_id == last


def test_pin_to_an_instance_that_cannot_run_the_shape_is_refused():
    vec8 = next(
        s.name
        for s in gc.registered_conv_grouped_instances(_ARCH)
        if not s.is_direct and s.identity.vector_size_a == 8 and s.dtype == "fp16"
    )
    with pytest.raises(PinRefused, match="not divisible by vector_size_a"):
        gc.dispatch_conv_grouped(_req(C=3, K=64, spec_id=vec8))


def test_algorithm_pin_keeps_one_family():
    req = _req(**_SHAPES["depthwise7x7"], algorithm="direct_depthwise_tiled")
    results = gc.dispatch_conv_grouped_all(req)
    assert results
    assert {r.spec.algorithm for r in results} == {"direct_depthwise_tiled"}
    assert gc.dispatch_conv_grouped(req).spec.algorithm == "direct_depthwise_tiled"


# ---------------------------------------------------------------------------
# The list file: canonical, and its loader refuses a broken one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", sorted(_LISTS.values()), ids=lambda p: p.name)
def test_instance_list_file_is_canonical(path):
    """Fix with ``instance_list.py format``."""
    lst = gc.load_instance_list(path)
    assert path.read_text(encoding="utf-8") == gc.format_instance_list(lst)


def _implicit_record() -> dict:
    return {
        "algorithm": "implicit_gemm",
        "dtype": "fp16",
        "tile": [32, 32, 64],
        "warps": [2, 2],
        "warp_tile": [16, 16, 32],
        "vector": [8, 8, 8],
        "pipeline": "mem",
        "epilogue": "cshuffle",
        "unroll_k": True,
        "grouped": True,
    }


def _write_list(tmp_path, records, **header) -> Path:
    doc = {
        "arch": _ARCH,
        "direction": "fwd",
        "defaults": {"llvm_flavor": "llvm22", "wave_size": 64},
        "instances": records,
    }
    doc.update(header)
    path = tmp_path / "list.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_instance_record_spells_the_label(tmp_path):
    path = _write_list(tmp_path, [_implicit_record()])
    (identity,) = gc.load_instance_list(path).identities
    assert identity.label() == (
        "fwd-implicit_gemm-fp16-2d-t32x32x64-w2x2-a16x16x32-v8x8x8-mem-cshuffle"
        "-grp-unroll-wave64-gfx950-llvm22"
    )


def test_a_dgrad_list_gives_its_direct_records_the_direct_direction(tmp_path):
    direct = {
        "algorithm": "direct_depthwise_dgrad",
        "dtype": "fp16",
        "filter": [3, 3],
        "pad": [1, 1],
        "stride": [1, 1],
        "dilation": [1, 1],
        "cpg": 1,
        "kpg": 1,
        "knobs": {"block_w": 8, "block_waves": 1},
    }
    path = _write_list(tmp_path, [_implicit_record(), direct], direction="dgrad")
    implicit, depthwise = gc.load_instance_list(path).identities
    assert (implicit.direction, depthwise.direction) == ("dgrad", "direct_dgrad")


@pytest.mark.parametrize("path", sorted(_LISTS.values()), ids=lambda p: p.name)
def test_every_listed_instance_round_trips_through_its_record(path):
    lst = gc.load_instance_list(path)
    for identity in lst.identities:
        record = lst.record(identity)
        assert lst.identity(json.loads(json.dumps(record)), "rt") == identity


def _with(**kw) -> dict:
    record = _implicit_record()
    record.update(kw)
    return {k: v for k, v in record.items() if v is not None}


@pytest.mark.parametrize(
    "records, header, match",
    [
        ([_implicit_record()] * 2, {}, r"already instances\[0\]"),
        # Spelling a field at its baseline is the same kernel.
        ([_implicit_record(), _with(split_k=1)], {}, "already"),
        ([_with(arch="gfx942")], {}, "the list's"),
        ([_with(tile=[32, 32])], {}, "tile is a list of 3"),
        ([_with(tile=[32, 32, "64"])], {}, "tile_k must be int"),
        ([_with(unroll_k=1)], {}, "unroll_k must be bool"),
        ([_with(tlie=[1, 2, 3])], {}, "unknown key 'tlie'"),
        ([_with(pipeline=None)], {}, "lacks pipeline"),
        ([_with(algorithm=None)], {}, "no algorithm"),
        ([_implicit_record()], {"defaults": {}}, "lacks llvm_flavor, wave_size"),
        ([_implicit_record()], {"defaults": {"arch": _ARCH}}, "the list's"),
        ([_implicit_record()], {"instance": []}, "unknown keys"),
        ([_with(pipeline="wavelet")], {}, "wavelet"),
        # Another well-formed forward kernel is a valid list.
        ([_with(unroll_k=None)], {}, None),
    ],
)
def test_instance_list_loader_rejects_broken_lists(tmp_path, records, header, match):
    path = _write_list(tmp_path, records, **header)
    if match is None:
        assert len(gc.load_instance_list(path).identities) == 1
        return
    with pytest.raises(gc.InstanceListError, match=match):
        gc.load_instance_list(path)


def test_registered_list_must_declare_its_arch_and_direction(tmp_path, monkeypatch):
    _write_list(tmp_path, [_implicit_record()], arch="gfx942")
    monkeypatch.setattr(gc, "_INSTANCE_DIR", tmp_path)
    monkeypatch.setattr(gc, "_INSTANCE_LISTS", {(_ARCH, "fwd"): "list.json"})
    with pytest.raises(ValueError, match="registered as gfx950 fwd"):
        gc._read_instance_list(_ARCH, "fwd")


def test_instance_list_tool_adds_labels_and_rebuilds_lists(tmp_path, capsys):
    names = _listed_names()
    labels = tmp_path / "labels.txt"
    labels.write_text("# picked\n" + "\n".join(names[:3]) + "\n", encoding="utf-8")
    path = tmp_path / "new.json"
    instance_list.main(["new", str(path), "--from", str(labels)])
    instance_list.main(["add", str(path), names[3], "--at", "1"])
    with pytest.raises(SystemExit, match="already listed"):
        instance_list.main(["add", str(path), names[0]])
    capsys.readouterr()
    instance_list.main(["labels", str(path)])
    assert capsys.readouterr().out.split() == [names[0], names[3], *names[1:3]]
    instance_list.main(["format", "--check", str(path)])
    lst = gc.load_instance_list(path)
    assert lst.defaults == gc.load_instance_list(_LIST).defaults
