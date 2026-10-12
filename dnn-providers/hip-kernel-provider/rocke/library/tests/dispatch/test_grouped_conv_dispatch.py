# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The grouped-convolution dispatcher registers its instance list and nothing else.

``dispatch.grouped_convolution`` registers one candidate per record of
``dispatch/grouped_convolution_instances/*.json`` -- shape-generic AOT
binaries, named by their lossless labels. These tests pin down that:

* the registry is exactly the list, in list order, and every instance has
  the AOT build job ``--compile-all`` makes its binary from;
* the list file is canonical and its loader refuses a broken one;
* nothing outside the list (another arch, another direction) dispatches;
* the dispatcher admits a shape exactly when the AOT cache's run path would
  launch that binary on it, with the same grid, block, signature and kernarg
  values -- the binaries were built and verified through that path;
* pins and the priority order behave.

CPU only: nothing is compiled or launched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common import direct_kernel_sweep as dks
from benchmarks.common import kernel_sweep as ks
from benchmarks.common.grouped_conv import instance_list
from dispatch import grouped_convolution as gc
from kernels.common.conv_kernel_identity import KernelIdentity
from rocke.dispatch.core import PinRefused

_ARCH = "gfx950"
_LIST = (
    Path(gc.__file__).resolve().with_name("grouped_convolution_instances")
    / "gfx950_fwd.json"
)


def _listed_names() -> list[str]:
    return [i.label() for i in gc.load_instance_list(_LIST).identities]


@pytest.fixture(autouse=True)
def _instance_flavor(monkeypatch):
    """Lower the flavor the instances were named for: admission checks it."""
    flavors = {s.identity.llvm_flavor for s in gc.registered_conv_grouped_instances()}
    assert len(flavors) == 1
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavors.pop())


def _req(**kw) -> gc.ConvGroupedRequest:
    base = dict(N=2, C=64, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1)
    base.update(kw)
    return gc.ConvGroupedRequest(arch=_ARCH, **base)


# Shapes that reach every registered family: dense (implicit GEMM and
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


# ---------------------------------------------------------------------------
# What is registered
# ---------------------------------------------------------------------------


def test_registry_is_exactly_the_instance_list_in_order():
    names = _listed_names()
    assert names, "the instance list is empty"
    assert [c.name for c in gc.conv_grouped_candidates()] == names
    assert [s.name for s in gc.registered_conv_grouped_instances(_ARCH)] == names
    for candidate in gc.conv_grouped_candidates():
        assert candidate.spec_id == candidate.name
        assert candidate.capability.arches == (_ARCH,)


def test_every_instance_is_a_canonical_forward_gfx950_name():
    for spec in gc.registered_conv_grouped_instances():
        assert KernelIdentity.from_label(spec.name) == spec.identity
        assert spec.identity.label() == spec.name
        assert spec.identity.arch == _ARCH
        assert spec.identity.direction == ("direct_fwd" if spec.is_direct else "fwd")
        assert spec.algorithm == spec.identity.algorithm


def test_every_instance_has_the_aot_build_job_compile_all_uses():
    """``--compile-all`` builds each instance from its family's job; a name the
    generators cannot reproduce would be an instance with no binary."""
    for spec in gc.registered_conv_grouped_instances():
        ident = spec.identity
        if ident.is_direct:
            job = dks.direct_job_for_identity(ident)
        else:
            job = ks.job_for_identity(ident)
        assert job.identity == ident


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(arch="gfx942"), "arch 'gfx942'"),
        (dict(arch="gfx1250"), "arch 'gfx1250'"),
        (dict(direction="wgrad"), "only forward instances"),
        (dict(direction="dgrad"), "only forward instances"),
        (dict(dtype="fp32"), "fp16 or bf16 only"),
        (dict(C=63, G=2), "divisible by G"),
        (dict(Di=8, Z=3, stride_d=1, pad_d=1, dilation_d=1), "3-D capability"),
    ],
)
def test_nothing_outside_the_list_dispatches(kw, match):
    fields = dict(N=2, C=64, K=64, Hi=28, Wi=28, Y=3, X=3, pad_h=1, pad_w=1, G=1)
    fields.update(kw)
    fields.setdefault("arch", _ARCH)
    req = gc.ConvGroupedRequest(**fields)
    with pytest.raises(ValueError, match=match):
        gc.dispatch_conv_grouped(req)
    assert not gc.dispatch_conv_grouped_all(req)


def test_no_backward_candidates():
    assert gc.conv_grouped_candidates("wgrad") == ()
    assert gc.conv_grouped_candidates("dgrad") == ()
    assert gc.registered_conv_grouped_instances(direction="wgrad") == ()


def test_another_llvm_flavor_admits_nothing(monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm-other")
    with pytest.raises(ValueError, match="LLVM flavor"):
        gc.dispatch_conv_grouped(_req())


# ---------------------------------------------------------------------------
# Admission and the launch contract agree with the AOT cache's run path
# ---------------------------------------------------------------------------


def _cache_side_admits(spec, req) -> bool:
    """Would the AOT run path launch ``spec``'s binary on ``req``?

    Built the way the implicit-GEMM and direct benchmarks build their
    problems, and asked through ``KernelCache.supports_problem`` and
    ``direct_kernel_sweep.plan_for``.
    """
    from benchmarks.common.kernel_cache import KernelCache
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_implicit_gemm import ConvProblem

    ident = spec.identity
    if ident.dtype_a != req.dtype:
        return False
    if not ident.is_direct:
        problem = ConvProblem(
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
        return KernelCache.supports_problem(ident, problem)[0]
    if req.stride_h != req.stride_w or req.pad_h != req.pad_w:
        return False
    if ident.algorithm == "direct_nongrouped" and req.G != 1:
        return False
    problem = DirectConvProblem(
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
    if not KernelCache.supports_problem(ident, problem)[0]:
        return False
    plan, _why = dks.plan_for(
        None, ident, Path("unused.hsaco"), {"kernel_name": "k"}, problem, _ARCH
    )
    return plan is not None


@pytest.mark.parametrize("shape", sorted(_SHAPES), ids=str)
def test_admission_matches_the_aot_run_path(shape):
    req = _req(**_SHAPES[shape])
    admitted = [r.spec.name for r in gc.dispatch_conv_grouped_all(req)]
    expected = [
        s.name
        for s in gc.registered_conv_grouped_instances()
        if _cache_side_admits(s, req)
    ]
    assert admitted == expected
    assert admitted, f"{shape}: no registered instance runs it"


def test_the_shapes_reach_every_registered_family():
    families = {s.algorithm for s in gc.registered_conv_grouped_instances()}
    reached = {
        r.spec.algorithm
        for kw in _SHAPES.values()
        for r in gc.dispatch_conv_grouped_all(_req(**kw))
    }
    assert reached == families


def _cache_side_launch(spec, req, ptrs, sizes):
    """``(grid, block, signature names, values)`` the AOT run path launches."""
    from benchmarks.common.benchmark_implicit_gemm_conv import (
        _grid_for,
        _signature_for_identity,
    )
    from kernels.common.conv_direct_grouped import DirectConvProblem
    from kernels.common.conv_implicit_gemm import ConvProblem

    ident = spec.identity
    if not ident.is_direct:
        problem = ConvProblem(
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
        return (
            _grid_for("fwd", problem, ident),
            (ident.warp_m * ident.warp_n * ident.wave_size, 1, 1),
            [a["name"] for a in _signature_for_identity(ident, ident.dtype_a)],
            ks._launch_values_for("fwd", problem, ident, ptrs, sizes, {}),
        )
    problem = DirectConvProblem(
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
    plan, why = dks.plan_for(
        None, ident, Path("unused.hsaco"), {"kernel_name": "k"}, problem, _ARCH
    )
    assert plan is not None, why
    (step,) = plan.steps
    assert step.buffers == ("x", "w", "y")
    return (
        step.grid,
        step.block,
        [a["name"] for a in step.signature],
        dks.launch_values(
            step, dict(zip(step.buffers, ptrs)), dict(zip(step.buffers, sizes))
        ),
    )


@pytest.mark.parametrize("shape", sorted(_SHAPES), ids=str)
def test_launch_contract_matches_the_aot_run_path(shape):
    req = _req(**_SHAPES[shape])
    ptrs, sizes = (0x1000, 0x2000, 0x3000), (111, 222, 333)
    for result in gc.dispatch_conv_grouped_all(req):
        values = gc.launch_values_for(
            result.request,
            result.spec,
            A_ptr=ptrs[0],
            B_ptr=ptrs[1],
            D_ptr=ptrs[2],
            A_bytes=sizes[0],
            B_bytes=sizes[1],
            D_bytes=sizes[2],
        )
        grid, block, names, want = _cache_side_launch(result.spec, req, ptrs, sizes)
        assert result.grid == tuple(grid), result.spec.name
        assert result.block == tuple(block), result.spec.name
        assert [a["name"] for a in result.signature] == names, result.spec.name
        assert values == want, result.spec.name
        assert set(values) == set(names), result.spec.name


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
        for s in gc.registered_conv_grouped_instances()
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


def test_instance_list_file_is_canonical():
    """Fix with ``instance_list.py format``."""
    lst = gc.load_instance_list(_LIST)
    assert _LIST.read_text(encoding="utf-8") == gc.format_instance_list(lst)


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


def test_every_listed_instance_round_trips_through_its_record():
    lst = gc.load_instance_list(_LIST)
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
