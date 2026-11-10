# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the opt-in Winograd forward route of the grouped-convolution
dispatcher (``conv_winograd_fwd``, :mod:`dispatch.conv_winograd_plan`).

* ``algorithm="auto"`` never picks it: a gfx1250 forward request stays refused
  unless it pins ``algorithm="winograd"``;
* the pinned request selects the selector's plan, the sweep entry points list
  every safe plan;
* each plan kind's launch plan: step order, kernargs spelled per signature,
  workspace sizes equal to the kernels' own, every step's kernel builds;
* eligibility refusals.
"""

from __future__ import annotations

import dataclasses

import pytest

import dispatch.grouped_convolution as gc
from dispatch.conv_winograd_plan import ConvWinogradSpec, winograd_launch_plan
from kernels.common import conv_winograd_fwd as W
from kernels.common import conv_winograd_gen as WG
from kernels.common.conv_winograd_fused_1d_wmma import FusedWino1DSpec
from kernels.common.conv_winograd_gemm_wmma import WinogradGemmProblem
from kernels.common.conv_winograd_select import WinogradPlan
from rocke.core.lower_llvm import lower_kernel_to_llvm

_ARCH = "gfx1250"
_PTRS = {
    "x": 1 << 20,
    "w": 2 << 20,
    "y": 3 << 20,
    "U": 4 << 20,
    "V": 5 << 20,
    "M": 6 << 20,
    "e": 7 << 20,
}


def _req(**kw):
    fields = dict(N=2, C=64, K=64, Hi=16, Wi=16, Y=3, X=3, pad_h=1, pad_w=1)
    fields.update(kw)
    fields.setdefault("arch", _ARCH)
    if fields["arch"] == _ARCH:
        # The K3 GEMM of the unfused plans lowers at llvm23 only.
        fields.setdefault("llvm_flavor", "llvm23")
    return gc.ConvGroupedRequest(**fields)


def _pin(req):
    return dataclasses.replace(req, algorithm="winograd")


def _check_plan(spec: ConvWinogradSpec, *, build: bool = True):
    plan = gc.launch_plan_for(None, spec)
    assert plan.steps
    for step in plan.steps:
        assert step.identity is None
        assert len(step.kernargs) == len(step.signature)
        vals = plan.values(step, _PTRS, {})
        assert list(vals) == [a["name"] for a in step.signature]
        for arg, spelled in zip(step.signature, step.kernargs):
            if spelled[0] == "buf":
                assert spelled[1] in ("x", "w", "y", *plan.workspaces)
            else:
                assert arg["name"] not in ("x", "w", "y", "u", "v", "mm")
        assert min(step.grid) >= 1 and step.block[0] >= 32
        if build:
            k = step.kernel.build(_ARCH)
            assert [p.name for p in k.params] == [a["name"] for a in step.signature]
    return plan


def test_auto_never_picks_winograd():
    req = _req()
    with pytest.raises(ValueError, match="pin algorithm='winograd'"):
        gc.dispatch_conv_grouped(req)
    # gfx950 routes stay on the listed instances.
    r950 = gc.dispatch_conv_grouped(_req(arch="gfx950", Hi=28, Wi=28))
    assert r950.spec.algorithm != "winograd"


def test_pinned_request_selects_a_plan():
    res = gc.dispatch_conv_grouped(_pin(_req()))
    assert isinstance(res.spec, ConvWinogradSpec)
    assert res.candidate.name == "conv_winograd_fwd" and res.candidate.opt_in
    assert res.spec.plan.kind == "unfused2d"
    last = gc.launch_plan_for(_pin(_req()), res.spec).steps[-1]
    assert res.grid == last.grid and res.block == last.block
    assert tuple(res.signature) == last.signature
    k = res.build()
    assert [p.name for p in k.params] == [a["name"] for a in last.signature]


def test_sweep_returns_the_opt_in_plans():
    req = _req()
    specs = gc.conv_grouped_sweep_space(req)
    assert specs and all(isinstance(s, ConvWinogradSpec) for s in specs)
    tiles = {s.plan.tile for s in specs}
    assert (4, 4) in tiles and (2, 2) in tiles and (6, 6) not in tiles
    combos = gc.registered_conv_grouped_combos(req)
    assert [s.name for _, s in combos] == [s.name for s in specs]
    results = gc.dispatch_conv_grouped_all(req)
    assert [r.spec.name for r in results] == [s.name for s in specs]
    assert [r.spec.name for r in gc.iter_dispatch_conv_grouped_all(req)] == [
        s.name for s in specs
    ]


def test_registered_instances_hold_no_winograd():
    for spec in gc.registered_conv_grouped_instances(direction="fwd"):
        assert spec.algorithm != "winograd"
    assert gc.registered_conv_grouped_instances("gfx1250") == ()


def _unfused2d_spec(req, tile):
    for s in gc.conv_grouped_sweep_space(req):
        if s.plan.kind == "unfused2d" and s.plan.tile == tile:
            return s
    raise AssertionError(f"no {tile} plan")


def test_3x3_fp16_plan():
    req = _req()
    spec = _unfused2d_spec(req, (4, 4))
    plan = _check_plan(spec)
    names = [st.kernel.name for st in plan.steps]
    assert names == ["MERGED_K2_K1", "K3", "K4"]
    ws = W.WinogradFwdSpec(
        problem=spec.problem, m=4, **W.tuned_transform_options(4, False, spec.problem)
    ).buffer_bytes()
    assert plan.workspaces == {"U": ws["U"], "V": ws["V"], "M": ws["M"], "e": 4}
    # K3 reads every batch of the GEMM.
    k3 = plan.steps[1]
    vals = plan.values(k3, _PTRS, {})
    prob = WinogradGemmProblem.from_conv(spec.problem, (4, 4))
    assert (vals["T"], vals["G"], vals["Cg"], vals["Kg"]) == (
        prob.T,
        prob.G,
        prob.Cg,
        prob.Kg,
    )
    assert (vals["g0"], vals["nb"], vals["splits"]) == (0, prob.G, 1)


def test_3x3_bf16_dilated_prescale_plan():
    req = _req(Hi=24, Wi=20, K=96, pad_h=2, pad_w=2, dilation_h=2, dilation_w=2)
    req = dataclasses.replace(req, dtype="bf16")
    spec = _unfused2d_spec(req, (4, 4))
    plan = _check_plan(spec)
    fs = W.WinogradFwdSpec(
        problem=spec.problem,
        m=4,
        io_dtype="bf16",
        prescale=True,
        **W.tuned_transform_options(4, True, spec.problem),
    )
    assert plan.workspaces["e"] == fs.buffer_bytes()["e"] > 4
    # K4 reads the per-tile exponents.
    k4 = plan.steps[-1]
    assert ("buf", "e", 0) in k4.kernargs
    # bf16 F(2,3) runs bf16 operands, F(4,3) fp16 ones.
    s22 = _unfused2d_spec(req, (2, 2))
    assert gc.launch_plan_for(None, s22).steps[-2].kernel.name == "K3"


def test_6x4_merged_plan():
    spec = _unfused2d_spec(_req(N=1, Hi=26, Wi=18), (6, 4))
    plan = _check_plan(spec)
    assert plan.steps[0].kernel.name.startswith("MERGED_")
    assert "w" in [k[1] for k in plan.steps[0].kernargs if k[0] == "buf"]


def test_batch_split_plan():
    p = _req(N=4, Hi=8, Wi=8)
    base = gc._winograd_spec(
        p,
        WinogradPlan(kind="unfused2d", tile=(2, 2), k3="128x128x128_w2x2_f4x4_s2_tdm"),
    )
    plan2 = WinogradPlan(
        kind="unfused2d", tile=(2, 2), k3="128x128x128_w2x2_f4x4_s2_tdm", chunks=2
    )
    split = winograd_launch_plan(base.problem, "fp16", plan2, _ARCH)
    names = [st.kernel.name for st in split.steps]
    assert names == ["K1[0]", "K2[0]", "K3", "K4[0]", "K2[1]", "K3", "K4[1]"]
    sub = dataclasses.replace(base.problem, N=2)
    sz = W.WinogradFwdSpec(
        problem=sub, m=2, **W.tuned_transform_options(2, False, sub)
    ).buffer_bytes()
    k2_1 = split.steps[4]
    v = split.values(k2_1, _PTRS, {})
    assert v["x"] == _PTRS["x"] + sz["x"] and v["x_bytes"] == sz["x"]
    k4_1 = split.steps[6]
    v = split.values(k4_1, _PTRS, {})
    assert v["y"] == _PTRS["y"] + sz["y"] and v["y_bytes"] == sz["y"]
    assert split.workspaces["V"] == sz["V"]


def test_1x11_plans():
    req = _req(N=2, C=64, K=64, Hi=1, Wi=300, Y=1, X=11, pad_h=0, pad_w=5)
    specs = gc.conv_grouped_sweep_space(req)
    kinds = {s.plan.kind for s in specs}
    assert kinds == {"fused1d", "unfused_gen"}
    for spec in specs:
        plan = _check_plan(spec)
        names = [st.kernel.name for st in plan.steps]
        if spec.plan.kind == "fused1d":
            assert names == ["K1", "FUSED"]
            assert set(plan.workspaces) == {"U"}
            fs = FusedWino1DSpec(
                problem=spec.problem,
                ax=WG.WinoAxis(4, 11, 3),
                **dict(spec.plan.fused_cfg),
            )
            assert plan.workspaces["U"] == fs.buffer_bytes()["u"]
        else:
            assert names == ["K1", "K2", "K3", "K4"]
            assert set(plan.workspaces) == {"U", "V", "M"}
    # bf16 I/O: the fused kernel is fp16 only.
    bf = gc.conv_grouped_sweep_space(dataclasses.replace(req, dtype="bf16"))
    assert bf and {s.plan.kind for s in bf} == {"unfused_gen"}


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22"])
def test_unfused_plans_need_llvm23(flavor):
    """The K3 GEMM lowers at llvm23 only: a request for another flavor admits
    no plan that runs it, and says why."""
    req = _pin(_req(llvm_flavor=flavor))
    ok, why = gc._WINOGRAD_CANDIDATE.admits(req)
    assert not ok and "llvm23" in why and flavor in why, why
    with pytest.raises(ValueError, match="needs LLVM flavor llvm23"):
        gc.dispatch_conv_grouped(req)
    assert not gc.conv_grouped_sweep_space(req)
    assert not gc.dispatch_conv_grouped_all(_req(llvm_flavor=flavor))
    # 1-D: only the fused plan (no K3) is left, and it lowers at the flavor.
    req1 = _pin(_req(llvm_flavor=flavor, Hi=1, Wi=300, Y=1, X=11, pad_h=0, pad_w=5))
    assert {s.plan.kind for s in gc.conv_grouped_sweep_space(req1)} == {"fused1d"}
    res = gc.dispatch_conv_grouped(req1)
    assert res.spec.plan.kind == "fused1d"
    for step in gc.launch_plan_for(req1, res.spec).steps:
        assert step.kernel.llvm_flavors is None
        lower_kernel_to_llvm(step.kernel.build(_ARCH), llvm_flavor=flavor, arch=_ARCH)
    # The K3 kernel the flavor check refers to does refuse to lower there.
    k3 = next(
        st
        for st in gc.launch_plan_for(None, _unfused2d_spec(_req(), (4, 4))).steps
        if st.kernel.name == "K3"
    )
    assert k3.kernel.llvm_flavors == ("llvm23",)
    with pytest.raises(ValueError, match="requires LLVM flavor llvm23"):
        lower_kernel_to_llvm(k3.kernel.build(_ARCH), llvm_flavor=flavor, arch=_ARCH)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(stride_h=2, stride_w=2), "stride"),
        (
            dict(
                Hi=128, Wi=128, C=128, pad_h=36, pad_w=36, dilation_h=36, dilation_w=36
            ),
            "dilation",
        ),
        (dict(Y=1, X=1, pad_h=0, pad_w=0), "1x1"),
        (dict(G=4), "C/g"),
        (dict(direction="dgrad"), "forward"),
    ],
)
def test_ineligible_requests_are_refused(kw, match):
    req = _pin(_req(**kw))
    ok, why = gc._WINOGRAD_CANDIDATE.admits(req)
    assert not ok and match.lower() in why.lower(), why
    assert not [
        r
        for r in gc.dispatch_conv_grouped_all(_req(**kw))
        if r.spec.algorithm == "winograd"
    ]


def test_wrong_arch_or_dtype_is_refused():
    for req in (_pin(_req(arch="gfx950")), _pin(_req(dtype="fp32"))):
        ok, _ = gc._WINOGRAD_CANDIDATE.admits(req)
        assert not ok
        with pytest.raises(ValueError):
            gc.dispatch_conv_grouped(req)
