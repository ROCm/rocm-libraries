# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the opt-in AOT Winograd forward route of the grouped-convolution
dispatcher (``conv_winograd_fwd_aot``, plan kind ``unfused2d_rt``).

* ``algorithm="auto"`` never picks it; ``algorithm="winograd_aot"`` selects a
  3x3 plan on the shape-generic transform binaries;
* every step of its launch plans is an AOT binary: the transforms are members
  of :func:`conv_winograd_fwd_rt.aot_stage_specs` and are named by their
  kernel names, every kernarg is spelled, every kernel builds;
* the batch split re-launches the same binaries on byte offsets of x and y;
* requests without a 3x3 plan are refused with a reason.
"""

from __future__ import annotations

import dataclasses

import pytest

import dispatch.grouped_convolution as gc
from dispatch.conv_winograd_plan import ConvWinogradSpec, winograd_launch_plan
from kernels.common import conv_winograd_fwd_rt as R
from kernels.common.conv_winograd_gemm_wmma import WinogradGemmProblem
from kernels.common.conv_winograd_select import WinogradPlan

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
_GRID = {s.kernel_name() for s in R.aot_stage_specs()}


def _req(**kw):
    fields = dict(N=2, C=64, K=64, Hi=16, Wi=16, Y=3, X=3, pad_h=1, pad_w=1)
    fields.update(kw)
    fields.setdefault("arch", _ARCH)
    fields.setdefault("llvm_flavor", "llvm23")  # the K3 GEMM lowers at llvm23
    return gc.ConvGroupedRequest(**fields)


def _pin(req):
    return dataclasses.replace(req, algorithm="winograd_aot")


def _check_plan(spec: ConvWinogradSpec, *, build: bool = True):
    plan = gc.launch_plan_for(None, spec)
    assert plan.steps
    for step in plan.steps:
        assert step.identity is None
        assert len(step.kernargs) == len(step.signature)
        vals = plan.values(step, _PTRS, {})
        assert list(vals) == [a["name"] for a in step.signature]
        if not step.kernel.name.startswith("K3"):
            assert step.kernel.name in _GRID
        assert min(step.grid) >= 1 and step.block[0] >= 32
        if build:
            k = step.kernel.build(_ARCH)
            assert [p.name for p in k.params] == [a["name"] for a in step.signature]
    return plan


def _aot_spec(req, tile):
    for s in gc.conv_grouped_sweep_space(req):
        if s.plan.kind == "unfused2d_rt" and s.plan.tile == tile:
            return s
    raise AssertionError(f"no AOT {tile} plan")


def test_auto_never_picks_the_aot_route():
    with pytest.raises(ValueError):
        gc.dispatch_conv_grouped(_req())


def test_pinned_request_selects_an_aot_plan():
    res = gc.dispatch_conv_grouped(_pin(_req()))
    assert isinstance(res.spec, ConvWinogradSpec)
    assert res.candidate.name == "conv_winograd_fwd_aot" and res.candidate.opt_in
    assert res.candidate.algorithm == "winograd_aot"
    assert res.spec.plan.kind == "unfused2d_rt" and not res.spec.plan.merge
    assert res.spec.name.startswith("conv_winograd_fwd_aot:unfused2d_rt:F(")
    last = gc.launch_plan_for(_pin(_req()), res.spec).steps[-1]
    assert res.grid == last.grid and res.block == last.block
    assert last.kernel.name.startswith("wino_rt_output_")


def test_sweep_lists_both_routes():
    specs = gc.conv_grouped_sweep_space(_req())
    kinds = {s.plan.kind for s in specs}
    assert {"unfused2d", "unfused2d_rt"} <= kinds
    aot = [s for s in specs if s.plan.kind == "unfused2d_rt"]
    jit = [s for s in specs if s.plan.kind == "unfused2d"]
    assert {s.plan.tile for s in aot} == {s.plan.tile for s in jit}


def test_fp16_plan_steps():
    req = _req()
    spec = _aot_spec(req, (4, 4))
    plan = _check_plan(spec)
    names = [st.kernel.name for st in plan.steps]
    assert names[0].startswith("wino_rt_filter_f4_fp16")
    assert names[1].startswith("wino_rt_input_f4_fp16") and names[2] == "K3"
    geo = R.RtGeometry(spec.problem, 4, 4)
    sizes = geo.buffer_bytes(False)
    assert plan.workspaces == {
        "U": sizes["U"],
        "V": sizes["V"],
        "M": sizes["M"],
        "e": 4,
    }
    vals = plan.values(plan.steps[2], _PTRS, {})
    prob = WinogradGemmProblem.from_conv(spec.problem, (4, 4))
    assert (vals["T"], vals["G"], vals["Cg"], vals["Kg"]) == (
        prob.T,
        prob.G,
        prob.Cg,
        prob.Kg,
    )
    # The input transform reads the problem from its kernargs.
    k2 = plan.values(plan.steps[1], _PTRS, {})
    assert (k2["C"], k2["Hi"], k2["Wi"], k2["pH"], k2["T"]) == (
        64,
        16,
        16,
        1,
        geo.num_tiles,
    )


def test_small_tile_uses_the_lds_region():
    spec = _aot_spec(_req(Hi=32, Wi=32), (2, 2))
    plan = _check_plan(spec)
    assert "_lds4x8_" in plan.steps[1].kernel.name


def test_bf16_prescale_plans():
    req = dataclasses.replace(_req(K=96), dtype="bf16")
    fused = _check_plan(_aot_spec(req, (4, 4)))
    assert fused.steps[1].kernel.name.endswith("_psf")
    assert fused.workspaces["e"] > 4
    assert ("buf", "e", 0) in fused.steps[-1].kernargs
    wide = dataclasses.replace(_req(C=16384, K=64, Hi=8, Wi=8), dtype="bf16")
    split = _check_plan(_aot_spec(wide, (4, 4)), build=False)
    assert split.steps[1].kernel.name.startswith("wino_rt_pmax_bf16")
    assert split.steps[2].kernel.name.endswith("_ps")


def test_batch_split_relaunches_the_same_binaries():
    p = _req(N=4, Hi=8, Wi=8)
    plan = WinogradPlan(
        kind="unfused2d_rt", tile=(2, 2), k3="128x128x128_w2x2_f4x4_s2_tdm", chunks=2
    )
    split = winograd_launch_plan(gc._conv_problem(p), "fp16", plan, _ARCH)
    names = [st.kernel.name for st in split.steps]
    assert len(names) == 7 and names[2] == names[5] == "K3"
    assert names[1] == names[4] and names[3] == names[6]
    sub = dataclasses.replace(gc._conv_problem(p), N=2)
    sz = R.RtGeometry(sub, 2, 2).buffer_bytes(False)
    v = split.values(split.steps[4], _PTRS, {})
    assert v["x"] == _PTRS["x"] + sz["x"] and v["x_bytes"] == sz["x"]
    v = split.values(split.steps[6], _PTRS, {})
    assert v["y"] == _PTRS["y"] + sz["y"] and v["y_bytes"] == sz["y"]


def test_no_3x3_plan_is_refused():
    req = _pin(_req(Y=1, X=11, Hi=1, Wi=200, pad_h=0, pad_w=5))
    with pytest.raises(ValueError, match="has no AOT kernels"):
        gc.dispatch_conv_grouped(req)
    strided = _pin(_req(stride_h=2, stride_w=2))
    with pytest.raises(ValueError):
        gc.dispatch_conv_grouped(strided)
