# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the Winograd plan selector (``conv_winograd_select``).

Structural checks only (no timing data):

* every reference layer is eligible and every plan offered or selected
  there is safe; the same on a grid of synthetic shapes;
* eligibility, the unsafe list, kernel limits and the batch split;
* the cost model's plumbing (sorted plans, the accuracy margin, parameter
  checks).

``REFERENCE``: ``N,Hi,Wi,C,K,Y,X,pH,pW,dH,dW,groups`` and the I/O dtype of
layers of public models the plans were validated on.
"""

from __future__ import annotations

import itertools
import math

import pytest

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_select import (
    TILES_2D,
    UNSAFE,
    WinogradCostModel,
    WinogradPlan,
    _buffers,
    _gen_axes,
    _Axis,
    _MAX_BUFFER_BYTES,
    plan_cost,
    plan_is_unsafe,
    select_winograd,
    winograd_ineligible_reason,
    winograd_plans,
)
from kernels.common import conv_winograd_select as SEL

REFERENCE_TSV = """\
16,128,128,512,512,3,3,4,4,4,4,1	fp16
16,128,128,2048,256,3,3,12,12,12,12,1	fp16
16,128,128,2048,256,3,3,24,24,24,24,1	fp16
16,128,128,256,256,3,3,2,2,2,2,1	fp16
16,128,128,256,256,3,3,1,1,1,1,1	fp16
16,65,65,2048,256,3,3,12,12,12,12,1	fp16
16,65,65,512,512,3,3,4,4,4,4,1	fp16
16,65,65,256,256,3,3,2,2,2,2,1	fp16
1,128,128,512,512,3,3,4,4,4,4,1	fp16
1,128,128,2048,256,3,3,12,12,12,12,1	fp16
1,128,128,2048,256,3,3,24,24,24,24,1	fp16
1,65,65,2048,256,3,3,12,12,12,12,1	fp16
1,65,65,512,512,3,3,4,4,4,4,1	fp16
1,65,65,256,256,3,3,2,2,2,2,1	fp16
128,120,120,48,192,3,3,1,1,1,1,1	fp16
128,96,96,48,192,3,3,1,1,1,1,1	fp16
128,60,60,64,256,3,3,1,1,1,1,1	fp16
8,192,192,512,512,3,3,1,1,1,1,1	bf16
4,256,256,512,512,3,3,1,1,1,1,1	bf16
8,384,384,256,256,3,3,1,1,1,1,1	bf16
8,768,768,128,128,3,3,1,1,1,1,1	bf16
8,384,384,512,512,3,3,1,1,1,1,1	bf16
8,768,768,256,256,3,3,1,1,1,1,1	bf16
1,256,256,512,512,3,3,1,1,1,1,1	bf16
1,1024,1024,128,128,3,3,1,1,1,1,1	bf16
1,192,192,512,512,3,3,1,1,1,1,1	bf16
1,384,384,256,256,3,3,1,1,1,1,1	bf16
1,768,768,128,128,3,3,1,1,1,1,1	bf16
1,384,384,512,512,3,3,1,1,1,1,1	bf16
16,1,25600,128,128,1,11,0,5,1,1,1	fp16
32,1,25600,128,128,1,11,0,5,1,1,1	fp16
32,1,25600,128,128,1,7,0,3,1,1,1	fp16
16,1,25600,128,128,1,7,0,3,1,1,1	fp16
32,1,12800,128,128,1,11,0,5,1,1,1	fp16
32,1,3200,256,256,1,11,0,5,1,1,1	fp16
32,1,51200,64,64,1,11,0,5,1,1,1	fp16
16,1,12800,128,128,1,11,0,5,1,1,1	fp16
16,1,3200,256,256,1,11,0,5,1,1,1	fp16
32,1,51200,64,64,1,7,0,3,1,1,1	fp16
32,1,25600,128,128,1,11,0,15,1,3,1	fp16
32,1,25600,128,128,1,11,0,25,1,5,1	fp16
32,1,102400,32,32,1,11,0,5,1,1,1	fp16
16,1,25600,128,128,1,11,0,15,1,3,1	fp16
16,1,102400,32,32,1,11,0,5,1,1,1	fp16
16,1,25600,128,128,1,7,0,15,1,5,1	fp16
16,1,25600,128,128,1,7,0,9,1,3,1	fp16
32,1,25600,128,128,1,7,0,15,1,5,1	fp16
32,1,102400,32,32,1,7,0,3,1,1,1	fp16
32,1,51200,64,64,1,3,0,1,1,1,1	fp16
32,1,25600,128,128,1,3,0,3,1,3,1	fp16
16,1,102400,32,32,1,11,0,15,1,3,1	fp16
32,1,102400,32,32,1,11,0,15,1,3,1	fp16
32,1,102400,32,32,1,7,0,15,1,5,1	fp16
16,1,400,80,512,1,7,0,3,1,1,1	fp16
16,1,200,80,512,1,7,0,3,1,1,1	fp16
1,1,200,80,512,1,7,0,3,1,1,1	fp16
128,73,73,80,192,3,3,0,0,1,1,1	fp16
128,35,35,96,96,3,3,1,1,1,1,1	fp16
128,17,17,192,192,1,7,0,3,1,1,1	fp16
128,17,17,192,192,7,1,3,0,1,1,1	fp16
128,35,35,48,64,5,5,2,2,1,1,1	fp16
128,17,17,160,160,1,7,0,3,1,1,1	fp16
128,17,17,160,160,7,1,3,0,1,1,1	fp16
64,17,17,192,192,1,7,0,3,1,1,1	fp16
64,17,17,192,192,7,1,3,0,1,1,1	fp16
64,35,35,48,64,5,5,2,2,1,1,1	fp16
128,8,8,384,384,1,3,0,1,1,1,1	fp16
128,8,8,384,384,3,1,1,0,1,1,1	fp16
64,8,8,384,384,1,3,0,1,1,1,1	fp16
64,8,8,384,384,3,1,1,0,1,1,1	fp16
16,17,17,192,192,1,7,0,3,1,1,1	fp16
16,17,17,192,192,7,1,3,0,1,1,1	fp16
16,35,35,48,64,5,5,2,2,1,1,1	fp16
16,8,8,384,384,1,3,0,1,1,1,1	fp16
16,8,8,384,384,3,1,1,0,1,1,1	fp16
128,5,5,128,768,5,5,0,0,1,1,1	fp16
1,17,17,192,192,1,7,0,3,1,1,1	fp16
1,17,17,192,192,7,1,3,0,1,1,1	fp16
1,35,35,48,64,5,5,2,2,1,1,1	fp16
1,8,8,384,384,1,3,0,1,1,1,1	fp16
1,8,8,384,384,3,1,1,0,1,1,1	fp16
16,5,5,128,768,5,5,0,0,1,1,1	fp16
16,128,128,192,64,3,3,1,1,1,1,1	fp16
128,16,16,896,896,3,3,1,1,1,1,16	fp16
128,32,32,448,448,3,3,1,1,1,1,8	fp16
128,16,16,256,256,3,3,1,1,1,1,1	fp16
128,14,14,256,256,3,3,1,1,1,1,1	fp16
128,32,32,128,128,3,3,1,1,1,1,1	fp16
8,48,48,1280,1280,3,3,1,1,1,1,1	bf16
8,32,32,1280,1280,3,3,1,1,1,1,1	bf16
8,96,96,640,640,3,3,1,1,1,1,1	bf16
8,64,64,640,640,3,3,1,1,1,1,1	bf16
8,24,24,1280,1280,3,3,1,1,1,1,1	bf16
8,24,24,2560,1280,3,3,1,1,1,1,1	bf16
1,32,32,1280,1280,3,3,1,1,1,1,1	bf16
1,64,64,640,640,3,3,1,1,1,1,1	bf16
1,48,48,1280,1280,3,3,1,1,1,1,1	bf16
1,96,96,640,640,3,3,1,1,1,1,1	bf16
8,192,192,320,320,3,3,1,1,1,1,1	bf16
8,40,40,1280,1280,3,3,1,1,1,1,1	bf16
8,128,128,320,320,3,3,1,1,1,1,1	bf16
8,160,160,320,320,3,3,1,1,1,1,1	bf16
8,80,80,640,640,3,3,1,1,1,1,1	bf16
8,48,48,2560,1280,3,3,1,1,1,1,1	bf16
4,192,192,320,320,3,3,1,1,1,1,1	bf16
1,128,128,320,320,3,3,1,1,1,1,1	bf16
1,40,40,1280,1280,3,3,1,1,1,1,1	bf16
1,160,160,320,320,3,3,1,1,1,1,1	bf16
1,80,80,640,640,3,3,1,1,1,1,1	bf16
1,64,64,2048,2048,3,3,1,1,1,1,8	fp16
1,128,128,1024,1024,3,3,1,1,1,1,8	fp16
16,256,256,64,256,3,3,1,1,1,1,1	fp16
16,128,128,64,256,3,3,1,1,1,1,1	fp16
16,256,256,128,128,3,3,1,1,1,1,1	fp16
16,64,64,512,512,3,3,1,1,1,1,1	fp16
16,64,64,1024,512,3,3,1,1,1,1,1	fp16
1,256,256,128,128,3,3,1,1,1,1,1	fp16
1,64,64,1024,512,3,3,1,1,1,1,1	fp16
4,240,416,384,192,3,3,1,1,1,1,1	bf16
4,480,832,192,96,3,3,1,1,1,1,1	bf16
8,160,160,384,192,3,3,1,1,1,1,1	bf16
8,320,320,192,96,3,3,1,1,1,1,1	bf16
2,120,208,384,192,3,3,1,1,1,1,1	bf16
1,240,416,384,192,3,3,1,1,1,1,1	bf16
1,480,832,192,96,3,3,1,1,1,1,1	bf16
1,160,160,384,192,3,3,1,1,1,1,1	bf16
1,320,320,192,96,3,3,1,1,1,1,1	bf16
1,120,208,384,192,3,3,1,1,1,1,1	bf16
"""


def _reference():
    out = []
    for line in REFERENCE_TSV.strip().splitlines():
        shp, dt = line.split("\t")
        f = [int(v) for v in shp.split(",")]
        p = ConvProblem(
            N=f[0],
            Hi=f[1],
            Wi=f[2],
            C=f[3],
            K=f[4],
            Y=f[5],
            X=f[6],
            pH=f[7],
            pW=f[8],
            dH=f[9],
            dW=f[10],
            groups=f[11],
        )
        out.append((p, dt))
    return out


REFERENCE = _reference()
# Any CU count: the checks below are structural.
MODEL = WinogradCostModel(num_cus=128)


def _assert_safe(plan: WinogradPlan, p: ConvProblem, dt: str) -> None:
    assert not plan_is_unsafe(plan, p), plan
    assert (plan.kind, plan.form) not in UNSAFE, plan
    if plan.kind == "unfused2d":
        assert plan.tile in TILES_2D and plan.tile != (6, 6)
        assert not (dt == "bf16" and plan.tile == (4, 6))
    if plan.kind == "fused1d":
        assert plan.variant == {11: "4s3", 7: "4s2", 3: "6"}[p.X]
        assert p.C in (32, 64, 128) and p.Hi == 1 and p.groups == 1
    if plan.kind in ("unfused_gen", "fused1d"):
        for a in _gen_axes(p, plan.variant):
            if a.r > 1 and a.seg == 1:
                # single-segment long filters: only F(4,7) (n = 10) and shorter
                assert a.n <= 10, plan
    assert max(p.dH, p.dW) < 36
    assert plan.cost > 0 and math.isfinite(plan.cost)
    assert plan.cost == pytest.approx(plan_cost(plan))


def test_reference_rows_eligible():
    for p, dt in REFERENCE:
        assert winograd_ineligible_reason(p, dt) is None, p
        assert winograd_plans(p, dt, model=MODEL), p


def test_reference_rows_never_unsafe():
    for p, dt in REFERENCE:
        for pl in winograd_plans(p, dt, model=MODEL):
            _assert_safe(pl, p, dt)
        for margin in (0.0, 0.03, 0.1):
            sel = select_winograd(p, dt, model=MODEL, accuracy_margin=margin)
            assert sel.kind != "none", p
            _assert_safe(sel, p, dt)


def _grid():
    for (y, x), n, c, hw, d, g, dt in itertools.product(
        [(3, 3), (1, 3), (3, 1), (1, 5), (1, 7), (7, 1), (1, 11), (5, 5)],
        [1, 16],
        [32, 64, 128, 256, 512, 2048],
        [1, 17, 64],
        [1, 2, 12, 36],
        [1, 8],
        ["fp16", "bf16"],
    ):
        if c % g:
            continue
        hi = 1 if (y == 1 and hw == 1) else hw
        wi = 4096 if hi == 1 else hw
        if hi == 1 and y != 1:
            continue
        dh = d if y > 1 else 1
        dw = d if x > 1 else 1
        yield ConvProblem(
            N=n,
            Hi=hi,
            Wi=wi,
            C=c,
            K=c,
            Y=y,
            X=x,
            pH=dh * (y // 2),
            pW=dw * (x // 2),
            dH=dh,
            dW=dw,
            groups=g,
        ), dt


def test_grid_never_unsafe():
    seen = set()
    for p, dt in _grid():
        for pl in winograd_plans(p, dt, model=MODEL):
            _assert_safe(pl, p, dt)
            seen.add((pl.kind, pl.form))
        sel = select_winograd(p, dt, model=MODEL)
        if sel.kind != "none":
            _assert_safe(sel, p, dt)
    # every implementation is reachable on the grid
    assert {k for k, _ in seen} == {"unfused2d", "fused1d", "unfused_gen"}
    assert not seen & UNSAFE


@pytest.mark.parametrize(
    "y,x,forbidden",
    [
        (3, 3, "F(6x6,3x3)"),
        (1, 11, "F(4,11)"),
        (1, 11, "F(6,11)"),
        (1, 11, "F(2,11)"),
        (1, 7, "F(6,7)"),
        (5, 5, "F(4x4,5x5)"),
    ],
)
def test_reported_unsafe_variants_never_offered(y, x, forbidden):
    hi = 1 if y == 1 else 64
    p = ConvProblem(
        N=8, Hi=hi, Wi=4096 if hi == 1 else 64, C=128, K=128, Y=y, X=x,
        pH=y // 2, pW=x // 2,
    )  # fmt: skip
    maths = {pl.form for pl in winograd_plans(p, "fp16", model=MODEL)}
    assert maths, (y, x)
    assert forbidden not in maths


def test_unsafe_plans_are_flagged():
    for c in (32, 64, 128, 192):
        p = ConvProblem(N=16, Hi=64, Wi=64, C=c, K=c, Y=3, X=3, pH=1, pW=1)
        assert plan_is_unsafe(WinogradPlan(kind="unfused2d", tile=(6, 6)), p)


@pytest.mark.parametrize(
    "problem,dtype,why",
    [
        (ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=1, X=1), "fp16", "1x1"),
        (
            ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=3, X=3, sH=2, sW=2),
            "fp16",
            "stride",
        ),
        (
            ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=3, X=3, groups=64),
            "fp16",
            "C/g",
        ),
        (ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=3, X=3), "fp32", "dtype"),
        (
            ConvProblem(
                N=1,
                Hi=128,
                Wi=128,
                C=2048,
                K=256,
                Y=3,
                X=3,
                pH=36,
                pW=36,
                dH=36,
                dW=36,
            ),
            "fp16",
            "dilation",
        ),  # fmt: skip
        (
            ConvProblem(N=1, Hi=8, Wi=8, C=64, K=64, Y=7, X=7, pH=3, pW=3),
            "fp16",
            "filter",
        ),
    ],
)
def test_ineligible(problem, dtype, why):
    r = winograd_ineligible_reason(problem, dtype)
    assert r is not None and why.lower() in r.lower(), r
    assert winograd_plans(problem, dtype, model=MODEL) == []
    assert select_winograd(problem, dtype, model=MODEL).kind == "none"


@pytest.mark.parametrize(
    "shape,dtype,tile,chunks",
    [
        ((16, 128, 128, 2048, 256, 12), "fp16", (6, 4), 2),
        ((16, 128, 128, 2048, 256, 24), "fp16", (6, 2), 2),
        ((16, 128, 128, 2048, 256, 24), "fp16", (2, 2), 4),
        ((8, 384, 384, 512, 512, 1), "bf16", (6, 4), 2),
        ((8, 768, 768, 256, 256, 1), "bf16", (6, 4), 4),
        ((8, 768, 768, 128, 128, 1), "bf16", (4, 4), 2),
    ],
)
def test_batch_split(shape, dtype, tile, chunks):
    n, h, w, c, k, d = shape
    p = ConvProblem(N=n, Hi=h, Wi=w, C=c, K=k, Y=3, X=3, pH=d, pW=d, dH=d, dW=d)
    plans = {
        pl.tile: pl
        for pl in winograd_plans(p, dtype, model=MODEL)
        if pl.kind == "unfused2d"
    }
    assert plans[tile].chunks == chunks
    for pl in plans.values():
        bufs = _buffers(p, _Axis(pl.tile[0], 3), _Axis(pl.tile[1], 3), n // pl.chunks)
        assert max(bufs.values()) < _MAX_BUFFER_BYTES
        assert n % pl.chunks == 0


def test_kernel_limits():
    for c, ok in ((32, True), (64, True), (128, True), (256, False), (96, False)):
        p = ConvProblem(N=16, Hi=1, Wi=8192, C=c, K=c, Y=1, X=11, pW=5)
        kinds = {pl.kind for pl in winograd_plans(p, "fp16", model=MODEL)}
        assert ("fused1d" in kinds) == ok, (c, kinds)
    one_d = ConvProblem(N=16, Hi=1, Wi=8192, C=128, K=128, Y=1, X=11, pW=5)
    assert all(
        pl.kind != "fused1d" for pl in winograd_plans(one_d, "bf16", model=MODEL)
    )  # the fused 1-D kernel has fp16 I/O only
    grouped = ConvProblem(N=16, Hi=1, Wi=8192, C=256, K=256, Y=1, X=11, pW=5, groups=2)
    assert all(
        pl.kind != "fused1d" for pl in winograd_plans(grouped, "fp16", model=MODEL)
    )
    bf = ConvProblem(N=8, Hi=64, Wi=64, C=1280, K=1280, Y=3, X=3, pH=1, pW=1)
    assert (4, 6) not in {pl.tile for pl in winograd_plans(bf, "bf16", model=MODEL)}
    assert (4, 6) in {pl.tile for pl in winograd_plans(bf, "fp16", model=MODEL)}


def test_accuracy_margin_prefers_lower_error():
    for p, dt in REFERENCE:
        plans = winograd_plans(p, dt, model=MODEL)
        sel = select_winograd(p, dt, model=MODEL, accuracy_margin=0.03)
        assert sel.cost <= plans[0].cost * 1.03 * (1 + 1e-12)
        if sel != plans[0]:
            assert sel.err_class < plans[0].err_class
        assert select_winograd(p, dt, model=MODEL, accuracy_margin=0.0) == plans[0]


def test_plans_sorted_and_k3_labels():
    for p, dt in REFERENCE:
        plans = winograd_plans(p, dt, model=MODEL)
        costs = [pl.cost for pl in plans]
        assert costs == sorted(costs)
        for pl in plans:
            if pl.kind in ("unfused2d", "unfused_gen"):
                assert pl.k3.endswith("_tdm") and pl.chunks >= 1
            else:
                assert pl.fused_cfg


def test_restrict_kinds_and_variants():
    p = ConvProblem(N=16, Hi=1, Wi=8192, C=128, K=128, Y=1, X=11, pW=5)
    plans = winograd_plans(p, "fp16", model=MODEL)
    assert {pl.kind for pl in plans} == {"fused1d", "unfused_gen"}
    for kind in ("fused1d", "unfused_gen"):
        assert select_winograd(p, "fp16", model=MODEL, kinds=(kind,)).kind == kind
    lab = plans[-1].label
    assert select_winograd(p, "fp16", model=MODEL, variants=(lab,)).label == lab
    none = select_winograd(p, "fp16", model=MODEL, kinds=("unfused2d",))
    assert none.kind == "none" and none.reason


def test_cost_model_parameters():
    with pytest.raises(ValueError):
        WinogradCostModel(num_cus=0)
    with pytest.raises(ValueError):
        WinogradCostModel(num_cus=8, flops_per_byte=0.0)
    # The small-GEMM rule of the K3 configuration follows the CU count.
    p = ConvProblem(N=1, Hi=32, Wi=32, C=512, K=512, Y=3, X=3, pH=1, pW=1)
    ax = SEL._Axis(4, 3)
    few = SEL.recommended_k3(p, ax, ax, WinogradCostModel(num_cus=1))
    many = SEL.recommended_k3(p, ax, ax, WinogradCostModel(num_cus=4096))
    assert few.startswith("256x256") and many.startswith("128x128")
    # More launch cost never lowers a plan's cost.
    cheap = winograd_plans(p, "fp16", model=WinogradCostModel(num_cus=128))
    dear = winograd_plans(
        p, "fp16", model=WinogradCostModel(num_cus=128, launch_bytes=1e12)
    )
    assert min(pl.cost for pl in dear) > min(pl.cost for pl in cheap)
