# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/conv_route_exec.py: route workspaces shared between
# bindings are reused only by plans they fit, and the caller's operand sizes
# must be the problem's. Host-only (a recording runtime stands in for the
# device; the plans have no launch steps).

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from benchmarks.common.conv_route_exec import (
    RouteBinding,
    SharedWorkspaces,
    operand_nbytes,
)
from dispatch.conv_routes import RouteWorkspace
from kernels.common._conv_implicit_gemm_common import ConvProblem

# 3-D, 3x3x3, pad 1, stride 1: X [1, 4, 6, 8, 16], W [32, 3, 3, 3, 16],
# Y [1, 4, 6, 8, 32] (bf16).
_PROBLEM = ConvProblem(
    N=1, Hi=6, Wi=8, C=16, K=32, Y=3, X=3, pH=1, pW=1, Di=4, Z=3, sD=1, pD=1, dD=1
)
_X, _W, _Y = 2 * 4 * 6 * 8 * 16, 2 * 32 * 27 * 16, 2 * 4 * 6 * 8 * 32


class _Rt:
    def __init__(self):
        self.allocs, self.memsets, self._next = [], [], 0x1000

    def alloc(self, nbytes):
        p = self._next
        self._next += max(nbytes, 1) + 0x100
        self.allocs.append((p, nbytes))
        return p

    def memset(self, p, v, nbytes):
        self.memsets.append((p, v, nbytes))

    def free(self, p):
        pass


def _plan(*ws, route="winograd", label="", direction="fwd", problem=_PROBLEM):
    return SimpleNamespace(
        case="fwd_s8",
        route=route,
        direction=direction,
        problem=problem,
        request=SimpleNamespace(dtype="bf16"),
        kernel_label=label,
        launches=("relayout", "main", "reduce"),
        pad_copy=None,
        winograd=f"plan[{route}]",
        workspaces=tuple(ws),
        steps=(),
    )


def _bind(plan, rt, shared, sizes=None):
    return RouteBinding(
        plan,
        None,
        rt,
        "gfx942",
        (1, 2, 3),
        operand_nbytes(plan) if sizes is None else sizes,
        workspaces=shared,
        B=object(),
    )


def test_same_plan_shares_buffers():
    rt, shared = _Rt(), SharedWorkspaces()
    plan = _plan(RouteWorkspace("P", 64), RouteWorkspace("X8", 32, zero_once=True))
    a = _bind(plan, rt, shared)
    b = _bind(plan, rt, shared)
    assert a.workspaces["P"] == b.workspaces["P"]
    assert len(rt.allocs) == 2 and len(rt.memsets) == 1
    assert shared.compatible(plan)


def test_larger_same_name_workspace_is_refused():
    # A variant whose P is larger (e.g. a deeper Winograd split) must never be
    # bound to the base's smaller P.
    rt, shared = _Rt(), SharedWorkspaces()
    base = _plan(RouteWorkspace("P", 64))
    big = _plan(RouteWorkspace("P", 128), label="ksplit8")
    _bind(base, rt, shared)
    assert not shared.compatible(big)
    with pytest.raises(ValueError, match="route workspace P"):
        _bind(big, rt, shared)
    # Its own SharedWorkspaces gets a buffer of its own size.
    own = SharedWorkspaces()
    _bind(big, rt, own)
    assert own.meta["P"][0] == 128


def test_smaller_plain_workspace_reuses_larger_buffer():
    rt, shared = _Rt(), SharedWorkspaces()
    _bind(_plan(RouteWorkspace("P", 128)), rt, shared)
    small = _plan(RouteWorkspace("P", 64), label="other")
    assert shared.compatible(small)
    _bind(small, rt, shared)
    assert len(rt.allocs) == 1


def test_zero_once_is_not_shared_across_routes():
    rt, shared = _Rt(), SharedWorkspaces()
    _bind(
        _plan(RouteWorkspace("Xp", 64, zero_once=True), route="padcopy_halo"),
        rt,
        shared,
    )
    other = _plan(
        RouteWorkspace("Xp", 64, zero_once=True), route="padcopy_halo", label="k2"
    )
    assert not shared.compatible(other)
    with pytest.raises(ValueError, match="zero_once"):
        _bind(other, rt, shared)
    flip = _plan(RouteWorkspace("Xp", 64), route="padcopy_halo")
    assert not shared.compatible(flip)


def test_plain_dict_is_refused():
    with pytest.raises(TypeError):
        _bind(_plan(RouteWorkspace("P", 64)), _Rt(), {})


def test_shipped_route_with_deeper_split_gets_its_own_workspaces(tmp_path):
    # The shipped deterministic fwd_s8 Winograd route beside a copy of it with
    # a deeper split: same workspace names, a larger P -- not shareable.
    import copy
    import dataclasses
    import json

    from benchmarks.common.conv_step_replay import load_route_cases
    from dispatch.conv_routes import shipped_conv_route_tables

    path = shipped_conv_route_tables()["gfx942_unet3d_bf16"]
    doc = json.loads(path.read_text())
    row = doc["modes"]["deterministic"]["cases"].get("fwd_s8_pad1")
    if row is None or "winograd" not in row:
        pytest.skip("no fwd_s8_pad1 Winograd route in the shipped table")
    var = copy.deepcopy(doc)
    var["modes"]["deterministic"]["cases"]["fwd_s8_pad1"]["winograd"]["spec"][
        "ksplit"
    ] = 8
    vpath = tmp_path / "variant.json"
    vpath.write_text(json.dumps(var))
    base = load_route_cases(f"{path}#deterministic@pad1", "gfx942")["fwd_s8"]
    deep = load_route_cases(f"{vpath}#deterministic@pad1", "gfx942")["fwd_s8"]
    bp, dp = base["route_plan"], deep["route_plan"]
    bsz = {w.name: w.nbytes for w in bp.workspaces}
    assert any(w.nbytes > bsz.get(w.name, 0) for w in dp.workspaces)
    # Bind the plans' workspaces only (their launch steps need a kernel cache).
    rt, shared = _Rt(), SharedWorkspaces()
    bp0, dp0 = (dataclasses.replace(p, steps=()) for p in (bp, dp))
    _bind(bp0, rt, shared)
    assert shared.compatible(bp0)
    assert not shared.compatible(dp0)
    with pytest.raises(ValueError, match="route workspace"):
        _bind(dp0, rt, shared)
    # The shipped problem's own operand sizes: NDHWC bf16 X / Y, W [K, 3, 3, 3, C].
    p = bp.problem
    x, y = 2 * p.N * p.Di * p.Hi * p.Wi * p.C, 2 * p.N * p.Do * p.Ho * p.Wo * p.K
    assert operand_nbytes(bp) == (x, 2 * p.K * 27 * p.C, y)
    with pytest.raises(ValueError, match="operand size mismatch"):
        _bind(bp0, rt, SharedWorkspaces(), sizes=(x, 2 * p.K * 27 * p.C, y - 2))
    # The same route resolved again (another load of the table) is shareable.
    again = load_route_cases(f"{path}#deterministic@pad1", "gfx942")["fwd_s8"]
    assert shared.compatible(dataclasses.replace(again["route_plan"], steps=()))


@pytest.mark.parametrize(
    "direction, sizes",
    [("fwd", (_X, _W, _Y)), ("dgrad", (_Y, _W, _X)), ("wgrad", (_Y, _X, _W))],
)
def test_operand_sizes_follow_the_direction(direction, sizes):
    assert operand_nbytes(_plan(direction=direction)) == sizes


def test_grouped_weight_is_per_group_channels():
    import dataclasses

    g = dataclasses.replace(_PROBLEM, groups=4)
    assert operand_nbytes(_plan(problem=g))[1] == _W // 4


@pytest.mark.parametrize(
    "direction, sizes, operand",
    [
        ("fwd", (_X - 2, _W, _Y), "A (X)"),  # undersized input
        ("fwd", (_X, _W, _Y // 2), "D (Y)"),  # undersized output
        ("fwd", (_X, _W + 2, _Y), "B (W)"),  # oversized weight
        ("dgrad", (_X, _W, _Y), "A (dY)"),  # fwd sizes for a dgrad call
        ("wgrad", (_Y, _X, _Y), "D (dW)"),  # output of another tensor
    ],
)
def test_mismatched_operand_size_is_refused_before_binding(direction, sizes, operand):
    rt = _Rt()
    plan = _plan(RouteWorkspace("P", 64, zero_once=True), direction=direction)
    with pytest.raises(
        ValueError, match=rf"operand size mismatch: .*{re.escape(operand)} is"
    ):
        _bind(plan, rt, SharedWorkspaces(), sizes=sizes)
    # Refused before any workspace is allocated or zeroed.
    assert rt.allocs == [] and rt.memsets == []


def test_operand_sizes_must_be_three():
    with pytest.raises(ValueError, match="sizes takes"):
        _bind(_plan(), _Rt(), SharedWorkspaces(), sizes=(_X, _W))
