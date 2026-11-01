# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Candidate routes, autotune mode and the tuned route table of the tensor GETT
product route (CPU only: planning, keys and the search driver; no device).

* :func:`plan_tensor_gett_candidates` lists the cost-model route first, then
  MN-layout routes on the selector's both-MN-contiguous bodies, neighbour
  mutations and K-contiguous routes; every GETT spec is valid and no route
  appears twice.
* :func:`tensor_gett_route_key` ignores mode names but not the problem.
* :func:`tune_tensor_gett` returns the fastest route that passes the check
  (a failing route never wins, whatever its time).
* A :data:`TENSOR_GETT_TUNED` entry routes the product plan to its body; an
  entry that no longer builds falls back to the cost-model route.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import rocke.dispatch.families.tensor_gett as TG
from rocke.dispatch.families.tensor_gett import (
    TENSOR_GETT_TUNED,
    TensorGettRequest,
    plan_tensor_gett,
    plan_tensor_gett_candidates,
    tensor_gett_neighbour_routes,
    tensor_gett_route_key,
    tune_tensor_gett,
)
from rocke.dispatch.gemm.selector import config_string
from rocke.instances.common.tensor import GettKnobs, TensorGettSpec, is_valid_spec

# hipTensor canonical contraction (row-major mode lists, last mode fastest):
# D[N.., M..] = A[K.., M..] B[K.., N..], M / N fastest -> both operands
# MN-contiguous (a canonical contraction at 4608 x 3840 x 576)
_A = ["K1", "K0", "M1", "M0"]
_B = ["K1", "K0", "N1", "N0"]
_D = ["N1", "N0", "M1", "M0"]
_EXT = {"K0": 24, "K1": 24, "M0": 72, "M1": 64, "N0": 60, "N1": 64}


def _req(dtype="bf16", bilinear=False, modes=None, ext=None):
    a, b, d = modes or (_A, _B, _D)
    p = TensorGettSpec.make(
        a, b, d, ext or _EXT, dtype=dtype, knobs=GettKnobs(bilinear=bilinear)
    )
    return TensorGettRequest(problem=p, arch="gfx950")


@pytest.fixture(autouse=True)
def _clean_table():
    saved = dict(TENSOR_GETT_TUNED)
    TENSOR_GETT_TUNED.clear()
    yield
    TENSOR_GETT_TUNED.clear()
    TENSOR_GETT_TUNED.update(saved)


def test_candidates_default_first_valid_and_unique():
    req = _req()
    routes = plan_tensor_gett_candidates(req, mn_top=4, kc_top=2, neighbours=1)
    assert routes[0].origin == "default"
    assert routes[0].plan.gett.spec == plan_tensor_gett(req).gett.spec
    kinds = {r.origin.split(":")[0] for r in routes}
    assert {"default", "mn", "nb0", "kc"} <= kinds
    keys = set()
    for r in routes:
        sp = r.plan.gett.spec
        assert is_valid_spec(sp, arch="gfx950")[0], r.origin
        assert r.route == ("kc" if sp.kc_body else "mn")
        if r.route == "kc":
            assert len(r.plan.packs) == 2
        key = (r.route, repr(sp))
        assert key not in keys, r.origin
        keys.add(key)


def test_neighbour_routes_change_one_lever():
    req = _req()
    body = plan_tensor_gett(req).gett.spec.body()
    nbr = tensor_gett_neighbour_routes(req, [body])
    assert nbr
    for r in nbr:
        lever, val = r.origin.split(":", 1)[1].split("=")
        diff = [
            f
            for f in body.trait.__dataclass_fields__
            if getattr(body.trait, f) != getattr(r.body.trait, f)
        ]
        assert lever in diff, (r.origin, diff)
        # a chiplet_wgm move switches the raster on with it, nothing else changes
        assert set(diff) <= {lever, "chiplet_swizzle"}, (r.origin, diff)
        assert r.body.tile == body.tile


def test_route_key_ignores_mode_names_only():
    ren = {
        "K0": "a",
        "K1": "b",
        "M0": "c",
        "M1": "d",
        "N0": "e",
        "N1": "f",
    }
    modes = tuple([ren[m] for m in ms] for ms in (_A, _B, _D))
    ext = {ren[m]: v for m, v in _EXT.items()}
    assert tensor_gett_route_key(_req()) == tensor_gett_route_key(
        _req(modes=modes, ext=ext)
    )
    k = tensor_gett_route_key(_req())
    assert k != tensor_gett_route_key(_req(dtype="fp16"))
    assert k != tensor_gett_route_key(_req(bilinear=True))
    assert k != tensor_gett_route_key(_req(ext=dict(_EXT, M0=80)))
    # a different D mode order is a different problem
    assert k != tensor_gett_route_key(_req(modes=(_A, _B, ["N1", "M1", "N0", "M0"])))


def test_tune_picks_fastest_passing_route():
    req = _req()
    routes = plan_tensor_gett_candidates(req, mn_top=3, kc_top=1)
    assert len(routes) >= 3
    times = {id(r.plan): 10.0 + i for i, r in enumerate(routes)}
    # the fastest route by time fails the numeric check: it must never win
    bad = routes[-1]
    times[id(bad.plan)] = 1.0
    good = routes[1]
    times[id(good.plan)] = 2.0

    res = tune_tensor_gett(
        req,
        measure=lambda pl, fid: [times[id(pl)]] * 3,
        check=lambda pl: pl is not bad.plan,
        mode="lite",
        routes=routes,
    )
    assert res.route is good
    key, (route, cfg) = res.table_entry(req)
    assert key == tensor_gett_route_key(req)
    assert route == good.route and cfg == config_string(good.body)


def test_tuned_table_routes_the_product_plan():
    req = _req()
    routes = plan_tensor_gett_candidates(req, mn_top=4, kc_top=2)
    dflt = plan_tensor_gett(req)
    other = next(
        r for r in routes if r.plan.gett.spec != dflt.gett.spec and r.route == "mn"
    )
    TENSOR_GETT_TUNED["gfx950"] = {
        tensor_gett_route_key(req): (other.route, config_string(other.body))
    }
    got = plan_tensor_gett(req)
    assert got.gett.spec == other.plan.gett.spec
    assert "tuned route table" in got.explanation
    # other problems and the explicit route policies ignore the entry
    assert plan_tensor_gett(_req(dtype="fp16")).gett.spec.body().trait is not None
    saved = TG.TENSOR_GETT_ROUTE
    try:
        TG.TENSOR_GETT_ROUTE = "mn"
        assert "tuned route table" not in plan_tensor_gett(req).explanation
    finally:
        TG.TENSOR_GETT_ROUTE = saved
    # the knob turns the table off
    TG.TENSOR_GETT_USE_TUNED = False
    try:
        assert plan_tensor_gett(req).gett.spec == dflt.gett.spec
    finally:
        TG.TENSOR_GETT_USE_TUNED = True


def test_tuned_kc_entry_and_stale_entries():
    req = _req()
    routes = plan_tensor_gett_candidates(req, mn_top=0, kc_top=2)
    kc = next(r for r in routes if r.route == "kc")
    key = tensor_gett_route_key(req)
    TENSOR_GETT_TUNED["gfx950"] = {key: ("kc", config_string(kc.body))}
    got = plan_tensor_gett(req)
    assert got.gett.spec == kc.plan.gett.spec and len(got.packs) == 2
    dflt_spec = None
    TENSOR_GETT_TUNED["gfx950"] = {key: ("mn", "not a config")}
    TG.TENSOR_GETT_USE_TUNED = False
    try:
        dflt_spec = plan_tensor_gett(req).gett.spec
    finally:
        TG.TENSOR_GETT_USE_TUNED = True
    assert plan_tensor_gett(req).gett.spec == dflt_spec
    # a body the GETT cannot run (no direct-to-LDS ring) falls back as well
    body = replace(kc.body.trait, direct_to_lds=False, dtl_prefetch=False)
    TENSOR_GETT_TUNED["gfx950"] = {
        key: ("mn", config_string(replace(kc.body, trait=body)))
    }
    assert plan_tensor_gett(req).gett.spec == dflt_spec


def test_shipped_table_entries_route_their_problems():
    """Every shipped tuned entry belongs to the problem stored with it and
    routes that problem's product plan to a valid GETT on the entry's body."""
    TENSOR_GETT_TUNED.update({a: dict(t) for a, t in TG._SHIPPED_TUNED.items()})
    n = 0
    for arch, table in TG._SHIPPED_TUNED.items():
        for key, (route, cfg) in table.items():
            prob = TG.tuned_entry_problem(arch, key)
            assert prob is not None, key
            p = TensorGettSpec.make(
                prob["modes_a"],
                prob["modes_b"],
                prob["modes_d"],
                prob["extents"],
                dtype=prob["dtype"],
                knobs=GettKnobs(bilinear=prob["bilinear"]),
            )
            req = TensorGettRequest(problem=p, arch=arch)
            assert tensor_gett_route_key(req) == key
            got = plan_tensor_gett(req)
            assert "tuned route table" in got.explanation, key
            assert config_string(got.gett.spec.body()) == cfg
            assert (route == "kc") == bool(got.gett.spec.kc_body)
            assert is_valid_spec(got.gett.spec, arch=arch)[0]
            n += 1
    assert n == sum(len(t) for t in TG._SHIPPED_TUNED.values())


def test_pack_kernel_knob():
    """TENSOR_GETT_PACK_KERNEL = "permute_tiled" moves contiguous-source pack
    steps onto the tiled permute (same workspace, params X / Y); strided
    sources keep tensor_pack; the default keeps tensor_pack everywhere."""
    from rocke.instances.common.permute_tiled import PermuteTiledSpec
    from rocke.instances.common.tensor.pack import TensorPackSpec

    req = _req()
    routes = plan_tensor_gett_candidates(req, mn_top=0, kc_top=1)
    kc = next(r for r in routes if r.route == "kc")
    assert all(isinstance(s.spec, TensorPackSpec) for s in kc.plan.packs)
    saved = TG.TENSOR_GETT_PACK_KERNEL
    try:
        TG.TENSOR_GETT_PACK_KERNEL = "permute_tiled"
        pl = TG._plan_kc(req, kc.body)
        assert [s.operand for s in pl.packs] == [s.operand for s in kc.plan.packs]
        for new, old in zip(pl.packs, kc.plan.packs):
            assert isinstance(new.spec, PermuteTiledSpec)
            assert new.workspace_bytes == old.workspace_bytes
            n = 1
            for e in new.spec.x_shape:
                n *= e
            assert 2 * n == old.workspace_bytes
            kd = TG.build_pack_step(new, "gfx950")
            assert [p.name for p in kd.params] == ["X", "Y"]
        # a strided source keeps tensor_pack
        st = {m: s for m, s in zip(_A[::-1], (1, 72, 72 * 64 + 8, (72 * 64 + 8) * 24))}
        p2 = replace(req.problem, strides_a=st)
        pl2 = TG._plan_kc(replace(req, problem=p2), kc.body)
        if pl2 is not None:
            kinds = {s.operand: type(s.spec).__name__ for s in pl2.packs}
            assert kinds.get("A") in (None, "TensorPackSpec"), kinds
    finally:
        TG.TENSOR_GETT_PACK_KERNEL = saved
