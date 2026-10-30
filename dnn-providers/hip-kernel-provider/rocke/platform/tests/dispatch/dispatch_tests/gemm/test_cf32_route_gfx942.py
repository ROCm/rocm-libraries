# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side contract of the gfx942 bodies of the emulated-f32 GEMM routes
(``rocke.dispatch.gemm.cf32``):

* the named register-staged bodies (``G<M>x<N>x<K><variant>``) and the body
  sets of the routes; every body of a set is valid with the route's prologue
  (split mode included), batched and in its 64-bit-address form;
* the "cost" body rule (:func:`gemm_body_cost`, :func:`gemm_bodies_by_cost`)
  and its pinned orders (hipTensor's planner computes the same orders from its
  kernel pack's route table: its test ``RockeGemmBodyRule.Gfx942CostOrder``);
* the x6 recipe's native route rule (:func:`native_first`);
* the launch limits (:func:`gemm_launch_limits`) agree with
  ``launch_shape_error`` / ``batched_launch_error``, and the launch
  adjustments of the register-staged bodies (padding, ``wide_addr``);
* the f32 / cf32 plans of gfx942 build.

No device needed (``build_universal_gemm`` lowers to IR only).
"""

from __future__ import annotations

import re

import pytest

from rocke.dispatch.gemm import cf32
from rocke.dispatch.gemm.cf32 import (
    CF32_BODY_RULE,
    CF32_ROUTE_ALIAS,
    CF32_ROUTE_BODIES,
    _adjust_k,
    _vp_needs_wide,
    cf32_body,
    cf32_body_for,
    f32_body_for,
    gemm_bodies_by_cost,
    gemm_body_cost,
    gemm_launch_limits,
    native_first,
    plan_cf32_gemm,
    plan_f32_gemm,
    route_bodies,
    split_mode,
    wide_body,
)
from rocke.instances.common.gemm_prologue import Prologue, attach_prologue
from rocke.instances.common.gemm_universal import (
    DTL_BUFFER_NUM_RECORDS,
    DataSpec,
    UniversalGemmSpec,
    batched_launch_error,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
)

ARCH = "gfx942"
ROUTES = ("h16", "x6", "cf32", "x3", "x4", "x9", "cf32x3", "cf32x4", "cf32x9")


def _pro(route):
    if route == "h16":
        return dict(src="same", recipe="cvt", out_f32=True)
    cplx = route.startswith("cf32")
    recipe = (route[len("cf32") :] or "x6") if cplx else route
    d = dict(src="f32", recipe=recipe, out_f32=True, split_at="use", psub=True)
    if cplx:
        d["cplx_b"] = True
    if split_mode(ARCH, route) != "rne":
        d["split_mode"] = split_mode(ARCH, route)
    return d


def _spec(body, route="x6", batched=False, wide=False):
    tile, tr, bs = cf32_body(body, ARCH)
    dt = "fp16" if route == "h16" else "bf16"
    sp = UniversalGemmSpec(
        name=f"t_{body.lower()}",
        tile=tile,
        trait=tr,
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
        block_size=bs,
        batched=batched,
    )
    if wide:
        sp = _adjust_k(sp, tile.tile_m, tile.tile_n, True)
    attach_prologue(sp, Prologue(**_pro(route)))
    return sp


def test_body_names_are_their_traits():
    """A name says the tile, the waves layout is the body's, and the variant letters say the
    levers: L3 / L1 the main-loop levers with that release window, U vp_unroll, S3
    lds_swizzle_l 3."""
    for name in cf32._VP_BODIES:
        tile, tr, bs = cf32_body(name, ARCH)
        m = re.fullmatch(r"G(\d+)x(\d+)x(\d+)(L[13])?(U)?(S3)?", name)
        assert m is not None, name
        var = "".join(g or "" for g in m.groups()[3:])
        assert (tile.tile_m, tile.tile_n, tile.tile_k) == tuple(
            int(g) for g in m.groups()[:3]
        )
        assert (
            tr.vgpr_prefetch == 4 and not tr.ab_double_buffer and tr.lds_swizzle_r == 0
        )
        assert (tr.vp_pair_reads == 1) == ("L" in var), name
        assert tr.vp_release_atoms == (3 if "L3" in var else 1), name
        assert tr.vp_unroll == (1 if "U" in var else 0), name
        assert tr.lds_swizzle_l == (3 if "S3" in var else -1), name
        assert tr.acc_agpr == (tile.tile_m == 256), name
        assert bs == 64 * tile.warp_m * tile.warp_n
        assert cf32_body(name, "gfx950") == (tile, tr, bs)  # target independent
    with pytest.raises(ValueError):
        cf32_body("G48x48x48", ARCH)


def test_route_sets_and_rules():
    assert CF32_BODY_RULE == {"gfx950": "waves", "gfx942": "cost"}
    sets = CF32_ROUTE_BODIES[ARCH]
    assert set(sets) == {"h16", "x6", "cf32"}
    for route in ROUTES:
        bodies = route_bodies(ARCH, route)
        assert bodies == sets[CF32_ROUTE_ALIAS.get(route, route)]
        assert len(set(bodies)) == len(bodies) and set(bodies) <= set(cf32._VP_BODIES)
        assert wide_body(ARCH, route) in bodies
        for b in bodies:
            assert cf32.gemm_body_coef(b, route) is not None, (route, b)
    assert wide_body("gfx950", "h16") == wide_body("gfx950", "x6") == "RIW"
    # the split: truncated parts for x6 / x9 (three parts hold an f32), rounded for x3 / x4
    assert [split_mode(ARCH, r) for r in ROUTES[1:]] == [
        "trunc",
        "trunc",
        "rne",
        "rne",
        "trunc",
        "rne",
        "rne",
        "trunc",
    ]
    assert all(split_mode("gfx950", r) == "rne" for r in ROUTES)


@pytest.mark.parametrize("route", ROUTES)
def test_every_pack_form_is_valid(route):
    """Every body of the route, non-batched and batched, and the 64-bit-address body, with the
    route's prologue (as hipTensor's kernel pack builds them)."""
    for b in route_bodies(ARCH, route):
        for batched in (False, True):
            ok, why = is_valid_spec(_spec(b, route, batched=batched), arch=ARCH)
            assert ok, (route, b, batched, why)
    w = _spec(wide_body(ARCH, route), route, wide=True)
    assert w.trait.wide_addr and is_valid_spec(w, arch=ARCH)[0]


def test_pair_view_needs_a_short_release_window():
    """A 64-deep K tile has two K atoms in the f32 pair view: the release window of 3 atoms
    (L3) is refused there, so the f32 routes take the L1 variant."""
    ok, why = is_valid_spec(_spec("G256x224x64L3", "x6"), arch=ARCH)
    assert not ok and "vp_release_atoms" in why
    assert is_valid_spec(_spec("G256x224x64L1", "x6"), arch=ARCH)[0]
    assert is_valid_spec(_spec("G256x224x64L3", "h16"), arch=ARCH)[0]


def test_cost_formula():
    # 1000 x 3000 (kernel N) on G128x128x128L3, K 777, 2 batches, 100 CUs
    a, b, occ = cf32.CF32_COST["x6"]["G128x128x128L3"]
    tiles = 8 * 24 * 2
    want = a + b * -(-tiles // (100 * occ)) * 777
    got = gemm_body_cost(
        "G128x128x128L3", 1000, 3000, 777, route="x9", batch=2, cus=100
    )
    assert got == want  # x9 on x6's coefficients
    assert gemm_body_cost("RIW", 1000, 3000, 777, cus=100) == float("inf")


def test_cost_ties_by_name(monkeypatch):
    same = {b: (1.0, 0.0, 1) for b in cf32._VP_BODIES}
    monkeypatch.setitem(cf32.CF32_COST, "x6", same)
    assert gemm_bodies_by_cost(64, 64, 64, cus=304) == sorted(route_bodies(ARCH, "x6"))


# The orders hipTensor's test RockeGemmBodyRule.Gfx942CostOrder expects (its planner, the
# coefficients of the gfx942 pack's route table): (CUs, route, M, N, K, bodies). When CF32_COST
# changes, regenerate this table and hipTensor's together.
_PINNED = [
    (
        304,
        "h16",
        64,
        64,
        64,
        "G64x32x128U G64x64x128L3 G32x32x128U G128x128x128L3 G192x128x64L3 "
        "G256x224x64L3 G256x256x64L3",
    ),
    (
        304,
        "h16",
        4096,
        4096,
        4096,
        "G256x224x64L3 G256x256x64L3 G128x128x128L3 G192x128x64L3 G64x64x128L3 "
        "G64x32x128U G32x32x128U",
    ),
    (
        304,
        "x6",
        64,
        64,
        64,
        "G128x64x64 G64x64x128L3 G32x32x128L3 G128x128x128L3 G64x32x128U "
        "G192x128x64L1S3 G256x224x64L1",
    ),
    (
        304,
        "x6",
        4096,
        4096,
        4096,
        "G256x224x64L1 G128x128x128L3 G192x128x64L1S3 G128x64x64 G64x64x128L3 "
        "G64x32x128U G32x32x128L3",
    ),
    (
        304,
        "cf32",
        4096,
        4096,
        4096,
        "G256x224x64L1 G192x128x64US3 G128x128x128L3 G128x64x64U G64x64x128L3 "
        "G64x32x128U G32x32x128L3",
    ),
    (
        228,
        "x6",
        4096,
        4096,
        4096,
        "G128x128x128L3 G256x224x64L1 G128x64x64 G192x128x64L1S3 G64x64x128L3 "
        "G64x32x128U G32x32x128L3",
    ),
    (
        228,
        "cf32",
        64,
        64,
        64,
        "G32x32x128L3 G64x32x128U G64x64x128L3 G128x64x64U G128x128x128L3 "
        "G192x128x64US3 G256x224x64L1",
    ),
]


@pytest.mark.parametrize("cus,route,M,N,K,want", _PINNED)
def test_cost_rule_pinned_orders(cus, route, M, N, K, want):
    Nk, Kk = (2 * N, 2 * K) if route == "cf32" else (N, K)
    got = gemm_bodies_by_cost(M, Nk, Kk, route=route, cus=cus)
    assert " ".join(got) == want, (
        "the cost rule's order changed: regenerate this table and hipTensor's "
        "RockeGemmBodyRule.Gfx942CostOrder"
    )


def test_native_first():
    assert native_first(512, 512, 512, cplx=False, arch=ARCH) is True
    assert native_first(4096, 4096, 4096, cplx=False, arch=ARCH) is False
    assert native_first(64, 64, 64, cplx=True, arch=ARCH) is True
    assert native_first(512, 512, 512, cplx=True, arch=ARCH) is False
    assert native_first(64, 64, 64, cplx=False, arch="gfx950") is None
    # the whole contraction's multiply-adds: L batches count
    assert native_first(512, 512, 512, cplx=False, arch=ARCH, batch=256) is False
    assert native_first(64, 64, 64, cplx=True, arch=ARCH, batch=16) is True
    assert native_first(64, 64, 64, cplx=True, arch=ARCH, batch=64) is False


def test_body_for_follows_the_rule():
    for M, N, K in ((64, 64, 64), (4096, 4096, 4096), (100, 30000, 8000)):
        assert f32_body_for(M, N, K, arch=ARCH) == gemm_bodies_by_cost(M, N, K)[0]
        assert f32_body_for(M, N, K, arch=ARCH, recipe="x3") == (
            gemm_bodies_by_cost(M, N, K, route="x3")[0]
        )
        assert cf32_body_for(M, N, K, arch=ARCH) == (
            gemm_bodies_by_cost(M, 2 * N, 2 * K, route="cf32")[0]
        )
    # operands past the 32-bit offsets: the 64-bit-address body
    assert f32_body_for(65536, 64, 16384, arch=ARCH) == wide_body(ARCH, "x6")


def test_needs_wide_boundaries():
    # 128 x 128 x 128: operand bytes max(M, N rows) * Kh * 2 < 2^32 (2^31 when ragged)
    b = "G128x128x128L3"
    assert not _vp_needs_wide(b, 1 << 15, 128, (1 << 16) - 128, 128)
    assert _vp_needs_wide(b, 1 << 15, 128, 1 << 16, 128)
    assert _vp_needs_wide(b, (1 << 15) + 1, 128, 1 << 15, 128)  # ragged: 2^31
    assert _vp_needs_wide(b, 1 << 16, (1 << 15) + 128, 128, 128)  # D past 2^31
    assert not _vp_needs_wide(b, 1 << 16, 1 << 15, 128, 128)


def test_adjust_register_staged():
    sp = _spec("G128x64x64")
    ragged = _adjust_k(sp, 1000, 1000, False)
    assert ragged.trait.pad_m and ragged.trait.pad_n and not ragged.trait.wide_addr
    w = _adjust_k(sp, 128, 64, True)
    assert w.trait.wide_addr and not (w.trait.pad_m or w.trait.pad_n)
    # no DMA-ring levers on the register-staged body
    for t in (ragged.trait, w.trait):
        assert not (t.dtl_soff0 or t.dtl_tight_rsrc or t.direct_to_lds)


@pytest.mark.parametrize("body", sorted(cf32._VP_BODIES))
def test_launch_limits_are_the_launch_rules(body):
    """The pack's limits (hipTensor declines a launch at them) against the launch rules, at
    extents padded to tile multiples."""
    route = "h16" if body.endswith("L3") and body.startswith("G256") else "x6"
    sp = _spec(body, route)
    lim = gemm_launch_limits(sp)
    assert lim == dict(
        limit_operand_bytes=1 << 32,
        limit_c_elements=1 << 31,
        limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
        limit_c_tile_elements=1 << 31,
        limit_grid_yz=65535,
    )
    t = sp.tile
    # A of about 2^20 rows (a tile multiple; B: one tile of rows): operand bytes M * K * 2
    # below 2^32; k_hi: the least K (a tile_k multiple) at the limit
    M, N = t.tile_m * ((1 << 20) // t.tile_m), t.tile_n

    def k_at(limit_halves):
        k = -(-limit_halves // M)
        return -(-k // t.tile_k) * t.tile_k

    k_hi = k_at(lim["limit_operand_bytes"] // 2)
    assert launch_shape_error(sp, M, N, k_hi) is not None
    assert launch_shape_error(sp, M, N, k_hi - t.tile_k) is None
    # batched: all batches' operand bytes below the limit (the i32 batch element offsets)
    bsp = _spec(body, route, batched=True)
    L = 4
    kb = k_at(lim["limit_operand_bytes"] // 2 // L)
    for k, ok in ((kb - t.tile_k, True), (kb, False)):
        err = batched_launch_error(bsp, M, N, k, L, M * k, N * k, M * N)
        assert (err is None) == ok, (k, err)
    # the 64-bit-address form: only one tile's operand rows within the DMA buffer range
    wsp = _spec(body, route, wide=True)
    kt = lim["limit_tile_operand_bytes"] // (2 * t.tile_m)
    kt -= kt % t.tile_k
    assert t.tile_n <= t.tile_m
    assert launch_shape_error(wsp, 4 * t.tile_m, N, kt) is None
    assert launch_shape_error(wsp, 4 * t.tile_m, N, kt + t.tile_k) is not None


def test_ring_limits_unchanged():
    tile, tr, bs = cf32_body("RIW", "gfx950")
    sp = UniversalGemmSpec(
        name="t",
        tile=tile,
        trait=tr,
        data=DataSpec(dtype_a="bf16", dtype_b="bf16", dtype_c="bf16"),
        block_size=bs,
    )
    assert gemm_launch_limits(sp) == dict(
        limit_operand_bytes=DTL_BUFFER_NUM_RECORDS,
        limit_c_elements=(1 << 31) - 1,
        limit_tile_operand_bytes=DTL_BUFFER_NUM_RECORDS,
        limit_c_tile_elements=1 << 31,
        limit_grid_yz=65535,
    )


@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("M,N,K", [(512, 384, 256), (1000, 300, 128), (64, 64, 64)])
def test_plans_build(cplx, M, N, K):
    plan = plan_cf32_gemm if cplx else plan_f32_gemm
    p = plan(M, N, K, arch=ARCH)
    route = "cf32" if cplx else "x6"
    assert p.body in route_bodies(ARCH, route)
    tr = p.spec.trait
    assert tr.vgpr_prefetch == 4 and not tr.wide_addr and p.workspace_bytes == 0
    assert p.spec._prologue.split_mode == "trunc"
    t = p.spec.tile
    Nk, Kh = (2 * N, 4 * K) if cplx else (N, 2 * K)
    assert tr.pad_m == bool(M % t.tile_m) and tr.pad_n == bool(Nk % t.tile_n)
    assert tr.pad_k == bool(Kh % t.tile_k)
    assert p.grid == (-(-Nk // t.tile_n), -(-M // t.tile_m), 1)
    assert launch_shape_error(p.spec, M, Nk, Kh) is None
    assert p.kernel.name.endswith("_smt")


def test_x3_keeps_the_rounded_split():
    p = plan_f32_gemm(512, 512, 512, arch=ARCH, recipe="x3")
    assert p.spec._prologue.split_mode == "rne"
    assert p.body in route_bodies(ARCH, "x3")


def test_wide_plan():
    p = plan_f32_gemm(65536, 64, 16384, arch=ARCH)
    assert p.body == wide_body(ARCH, "x6") and p.spec.trait.wide_addr
    assert launch_shape_error(p.spec, 65536, 64, 2 * 16384) is None
    assert build_universal_gemm(p.spec, ARCH).name
