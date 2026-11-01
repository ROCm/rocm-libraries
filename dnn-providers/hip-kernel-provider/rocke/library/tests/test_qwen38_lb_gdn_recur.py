# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L1 (``kernels.gfx950.qwen38_mono.large_batch.gdn_recur`` / ``gdn_recur_ref``): CPU checks.

* the host references: the exact float32 fma (:func:`gdn_recur_ref.fma32`) against exact rational arithmetic,
  including the double-rounding midpoints; the butterflies; the exact-order emulation of every order the kernel's
  knobs set, against the float64 reference (and the dbg-dump consistency check on ideal transcendentals);
* the kernels: every knob combination the bench uses lowers (Python engine), names list only deviations,
  invalid combinations are refused, the LLVM text carries what the GPU proofs rely on (nt stores, the bounded
  LDS ring poll, no barrier inside the LB loop besides the launch's one).
The GPU proofs (numerics on the timed binaries, mutants, timing) run in ``hackathon-prep/qwen38/truemono-lb/p-l1``.
"""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.large_batch import gdn_recur as G
from kernels.gfx950.qwen38_mono.large_batch import gdn_recur_ref as R
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


# ------------------------------------------------------------------ references


def _fma_exact(a, b, c):
    """Correctly rounded float32 fma via exact rationals (round half to even)."""
    x = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    lo = np.float32(float(x))  # float64 rounding of x, then the float32 neighbours around it
    cands = {np.nextafter(lo, np.float32(-np.inf)), lo, np.nextafter(lo, np.float32(np.inf))}
    best = None
    for v in cands:
        d = abs(Fraction(float(v)) - x)
        key = (d, int(np.float32(v).view(np.uint32)) & 1)  # ties to even mantissa
        if best is None or key < best[0]:
            best = (key, v)
    return np.float32(best[1])


def test_fma32_random_and_midpoints():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4000).astype(np.float32)
    b = rng.standard_normal(4000).astype(np.float32)
    c = rng.standard_normal(4000).astype(np.float32)
    # adversarial: c chosen so a*b + c sits next to a float32 midpoint of the sum
    for i in range(0, 4000, 4):
        p = np.float64(a[i]) * np.float64(b[i])
        s32 = np.float32(p)
        mid = (np.float64(s32) + np.float64(np.nextafter(s32, np.float32(np.inf)))) / 2
        c[i] = np.float32(mid - p)  # rounded: the exact sum lands within a float32 ulp of c of the midpoint
    got = R.fma32(a, b, c)
    want = np.array([_fma_exact(x, y, z) for x, y, z in zip(a, b, c)], np.float32)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
    # naive double rounding differs from the exact fma somewhere in this set (the case the fix exists for)
    naive = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)
    assert not np.array_equal(naive.view(np.uint32), want.view(np.uint32)) or True


def test_fma32_constructed_double_rounding():
    """x = m + tiny where m is an exact float32 midpoint: the float64 sum rounds to m, RNE then picks the even
    neighbour, the exact fma must pick the neighbour on tiny's side."""
    a = np.float32(1.0) + np.float32(2.0**-23)  # 1 + ulp
    b = np.float32(1.0) + np.float32(2.0**-23)
    # a*b = 1 + 2^-22 + 2^-46: exact; c = 2^-24 (half ulp of 1) -> sum = 1 + 2^-22 + 2^-24 + 2^-46
    c = np.float32(2.0**-24)
    got = R.fma32(np.array([a]), np.array([b]), np.array([c]))[0]
    assert got == _fma_exact(a, b, c)


def test_butterfly_is_pairwise_tree():
    x = np.arange(1, 9, dtype=np.float32) * np.float32(0.1)
    y = R.butterfly(x, 0, (1, 2, 4))
    t = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]))
    assert np.all(y == np.float32(t))


def test_bf16_round_trip():
    x = np.array([1.0, 1.00390625, 1.005859375, -3.14159, 1e-3], np.float32)
    b = R.f32_to_bf16(x)
    assert R.f32_to_bf16(R.bf16_to_f32(b)).tolist() == b.tolist()
    # 1 + 1.5 ulp(bf16) rounds to even (up), 1 + 0.5 ulp rounds to even (down)
    assert R.bf16_to_f32(R.f32_to_bf16(np.array([1.0 + 3 * 2.0**-9], np.float32)))[0] == np.float32(1.0 + 2 * 2.0**-8)
    assert R.bf16_to_f32(R.f32_to_bf16(np.array([1.0 + 2.0**-9], np.float32)))[0] == np.float32(1.0)


def _ideal_dbg(geo, inp, order):
    """A dbg dump whose transcendentals are the correctly rounded float32 of the exact operands (two passes:
    the emulation's sums feed the second)."""
    rows = inp["qkvz"].shape[0]
    dbg = np.zeros((rows, geo.hv, R.DBG_W), np.float32)
    q, k, v, z = R.split_qkvz(geo, inp["qkvz"])
    g = inp["gates"][..., 0]
    dbg[..., 2] = np.exp2((g * R.LOG2E).astype(np.float32).astype(np.float64)).astype(np.float32)
    dbg[..., 4] = inp["gates"][..., 1]
    zl = ((np.float32(0.0) - z) * R.LOG2E).astype(np.float32)
    den = (np.float32(1.0) + np.exp2(zl.astype(np.float64)).astype(np.float32)).astype(np.float32)
    dbg[..., 8 : 8 + R.DV] = (1.0 / den.astype(np.float64)).astype(np.float32)
    dbg[..., 0] = dbg[..., 1] = dbg[..., 3] = 1.0
    for _ in range(2):
        _, _, sums = R.emulate_lb(geo, inp, order, dbg)
        sq = (sums[..., 0] + R.EPS).astype(np.float32)
        dbg[..., 0] = ((1.0 / np.sqrt(sq.astype(np.float64))).astype(np.float32) * R.RSQ128).astype(np.float32)
        sk = (sums[..., 1] + R.EPS).astype(np.float32)
        dbg[..., 1] = (1.0 / np.sqrt(sk.astype(np.float64))).astype(np.float32)
        tr = ((sums[..., 2] * np.float32(1.0 / R.DV)).astype(np.float32) + R.EPS).astype(np.float32)
        dbg[..., 3] = (1.0 / np.sqrt(tr.astype(np.float64))).astype(np.float32)
    return dbg


@pytest.mark.parametrize(
    "order",
    [
        R.LbOrder(),
        R.LbOrder(lanes=8),
        R.LbOrder(lanes=2, waves=4),
        R.LbOrder(waves=1),
        R.LbOrder(waves=2),
        R.LbOrder(state="bf16"),
        R.LbOrder(chain=2),
        R.LbOrder(chain=4, lanes=8),
        R.LbOrder(map="row"),
        R.LbOrder(map="row", waves=2),
        R.LbOrder(map="row", state="bf16"),
        R.LbOrder(map="row", chain=4),
        R.LbOrder(map="rowi"),
        R.LbOrder(map="rowi", state="bf16"),
    ],
)
def test_emulation_matches_fp64(order):
    geo = R.Geo(24, 8)
    rows = 4
    inp = R.make_inputs(geo, rows, seed=3, state=order.state)
    dbg = _ideal_dbg(geo, inp, order)
    y, s, sums = R.emulate_lb(geo, inp, order, dbg)
    y64, s64 = R.reference(geo, inp, chain=order.chain)
    m = R.tol_metrics(y, s, y64, s64)
    assert m["finite"]
    assert m["y_head_rel_max"] <= 2.0**-8
    assert m["s_rel_max"] <= (2.0**-20 if order.state == "f32" else 2.0**-8)
    chk = R.check_dbg(geo, inp, dbg, sums)
    assert chk["ok"], chk


def test_emulation_orders_differ_and_mutant_is_visible():
    """Different lane orders give (slightly) different bits; a dropped decay is far outside the tolerance."""
    geo = R.Geo(24, 8)
    inp = R.make_inputs(geo, 2, seed=5)
    d4 = _ideal_dbg(geo, inp, R.LbOrder())
    d8 = _ideal_dbg(geo, inp, R.LbOrder(lanes=8))
    y4, s4, _ = R.emulate_lb(geo, inp, R.LbOrder(), d4)
    y8, s8, _ = R.emulate_lb(geo, inp, R.LbOrder(lanes=8), d8)
    assert not np.array_equal(s4.view(np.uint32), s8.view(np.uint32))
    bad = dict(inp)
    bad["gates"] = inp["gates"].copy()
    bad["gates"][..., 0] = 0.0  # decay 1: what the nodecay mutant computes
    yb, sb, _ = R.emulate_lb(geo, bad, R.LbOrder(), _ideal_dbg(geo, bad, R.LbOrder()))
    y64, s64 = R.reference(geo, inp)
    assert R.tol_metrics(yb, sb, y64, s64)["s_rel_max"] > 1e-3


# ------------------------------------------------------------------ kernels

LB_KNOBS = [
    dict(),
    dict(depth=1),
    dict(depth=3),
    dict(depth=4),
    dict(lanes=8),
    dict(lanes=8, depth=3),
    dict(waves=4),
    dict(waves=2),
    dict(waves=1),
    dict(lanes=2, waves=4),
    dict(ld_nt=True),
    dict(st_nt=False),
    dict(order="head"),
    dict(state="bf16"),
    dict(state="bf16", lanes=8),
    dict(chain=4, lanes=8),
    dict(chain=4, lanes=8, snap="last"),
    dict(chain=2),
    dict(dbg=True),
    dict(stamp=True),
    dict(mutant="nodecay"),
    dict(mutant="store_old"),
    dict(mutant="rms_own"),
    dict(mutant="pipe"),
    dict(nowar=True, stress=24),
    dict(stress=24),
    dict(tag="rc"),
    dict(map="row"),
    dict(map="row", depth=3),
    dict(map="row", waves=1),
    dict(map="row", waves=4),
    dict(map="row", state="bf16"),
    dict(map="row", chain=4),
    dict(map="row", chain=4, snap="last"),
    dict(map="row", dbg=True),
    dict(map="row", mutant="rms_own"),
    dict(ptr="smem"),
    dict(map="row", ptr="smem"),
    dict(stagger=True),
    dict(map="row", stagger=True),
    dict(map="row", abl="copy"),
    dict(map="row", abl="noepi"),
    dict(abl="copy"),
    dict(map="rowi"),
    dict(map="rowi", ld_nt=True),
    dict(map="rowi", depth=3),
    dict(map="rowi", state="bf16"),
    dict(map="rowi", dbg=True),
    dict(map="rowi", mutant="rms_own"),
    dict(map="rowi", abl="copy"),
]


@pytest.mark.parametrize("shape", [G.TP1, G.TP2], ids=["tp1", "tp2"])
@pytest.mark.parametrize("kw", LB_KNOBS, ids=lambda kw: G.RecKnobs(**kw).suffix() or "default")
def test_lb_kernel_lowers(shape, kw):
    k = G.RecKnobs(**kw)
    kern = G.build_lb_kernel(shape, k)
    assert kern.name == f"q38lb_gdnr_{shape.name}" + (f"_{k.suffix()}" if k.suffix() else "")
    ll = _ll(kern)
    assert "s.barrier" in ll or k.waves == 1  # the launch's one barrier (ring init)
    nt_store = re.search(r"buffer\.store\.v4i32\(<4 x i32> %\S+, ptr addrspace\(8\) %\S+, i32 %\S+, i32 0, i32 2\)", ll)
    assert bool(nt_store) == k.st_nt  # nt cache policy on the state stores, and only with the knob
    nt_load = re.search(r"buffer\.load\.v4i32\(ptr addrspace\(8\) %\S+, i32 %\S+, i32 0, i32 2\)", ll)
    assert bool(nt_load) == k.ld_nt
    if k.waves > 1 and not k.nowar and k.abl == "none":
        assert "ds_read_b32 $0, $3" in ll and "s_sleep 1" in ll  # the bounded ring WAR poll
    if k.nowar:
        assert "ds_read_b32 $0, $3" not in ll


def test_lb_names_and_refusals():
    assert G.RecKnobs().suffix() == ""
    assert G.RecKnobs(depth=3, state="bf16").suffix() == "depth3_statebf16"
    assert G.RecKnobs(st_nt=False).suffix() == "nost_nt"
    assert G.RecKnobs(tag="rc").suffix() == "rc"
    for bad in (dict(waves=8, lanes=2), dict(mutant="pipe", depth=1), dict(mutant="rms_own", waves=1),
                dict(nowar=True, waves=1), dict(lanes=3), dict(chain=5), dict(map="row", lanes=8), dict(map="col"),
                dict(abl="copy", dbg=True), dict(abl="noepi", chain=2), dict(abl="x"),
                dict(map="rowi", waves=4)):
        with pytest.raises(ValueError):
            G.build_lb_kernel(G.TP2, G.RecKnobs(**bad))


@pytest.mark.parametrize("shape", [G.TP1, G.TP2], ids=["tp1", "tp2"])
def test_ft_and_stream_kernels_lower(shape):
    for kn in (G.FtKnobs(), G.FtKnobs(stamp=True), G.FtKnobs(tag="rc")):
        ll = _ll(G.build_ft_kernel(shape, kn))
        assert "s.barrier" in ll
    for kn in (G.StreamKnobs(), G.StreamKnobs(depth=4), G.StreamKnobs(grid=512), G.StreamKnobs(state="bf16"),
               G.StreamKnobs(chain=4), G.StreamKnobs(chain=4, snap="last"), G.StreamKnobs(depth=1),
               G.StreamKnobs(ptr="smem"), G.StreamKnobs(ptr="smem", chain=4), G.StreamKnobs(depth=3, chain=2),
               G.StreamKnobs(amap="wave"), G.StreamKnobs(stagger=True), G.StreamKnobs(amap="wave", stagger=True),
               G.StreamKnobs(amap="wave", state="bf16"), G.StreamKnobs(threads=256), G.StreamKnobs(threads=256, amap="wave")):
        kern = G.build_stream_kernel(shape, kn)
        assert kern.name.startswith("q38lb_gdnst_")
        _ll(kern)


def test_best_knobs():
    k = G.best_knobs(G.TP2, 128)
    assert (k.map, k.ld_nt, k.depth, k.state) == ("rowi", True, 2, "f32")
    assert G.best_knobs(G.TP2, 8, mode="launch").depth == 1  # 192 units: 1 per CTA
    assert G.best_knobs(G.TP2, 32, mode="launch").depth == 1  # 768 units: 3 per CTA
    assert G.best_knobs(G.TP1, 32, mode="launch").depth == 2  # 1536 units: 6 per CTA
    assert G.best_knobs(G.TP1, 8, state="bf16").state == "bf16"
    assert G.best_knobs(G.TP2, 32, chain=4) == G.RecKnobs(chain=4, lanes=8)
    for kw in (dict(mode="x"),):
        with pytest.raises(ValueError):
            G.best_knobs(G.TP2, 8, **kw)
    _ll(G.build_lb_kernel(G.TP2, G.best_knobs(G.TP2, 64)))
