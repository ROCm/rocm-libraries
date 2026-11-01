# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L4 (``kernels.gfx950.qwen38_mono.large_batch.moe_lb``): CPU checks.

* knobs: names list only deviations, invalid combinations are refused, derived shapes (intermediate split, chunks,
  units, the ragged TP2 down chunk, LDS table bounds) match the model;
* the unit deal (``unit_plan``, the host model of the in-kernel ``deal``): every gate_up and down unit is dealt
  exactly once, ranges are monotone, the ``line`` map balances weight bytes per CTA, and its 8-step running maximum
  equals the full running maximum for every tile count the kernel can see;
* the kernels: every knob value the bench uses lowers (Python engine); the LLVM text carries what the GPU proofs rely
  on (scaled MFMAs per unit, nt expert loads, sc1 stores of the exchanged data, the null build's zero-length
  resources with the same instruction mix, mutants differ from the base).
The GPU proofs (numerics on the timed binaries, mutants, timing) run in ``hackathon-prep/qwen38/truemono-lb/p-l4``.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono.large_batch import moe_lb as M
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(M.build_moe_kernel(k), arch="gfx950", llvm_flavor=FLAVOR)


# ------------------------------------------------------------------ knobs and shapes


def test_names_list_only_deviations():
    assert M.kernel_name(M.MoeKnobs()) == "q38lb_moe_tp1_fp8_b128"
    assert M.kernel_name(M.MoeKnobs(tp=2, act="fp4", bucket=16)) == "q38lb_moe_tp2_fp4_b16"
    assert M.kernel_name(M.MoeKnobs(depth=2, map="split", tag="rc")) == "q38lb_moe_tp1_fp8_b128_depth2_mapsplit_rc"
    assert M.kernel_name(M.MoeKnobs(sb=False, null="oob")) == "q38lb_moe_tp1_fp8_b128_nulloob_nosb"


@pytest.mark.parametrize(
    "kw",
    [dict(tp=3), dict(act="fp6"), dict(bucket=8), dict(null="zero"), dict(depth=0), dict(depth=7), dict(dt=3),
     dict(dt=8, tp=1), dict(map="rr"), dict(wcp="sc1"), dict(xcp="nt"), dict(mutant="bogus"), dict(combine="x"),
     dict(diag="nothing")],
)
def test_invalid_knobs_refused(kw):
    with pytest.raises(ValueError):
        M.MoeKnobs(**kw).check()


@pytest.mark.parametrize("tp", [1, 2])
def test_shapes(tp):
    k = M.MoeKnobs(tp=tp)
    i = 640 // tp
    assert k.inter == i and k.ngu == i // 32 and k.nqd == -(-i // 128) and k.nrb == M.H // 64
    assert [k.chunk_bytes(c) for c in range(k.nqd)] == ([64] * 5 if tp == 1 else [64, 64, 32])
    d = M.dims(k)
    assert d["gub"] == 2 * i * M.H // 2 and d["dnb"] == M.H * i // 2
    assert d["gus"] == d["gub"] // 16 and d["dns"] == (M.H // 16) * k.nqd * 64
    # one expert's bytes: MXFP4 codes + one E8M0 per 32 (the down scales of the ragged chunk are padded)
    assert d["expert"] == pytest.approx(3 * i * M.H * 0.53125, rel=0.04)


@pytest.mark.parametrize("bucket", [16, 32, 64, 128])
def test_lds_table_bounds(bucket):
    """The tile table holds the worst case: every token's 10 experts distinct across tokens, plus the shared tiles."""
    k = M.MoeKnobs(bucket=bucket)
    pairs = 10 * bucket
    worst = sum(-(-c // 16) for c in [1] * min(512, pairs))  # many single-token experts
    worst = max(worst, -(-pairs // 16) + min(512, pairs))
    assert k.tt_max >= worst + k.tpe
    assert k.tpe <= M.MAX_TPE and bucket <= 256  # tokens fit the 8-bit list field
    assert M.set_words(k) >= M.S_CMB + bucket * M.NRB_MAX * 4 and k.nrb <= M.NRB_MAX
    assert M.counter_words(k) == M.W_SET + 2 * M.set_words(k)


# ------------------------------------------------------------------ unit deal


def _check_plan(k, nt):
    G, Dn = nt * k.ngu, nt * k.nrb
    plan = M.unit_plan(k, nt)
    assert plan[0][0] == 0 and plan[0][2] == 0 and plan[-1][1] == G and plan[-1][3] == Dn
    for (g0, g1, d0, d1), nxt in zip(plan, plan[1:] + [(G, G, Dn, Dn)]):
        assert g0 <= g1 == nxt[0] and d0 <= d1 == nxt[2]
    return plan


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("mp", ["line", "split"])
def test_plan_covers_every_unit_once(tp, mp):
    k = M.MoeKnobs(tp=tp, map=mp)
    for nt in (1, 3, 11, 47, 139, 240, 368, 480, 600):
        _check_plan(k, nt)


def _line_full(k, nt, grid=256):
    """The line map with the full running maximum (moe_fused's rule)."""
    G, Dn = nt * k.ngu, nt * k.nrb
    r = k.gu_weight
    W = r * G + Dn
    f = [t * W // grid - r * (t * G // grid) for t in range(grid)]
    out, best = [], 0
    for t in range(grid):
        best = max(best, f[t])
        out.append(min(max(best, 0), Dn))
    return out + [Dn]


@pytest.mark.parametrize("tp", [1, 2])
def test_line_window_equals_full_running_max(tp):
    k = M.MoeKnobs(tp=tp)
    for nt in list(range(1, 40)) + list(range(40, 640, 7)):
        plan = M.unit_plan(k, nt)
        full = _line_full(k, nt)
        assert [p[2] for p in plan] == full[:256], nt


@pytest.mark.parametrize("tp", [1, 2])
def test_line_balances_bytes(tp):
    k = M.MoeKnobs(tp=tp)
    r = k.gu_weight
    for nt in (139, 240, 480):
        plan = M.unit_plan(k, nt)
        w = [r * (g1 - g0) + (d1 - d0) for g0, g1, d0, d1 in plan]
        mean = sum(w) / len(w)
        assert max(w) - mean <= r + 1  # within one gate_up unit of the mean
        split = M.unit_plan(replace(k, map="split"), nt)
        ws = [r * (g1 - g0) + (d1 - d0) for g0, g1, d0, d1 in split]
        assert max(w) <= max(ws)


# ------------------------------------------------------------------ kernels lower


# mutants that need a particular build: noacq (no acquire after a grid barrier) on the stale-read twin, sopsel on sop
MUT_BASE = {"noacq": dict(hand="gbar", combine="gbar", pretouch=True, drot=128), "sopsel": dict(sop=True)}

ARMS = [
    dict(), dict(tp=2), dict(act="fp4"), dict(tp=2, act="fp4", bucket=16), dict(bucket=32, depth=2),
    dict(bucket=64, depth=5), dict(dt=2), dict(tp=2, dt=8), dict(map="split"), dict(wcp="default"),
    dict(xcp="default"), dict(xcp="sc1"), dict(sb=False), dict(stamp=True), dict(null="oob"), dict(tag="rc"),
    dict(combine="last"), dict(diag="nopoll+nocmb"), dict(diag="nogu"), dict(diag="nodn"),
    # W6 levers (P-L4): grid-barrier hand-off / combine, op_sel scales, gcb, wbal, xu, gap, the hz=0 control
    dict(combine="gbar"), dict(hand="gbar", combine="gbar"), dict(sop=True), dict(sop=True, dsop=True),
    dict(gcb=2, depth=2), dict(wbal=True), dict(hand="gbar", combine="gbar", xu=3), dict(gap=1024), dict(hz=False),
] + [dict(MUT_BASE.get(m, {}), mutant=m) for m in M.MUTANTS[1:]]


@pytest.mark.parametrize("kw", ARMS, ids=lambda kw: ",".join(f"{a}={b}" for a, b in kw.items()) or "default")
def test_lowers(kw):
    k = M.MoeKnobs(**kw)
    ll = _ll(k)
    assert M.kernel_name(k) in ll
    mfma = len(re.findall(r"mfma\.scale\.f32\.16x16x128\.f8f6f4", ll))
    if k.diag == "none":  # diagnosis builds drop unit bodies
        assert mfma >= 4 * M.NQG + k.dt * k.nqd  # one gate_up unit body + one down unit body


def test_expert_loads_nt_and_exchange_sc1():
    ll = _ll(M.MoeKnobs())
    # raw.ptr.buffer.load aux: nt = 2 on the expert weights, sc1 = 16 on the exchanged data (route, h, partials)
    assert re.search(r"buffer\.load\.v4i32\(.*i32 2\)", ll)
    assert re.search(r"buffer\.store\.(v4i32|i32)\(.*i32 16\)", ll)
    ll_def = _ll(M.MoeKnobs(wcp="default"))
    assert len(re.findall(r"i32 2\)", ll_def)) < len(re.findall(r"i32 2\)", ll))


def test_null_build_same_instruction_mix():
    a, b = _ll(M.MoeKnobs(tp=2, bucket=64)), _ll(M.MoeKnobs(tp=2, bucket=64, null="oob"))

    def count(ll, pat):
        return len(re.findall(pat, ll))

    for pat in (r"buffer\.load", r"mfma\.scale", r"buffer\.store"):
        assert count(a, pat) == count(b, pat), pat


@pytest.mark.parametrize("m", M.MUTANTS[1:])
def test_mutants_differ(m):
    b0 = M.MoeKnobs(**MUT_BASE.get(m, {}))
    base = _ll(b0).replace(M.kernel_name(b0), "K")
    mk = replace(b0, mutant=m)
    mut = _ll(mk).replace(M.kernel_name(mk), "K")
    assert base != mut


def test_stream_kernel_lowers():
    ll = _lower_kernel_to_llvm_python(M.build_stream_kernel(M.StreamKnobs()), arch="gfx950", llvm_flavor=FLAVOR)
    assert "q38lb_mst_d4_nt" in ll


_TWIN = dict(hand="gbar", combine="gbar", pretouch=True, drot=128)


@pytest.mark.parametrize("kw", [dict(ptall=True), dict(tdelay=8), dict(_TWIN, pretouch=False, ptall=True),
                                dict(_TWIN, hand="flow", ptall=True), dict(_TWIN, tdelay=257)])
def test_twin_only_knobs_refused(kw):
    """``ptall`` / ``tdelay`` (P-L4 fix round) exist only on the stale-read twin (pretouch; ptall also hand=gbar)."""
    with pytest.raises(ValueError):
        M.MoeKnobs(**kw).check()


@pytest.mark.parametrize("kw", [dict(ptall=True), dict(tdelay=8), dict(ptall=True, tdelay=32, mutant="noacq")])
def test_twin_only_knobs_lower_and_differ(kw):
    tw = M.MoeKnobs(tp=2, act="fp4", bucket=32, **_TWIN)
    k = replace(tw, **kw)
    k.check()
    base = _ll(tw).replace(M.kernel_name(tw), "K")
    ll = _ll(k).replace(M.kernel_name(k), "K")
    assert base != ll
    if k.tdelay:
        assert ll.count('"s_sleep 127"') == 2  # before the gate_up units and after the hand-off barrier
