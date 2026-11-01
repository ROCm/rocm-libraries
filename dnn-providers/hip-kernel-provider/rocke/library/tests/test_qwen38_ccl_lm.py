# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the CCL-LM (large-message) levers of the in-kernel CCL
(``kernels.gfx950.qwen38_mono.ccl``): knob validation and names (defaults
leave every earlier name and build unchanged), the arena map with wider
units, the lamport / ``ar_dtype`` CPU references, the slot-reuse checker with
lamport resets (and its no-drain mutant), the peer-address lint and the LLVM
text of every new build, and the link probe kernel. The GPU proofs run in
``rocke.examples.gfx950.ccl.bench_ccl`` (``lm_*`` tests) and
``bench_ccl_lm``."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.ccl import collectives as C
from kernels.gfx950.qwen38_mono.ccl import layout as L
from kernels.gfx950.qwen38_mono.ccl import lint as LI
from kernels.gfx950.qwen38_mono.ccl import lm_probe as LP
from kernels.gfx950.qwen38_mono.ccl import ref as RF
from kernels.gfx950.qwen38_mono.ccl import war as WA
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
K = L.CclKnobs


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


LM_KNOBS = [
    K(form="lamport"),
    K(form="lamport", ar="twoshot"),
    K(form="lamport", unit_kib=2),
    K(form="lamport", ar="twoshot", unit_kib=4),
    K(form="lamport", self_local=True),
    K(form="lamport", self_local=True, unit_kib=4),
    K(unit_kib=2),
    K(unit_kib=4, ar="twoshot"),
    K(publish="counted"),
    K(publish="comm_wave"),
    K(sync="grid", gsync="stripe"),
    K(form="lamport", sync="grid", gsync="stripe"),
    K(sync="grid", gsync="stripe", hooks=True, mutant="gsearly"),
    K(sync="grid", diag=True),
    K(form="lamport", sync="grid", gsync="stripe", diag=True, unit_kib=2),
    K(form="lamport", mutant="noreset"),
    K(form="lamport", mutant="nocanon"),
    K(form="lamport", hooks=True),
    K(form="lamport", waves=16),
    K(form="lamport", sync="grid", gsync="stripe", place="table"),
    K(place="table", sync="grid", ar="twoshot", unit_kib=2),
]


# ---------------------------------------------------------------- knobs / names


def test_defaults_unchanged():
    d = K()
    assert d.suffix() == "" and not d.lm_path and not d.diag and d.gsync == "word"
    assert L.AllReduceSpec(2, 4, 2560, "bf16").name() == "q38ccl_ar_n2_r4_c2560_bf16"
    assert L.stamp_cols(d) == 3
    lay = L.make_layout([L.AllReduceSpec(2, 4, 2560, "bf16")])
    assert all(e.unit_bytes == L.UNIT_BYTES for e in lay.exchanges)


@pytest.mark.parametrize(
    "kw,token",
    [(dict(form="lamport"), "lamport"), (dict(unit_kib=2), "u2"), (dict(publish="counted"), "pubcnt"), (dict(publish="comm_wave"), "pubcw"),
     (dict(form="lamport", self_local=True), "selfl"), (dict(sync="grid", gsync="stripe"), "gsstripe"), (dict(sync="grid", diag=True), "diag")],
)
def test_names(kw, token):
    sp = L.AllReduceSpec(2, 16, 2560, "bf16", K(**kw))
    assert token in sp.name().split("_")
    sp2 = L.AllReduceSpec(2, 16, 2560, "f32", K(ar_dtype="bf16"))
    assert "wbf16" in sp2.name()


@pytest.mark.parametrize(
    "kw",
    [dict(form="lamport", homing="dual"), dict(form="lamport", mutant="ge"), dict(form="lamport", poll_cpol="sc1"), dict(gsync="stripe"),
     dict(form="pair", unit_kib=2), dict(publish="counted", ar="twoshot"), dict(self_local=True), dict(mutant="noreset"),
     dict(mutant="gsearly"), dict(unit_kib=3), dict(publish="x"), dict(form="pair", ar_dtype="bf16"), dict(place="x"),
     dict(form="pair", place="table"), dict(publish="counted", place="table")],
)
def test_invalid_knobs(kw):
    with pytest.raises(ValueError):
        K(**kw)


def test_invalid_specs():
    with pytest.raises(ValueError):
        L.AllReduceSpec(2, 4, 2560, "bf16", K(ar_dtype="bf16"))  # bf16 data
    with pytest.raises(ValueError):
        L.AllReduceSpec(2, 1, 2560, "bf16", K(form="lamport", unit_kib=2))  # 5 KiB is not a whole number of 2 KiB units
    with pytest.raises(ValueError):
        L.AllGatherSpec(2, 4, 1280, "bf16", K(form="lamport"))
    with pytest.raises(ValueError):
        L.AllReduceSpec(8, 128, 10240, "bf16", K(publish="comm_wave"))  # 8 x 8 flag lanes ok, but units > grid x waves


def test_wire_dtype_and_units():
    sp = L.AllReduceSpec(2, 32, 2560, "f32", K(ar_dtype="bf16", unit_kib=2))
    assert sp.wire_dtype == "bf16" and sp.payload_dwords == 32 * 2560 // 2 and sp.in_dwords == 32 * 2560
    assert sp.unit_dwords == 512 and sp.units == sp.payload_dwords // 512


# ---------------------------------------------------------------- layout


@pytest.mark.parametrize("kn", [K(form="lamport", unit_kib=4), K(unit_kib=2, homing="dual"), K(form="lamport", ar="twoshot", unit_kib=2)])
def test_layout_wide_units(kn):
    sp = L.AllReduceSpec(4, 128, 10240, "bf16", kn)
    lay = L.make_layout([sp])
    lay.validate()
    for e in lay.exchanges:
        assert e.unit_bytes == 1024 * kn.unit_kib
        assert e.payload_off(e.slots - 1, e.srcs - 1, e.units - 1) + e.unit_bytes == e.payload + e.payload_bytes
        assert (e.flags >= 0) == (kn.form == "line")


# ---------------------------------------------------------------- references


def test_ref_ar_dtype_bf16():
    rng = np.random.default_rng(3)
    xs = [rng.standard_normal(64).astype(np.float32) for _ in range(3)]
    sc = RF.chain_scales(3)
    out, nxt = RF.allreduce_chain(xs, "f32", 2, sc, wire="bf16")
    x = xs
    for _ in range(2):
        s = RF.rank_sum([RF.bf16_to_f32(RF.f32_to_bf16(v)) for v in x])
        x = [(s * np.float32(c)).astype(np.float32) for c in sc]
    assert np.array_equal(out.view(np.uint32), s.view(np.uint32))
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(nxt, x))
    # the default wire reproduces the earlier reference bit for bit
    o1, _ = RF.allreduce_chain(xs, "f32", 2, sc)
    o2, _ = RF.allreduce_chain(xs, "f32", 2, sc, wire="f32")
    assert np.array_equal(o1.view(np.uint32), o2.view(np.uint32))


def test_chain_scales_growth_default_unchanged():
    # growth = 1 is W3's chain bit for bit (the kernarg values of every earlier run)
    for n in (1, 2, 3, 4, 8):
        t = n * (n + 1) / 2
        assert RF.chain_scales(n) == [float(np.float32((r + 1) / t)) for r in range(n)]
        assert RF.chain_scales(n, 1.0) == RF.chain_scales(n)
        assert abs(sum(RF.chain_scales(n, RF.LM_GROWTH)) - RF.LM_GROWTH) < 1e-6


def _stale_chain(xs, dtype, iters, scales, lag):
    """The chain a build produces when every collective from the (lag)th on
    sums the partials of collective k - lag instead of k (a stale-slot read)."""
    hist = [[np.asarray(v) for v in xs]]
    out = None
    for k in range(iters):
        src = hist[k - lag] if k >= lag else hist[k]
        s = RF.rank_sum([RF.wire_to_f32(v, dtype) for v in src])
        out = s
        hist.append([RF.f32_to_wire((s * np.float32(c)).astype(np.float32), dtype) for c in scales])
    return out


@pytest.mark.parametrize("dtype", ["bf16", "f32"])
@pytest.mark.parametrize("ranks", [2, 4])
def test_contracting_chain_hides_stale_reads_growth_does_not(dtype, ranks):
    # why the CCL-LM gates use LM_GROWTH: with W3's scales the 96-collective chain reaches a bitwise fixed point, so a
    # build that reads the previous collective's partial matches the reference; with growth it never does
    rng = np.random.default_rng(11 + ranks)
    xs = [RF.f32_to_wire(rng.standard_normal(512).astype(np.float32), dtype) for _ in range(ranks)]
    for g, hidden in ((1.0, True), (RF.LM_GROWTH, False)):
        sc = RF.chain_scales(ranks, g)
        want, _ = RF.allreduce_chain(xs, dtype, 96, sc)
        for lag in (1, 2):
            got = _stale_chain(xs, dtype, 96, sc, lag)
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)) == hidden, (g, lag)
    # and the grown chain stays finite at the timed length
    want, _ = RF.allreduce_chain(xs, dtype, 96, RF.chain_scales(ranks, RF.LM_GROWTH))
    assert np.isfinite(want).all()


def test_ref_lamport_canon():
    x = np.array([0xFFFF, 0x3F80, 0x7FC0], np.uint16)
    out, _ = RF.allreduce_chain([x, np.array([0x3F80, 0x3F80, 0x3F80], np.uint16)], "bf16", 1, [1.0, 1.0], canon=True)
    assert np.isnan(out[0]) and out[1] == 2.0 and np.isnan(out[2])
    f = np.array([0xFFFFFFFF, 0x3F800000], np.uint32).view(np.float32)
    out, _ = RF.allreduce_chain([f, f.copy()], "f32", 1, [1.0, 1.0], canon=True)
    assert np.isnan(out[0]) and out[1] == 2.0


# ---------------------------------------------------------------- slot reuse


@pytest.mark.parametrize("slots", [2, 3])
def test_war_lamport_resets(slots):
    assert WA.check(WA.oneshot_schedule(2, 4, 8, 4, 2, slots, lamport=True)) == []
    assert WA.check(WA.twoshot_schedule(4, 4, 8, 4, 2, slots, lamport=True)) == []
    # without the drain before the next push, the reset store is not ordered before the peer's next push
    assert WA.check(WA.oneshot_schedule(2, 4, 8, 4, 2, slots, lamport=True, reset_unordered=True))
    assert WA.check(WA.twoshot_schedule(4, 4, 8, 4, 2, slots, lamport=True, reset_unordered=True))


@pytest.mark.parametrize("kn", [K(form="lamport"), K(form="lamport", ar="twoshot", unit_kib=2)])
def test_war_spec_lamport(kn):
    sp = L.AllReduceSpec(2, 17 if kn.unit_kib == 1 else 16, 2560, "bf16", kn)
    assert WA.check(WA.spec_schedule(sp, waves_total=8)) == []


# ---------------------------------------------------------------- builds


@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("kn", LM_KNOBS)
def test_lm_build_lint_and_ir(ranks, kn):
    rows = 128 if kn.publish != "comm_wave" else 16
    sp = L.AllReduceSpec(ranks, rows, 2560, "bf16", kn)
    k = C.build_allreduce_kernel(sp)
    assert C.backend_for(k) == "python"
    assert LI.lint_peer_access(k) == []
    names = [p.name for p in k.params]
    assert ("gsb" in names) == (kn.gsync == "stripe")
    ll = _ll(k)
    if kn.form == "lamport":
        assert "v_bfi_b32" in ll  # the lamport poll's per-half empty test
        assert ll.count("global_store_dwordx4") >= 2
        assert "{v[96:99]}" in ll or "{v[64:67]}" in ll  # fixed poll registers
    if kn.diag:
        assert re.search(rf"mul nsw i32 %\w+, {L.stamp_cols(kn)}\b", ll)  # the wider stamps row


def test_lm_ar_dtype_build():
    sp = L.AllReduceSpec(2, 32, 10240, "f32", K(ar_dtype="bf16", form="lamport", ar="twoshot"))
    k = C.build_allreduce_kernel(sp)
    assert LI.lint_peer_access(k) == []
    assert "bfloat" in _ll(k)


@pytest.mark.parametrize("mode,nt", [("push", False), ("push", True), ("pull", False)])
def test_link_probe(mode, nt):
    k = LP.build_linkbw_kernel(LP.LinkBwSpec(2, mode, nt, 8, 4))
    assert LI.lint_peer_access(k) == []
    ll = _ll(k)
    assert ("global_store_dwordx4" in ll) == (mode == "push")
    if nt:
        assert "sc0 sc1 nt" in ll
    with pytest.raises(ValueError):
        LP.LinkBwSpec(2, "pull", True)


@pytest.mark.parametrize("units,nctas,waves", [(640, 80, 8), (2560, 256, 8), (85, 16, 8), (5, 8, 16), (160, 20, 8), (80, 10, 8)])
def test_xcd_table(units, nctas, waves):
    for shares in (None, [1, 1, 1.3, 1.3, 1.6, 1.3, 1.1, 1], [0, 1, 0, 0, 0, 0, 0, 0]):
        tab = L.xcd_table(units, nctas, waves, shares)
        assert len(tab) == nctas * waves + 1 and tab[0] == 0 and tab[-1] == units
        d = np.diff(tab)
        assert (d >= 0).all()
        per_cta = d.reshape(nctas, waves).sum(axis=1)
        per_xcd = np.bincount(np.arange(nctas) % 8, weights=per_cta, minlength=8)
        assert per_xcd.sum() == units
        if nctas % 8 == 0:
            sh = np.full(8, 1 / 8) if shares is None else np.asarray(shares, float) / np.sum(shares)
            assert np.abs(per_xcd - units * sh).max() <= 1.0 + 1e-9
    tab = L.xcd_table(64, 3, 8, [1, 1, 1, 5, 5, 5, 5, 5])  # XCDs 3-7 have no CTA: only 0-2 own units
    assert np.diff(tab).sum() == 64 and len(tab) == 25
    with pytest.raises(ValueError):
        L.xcd_table(64, 8, 8, [0, 0, 0, 0, 0, 0, 0, -1])


def test_place_table_build():
    sp = L.AllReduceSpec(2, 128, 2560, "bf16", K(form="lamport", sync="grid", place="table"))
    assert not sp.reg_carry and "ptable" in sp.name()
    k = C.build_allreduce_kernel(sp)
    assert "utab" in [p.name for p in k.params] and LI.lint_peer_access(k) == []


def test_lint_peer_loaded_data_is_not_an_address():
    from rocke.core.ir import I32, I64, IRBuilder, PtrType

    def k(kind):
        b = IRBuilder(f"lintp_{kind}")
        p = b.param("peer1", I64)
        dst = b.param("dst", PtrType(I32, "global"))
        off = b.const_i32(16)
        if kind == "data":  # a peer read's DATA stored with plain IR: clean
            v = b.inline_asm("global_load_dword $0, $1, $2 sc0 sc1\ns_waitcnt vmcnt(0)", "=&v,v,s", [off, p], I32)
            b.store(v, dst, off)
        elif kind == "nobits":  # the read itself without system scope: caught
            v = b.inline_asm("global_load_dword $0, $1, $2\ns_waitcnt vmcnt(0)", "=&v,v,s", [off, p], I32)
            b.store(v, dst, off)
        else:  # a peer address through a non-load asm block stays tainted: caught
            v = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [b.trunc(p, I32)], I32)
            b.global_store_i32(dst, v, v) if hasattr(b, "global_store_i32") else b.store(off, dst, v)
        b.ret()
        return b.kernel

    assert LI.lint_peer_access(k("data")) == []
    assert LI.lint_peer_access(k("nobits"))
    assert LI.lint_peer_access(k("addr"))


def test_pull_builds():
    for kw in (dict(form="pull"), dict(form="pull", sync="grid", gsync="stripe", unit_kib=4), dict(form="pull", mutant="slot"),
               dict(form="pull", place="table", sync="grid")):
        sp = L.AllReduceSpec(2, 128, 10240, "bf16", K(**kw))
        k = C.build_allreduce_kernel(sp)
        assert LI.lint_peer_access(k) == [], sp.name()
        assert "pull" in sp.name()
        ll = _ll(k)
        assert ll.count("global_load_dwordx4") >= 1
    lay = L.make_layout([L.AllReduceSpec(2, 128, 10240, "bf16", K(form="pull"))])
    assert all(e.flags >= 0 for e in lay.exchanges)
    for kw in (dict(form="pull", ar="twoshot"), dict(form="pull", homing="dual"), dict(form="pull", mutant="ge")):
        with pytest.raises(ValueError):
            K(**kw)


def test_ctas_knob_and_lm_select():
    sp = L.AllReduceSpec(2, 128, 10240, "bf16", K(form="pull", unit_kib=4, ctas=48, sync="grid"))
    assert sp.grid() == 48 and "g48" in sp.name() and not sp.reg_carry
    assert L.AllReduceSpec(2, 1, 2560, "bf16", K(ctas=48)).grid() == 1  # never more CTAs than needed
    assert L.AllReduceSpec(2, 128, 2560, "bf16").grid() == 80  # default cap unchanged
    with pytest.raises(ValueError):
        K(ctas=300)
    with pytest.raises(ValueError):
        L.AllGatherSpec(2, 4, 1280, "bf16", K(ctas=8))
    # the declared TP2 selection, by per-rank bytes and dtype
    exp = {(16, 2560, "bf16"): ("lamport", 1, 0, 4, "stripe"), (32, 2560, "bf16"): ("lamport", 1, 0, 8, "stripe"),
           (128, 2560, "bf16"): ("pull", 2, 32, 8, "stripe"), (64, 10240, "bf16"): ("pull", 2, 64, 8, "stripe"),
           (128, 10240, "bf16"): ("pull", 4, 64, 8, "word"), (32, 2560, "f32"): ("lamport", 2, 0, 8, "stripe"),
           (64, 2560, "f32"): ("lamport", 2, 0, 8, "stripe"), (128, 2560, "f32"): ("pull", 4, 32, 8, "stripe"),
           (64, 10240, "f32"): ("pull", 4, 64, 8, "word"), (128, 10240, "f32"): ("pull", 4, 64, 8, "word")}
    for (r, c, d), (form, u, g, w, gs) in exp.items():
        kn = L.auto_allreduce_spec(2, r, c, d, lm=True).knobs
        assert (kn.form, kn.unit_kib, kn.ctas, kn.waves, kn.sync, kn.gsync) == (form, u, g, w, "none", "word"), (r, c, d, kn)
        assert kn.publish == ("counted" if (d, r * c * 4) == ("f32", 1280 << 10) else "unit_end"), (r, c, d, kn)
        k = C.build_allreduce_kernel(L.auto_allreduce_spec(2, r, c, d, lm=True))
        assert LI.lint_peer_access(k) == []
        lk = L.lm_select(2, r * c * L.DTYPES[d], d, layer=True)
        assert (lk.form, lk.unit_kib, lk.ctas, lk.waves, lk.sync, lk.gsync) == (form, u, g, w, "grid", gs), (r, c, d, lk)
        assert LI.lint_peer_access(C.build_allreduce_kernel(L.AllReduceSpec(2, r, c, d, replace(lk, header=False)))) == []
    # the layer shape below 80 KiB / at other TPs: W3's selection with the striped grid sync (what the bench timed)
    assert L.lm_select(2, 8 * 2560 * 2, "bf16", layer=True) == L.select_knobs(2, 8 * 2560 * 2, shape="layer", sync="grid", gsync="stripe")
    # below 80 KiB and at other TPs: the earlier selection, unchanged
    assert L.auto_allreduce_spec(2, 4, 2560, "bf16", lm=True).knobs == L.auto_allreduce_spec(2, 4, 2560, "bf16").knobs
    assert L.auto_allreduce_spec(4, 128, 2560, "bf16", lm=True).knobs == L.auto_allreduce_spec(4, 128, 2560, "bf16").knobs
