# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2f (``large_batch.head_lb`` and ``large_batch.ple_lb``): CPU checks.

* the head subclasses E1 without touching it: an E1 build lowers to the same LLVM IR before and after a head build
  (no module state leaks), and every head build carries E1's GEMM (its MFMAs) plus the argmax epilogue's markers
  (``global_atomic_umax_x2``, the returning ``global_atomic_swap_x2``; TP2: the tagged pair push / poll);
* every head knob (``akey``, ``lout``, ``amrg``, ``xdelay``) and every mutant lowers, names list only deviations, invalid
  combinations refuse, and ``plan`` is valid for every TP x rows 1..128;
* the host key twins order like the device key (``-0 == +0``, NaN above every finite value, ties to the lower row);
* every embed / PLE gather / PLE apply build and mutant lowers; ``hz`` pads inline-asm VMEM with SGPR operands.
GPU proofs (numerics on random and planted data, must-fail mutants, the TP2 stale twin, shuffled timing, FlyTape
brackets) run in ``hackathon-prep/qwen38/truemono-lb/wp-l2f``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.large_batch import e1 as E
from kernels.gfx950.qwen38_mono.large_batch import e1_plans as EP
from kernels.gfx950.qwen38_mono.large_batch import head_lb as H
from kernels.gfx950.qwen38_mono.large_batch import ple_lb as P
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


def test_e1_untouched_by_head_builds():
    e = EP.plan("head", 1, 128)
    before = _ll(E.build_e1_kernel(e))
    _ll(H.build_head_kernel(H.HeadKnobs(e1=e, tp=2)))
    assert _ll(E.build_e1_kernel(e)) == before


@pytest.mark.parametrize("tp,rows", [(1, 5), (1, 16), (1, 128), (2, 5), (2, 8), (2, 64), (2, 128)])
def test_head_markers(tp, rows):
    hk = H.plan(tp, rows)
    t = _ll(H.build_head_kernel(hk))
    assert "mfma" in t
    assert "global_atomic_umax_x2" in t and "global_atomic_swap_x2" in t
    assert ("global_store_dwordx4" in t) == (tp == 2)  # the TP2 pair push
    assert H.head_kernel_name(hk) in t


@pytest.mark.parametrize("kw", [dict(akey="f32"), dict(lout=True), dict(amrg="cta"), dict(xdelay=4)])
def test_head_knobs_lower(kw):
    hk = replace(H.plan(2, 32), **kw)
    hk.check()
    name = H.head_kernel_name(hk)
    assert any(k in name for k in kw)
    _ll(H.build_head_kernel(hk))


@pytest.mark.parametrize("m", [m for m in H.HEAD_MUTANTS if m != "none"])
def test_head_mutants_lower(m):
    hk = replace(H.plan(2, 16), mutant=m)
    t = _ll(H.build_head_kernel(hk))
    assert f"mutant{m}" in H.head_kernel_name(hk)
    if m == "xstale":
        assert "s_sleep 127" in t


def test_head_refusals():
    e = EP.plan("head", 1, 32)
    with pytest.raises(ValueError):
        H.HeadKnobs(e1=replace(e, splits=2), tp=1).check()
    with pytest.raises(ValueError):
        H.HeadKnobs(e1=e, tp=1, mutant="xpeer").check()
    with pytest.raises(ValueError):
        H.HeadKnobs(e1=e, tp=3).check()
    with pytest.raises(ValueError):
        H.HeadKnobs(e1=replace(e, tb=1), tp=1, amrg="cta").check()


def test_head_plan_every_rows():
    for tp in (1, 2):
        for rows in range(1, 129):
            hk = H.plan(tp, rows)
            assert hk.e1.tiles * 16 >= rows


def test_host_keys():
    v = np.array([[0.0, -0.0, 1.0, -1.0, np.nan, 1.0]], dtype=np.float32)
    k = H.ref_keys_f32(v)
    assert k[0, 0] == k[0, 1]  # -0 == +0
    assert k[0, 4] > k[0, 2] > k[0, 0] > k[0, 3]  # NaN above every finite value, then signed order
    assert int(k.argmax(axis=1)[0]) == 4
    bits = (v.view(np.uint32) >> 16).astype(np.uint16)
    kb = H.ref_keys_bf16(bits)
    assert (kb == k).all()
    w = np.array([[1.0, 2.0, 2.0]], dtype=np.float32)
    assert int(H.ref_keys_f32(w).argmax(axis=1)[0]) == 1  # ties to the lower row


@pytest.mark.parametrize("kind,threads", [("embed", 256), ("embed", 512), ("pgather", 192), ("pgather", 256), ("papply", 320)])
def test_ple_builds(kind, threads):
    build = {"embed": P.build_embed_kernel, "pgather": P.build_pgather_kernel, "papply": P.build_papply_kernel}[kind]
    k = P.WalkKnobs(kind=kind, threads=threads)
    t = _ll(build(k))
    assert P.kname(k) in t
    muts = {"embed": P.E_MUTANTS, "pgather": P.G_MUTANTS, "papply": P.A_MUTANTS}[kind]
    for m in muts[1:]:
        km = replace(k, mutant=m)
        _ll(build(km))
        assert f"mutant{m}" in P.kname(km)


def test_ple_refusals():
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="papply", threads=256).check()
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="embed", mutant="gate").check()


def test_pproj_plan_every_rows():
    for rows in range(1, 129):
        kn = P.pproj_plan(rows)
        assert kn.tiles * 16 >= rows and kn.k == P.HID


def test_papply_upc2_lowers():
    k = P.WalkKnobs(kind="papply", threads=640, upc=2)
    t = _ll(P.build_papply_kernel(k))
    assert P.kname(k) in t
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="papply", threads=320, upc=2).check()


def test_prologue_lowers():
    k = P.WalkKnobs(kind="prologue", threads=320)
    t = _ll(P.build_prologue_kernel(k))
    assert P.kname(k) in t
    assert len(P.prologue_params()) == len(P.pgather_params()) + 2


def test_papply_plan_every_rows():
    for rows in range(1, 129):
        k = P.papply_knobs(rows)
        assert k.threads == 320 * k.upc


# ------------------------------------------------------------------ fix round: W4A16 head, A's slay / stats levers


@pytest.mark.parametrize("tp,rows", [(1, 5), (1, 16), (1, 64), (2, 8), (2, 128)])
def test_head_w4a16_and_w16_lower(tp, rows):
    hk = H.plan_w4a16(tp, rows)
    assert hk.w4a16 and hk.e1.w == "bf16" and not hk.e1.sop
    t = _ll(H.build_head_kernel(hk))
    n_cvt = t.count("call <2 x bfloat> @llvm.amdgcn.cvt.scalef32.pk.bf16.fp4")
    assert n_cvt == 16 * hk.e1.mt * hk.e1.kc  # 4 dwords x 4 bytes per weight tile and chunk
    assert "w4a16" in H.head_kernel_name(hk) and "_w16" not in H.head_kernel_name(hk)
    hw = H.plan_w4a16(tp, rows, w16=True)
    tw = _ll(H.build_head_kernel(hw))
    assert "cvt.scalef32.pk.bf16.fp4" not in tw and "_w16" in H.head_kernel_name(hw)
    # the bf16 head's MFMAs (4 K pieces per weight tile, token tile and chunk) on decoded fragments
    mf = "call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf16"
    assert t.count(mf) == 4 * hk.e1.mt * hk.e1.tbt * hk.e1.kc
    assert tw.count(mf) == 4 * hw.e1.mt * hw.e1.tbt * hw.e1.kc
    assert H.head_kernel_name(H.plan(tp, rows)) not in (H.head_kernel_name(hk), H.head_kernel_name(hw))


def test_head_w4a16_mutants_and_refusals():
    hk = H.plan_w4a16(1, 16)
    for m in ("w4scale", "tie", "pad"):
        _ll(H.build_head_kernel(replace(hk, mutant=m)))
    _ll(H.build_head_kernel(replace(hk, e1=replace(hk.e1, mutant="kskip"), lout=True)))
    with pytest.raises(ValueError):
        replace(H.plan(1, 16), mutant="w4scale").check()  # w4scale needs w4a16
    with pytest.raises(ValueError):
        replace(H.plan(1, 16), w4a16=True).check()  # e1.w must be bf16 (X)
    with pytest.raises(ValueError):
        replace(hk, e1=replace(hk.e1, xsrc="global")).check()


@pytest.mark.parametrize("upc", [1, 2])
@pytest.mark.parametrize("slay", ["reg", "lds", "ldsl"])
@pytest.mark.parametrize("stats", ["atom", "fused"])
def test_papply_slay_stats_lower(upc, slay, stats):
    k = P.WalkKnobs(kind="papply", threads=320 * upc, upc=upc, slay=slay, stats=stats)
    t = _ll(P.build_papply_kernel(k))
    assert P.kname(k) in t
    for m in P.A_MUTANTS[1:]:
        _ll(P.build_papply_kernel(replace(k, mutant=m)))


def test_papply_lever_refusals_and_names():
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="embed", slay="lds").check()
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="pgather", threads=192, stats="fused").check()
    with pytest.raises(ValueError):
        P.WalkKnobs(kind="papply", threads=320, slay="x").check()
    assert P.kname(P.WalkKnobs(kind="papply", threads=320)) == "q38lbp_papply_t320"
    assert P.kname(P.WalkKnobs(kind="papply", threads=320, slay="lds", stats="fused")) == \
        "q38lbp_papply_t320_slaylds_statsfused"
