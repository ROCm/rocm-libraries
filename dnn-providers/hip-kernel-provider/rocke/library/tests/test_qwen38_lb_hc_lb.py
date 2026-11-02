# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of WP-L2b (``kernels.gfx950.qwen38_mono.large_batch.hc_lb``): the parents (E1 ``e1.py``, the WP-L2c chain
``gdn_lb.py``) lower to the same LLVM IR before and after any HC build; every chain variant (TP1 / TP2, every bucket,
both norms, 4 / 8 waves, dbg twin, every mutant, every chain lever, ablations) lowers; the knob checks refuse what is not
built; the up-weight row permutation and the parameter order."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kernels.gfx950.qwen38_mono.large_batch import e1 as E
from kernels.gfx950.qwen38_mono.large_batch import gdn_lb as G
from kernels.gfx950.qwen38_mono.large_batch import hc_lb as H
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
SOP = dict(wlay="tile", sop=True, xslay="chunk", xscale="glb")


def ll(kd) -> str:
    return _lower_kernel_to_llvm_python(kd, arch="gfx950", llvm_flavor=FLAVOR)


PARENTS = [
    lambda: E.build_e1_kernel(E.E1Knobs(k=2560, bucket=128, mt=2, tb=4, **SOP)),
    lambda: E.build_e1_kernel(E.E1Knobs(k=10240, bucket=64, splits=10, yq=True, kseg=0, **SOP)),
    lambda: G.build_chain_kernel(G.ChainKnobs(tp=2, bucket=32)),
]


def test_parents_unchanged_by_composition():
    before = [ll(f()) for f in PARENTS]
    H.build_hc_kernel(H.HcKnobs(tp=1, bucket=64, prow=64))
    H.build_hc_kernel(H.HcKnobs(tp=2, bucket=16, prow=8, norm="reassoc", dbg=True))
    after = [ll(f()) for f in PARENTS]
    assert before == after
    assert E.IRBuilder is G.IRBuilder


CELLS = [(1, 5), (1, 8), (1, 16), (1, 32), (1, 64), (2, 128), (2, 16)]


def bucket(rows):
    return next(b for b in (16, 32, 64, 128) if rows <= b)


@pytest.mark.parametrize("tp,rows", CELLS)
@pytest.mark.parametrize("norm", ["faithful", "reassoc"])
def test_default_cells_lower(tp, rows, norm):
    hk = H.HcKnobs(tp=tp, bucket=bucket(rows), prow=rows, norm=norm)
    text = ll(H.build_hc_kernel(hk))
    assert H.hc_kernel_name(hk) in text
    d, u = hk.parts()
    assert d.k == H.HCW and d.yq and d.kseg == (4 if norm == "reassoc" else 0)
    assert u.k == H.LR and u.splits == 1 and u.rows_per_unit % 128 == 0


def test_waves8_and_cw2_lower():
    for c in (H.CKnobs(cw=1), H.CKnobs(cw=2), H.CKnobs(cw=2, inv="rsq")):
        hk = H.HcKnobs(tp=1, bucket=32, prow=32, waves=8, c=c)
        assert "define" in ll(H.build_hc_kernel(hk))


def test_mutants_levers_ablations_lower_and_are_named():
    hk = H.HcKnobs(tp=1, bucket=64, prow=64)
    d, u = hk.parts()
    arms = [replace(hk, c=replace(hk.c, mutant=m)) for m in H.C_MUTANTS if m != "none"]
    arms += [replace(hk, dmut="silu"), replace(hk, d=replace(d, mutant="kskip"))]
    arms += [replace(hk, umut=m) for m in ("mix", "sig", "xs")] + [replace(hk, u=replace(u, mutant="kskip"))]
    arms += [replace(hk, mutant=m) for m in ("early", "noacq")] + [replace(hk, delay=True), replace(hk, dbg=True)]
    arms += [replace(hk, stages=s) for s in ("C", "D", "U", "")]
    arms += [replace(hk, epre=False), replace(hk, pre="du"), replace(hk, acq="wave"), replace(hk, acq="cta"),
             replace(hk, lpad="fit"), replace(hk, ticket=False), replace(hk, tag="rc")]  # fmt: skip
    names = {H.hc_kernel_name(hk)}
    for a in arms:
        n = H.hc_kernel_name(a)
        assert n not in names, n
        names.add(n)
        assert n in ll(H.build_hc_kernel(a))


@pytest.mark.parametrize("cp", [1, 2, 5, 10])
@pytest.mark.parametrize("tp,rows", [(1, 8), (2, 128)])
def test_cstat_fold_lowers(cp, tp, rows):
    hk = H.HcKnobs(tp=tp, bucket=bucket(rows), prow=rows, norm="reassoc", cstat="fold", cp=cp)
    assert H.hc_kernel_name(hk) in ll(H.build_hc_kernel(hk))
    for m in ("eps", "inj", "xblk"):
        assert "define" in ll(H.build_hc_kernel(replace(hk, c=replace(hk.c, mutant=m))))


@pytest.mark.parametrize("tp,rows", [(1, 5), (1, 16), (1, 64), (2, 128)])
def test_cfuse_lowers(tp, rows):
    hk = H.HcKnobs(tp=tp, bucket=bucket(rows), prow=rows, norm="reassoc", cfuse=True)
    d, u = hk.parts()
    assert d.splits % 4 == 0 and not d.kseg and d.xsrc == "lds" and not d.xu
    assert H.hc_kernel_name(hk) in ll(H.build_hc_kernel(hk))
    for m in ("eps", "inj", "xblk", "ygate", "rskip"):
        assert "define" in ll(H.build_hc_kernel(replace(hk, c=replace(hk.c, mutant=m))))
    assert "define" in ll(H.build_hc_kernel(replace(hk, d=replace(d, mutant="fold"))))
    assert "define" in ll(H.build_hc_kernel(replace(hk, mutant="early", stages="DU")))


@pytest.mark.parametrize("tp,rows", [(1, 8), (2, 128)])
def test_late_levers_lower(tp, rows):
    hk = H.HcKnobs(tp=tp, bucket=bucket(rows), prow=rows)
    d, u = hk.parts()
    arms = [replace(hk, cvec="pk"), replace(hk, uvec="pk"), replace(hk, gplace="d"), replace(hk, gplace="u"),
            replace(hk, sig="fast"), replace(hk, eload="t"), replace(hk, stamp=True), replace(hk, pre="dun"),
            replace(hk, pre="D", d=replace(d, xu=True)), replace(hk, pre="DU", d=replace(d, xu=True), u=replace(u, xu=True)),
            replace(hk, norm="reassoc", cstat="dfold", cp=2), replace(hk, norm="reassoc", cstat="cfold", cp=2),
            replace(hk, gplace="d", c=replace(hk.c, mutant="inj"))]  # fmt: skip
    for a in arms:
        assert H.hc_kernel_name(a) in ll(H.build_hc_kernel(a))


def test_refusals():
    hk = H.HcKnobs(tp=1, bucket=64, prow=64)
    d, u = hk.parts()
    bad = [
        replace(hk, tp=3),
        replace(hk, waves=6),
        replace(hk, norm="other"),
        replace(hk, c=H.CKnobs(cw=2)),  # 4 waves cannot hold 2 waves per (token, stream) x 4 streams
        replace(hk, u=replace(u, mt=1, waves=4)),  # 64-row up units: no 32-column x blocks
        replace(hk, u=replace(u, splits=2)),
        replace(hk, d=replace(d, yq=False)),
        replace(hk, d=replace(d, kseg=4)),  # faithful norm with segment scales
        replace(hk, d=replace(d, waves=8)),
        replace(hk, d=replace(d, bar="persist")),
        replace(hk, stages="CX"),
        replace(hk, pre="x"),
        replace(hk, umut="x"),
        replace(hk, cstat="fold"),  # needs norm=reassoc
        replace(hk, norm="reassoc", cstat="fold", cp=3),
        replace(hk, cfuse=True),  # needs norm=reassoc
        replace(hk, norm="reassoc", cfuse=True, d=replace(H.default_dfuse(64, 64), splits=5)),
        replace(hk, norm="reassoc", cfuse=True, d=replace(H.default_dfuse(64, 64), xu=True)),
        replace(hk, gplace="x"),
        replace(hk, gplace="d", cvec="pk"),
        replace(hk, cvec="pk", norm="reassoc", cstat="cfold"),
        replace(hk, stamp=True, dbg=True),
        replace(hk, pre="Dd", d=replace(d, xu=True)),
        replace(hk, pre="D", d=replace(d, xu=False)),
    ]
    for b in bad:
        with pytest.raises((ValueError, NotImplementedError)):
            b.check()


def test_up_perm_and_pack():
    import torch

    p = H.up_perm()
    assert p.shape == (H.HCW,) and sorted(p.tolist()) == list(range(H.HCW))
    # new row 4 col + i = checkpoint row i H + col
    for col, i in ((0, 0), (0, 3), (17, 2), (H.HID - 1, 1)):
        assert int(p[4 * col + i]) == i * H.HID + col
    codes = torch.randint(0, 256, (H.HCW, H.LR // 2), dtype=torch.uint8)
    scales = torch.randint(119, 122, (H.HCW, H.LR // 32), dtype=torch.uint8)
    ct, so = H.pack_up_perm(codes, scales)
    ct2, so2 = E.pack_weights_tile_sop(codes[p], scales[p], H.HCW, H.LR)
    assert torch.equal(ct, ct2) and torch.equal(so, so2)
    wl, sl = G._wstride_mx(H.HCW, H.LR)
    assert ct.numel() == wl and so.numel() == sl


def test_params_and_strides():
    assert H.hc_params()[: len(H.HC_I64)] == H.HC_I64 and len(H.hc_params()) == 34
    s = H.strides(H.HcKnobs(tp=2, bucket=128))
    assert s["y"] == 2 * 128 * H.HID * 2 and s["xqs"] == 20 * 128 * 16 and s["aqs"] == 128 * 16
