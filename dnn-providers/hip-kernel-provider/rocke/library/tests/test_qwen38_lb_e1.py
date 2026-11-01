# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2a (``kernels.gfx950.qwen38_mono.large_batch.e1``): CPU checks.

* the fork is faithful: with every new knob at its default and ``hz=False`` an E1 build lowers to the same LLVM IR as
  the matching P-L3 ``gemm.py`` build (kernel name substituted), for P-L3's arm shapes, so P-L3's hardware proofs
  carry over to the default arm;
* every new lever (``kw``, short / funnel op_sel ranges, ``xsrc=global`` with op_sel, ``act=fp4``, ``bar=persist``,
  ``fg``, ``kseg``, ``yq``, ``chain``) and every new mutant builds and lowers, with the markers the GPU proofs rely on;
* ``hz`` pads every inline-asm VMEM block with an SGPR operand; names list only deviations; invalid combinations refuse;
* the host repacks (``pack_weights_tile_sop``, ``pack_act_scales_sop``) equal P-L3's bench layouts.
The GPU proofs (tolerance vs float64, bitwise across order-preserving knobs, determinism, mutants, the chain test with
stale-read twins, shuffled timing) run in ``hackathon-prep/qwen38/truemono-lb/wp-l2a``.
"""

from __future__ import annotations

import re

import pytest
from kernels.gfx950.qwen38_mono.large_batch import e1 as E
from kernels.gfx950.qwen38_mono.large_batch import gemm as G
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
SOP = dict(wlay="tile", xslay="chunk", xscale="glb", sop=True)


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


SAME = [
    dict(),
    dict(w="bf16", k=3072, waves=4, splits=6, xu=False, sb=False),
    dict(k=3072, splits=6, map="xcd", fold="none", **SOP),
    dict(k=2560, bucket=16, xsrc="global", depth=4),
    dict(w="bf16", k=320, bucket=128, xa=2, wlay="tile", bpre=2),
    dict(k=320, bucket=32, **SOP),
    dict(k=10240, bucket=128, splits=20, tb=4, waves=4, **SOP),
    dict(k=2560, mt=2, bpre=4, splits=5, **SOP),
    dict(k=3072, splits=6, mutant="fold"),
    dict(k=2560, mutant="scale"),
    dict(w="bf16", k=2560, mutant="kskip", wlay="tile"),
    dict(k=2560, mutant="sopsel", **SOP),
    dict(k=3072, bucket=16, splits=6, depth=4, waves=4, **SOP),
]


@pytest.mark.parametrize("kw", SAME, ids=[G.kernel_name(G.GemmKnobs(**k)) for k in SAME])
def test_ir_identical_to_gemm(kw):
    g, e = G.GemmKnobs(**kw), E.E1Knobs(hz=False, **kw)
    a = _ll(G.build_gemm_kernel(g)).replace(G.kernel_name(g), "KN")
    b = _ll(E.build_e1_kernel(e)).replace(E.kernel_name(e), "KN")
    assert a == b


NEW = [
    dict(k=3072, bucket=16, splits=12, depth=4, **SOP),  # short op_sel range (kc 2)
    dict(k=3072, bucket=16, splits=24, depth=4, **SOP),  # kc 1
    dict(k=3072, bucket=16, splits=8, depth=4, **SOP),  # funnel (kc 3)
    dict(k=3072, bucket=16, kw=8, depth=4, **SOP),  # kw: 8 groups of 3 chunks (funnel)
    dict(k=3072, bucket=64, kw=4, splits=2, depth=4, **SOP),
    dict(k=3072, bucket=128, kw=8, tb=2, depth=4, **SOP),
    dict(k=3072, bucket=16, kw=8, depth=3, xsrc="global", wlay="tile", sop=True),
    dict(k=10240, bucket=32, kw=8, splits=10, **SOP),
    dict(k=10240, bucket=16, kw=8, splits=10, kseg=4, depth=4, **SOP),
    dict(k=2560, bucket=64, act="fp4", depth=4, **SOP),
    dict(k=2560, bucket=64, act="fp4", depth=4, xsrc="global", wlay="tile", sop=True),
    dict(k=2560, bucket=32, bar="persist", splits=5, depth=4, **SOP),
    dict(k=2560, bucket=128, yq=True, depth=4, **SOP),
    dict(k=2560, bucket=32, yq=True, splits=5, depth=4, **SOP),
    dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, bar="persist", **SOP),
    dict(k=3072, bucket=16, splits=12, depth=4, fg=16, **SOP),
    dict(w="bf16", k=3072, bucket=32, kw=8, depth=2, xsrc="global"),
    dict(k=3072, bucket=16, kw=8, depth=4, mutant="kwred", **SOP),
    dict(k=10240, bucket=16, kw=8, splits=10, kseg=4, depth=4, mutant="seg", **SOP),
    dict(k=3072, bucket=16, splits=12, depth=4, mutant="sopsel", **SOP),
    dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, bar="persist", mutant="bardrop", **SOP),
    dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, mutant="barnone", **SOP),
    dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, mutant="noacq", **SOP),
    dict(k=10240, bucket=64, splits=20, tb=2, depth=4, fold="spread", xjunk=True, **SOP),  # spread fold
    dict(k=10240, bucket=64, splits=20, tb=2, depth=4, fold="spread", mutant="nowait", xjunk=True, **SOP),
    dict(k=3072, bucket=128, splits=6, tb=4, depth=4, fold="spread", mutant="fold", **SOP),
    dict(k=2560, bucket=16, depth=4, xjunk=True, **SOP),  # branch-free staging
    dict(k=2560, bucket=64, mfma=32, depth=4, **SOP),  # 32x32x64 MFMA
    dict(k=2560, bucket=128, mfma=32, tb=4, splits=5, depth=4, **SOP),
    dict(k=3072, bucket=64, mfma=32, kw=4, depth=4, **SOP),
    dict(k=2560, bucket=64, mfma=32, act="fp4", depth=4, xsrc="global", wlay="tile", sop=True),
    dict(k=3072, bucket=64, mfma=32, splits=8, depth=4, **SOP),  # funnel scales on the 32 shape
    dict(k=3072, bucket=64, mfma=32, splits=12, depth=4, mutant="sopsel", **SOP),
    dict(k=3072, bucket=64, mfma=32, kw=4, depth=4, mutant="kwred", **SOP),
]


@pytest.mark.parametrize("kw", NEW, ids=[E.kernel_name(E.E1Knobs(**k)) for k in NEW])
def test_new_levers_lower(kw):
    kn = E.E1Knobs(**kw)
    text = _ll(E.build_e1_kernel(kn))
    assert E.kernel_name(kn) in text
    defs = re.findall(r"^\s*(%[\w.]+) = ", text, re.M)
    assert len(defs) == len(set(defs))
    if kn.w == "mx" and kn.mfma == 16:
        assert "mfma.scale.f32.16x16x128.f8f6f4" in text
    if kn.bar == "persist":
        assert not re.search(r"s_sleep 1(?![0-9])", text)  # the persist poll pauses with s_nop
    if kn.yq:
        assert "global_store_byte" in text and "v_cvt_scalef32_pk_fp8_f32" in text
    if kn.mfma == 32:
        assert "mfma.scale.f32.32x32x64.f8f6f4" in text
    if kn.hz:
        # every inline-asm VMEM block with an SGPR operand starts with s_nop 4
        for m in re.finditer(r'asm sideeffect "([^"]*)", "([^"]*)"', text):
            tmpl, cons = m.group(1), m.group(2)
            if any(v in tmpl for v in ("global_", "buffer_")) and "s" in [c.strip() for c in cons.split(",")]:
                assert tmpl.startswith("s_nop 4"), tmpl[:80]


def test_mutants_change_the_body():
    for base, mut in (
        (dict(k=3072, bucket=16, kw=8, depth=4, **SOP), "kwred"),
        (dict(k=10240, bucket=16, kw=8, splits=10, kseg=4, depth=4, **SOP), "seg"),
        (dict(k=3072, bucket=16, splits=12, depth=4, **SOP), "sopsel"),
        (dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, bar="persist", **SOP), "bardrop"),
        (dict(k=2560, bucket=32, yq=True, chain=True, splits=5, depth=4, **SOP), "barnone"),
        (dict(k=10240, bucket=64, splits=20, tb=2, depth=4, fold="spread", **SOP), "nowait"),
        (dict(k=2560, bucket=64, mfma=32, depth=4, **SOP), "kskip"),
        (dict(k=3072, bucket=64, mfma=32, kw=4, depth=4, **SOP), "kwred"),
    ):
        a = E.E1Knobs(**base)
        b = E.E1Knobs(mutant=mut, **base)
        ta = _ll(E.build_e1_kernel(a)).replace(E.kernel_name(a), "KN")
        tb = _ll(E.build_e1_kernel(b)).replace(E.kernel_name(b), "KN")
        assert ta != tb, mut


def test_names_list_deviations():
    assert E.kernel_name(E.E1Knobs()) == "q38lb_e1_mx_k2560_b128"
    assert E.kernel_name(E.E1Knobs(hz=False)) == "q38lb_e1_mx_k2560_b128_nohz"
    assert E.kernel_name(E.E1Knobs(kw=4, bar="persist")).endswith("_kw4_barpersist")
    names = {E.kernel_name(E.E1Knobs(**k)) for k in NEW}
    assert len(names) == len(NEW)


@pytest.mark.parametrize(
    "kw",
    [
        dict(kw=3),  # kw must divide waves
        dict(k=3072, kw=8, splits=4, **SOP),  # 24 chunks do not split 32 ways
        dict(k=3072, kw=2),  # kw > 1 (mx) needs sop
        dict(act="fp4"),  # fp4 needs sop
        dict(w="bf16", act="fp4"),
        dict(bar="mesh"),
        dict(k=10240, kseg=4, splits=2, **SOP),  # splits x kw not a multiple of kseg
        dict(yq=True, waves=4, mt=1, **SOP),  # 16 wr mt = 64 ok ... (valid; see below)
        dict(chain=True, **SOP),  # chain needs yq
        dict(mutant="kwred", **SOP),
        dict(mutant="bardrop", **SOP),
        dict(w="bf16", k=3072, bucket=32, kw=4, depth=2),  # 80 KiB of LDS: neither the pad fits nor above half
        dict(fold="spread", **SOP),  # spread needs splits > 1
        dict(k=3072, splits=6, fold="spread", yq=True, **SOP),
        dict(k=3072, splits=6, mutant="nowait", **SOP),  # nowait needs fold=spread
        dict(k=2560, bucket=16, mfma=32, **SOP),  # one 16-token tile: no 32-token tile
        dict(k=2560, bucket=64, mfma=32),  # the 32 shape needs op_sel scales
        dict(k=2560, bucket=64, mfma=32, yq=True, **SOP),
        dict(mfma=64, **SOP),
    ],
)
def test_refusals(kw):
    if kw == dict(yq=True, waves=4, mt=1, **SOP):
        E.E1Knobs(**kw).check()
        with pytest.raises(ValueError):
            E.E1Knobs(yq=True, waves=4, kw=4, mt=1, **SOP).check()  # 16 rows per unit: not a 32-row block multiple
        return
    with pytest.raises(ValueError):
        E.E1Knobs(**kw).check()


def test_host_repacks_match_pl3_layouts():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(3)
    M, K = 37, 320  # ragged rows and K
    nq, nq4, nsc, KSP = 3, 1, 10, E.ksp(320)
    codes = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, generator=g)
    sc = torch.full((M, KSP), 127, dtype=torch.uint8)
    sc[:, :nsc] = torch.randint(118, 130, (M, nsc), dtype=torch.uint8, generator=g)
    ct, so = E.pack_weights_tile_sop(codes, sc, M, K)
    # P-L3 bench layouts (pl3_bench.Cell._tile, one layer)
    MT = -(-M // 16)
    c = torch.zeros(MT * 16, nq * 64, dtype=torch.uint8)
    c[:M, : K // 2] = codes
    ref_ct = c.view(MT, 16, nq, 64).permute(0, 2, 1, 3).contiguous().view(-1)
    s2 = torch.full((MT * 16, nq4 * 16), 127, dtype=torch.uint8)
    s2[:M, : min(nq4 * 16, KSP)] = sc[:, : nq4 * 16]
    ref_so = s2.view(MT, 16, nq4, 4, 4).permute(0, 2, 1, 4, 3).contiguous().view(-1)
    assert torch.equal(ct, ref_ct)
    assert torch.equal(so, ref_so)
    TP = 32
    xsc = torch.full((TP, KSP), 127, dtype=torch.uint8)
    xsc[:, :nsc] = torch.randint(118, 123, (TP, nsc), dtype=torch.uint8, generator=g)
    xpad = torch.full((TP, nq4 * 16), 127, dtype=torch.uint8)
    xpad[:, : min(nq4 * 16, KSP)] = xsc[:, : nq4 * 16]
    ref_x = xpad.view(TP, nq4, 4, 4).permute(1, 0, 3, 2).contiguous().view(-1)
    assert torch.equal(E.pack_act_scales_sop(xsc, TP, K), ref_x)


# ------------------------------------------------------------------ planner (e1_plans)


def test_plan_every_projection_and_row_count_is_buildable():
    from kernels.gfx950.qwen38_mono.large_batch import e1_plans as P

    for (proj, tp) in P.SHAPES:
        for rows in (1, 5, 6, 8, 9, 16, 17, 32, 33, 64, 65, 100, 128):
            kn = P.plan(proj, tp or 1, rows)
            kn.check()
            assert kn.k == P.SHAPES[(proj, tp)][1]
            assert kn.bucket >= rows and kn.sop and kn.fold != "none"  # complete output, op_sel scales
    assert P.plan("qsa_out", 2, 8).k == 3072  # qsa_out has gdn_out's shape


def test_plan_table_entries_build_and_lower():
    from kernels.gfx950.qwen38_mono.large_batch import e1_plans as P

    for (proj, tp, rows), dev in P.PLANS.items():
        kn = P.plan(proj, tp or 1, rows)
        for k, v in dev.items():
            if k != "tb" or v <= kn.tiles:
                assert getattr(kn, k) == v, (proj, tp, rows, k)
    if P.PLANS:
        key = sorted(P.PLANS)[0]
        kn = P.plan(key[0], key[1] or 1, key[2])
        assert E.kernel_name(kn) in _ll(E.build_e1_kernel(kn))


def test_row_key_and_model():
    from kernels.gfx950.qwen38_mono.large_batch import e1_plans as P

    assert [P.row_key(r) for r in (1, 5, 6, 8, 9, 16, 17, 33, 128)] == [5, 5, 8, 8, 16, 16, 32, 64, 128]
    with pytest.raises(ValueError):
        P.row_key(129)
    # grid fill first: 240 units on 128 CTAs (two rounds) cost about twice the same plan on 256 CTAs
    b = E.E1Knobs(k=3072, bucket=16, splits=12, depth=4, **SOP)
    assert P.model_cost(2560, 3072, 8, b, grid=128) > 1.8 * P.model_cost(2560, 3072, 8, b)
    # spread fold reads 1 / S of the partials per unit
    c = E.E1Knobs(k=10240, bucket=64, splits=20, tb=2, depth=4, **SOP)
    assert P.model_cost(324, 10240, 64, E.E1Knobs(**{**c.__dict__, "fold": "spread"})) < P.model_cost(324, 10240, 64, c)
