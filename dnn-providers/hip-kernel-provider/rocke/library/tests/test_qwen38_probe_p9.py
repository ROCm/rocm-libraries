# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of probe P9 (``kernels.gfx950.qwen38_mono.probes.p9_tail``): every build lowers, names are
unique and name only non-default knobs, invalid specs are refused, the LDS plan keeps the wave groups apart,
and P2's builds are unaffected by the emitter hooks P9 added to ``p2_head.emit_head``."""

import pytest

from kernels.gfx950.qwen38_mono.probes import p2_head as P2
from kernels.gfx950.qwen38_mono.probes import p9_tail as P
from rocke.core.lower_llvm import lower_kernel_to_llvm

BUILDS = {
    "TV": P.P9Spec(mode="verify"),
    "TV_es0": P.P9Spec(mode="verify", espec=0),
    "TD4": P.P9Spec(mode="draft", rows_in=4),
    "TD4rej": P.P9Spec(mode="draft", rows_in=4, reject=1),
    "TD1": P.P9Spec(mode="draft", rows_in=1),
    "TVt": P.P9Spec(mode="verify", tmo=1),
    "TV_spin": P.P9Spec(mode="verify", xwait="spin", idwait="all", pause="s_nop"),
    "TV_v2": P.P9Spec(mode="verify", lat="v2"),
    "TV_v2w": P.P9Spec(mode="verify", lat="v2w"),
    "TD4rej_v2": P.P9Spec(mode="draft", rows_in=4, reject=1, lat="v2"),
    "TD1_v2w": P.P9Spec(mode="draft", rows_in=1, lat="v2w"),
    "TV_v2w_ml": P.P9Spec(mode="verify", lat="v2w", mloop=1, xbt=8),
    "TV_v2w_st": P.P9Spec(mode="verify", lat="v2w", stamps=True),
    "TD1_v2_st": P.P9Spec(mode="draft", rows_in=1, lat="v2", stamps=True, mloop=1),
    "TV_v2w_row": P.P9Spec(mode="verify", lat="v2w", prer="row", xbt=8),
    "TV_v2w_direct": P.P9Spec(mode="verify", lat="v2w", fold="direct"),
    "TD4rej_v2w_direct": P.P9Spec(mode="draft", rows_in=4, reject=1, lat="v2w", fold="direct"),
    "TD1_direct_st": P.P9Spec(mode="draft", rows_in=1, fold="direct", stamps=True),
    "TVt_direct": P.P9Spec(mode="verify", fold="direct", tmo=1),
    "TV_v2w_pure": P.P9Spec(mode="verify", lat="v2w", opq="pure", fold="direct", prer="row", xbt=8),
    "TD1_v2w_pure": P.P9Spec(mode="draft", rows_in=1, lat="v2w", opq="pure"),
    "TV_fnpre": P.P9Spec(mode="verify", lat="v2w", fnorm="pre", fold="direct"),
    "TV_fnpre_es0": P.P9Spec(mode="verify", fnorm="pre", espec=0),
    "TD1_fnpre": P.P9Spec(mode="draft", rows_in=1, fnorm="pre"),
    "TV_hd4": P.P9Spec(mode="verify", lat="v2w", hdelay=4),
    "TD1_bs4": P.P9Spec(mode="draft", rows_in=1, lat="v2w", bsleep=4),
    "TV_ecta": P.P9Spec(mode="verify", lat="v2w", fold="direct", fnorm="pre", ectr="cta", xbt=8),
    "TD1_ecta_st": P.P9Spec(mode="draft", rows_in=1, ectr="cta", stamps=True),
    "BEST_V": P.best_spec("verify"),
    "BEST_D4": P.best_spec("draft", rows_in=4, reject=1),
    "BEST_D1": P.best_spec("draft", rows_in=1),
}
MUTS = {(m, lt, fo): P.P9Spec(mode="verify" if m in ("accept_ooo", "stale_ids", "sumsq_order") else "draft",
                              rows_in=4 if m in ("accept_ooo", "stale_ids", "sumsq_order") else 1, mutant=m, lat=lt,
                              fold=fo)
        for m in P.MUTANTS if m != "none" for lt in ("v1", "v2") for fo in ("fused", "direct")}


@pytest.mark.parametrize("name", sorted(BUILDS) + sorted(MUTS))
def test_build_lowers(name):
    spec = BUILDS[name] if isinstance(name, str) else MUTS[name]
    ok, why = P.is_valid_p9(spec)
    assert ok, why
    k = P.build_p9(spec)
    ll = lower_kernel_to_llvm(k, arch="gfx950")
    assert k.name in ll
    # no hardware barrier after the entry barrier: exactly one s_barrier in the module
    assert ll.count("call void @llvm.amdgcn.s.barrier()") == 1


def test_names_unique_and_default_free():
    names = [P.build_p9(s).name for s in list(BUILDS.values()) + list(MUTS.values())]
    assert len(names) == len(set(names))
    assert P.P9Spec().kernel_name() == "p9_verify_ri4_pyonly"


@pytest.mark.parametrize("kw", [
    dict(mode="bogus"), dict(mode="verify", rows_in=1), dict(rows_in=3, mode="draft"), dict(toff=100),
    dict(fpre="x"), dict(espec=2), dict(mutant="accept_ooo", mode="draft", rows_in=1), dict(reject=1),
    dict(head_dg=9), dict(mutant="nope"), dict(lat="v3"), dict(lat="v2", prex="redund"), dict(lat="v2", tmo=1), dict(mloop=1), dict(lat="v2", mloop=2), dict(xbt=6), dict(prer="row"), dict(lat="v2", prer="x"), dict(fold="x"), dict(fold="direct", idwait="all"), dict(fold="direct", dbg=2), dict(opq="pure"), dict(lat="v2", opq="x"), dict(fnorm="x"), dict(hdelay=17), dict(bsleep=17), dict(bsleep=2, tmo=1), dict(ectr="x"),
])
def test_invalid(kw):
    ok, _ = P.is_valid_p9(P.P9Spec(**kw))
    assert not ok
    with pytest.raises(ValueError):
        P.build_p9(P.P9Spec(**kw))


def test_scratch_layout_disjoint():
    regions = sorted(P.SCR.items(), key=lambda kv: kv[1])
    sizes = {"epoch": 8, "part": 4 * P.HC * P.R * 8, "rstd": 16 * 8, "xbox": 4 * (P.H // 2) * 8, "rstdh": 4 * 8,
             "ids": 8 * 8, "keys": 4 * P.HEAD_CTAS * 16, "err": 64 * 256 * 16, "dbg": 2048 * 16, "dbgw": 2 * 4 * 4 * 32 * 4}
    for (n0, o0), (n1, o1) in zip(regions, regions[1:]):
        assert o0 + sizes[n0] <= o1, (n0, n1)
    assert regions[-1][1] + sizes[regions[-1][0]] <= P.SCR_BYTES


def test_p2_head_hooks_default_free():
    """emit_head with the P9 hooks left at None emits P2's IR (the clone test covers mx_decode_gemv)."""
    from kernels.gfx950 import mx_decode_gemv as M

    h = M.MxDecodeGemvSpec(n=248320, k=2560, rows_max=4, epilogue="fused", waves_per_eu=1)
    a = lower_kernel_to_llvm(P2.build_head_clone(h), arch="gfx950")
    b = lower_kernel_to_llvm(M.build_mx_decode_gemv(h), arch="gfx950")
    assert a == b


def test_best_spec_knobs():
    v, d = P.best_spec("verify"), P.best_spec("draft", rows_in=1)
    assert (v.lat, v.fold, v.fnorm, v.ectr, v.xbt, v.hdelay) == ("v2w", "direct", "pre", "cta", 8, 8)
    assert (d.lat, d.fold, d.fnorm, d.ectr, d.xbt, d.hdelay) == ("v2w", "direct", "pre", "cta", 4, 0)
