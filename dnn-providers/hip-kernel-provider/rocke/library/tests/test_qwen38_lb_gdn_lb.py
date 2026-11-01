# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of WP-L2c (``kernels.gfx950.qwen38_mono.large_batch.gdn_lb``): the parents (E1 ``e1.py``, the P-L1
recurrence ``gdn_recur.py``) lower to the same LLVM IR before and after any WP-L2c build (the composition patches
nothing permanently); E4 with the hand-off off is P-L1's kernel; every stage / chain variant (TP1 / TP2, every bucket,
dbg twins, mutants, ablations) lowers; knob checks refuse what is not built; LDS budget and parameter order."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kernels.gfx950.qwen38_mono.large_batch import e1 as E
from kernels.gfx950.qwen38_mono.large_batch import gdn_lb as G
from kernels.gfx950.qwen38_mono.large_batch import gdn_recur as GR
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
SOP = dict(wlay="tile", sop=True, xslay="chunk", xscale="glb")


def ll(kd) -> str:
    return _lower_kernel_to_llvm_python(kd, arch="gfx950", llvm_flavor=FLAVOR)


def ll_unnamed(kd, name) -> str:
    return ll(kd).replace(name, "@K@")


PARENTS = [
    lambda: E.build_e1_kernel(E.E1Knobs(k=2560, bucket=128, mt=2, tb=4, **SOP)),
    lambda: E.build_e1_kernel(E.E1Knobs(w="bf16", k=2560, bucket=32, splits=4)),
    lambda: GR.build_lb_kernel(GR.TP1, GR.best_knobs(GR.TP1, 64)),
    lambda: GR.build_lb_kernel(GR.TP2, GR.RecKnobs(map="rowi", ld_nt=True, dbg=True, state="bf16")),
]


def test_parents_unchanged_by_composition():
    before = [ll(f()) for f in PARENTS]
    G.build_chain_kernel(G.ChainKnobs(tp=1, bucket=64))
    G.build_e3_kernel(G.E3Knobs(e1=E.E1Knobs(k=2560, bucket=32, splits=2, **SOP), tp=2, dbg=True))
    G.build_e4_kernel(G.E4Knobs(tp=2, bucket=32))
    after = [ll(f()) for f in PARENTS]
    assert before == after
    assert E.IRBuilder is G.IRBuilder and GR.IRBuilder is G.IRBuilder
    assert GR._pad_lds.__module__ == GR.__name__ and GR._touch_lds.__module__ == GR.__name__


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("state", ["f32", "bf16"])
def test_e4_hand_none_is_pl1(tp, state):
    """The subclass with the hand-off off emits P-L1's kernel (its copied finisher and dummy stores are the parent's)."""
    rk = GR.RecKnobs(map="rowi", ld_nt=True, state=state)
    k4 = G.E4Knobs(rec=rk, tp=tp, hand="none")
    sub = G._E4(k4).emit()
    ref = GR.build_lb_kernel(k4.shape.rec, rk)
    assert ll_unnamed(sub, sub.name) == ll_unnamed(ref, ref.name)
    # the build function's identity arm is the parent build renamed
    kd = G.build_e4_kernel(replace(k4, hz=False))
    assert kd.name == G.e4_kernel_name(replace(k4, hz=False))


def test_shapes():
    s1, s2 = G.GdnShape(1), G.GdnShape(2)
    assert (s1.qkv, s1.m_in, s1.m_ba, s1.val) == (10240, 16384, 96, 6144)
    assert (s2.qkv, s2.m_in, s2.m_ba, s2.val) == (5120, 8192, 48, 3072)
    assert s1.m_in == GR.TP1.row_elems and s2.m_in == GR.TP2.row_elems


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("bucket", [16, 32, 64, 128])
def test_chain_lowers(tp, bucket):
    kc = G.ChainKnobs(tp=tp, bucket=bucket)
    kc.check()
    txt = ll(G.build_chain_kernel(kc))
    assert "define amdgpu_kernel" in txt
    assert G.chain_lds_bytes(kc) <= G.LDS_MAX


@pytest.mark.parametrize("variant", ["dbg", "early", "noacq", "delay", "s3", "s4", "s5", "edges", "tap", "state",
                                     "dtb", "swap", "hscale", "hpair", "nodecay", "kskip3", "kskip5", "bf16", "rc",
                                     "acq_wave", "acq_load", "cpre", "pre_s5", "cepi", "pre_s4",
                                     "pre_s45", "pre_s4w2", "pre_s4_bf16", "lpad", "lpad_cepi",
                                     "cbatch", "cbatch_g4", "cbatch_cpre", "cbatch_split"])
def test_chain_variants_lower(variant):
    kc = G.ChainKnobs(tp=2, bucket=32)
    e3, e3g, e4, e5 = kc.parts()
    v = {
        "dbg": replace(kc, e3=replace(e3, dbg=True), e3g=replace(e3g, dbg=True), e4=replace(e4, rec=replace(e4.rec, dbg=True))),
        "early": replace(kc, mutant="early"), "noacq": replace(kc, mutant="noacq"), "delay": replace(kc, delay=True),
        "s3": replace(kc, stages="3"), "s4": replace(kc, stages="4"), "s5": replace(kc, stages="5"),
        "edges": replace(kc, stages=""),
        "tap": replace(kc, e3=replace(e3, mutant="tap")), "state": replace(kc, e3=replace(e3, mutant="state")),
        "dtb": replace(kc, e3g=replace(e3g, mutant="dtb")), "swap": replace(kc, e3g=replace(e3g, mutant="swap")),
        "hscale": replace(kc, e4=replace(e4, mutant="hscale")), "hpair": replace(kc, e4=replace(e4, mutant="hpair")),
        "nodecay": replace(kc, e4=replace(e4, rec=replace(e4.rec, mutant="nodecay"))),
        "kskip3": replace(kc, e3=replace(e3, e1=replace(e3.e1, mutant="kskip"))),
        "kskip5": replace(kc, e5=replace(e5, mutant="kskip")),
        "bf16": replace(kc, e4=replace(e4, rec=replace(e4.rec, state="bf16"))),
        "rc": replace(kc, tag="rc"),
        "acq_wave": replace(kc, acq="wave"),
        "acq_load": replace(kc, acq="load"),
        "pre_s5": replace(kc, pre="s5", acq="load"),
        "pre_s4": replace(kc, pre="s4"),
        "lpad": replace(kc, lpad="fit"),
        "cbatch": replace(kc, e3=replace(e3, cepi="batch")),
        "cbatch_g4": replace(kc, e3=replace(e3, cepi="batch", cgrp=4)),
        "cbatch_cpre": replace(kc, e3=replace(e3, cepi="batch", cpre=e3.e1.splits == 1)),
        "cbatch_split": replace(kc, e3=replace(e3, cepi="batch", e1=replace(e3.e1, splits=2))),
        "lpad_cepi": replace(kc, lpad="fit", e3=replace(e3, cepi="lds")) if e3.e1.splits == 1 else replace(kc, lpad="fit"),
        "pre_s45": replace(kc, pre="s45"),
        "pre_s4w2": replace(kc, pre="s4", warm=2 * e4.rec.chunks),
        "pre_s4_bf16": replace(kc, pre="s4", e4=replace(e4, rec=replace(e4.rec, state="bf16"))),
        "cepi": replace(kc, e3=replace(e3, cepi="lds")) if e3.e1.splits == 1 else kc,
        "cpre": replace(kc, e3=replace(e3, cpre=True)) if e3.e1.splits == 1 else kc,
    }[variant]
    v.check()
    kd = G.build_chain_kernel(v)
    assert kd.name == G.chain_kernel_name(v)
    assert "define amdgpu_kernel" in ll(kd)


def test_chain_names_distinct():
    kc = G.ChainKnobs(tp=1, bucket=32)
    e3, e3g, e4, e5 = kc.parts()
    names = {G.chain_kernel_name(x) for x in (kc, replace(kc, tag="rc"), replace(kc, mutant="early"), replace(kc, delay=True),
                                               replace(kc, e4=replace(e4, mutant="hpair")), replace(kc, stages="4"))}
    assert len(names) == 6


@pytest.mark.parametrize("splits,cpre", [(1, False), (1, True), (2, False)])
def test_e3_lowers(splits, cpre):
    k3 = G.E3Knobs(e1=E.E1Knobs(k=2560, bucket=64, splits=splits, **SOP), tp=1, cpre=cpre, dbg=True)
    assert "define amdgpu_kernel" in ll(G.build_e3_kernel(k3))
    assert G.e3_params(k3)[-4:] == ["tab", "convw", "qkvz", "dbg3"]


def test_e3g_e4_params():
    kg = G.E3gKnobs(dbg=True)
    assert G.e3g_params(kg)[-4:] == ["alog", "dtb", "gates", "dbg3g"]
    k4 = G.E4Knobs(rec=GR.RecKnobs(map="rowi", ld_nt=True, dbg=True))
    assert G.e4_params(k4) == ["tab", "qkvz", "gates", "normw", "out", "dbg", "rows", "layers", "oq", "oqs"]
    assert G.e4_params(replace(k4, hand="none")) == GR.lb_params(k4.rec)
    assert G.chain_params()[-5:] == ["rows", "l0", "nl", "maxit", "gridn"]


def test_refusals():
    with pytest.raises(ValueError):
        G.E3Knobs(e1=E.E1Knobs(w="bf16", k=2560)).check()
    with pytest.raises(NotImplementedError):
        G.E3Knobs(e1=E.E1Knobs(k=2560, yq=True, **SOP)).check()
    with pytest.raises(ValueError):
        G.E3Knobs(e1=E.E1Knobs(k=2560, splits=2, **SOP), cpre=True).check()
    with pytest.raises(ValueError):
        G.E3gKnobs(e1=E.E1Knobs(k=2560)).check()
    with pytest.raises(ValueError):
        G.E4Knobs(hand="none", mutant="hpair").check()
    with pytest.raises(ValueError):
        G.E4Knobs(hand="none", obf=False).check()
    kc = G.ChainKnobs(tp=1, bucket=64)
    e3, e3g, e4, e5 = kc.parts()
    with pytest.raises(ValueError):  # 256-thread stages cannot run in the 512-thread chain
        replace(kc, e5=replace(e5, waves=4)).check()
    with pytest.raises(ValueError):
        replace(kc, e5=replace(e5, bar="persist")).check()
    with pytest.raises(ValueError):
        replace(kc, e4=replace(e4, hand="none")).check()
    with pytest.raises(ValueError):
        replace(kc, stages="35x").check()
    with pytest.raises(ValueError):  # the XCD-leader acquire is unsound (gdn_lb_stale): refused
        replace(kc, acq="xcd").check()
    with pytest.raises(ValueError):  # warm without the s4 prefetch
        replace(kc, warm=2).check()
    with pytest.raises(ValueError):
        replace(kc, pre="s3").check()
    with pytest.raises(ValueError):  # LDS: two big staging buffers next to each other do not fit
        replace(kc, e3=replace(e3, e1=replace(e3.e1, kw=4, xsrc="lds", xslay="chunk", xscale="glb")),
                e5=replace(e5, kw=4, xsrc="lds", xslay="chunk", xscale="glb", xjunk=True, xpad=32)).check()


def test_acq_stale_kernels_lower():
    from kernels.gfx950.qwen38_mono.large_batch import gdn_lb_stale as AS

    for p in AS.PROTOS:
        assert "define amdgpu_kernel" in ll(AS.build_acq_stale_kernel(p))
    with pytest.raises(ValueError):
        AS.build_acq_stale_kernel("nope")


def test_rec_arena_words():
    assert G.rec_arena_words(GR.RecKnobs(map="rowi")) == 2 * (128 + 8 + 4)
    assert G.rec_arena_words(GR.RecKnobs(chain=4, lanes=8)) == 8 * 2 * 128 * 4 + 2 * (4 * 128 + 32 + 4)


def test_pre_s4_keeps_base_ir():
    """The S4 state warm builds the recurrence before the edge only when it is on: a base build's IR does not depend on
    the knob's existence (names and bodies equal to a chain built with pre="none" explicitly)."""
    kc = G.ChainKnobs(tp=2, bucket=16)
    a = ll(G.build_chain_kernel(kc))
    b = ll(G.build_chain_kernel(replace(kc, pre="none", warm=0)))
    assert a == b
    w = ll_unnamed(G.build_chain_kernel(replace(kc, pre="s4")), G.chain_kernel_name(replace(kc, pre="s4")))
    assert w != ll_unnamed(G.build_chain_kernel(kc), G.chain_kernel_name(kc))


@pytest.mark.parametrize("tp,bucket", [(1, 32), (2, 64), (1, 128)])
def test_lpad_fit_pool(tp, bucket):
    """lpad="fit": the pool is the stages' own pool plus an exclusive pad that takes it just past half a CU (one CTA
    per CU), never more than the entry pad's pool."""
    from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

    kc = G.ChainKnobs(tp=tp, bucket=bucket)
    fit = G._pool_bytes(G.build_chain_kernel(replace(kc, lpad="fit")))
    entry = G._pool_bytes(G.build_chain_kernel(kc))
    assert ONE_CTA_PER_CU_WORDS * 4 < fit <= entry
    with pytest.raises(ValueError):
        replace(kc, lpad="none").check()
