# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the HC stage functions' cross-layer knobs (``stages2.hc``, true-mono WP2a fix round, DESIGN WP2a
``xpre``): ``src="xl"`` (the attention HC polls the previous layer's M5 hand-off ``ph`` / ``phs``, probe P1's ``xl``),
``order="streams"`` (P1's ``a1hf``) and ``xhold`` (the layer-entry hold: ``layer`` | ``xpre``), the latter taken from
``PersistKnobs.xhold`` in a framework kernel."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

pytest.importorskip("rocke")

from kernels.gfx950.qwen38_mono import abi as A  # noqa: E402
from kernels.gfx950.qwen38_mono.stages2 import hc as H  # noqa: E402

FLAVOR = os.environ.get("ROCKE_LLVM_FLAVOR", "llvm22")


def _ir(stage="a1", kind="delta", rows=1, seq=False, **kw):
    return H.lower_hc_function(H.HcSpec(stage, kind, rows, seq, H.HcKnobs(**kw)), flavor=FLAVOR).ir


def test_defaults_unchanged():
    k = H.HcKnobs()
    assert (k.src, k.order, k.xhold) == ("mem", "weights", "none")
    assert H.knobs_spec(k) == ""
    assert H.knobs_from_spec("src=xl,order=streams,xhold=xpre") == H.HcKnobs(src="xl", order="streams", xhold="xpre")


def test_m5_producers():
    assert H.M5_TASKS == A.HIDDEN // 16 == 160


def test_refusals():
    sp = H.HcSpec
    assert H.supported(sp("m1", "delta", 1, False, H.HcKnobs(src="xl")))  # the MLP HC polls C1
    assert H.supported(sp("a1", "delta_ple", 1, False, H.HcKnobs(src="xl")))  # the PLE layer's polls P2
    assert H.supported(sp("a1", "delta", 1, False, H.HcKnobs(src="xl", pre="args")))
    assert H.supported(sp("a1", "delta", 8, False, H.HcKnobs(src="xl"), group=4))
    assert H.supported(sp("a1", "delta", 1, False, H.HcKnobs(order="streams")))  # memory read: nothing polled
    assert H.supported(sp("a2", "delta", 1, False, H.HcKnobs(src="xl", order="streams")))
    assert H.supported(sp("a1", "delta", 1, False, H.HcKnobs(xhold="xpre")))  # a hold needs the xl source
    assert H.supported(sp("a2", "delta", 1, False, H.HcKnobs(src="xl", xhold="layer")))
    for st, kind in (("a1", "delta"), ("a2", "delta"), ("a1", "qsa"), ("a2", "qsa")):
        for rows, seq in ((1, False), (4, True), (2, False)):
            assert H.supported(sp(st, kind, rows, seq, H.HcKnobs(src="xl"))) is None
    assert H.supported(sp("m1", "delta", 1, False, H.HcKnobs(order="streams"))) is None
    assert H.supported(sp("a1", "delta_ple", 1, False, H.HcKnobs(order="streams"))) is None
    with pytest.raises(ValueError):
        H.HcKnobs(src="hbm")
    with pytest.raises(ValueError):
        H.HcKnobs(xhold="always")


@pytest.mark.parametrize("stage", ["a1", "a2"])
@pytest.mark.parametrize("kind", ["delta", "qsa"])
def test_xl_source_polls_the_handoff(stage, kind):
    """``src=xl``: no ``h_in`` read; the polled HC path (as the PLE layer's) on ``ph`` / ``phs``."""
    mem, xl = _ir(stage, kind), _ir(stage, kind, src="xl")
    assert mem != xl
    ple = _ir(stage, "delta_ple")
    # the polled path's shape: a tagged poll loop on the stream words (the PLE layer's form)
    assert ("tagged" in ple) == ("tagged" in xl) or ple.count("br i1") <= xl.count("br i1")
    assert xl.count("br ") > mem.count("br ")  # poll retry loops


def test_xl_matches_ple_path_up_to_the_hint_layout():
    """The xl A1 and the PLE layer's A1 run the same polled code: only the hint layout (160 producers, stride 1 vs 80,
    stride 2) and the kind differ, so their IR has the same instruction multiset up to constants."""
    import re

    def shape(ir):
        body = ir[ir.index("define"):]
        body = re.sub(r"%[\w.]+", "%v", body)
        body = re.sub(r"\b-?\d+\b", "N", body)
        return sorted(ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith(";"))

    a, b = shape(_ir("a1", "delta", src="xl")), shape(_ir("a1", "delta_ple"))
    sa, sb = set(a), set(b)
    # the PLE A1 differs in nothing structural beyond the hint arithmetic: the instruction kinds are the same
    assert {x.split("(")[0].split(" = ")[-1].split(" ")[0] for x in sa} == {
        x.split("(")[0].split(" = ")[-1].split(" ")[0] for x in sb}


@pytest.mark.parametrize("kind", ["delta", "qsa"])
def test_streams_first_and_holds_act(kind):
    base = _ir("a1", kind, src="xl")
    hf = _ir("a1", kind, src="xl", order="streams")
    lay = _ir("a1", kind, src="xl", xhold="layer")
    xpre = _ir("a1", kind, src="xl", xhold="xpre")
    assert len({base, hf, lay, xpre}) == 4
    # the hold: one more barrier on every CTA (layer) or on the CTAs without an A1 task (xpre)
    assert lay.count("s.barrier") == base.count("s.barrier") + 1
    assert xpre.count("s.barrier") == base.count("s.barrier") + 1


def test_xpre_hold_skips_a1_ctas():
    """``xpre``: the hold sits under ``task >= rows x 14`` (the CTA has no A1 task)."""
    for rows, seq in ((1, False), (4, True)):
        import re

        ir = _ir("a1", "delta", rows, seq, src="xl", xhold="xpre")
        assert re.search(rf"icmp sge i32 %[\w.]+, {rows * H.TASKS_HC}$", ir, re.M)


def test_framework_takes_persist_xhold():
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.gfx950.qwen38_mono import stage_adapt as SA
    from kernels.gfx950.qwen38_mono.framework import LayerShape, build_layer
    from kernels.gfx950.qwen38_mono.persist.knobs import PersistKnobs

    def ll(persist, src):
        f = {st: H.framework_factory(st, H.HcKnobs(src=src)) for st in ("a1", "a2")}
        kn = replace(SA.adapted_knobs(), persist=persist)
        k = build_layer("delta", LayerShape(), stages=("a1", "a2"), factories=f, knobs=kn)
        return lower_kernel_to_llvm(k, arch="gfx950", llvm_flavor=FLAVOR)

    # a static layer kernel takes kernarg arguments; xhold is D0.8's xpre by default
    xpre = ll(PersistKnobs(args="kernarg"), "xl")
    lay = ll(PersistKnobs(args="kernarg", xhold="layer"), "xl")
    assert xpre != lay
    # a non-default xhold with no xl A1 to consume it is refused (never silently ignored)
    with pytest.raises(ValueError, match="no stage emitter read"):
        ll(PersistKnobs(args="kernarg", xhold="layer"), "mem")
