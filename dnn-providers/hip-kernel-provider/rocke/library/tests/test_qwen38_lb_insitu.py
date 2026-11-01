# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-C5 (``kernels.gfx950.qwen38_mono.large_batch.insitu``): CPU checks.

* every form x mutant x plan the WP-C5 bench uses builds and lowers (Python engine) and passes the CCL peer-address
  lint (peer stores / loads with ``sc0 sc1``, no atomics / buffer ops / scalar loads on peer or arena addresses);
* names are unique per knob set, invalid combinations are refused, the arena regions do not overlap;
* the LLVM text carries what the hardware proofs rely on: the GEMM's scaled MFMA, the 8-byte system-scope pushes to
  every peer base (lamport) or to the own arena plus a per-lane flag store (pull), the bounded polls;
* the plain GEMM's builds are untouched by this module (it subclasses, it does not edit ``gemm.py``).
The GPU proofs (bitwise against the host rank-order sum over 3 launches, must-fail stale-read twins, shuffled timing)
run in ``hackathon-prep/qwen38/truemono-lb/wp-c5``.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from kernels.gfx950.qwen38_mono.ccl import lint as LI
from kernels.gfx950.qwen38_mono.large_batch import gemm as G
from kernels.gfx950.qwen38_mono.large_batch import insitu as X
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"
SOP = dict(w="mx", k=3072, depth=4, wlay="tile", xslay="chunk", xscale="glb", sop=True, splits=6)
PLANS = [
    G.GemmKnobs(bucket=16, xu=False, **SOP),
    G.GemmKnobs(bucket=128, xa=2, tb=4, **SOP),
    G.GemmKnobs(bucket=64, waves=4, **{**SOP, "k": 1536, "splits": 3}),
    G.GemmKnobs(bucket=16, k=2560),  # splits=1: the unit's epilogue pushes
]


def _ll(k):
    return _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor=FLAVOR)


CASES = [X.InsituKnobs(gemm=g, ranks=n, form=f) for g in PLANS for f in ("lamport", "pull") for n in (2, 4)]
CASES += [X.mutant(CASES[0]), X.mutant(CASES[2]), replace(CASES[0], gemm=replace(PLANS[0], mutant="kskip")), replace(CASES[0], ranks=1)]
FOLD = [X.InsituKnobs(gemm=g, ranks=n, form=f, cons="fold") for g in PLANS[:3] for f in ("lamport", "pull") for n in (2, 4)]
FOLD += [X.mutant(FOLD[0]), X.mutant(FOLD[2]), replace(FOLD[0], ranks=1)]
CASES += FOLD
LITE = [X.InsituKnobs(gemm=PLANS[1], form=f, gb="lite") for f in X.FORMS] + [X.InsituKnobs(gemm=PLANS[0], form="none")]


@pytest.mark.parametrize("kn", CASES, ids=[X.insitu_name(k) for k in CASES])
def test_builds_lowers_and_lints(kn):
    k = X.build_insitu_kernel(kn)
    assert k.name == X.insitu_name(kn)
    assert LI.lint_peer_access(k) == []
    text = _ll(k)
    assert "mfma.scale.f32.16x16x128.f8f6f4" in text
    if kn.cons == "fold":  # 16-byte pushes from the fold (lamport: to the N - 1 other ranks), no consumer pass
        if kn.form == "lamport" and kn.ranks > 1:
            assert f"global_store_dwordx4 $0, $1, ${kn.ranks} sc0 sc1" in text and f"${kn.ranks + 1} sc0 sc1" not in text
    elif kn.form == "lamport":
        assert text.count(f"global_store_dwordx2 $0, $1, ${kn.ranks + 1} sc0 sc1") >= 1  # push to every rank
        # the consumer's reset (absent exactly in the noreset must-fail twin)
        assert ("global_store_dwordx4" in text) == (kn.mutant != "noreset")
    else:
        assert "global_store_dwordx2 $0, $1, $2 sc0 sc1\\0As_nop 1" in text or "global_store_dwordx2 $0, $1, $2 sc0 sc1" in text
        assert "global_store_dword $0, $1, off sc0 sc1" in text  # the flag
    names = [p.name for p in k.params]
    assert names[-(kn.ranks + 6):] == [f"peer{r}" for r in range(kn.ranks)] + ["ctrl", "arena", "dst", "rank", "dl_first", "dl"]


def test_names_unique_and_refusals():
    names = [X.insitu_name(k) for k in CASES]
    assert len(names) == len(set(names))
    with pytest.raises(ValueError):
        X.InsituKnobs(gemm=PLANS[0], form="pull", mutant="noreset").check()
    with pytest.raises(ValueError):
        X.InsituKnobs(gemm=PLANS[0], form="lamport", mutant="slot").check()
    with pytest.raises(ValueError):
        X.InsituKnobs(gemm=replace(PLANS[0], fold="none")).check()
    with pytest.raises(ValueError):
        X.InsituKnobs(gemm=PLANS[0], m=2500).check()
    with pytest.raises(ValueError):  # cons=fold needs split-K
        X.InsituKnobs(gemm=PLANS[3], cons="fold").check()
    with pytest.raises(ValueError):  # cons=fold needs at most one unit per CTA per layer (20 x 8 x 6 > 256)
        X.InsituKnobs(gemm=replace(PLANS[1], tb=1), cons="fold").check()


@pytest.mark.parametrize("kn", CASES[:8], ids=[X.insitu_name(k) for k in CASES[:8]])
def test_arena_map(kn):
    lay = X.arena_map(kn)
    spans = [(0, 256), (lay["ones"], lay["ones"] + 1024), (lay["junk"], lay["junk"] + lay["junk_bytes"]),
             (lay["flag"], lay["flag"] + lay["flags"]), (lay["pay_lam"], lay["pay_lam"] + lay["region"]),
             (lay["pay_pull"], lay["pay_pull"] + lay["region"])]
    spans.sort()
    for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        assert a1 <= b0
    assert spans[-1][1] <= lay["nbytes"] < (1 << 31)
    assert lay["blkb"] % 1024 == 0


def test_gemm_builds_unchanged():
    """Building in-situ kernels does not change the plain GEMM's IR (subclass, no shared mutable state)."""
    before = [hashlib.sha256(_ll(G.build_gemm_kernel(g)).encode()).hexdigest() for g in PLANS]
    for kn in CASES[:4]:
        X.build_insitu_kernel(kn)
    after = [hashlib.sha256(_ll(G.build_gemm_kernel(g)).encode()).hexdigest() for g in PLANS]
    assert before == after


@pytest.mark.parametrize("kn", [k for k in FOLD if k.form == "lamport" and k.mutant == "none" and k.ranks > 1][:2])
def test_fold_noreset_twin_drops_only_the_reset(kn):
    real, mut = _ll(X.build_insitu_kernel(kn)), _ll(X.build_insitu_kernel(X.mutant(kn)))
    # the resets (one 16-byte store per other rank and fold iteration) go; the pushes and polls stay
    assert mut.count("global_store_dwordx4") == real.count("global_store_dwordx4") - (kn.ranks - 1) * -(-kn.gemm.tbt * 16 * (kn.R // 8) // kn.gemm.threads)
    assert mut.count("_abort") == real.count("_abort")


@pytest.mark.parametrize("kn", LITE, ids=[X.insitu_name(k) for k in LITE])
def test_lite_barrier_and_exchange_free_twin(kn):
    k = X.build_insitu_kernel(kn)
    assert LI.lint_peer_access(k) == []
    text = _ll(k)
    assert "mfma.scale.f32.16x16x128.f8f6f4" in text
    if kn.form == "none":  # the plain GEMM's fold (bf16 Y stores), no push, no poll
        assert "sc0 sc1\0As_nop 1" not in text and "global_store_dwordx2 $0" not in text and "_abort" not in text
    if kn.gb == "lite":  # non-returning stripe arrival; the launch counter advanced by the last CTA
        assert "global_atomic_add" in text
    with pytest.raises(ValueError):
        X.mutant(X.InsituKnobs(gemm=PLANS[0], form="none"))


# ---------------------------------------------------------------------------------------------------- WP-C5 fix round
# form=line x publish (unit_end / comm_wave / counted), ar=twoshot, cons=phase (FlyTape's shape), the hz pad, ar=auto
SPLIT_PLANS = PLANS[:3]
FIX = [X.InsituKnobs(gemm=g, ranks=n, form="line", publish=p) for g in SPLIT_PLANS for n in (2, 4) for p in X.PUBLISH]
FIX += [X.InsituKnobs(gemm=g, ranks=n, form="lamport", ar="twoshot") for g in PLANS for n in (2, 4)]
FIX += [X.InsituKnobs(gemm=g, ranks=n, form=f, cons="phase") for g in SPLIT_PLANS for n in (2, 4) for f in ("lamport", "line")]
FIX_MUT = [X.mutant(k) for k in FIX[::3]]


def _asm_ops(kd):
    out = []

    def walk(region):
        for op in region.ops:
            if op.name == "tile.inline_asm":
                out.append(op)
            for r in op.regions:
                walk(r)

    walk(kd.body)
    return out


@pytest.mark.parametrize("kn", FIX + FIX_MUT, ids=[X.insitu_name(k) for k in FIX + FIX_MUT])
def test_fix_round_builds_lower_and_lint(kn):
    k = X.build_insitu_kernel(kn)
    assert k.name == X.insitu_name(kn)
    assert LI.lint_peer_access(k) == []
    text = _ll(k)
    assert "mfma.scale.f32.16x16x128.f8f6f4" in text
    N = kn.ranks
    if kn.form == "line":  # 16-byte line pushes to every rank's line payload, flags with the per-lane 64-bit store
        assert f"global_store_dwordx4 $0, $1, ${N + 1} sc0 sc1" in text
        assert "global_store_dword $0, $1, off sc0 sc1" in text
        if kn.publish == "comm_wave":  # the LDS block tile: more LDS traffic than the unit_end twin
            ue = _ll(X.build_insitu_kernel(replace(kn, publish="unit_end")))
            assert text.count("addrspace(3)") > ue.count("addrspace(3)")
        if kn.mutant == "slot":
            assert X.mutant(replace(kn, mutant="none")) == kn
    if kn.ar == "twoshot":  # reduce-scatter: the fold pushes to ONE (owner) base; the all-gather pushes to every rank
        assert "global_store_dwordx2 $0, $1, $2 sc0 sc1\\0As_nop 1" in text
        assert f"global_store_dwordx2 $0, $1, ${N + 1} sc0 sc1" not in text
        assert f"global_store_dwordx4 $0, $1, ${N + 1} sc0 sc1" in text


@pytest.mark.parametrize("kn", [FIX[0], FIX[2], FIX[len(FIX) // 2]], ids=lambda k: X.insitu_name(k))
def test_hz_pads_every_sgpr_vmem_asm_and_nothing_else(kn):
    """hz (default): every inline asm issuing VMEM with an SGPR input starts with ``s_nop 4``; hz=False is the same
    kernel without exactly those pads (the pre-fix emission; the harness checks it against the pre-fix module)."""
    on, off = X.build_insitu_kernel(kn), X.build_insitu_kernel(replace(kn, hz=False))
    a_on, a_off = _asm_ops(on), _asm_ops(off)
    assert len(a_on) == len(a_off)
    padded = 0
    for x, y in zip(a_on, a_off):
        t, t0 = x.attrs["template"], y.attrs["template"]
        sg = any(v in t0 for v in X._VMEM) and "s" in [c.strip() for c in y.attrs["constraints"].split(",")]
        if sg:
            assert t == "s_nop 4\n" + t0
            padded += 1
        else:
            assert t == t0
    assert padded > 0
    off.name = on.name
    assert _ll(off) != _ll(on)
    assert X.insitu_name(replace(kn, hz=False)) == X.insitu_name(kn) + "_nohz"


def test_fix_round_names_unique_and_refusals():
    names = [X.insitu_name(k) for k in CASES + FIX + FIX_MUT]
    assert len(names) == len(set(names))
    for bad in (dict(form="lamport", publish="counted"), dict(form="pull", publish="comm_wave"),
                dict(form="pull", ar="twoshot"), dict(form="line", cons="fold"), dict(form="lamport", ar="twoshot", cons="fold"),
                dict(form="lamport", cons="phase", gb="lite"), dict(form="none", cons="phase"), dict(form="lamport", publish="x"),
                dict(form="lamport", ar="threeshot"), dict(form="line", mutant="noreset"), dict(form="lamport", ar="auto")):
        with pytest.raises(ValueError):
            X.InsituKnobs(gemm=PLANS[0], **bad).check()
    with pytest.raises(ValueError):  # line needs split-K (the fold writes the 16-byte lines)
        X.InsituKnobs(gemm=PLANS[3], form="line").check()
    with pytest.raises(ValueError):  # twoshot needs a peer
        X.InsituKnobs(gemm=PLANS[0], form="lamport", ar="twoshot", ranks=1).check()


def test_ar_auto_resolves_from_the_table(monkeypatch):
    kn = X.InsituKnobs(gemm=PLANS[1], ar="auto")
    assert X.resolve(kn) == replace(kn, ar="oneshot")  # no table entry: the defaults
    monkeypatch.setitem(X.AUTO, (2, 128), dict(form="lamport", cons="fold"))
    r = X.resolve(kn)
    assert (r.ar, r.cons, r.form) == ("oneshot", "fold", "lamport")
    assert X.insitu_name(kn) == X.insitu_name(r)
    assert X.build_insitu_kernel(kn).name == X.insitu_name(r)


@pytest.mark.parametrize("kn", [FIX[0], FIX[len(FIX) // 2]], ids=lambda k: X.insitu_name(k))
def test_fix_round_arena_map(kn):
    lay = X.arena_map(kn)
    spans = [(0, 256), (lay["ones"], lay["ones"] + 1024), (lay["junk"], lay["junk"] + lay["junk_bytes"]),
             (lay["flag"], lay["flag"] + lay["flags"]), (lay["pay_lam"], lay["pay_lam"] + lay["region"]),
             (lay["pay_pull"], lay["pay_pull"] + lay["region"]), (lay["pay_line"], lay["pay_line"] + lay["region"]),
             (lay["lflag"], lay["lflag"] + lay["lflags"]), (lay["pay_ag"], lay["pay_ag"] + lay["ag_region"])]
    spans.sort()
    for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
        assert a1 <= b0
    assert spans[-1][1] <= lay["nbytes"] < (1 << 31)
    assert lay["lpb"] * 1024 == lay["blkb"] == kn.blkb
    # the original regions keep their offsets (the pre-fix builds' constants)
    old = X.arena_map(replace(kn, form="lamport", publish="unit_end"))
    assert (old["pay_lam"], old["pay_pull"], old["flag"]) == (lay["pay_lam"], lay["pay_pull"], lay["flag"])


def test_fix_round_leaves_gemm_and_default_builds_alone():
    before = [hashlib.sha256(_ll(G.build_gemm_kernel(g)).encode()).hexdigest() for g in PLANS]
    d0 = _ll(X.build_insitu_kernel(CASES[0]))
    for kn in FIX[:6]:
        X.build_insitu_kernel(kn)
    after = [hashlib.sha256(_ll(G.build_gemm_kernel(g)).encode()).hexdigest() for g in PLANS]
    assert before == after
    assert _ll(X.build_insitu_kernel(CASES[0])) == d0
