# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of WP-L2e (``large_batch/qsa_lb2.py``, ``qsa_ref2.py``): the default build's LLVM IR equals P-L5's
``qsa_lb`` (kernel name aside), knob validation, the parameter list, the LDS map, every new knob and mutant lowers with
the asm hazard pad, and the ATOM-semantics references' own rules. Hardware numerics (every stage vs ``qsa_ref`` /
``qsa_ref2``, mutants, stale twins, the composed E1 -> core -> E1 chain) run in
``hackathon-prep/qwen38/truemono-lb/wp-l2e/l2e_bench.py``."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kernels.gfx950.qwen38_mono.large_batch import qsa_lb as A
from kernels.gfx950.qwen38_mono.large_batch import qsa_lb2 as Q
from kernels.gfx950.qwen38_mono.large_batch import qsa_ref as R
from kernels.gfx950.qwen38_mono.large_batch import qsa_ref2 as R2
from rocke.core.lower_llvm import lower_kernel_to_llvm


def _ir(mod, shape, kd):
    k = mod.QsaKnobs(**kd)
    return str(lower_kernel_to_llvm(mod.build_qsa_kernel(shape, k), arch="gfx950")).replace(mod.kernel_name(shape, k), "K")


@pytest.mark.parametrize("shape,kd", [
    (Q.TP2, dict(batch=8, ctxcap=192)),
    (Q.TP1, dict(batch=8, ctxcap=8256, sel="v2", pf=1, scan_its=4, mpad=True, sela=True, spf=True, qloc=True)),
    (Q.TP2, dict(batch=128, ctxcap=32832, waves=8, kvcp="ntsc1", sel="v1", pf=1, scan_its=8, score="mfma", kw16=True,
                 mpad=True, spf=True, qloc=True, qlds=True)),
    (Q.TP2, dict(batch=16, ctxcap=192, gang=4, pf=1, mpad=True, gearly=True, qloc=True, walk="phase")),
    (Q.TP1, dict(batch=8, ctxcap=8256, sel="hist", mutant="tie", stress=40, acq="plain", stamp=True)),
])
def test_default_ir_equals_pl5(shape, kd):
    """New knobs off: the same LLVM IR as P-L5's qsa_lb (P-L5's numerics and verdicts carry over)."""
    assert _ir(A, shape, kd) == _ir(Q, shape, kd)


def test_params_default_unchanged_and_extras_appended():
    k = Q.QsaKnobs()
    assert Q.qsa_params(k) == list(A.PARAMS) + list(A.SCAL)
    k2 = Q.QsaKnobs(kvl="page16", sem="atom", oq="mx", pio="e1")
    p = Q.qsa_params(k2)
    assert p[: len(A.PARAMS)] == list(A.PARAMS)
    assert p[len(A.PARAMS): len(A.PARAMS) + 6] == ["pages", "kvt", "cos", "sin", "oqc", "oqs"]
    assert p[len(A.PARAMS) + 6:] == list(A.SCAL) + ["maxp", "oqb", "prs", "pls"]


@pytest.mark.parametrize("kw", [dict(sem="atom"), dict(score="f64"), dict(kvl="page16", sem="atom", sel="hist"),
                                dict(kvl="page16", sem="atom", score="mfma", kw16=True), dict(oq="x"), dict(pio="x"),
                                dict(kvl="x"), dict(pgt="x"), dict(mutant="page"), dict(mutant="qnorm", kvl="page16"),
                                dict(mutant="oqs"), dict(hsel=True), dict(kvl="page16", sem="atom", hsel=True, sel="nib"),
                                dict(kvl="page16", sem="atom", fma64=True), dict(kvl="page16", sem="atom", mutant="hneed")])
def test_knob_validation(kw):
    with pytest.raises(ValueError):
        Q.resolve(Q.QsaKnobs(**kw))


def test_kernel_name():
    assert Q.kernel_name(Q.TP2, Q.QsaKnobs(batch=8, ctxcap=192)) == "q38lb_qsa2_h12k1_b8c192"
    n = Q.kernel_name(Q.TP1, Q.QsaKnobs(batch=8, ctxcap=8256, kvl="page16", sem="atom", score="f64", oq="mx"))
    assert n.startswith("q38lb_qsa2_h24k2_") and "kvlpage16" in n and "sematom" in n and "scoref64" in n and "oqmx" in n


@pytest.mark.parametrize("k", [
    Q.QsaKnobs(batch=8, ctxcap=32832, kvl="page16", sem="atom"),
    Q.QsaKnobs(batch=128, ctxcap=32832, kvl="page16", sem="atom", waves=8, pf=1, qloc=True, qlds=True, mpad=True,
               sela=True, sel="v2"),
    Q.QsaKnobs(batch=8, ctxcap=32832, kvl="page16", sem="atom", qloc=True, sela=True)])
def test_lds_fits(k):
    words = Q.lds_words(k)
    assert 80 * 1024 < words * 4 <= 160 * 1024


ATOM = dict(batch=8, ctxcap=8256, kvl="page16", sem="atom")


@pytest.mark.parametrize("shape,kd", [
    (Q.TP2, dict(batch=8, ctxcap=8256, kvl="page16")),
    (Q.TP2, dict(ATOM, pgt="glb", kvcp="ntsc1")),
    (Q.TP1, dict(ATOM, score="f64", sel="v2", sela=True, spf=True, qloc=True, mpad=True, oq="mx", pio="e1")),
    (Q.TP2, dict(ATOM, score="mfma", scan_its=4, waves=8, pf=1, qloc=True, qlds=True, oq="both", walk="phase")),
    (Q.TP2, dict(batch=8, ctxcap=192, oq="both")),
    (Q.TP1, dict(ATOM, score="f64", fma64=True, hsel=True, sel="v1", sela=True, scan_its=4, mpad=True)),
    (Q.TP2, dict(ATOM, score="mfma", hsel=True, sel="v2", scan_its=8, mutant="hneed")),
] + [(Q.TP2, dict(ATOM, score="f64", oq="both", mutant=m)) for m in ("page", "qnorm", "ikey", "gate", "oqs", "sel16",
                                                                      "cmp", "rope", "scan", "iqrope")])
def test_new_knobs_lower_with_hazard_pad(shape, kd):
    kern = Q.build_qsa_kernel(shape, Q.QsaKnobs(**kd))
    txt = str(lower_kernel_to_llvm(kern, arch="gfx950"))
    assert "q38lb_qsa2_" in txt
    # every inline-asm VMEM with an SGPR operand carries the s_nop 4 pad (lbf_hazard)
    for line in txt.splitlines():
        if "asm" in line and "global_store_dword" in line and "$2" in line:
            assert "s_nop 4" in line


# ------------------------------------------------------------------ references


def test_rope_tables_and_head_match_f64():
    """ATOM's bf16-rounded norm + RoPE is within bf16 rounding of the f64 form of the same math."""
    rng = np.random.default_rng(1)
    cs, sn = R2.rope_tables(300)
    cos, sin = R.bf16_to_f32(cs), R.bf16_to_f32(sn)
    x = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal(256).astype(np.float32)))
    w = R.bf16_to_f32(R.f32_to_bf16((1 + 0.1 * rng.standard_normal(256)).astype(np.float32)))
    y = R2.head(x, w, cos[277], sin[277])
    xd = x.astype(np.float64)
    n = xd / np.sqrt((xd * xd).mean() + 1e-6) * w
    ref = n.copy()
    d = np.arange(64)
    ref[:64] = np.where(d < 32, n[:64] * cos[277] - n[d ^ 32] * sin[277], n[:64] * cos[277] + n[d ^ 32] * sin[277])
    assert np.abs(y - ref).max() <= 4 * 2 ** -8 * np.abs(ref).max()
    assert (R.f32_to_bf16(y) == R.f32_to_bf16(R.bf16_to_f32(R.f32_to_bf16(y)))).all()  # bf16 values


def test_scores_exact_close_to_f64_and_select_rule():
    rng = np.random.default_rng(2)
    iq = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((4, 128)).astype(np.float32)))
    keys = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((900, 128)).astype(np.float32)))
    sx = R2.scores_exact(iq, keys)
    sf = R2.scores_f64(iq, keys)
    assert np.abs(sx - sf).max() <= 2 ** -21 * np.abs(sf).max()
    sel = R2.select(sx, 900)
    assert len(sel) == 512 and (np.diff(sel) > 0).all()
    kth = np.sort(sx)[::-1][511]
    assert sx[sel].min() >= kth
    tied = np.ones(900, np.float32)
    assert (R2.select(tied, 900) == np.arange(512)).all()  # ties to the lower block
    # the 16-bit keys (FlyTape's) coarsen the selection: they differ from the 31-bit rule on near-equal scores
    near = (1.0 + np.arange(900) * 2.0 ** -20).astype(np.float32)  # the exact rule takes the upper 512
    assert not (R2.select(near, 900, key16=True) == R2.select(near, 900)).all()


def test_block_key_pooling_and_rotation():
    rng = np.random.default_rng(4)
    cs, sn = R2.rope_tables(64)
    raw = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((4, 128)).astype(np.float32)))
    w = R.bf16_to_f32(R.f32_to_bf16((0.1 * rng.standard_normal(128)).astype(np.float32)))
    k = R2.block_key(raw, w, R.bf16_to_f32(cs[8]), R.bf16_to_f32(sn[8]))
    k2 = R2.block_key(raw[[1, 0, 3, 2]], w, R.bf16_to_f32(cs[8]), R.bf16_to_f32(sn[8]))
    assert (k == k2).all()  # (r0 + r1) + (r2 + r3) is symmetric in each pair
    k3 = R2.block_key(raw, w, R.bf16_to_f32(cs[9]), R.bf16_to_f32(sn[9]))
    assert not (k == k3).all()  # rotated at the block's first token


def test_atom_model_close_to_golden():
    rng = np.random.default_rng(5)
    q = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((12, 256)).astype(np.float32)))
    k = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((700, 256)).astype(np.float32)))
    v = R.bf16_to_f32(R.f32_to_bf16(rng.standard_normal((700, 256)).astype(np.float32)))
    g = rng.standard_normal((12, 256)).astype(np.float32)
    gold = R2.attention_golden(q, k, v, g)
    mod = R2.attention_atom_model(q, k, v, g)
    mx, l2 = R2.drift(mod, gold)
    assert mx < 2 ** -6 and l2 < 2 ** -7
