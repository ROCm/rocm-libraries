# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of probe P-L5 (``large_batch/qsa_lb.py``, ``qsa_ref.py``): knob validation, FlyTape's gang rule, the LDS
map, the host references' selection / position rules, and IR lowering of representative builds. Hardware numerics
(every stage vs ``qsa_ref``, mutants, stale twins) run in ``hackathon-prep/qwen38/truemono-lb/p-l5/pl5_bench.py``."""

from __future__ import annotations

import numpy as np
import pytest

from kernels.gfx950.qwen38_mono.large_batch import qsa_lb as Q
from kernels.gfx950.qwen38_mono.large_batch import qsa_ref as R


def test_ft_gang_matches_flytape_rule():
    # FlyTape bodies_qsa.gang_size: max(2, min(32, grid // batch, rounds, grid & -grid)) floored to a power of two
    for batch in (1, 4, 8, 16, 32, 40, 64, 128):
        for ctx in (64, 192, 2112, 8256, 32832):
            rounds = -(-min(ctx, 2051) // 64)
            g = max(2, min(32, 256 // batch, rounds, 256))
            want = 1 << (g.bit_length() - 1)
            assert Q.ft_gang(batch, ctx) == want


@pytest.mark.parametrize("kw", [dict(waves=3), dict(gang=3), dict(walk="x"), dict(sel="x"), dict(pf=3),
                                dict(qlds=True), dict(kw16=True), dict(score="mfma", scan_its=2), dict(mutant="nope")])
def test_knob_validation(kw):
    with pytest.raises(ValueError):
        Q.resolve(Q.QsaKnobs(**kw))


def test_kernel_name_lists_deviations_only():
    assert Q.kernel_name(Q.TP2, Q.QsaKnobs(batch=8, ctxcap=192)) == "q38lb_qsa_h12k1_b8c192"
    n = Q.kernel_name(Q.TP1, Q.QsaKnobs(batch=8, ctxcap=8256, sel="v2", qloc=True, tag="rc"))
    assert n.startswith("q38lb_qsa_h24k2_") and "selv2" in n and "qloc" in n and n.endswith("_rc")


@pytest.mark.parametrize("k", [Q.QsaKnobs(batch=8, ctxcap=8256), Q.QsaKnobs(batch=8, ctxcap=8256, waves=8, pf=1,
                                                                            qloc=True, qlds=True, mpad=True, sela=True),
                               Q.QsaKnobs(batch=128, ctxcap=32832, sel="nib", qloc=True)])
def test_lds_fits(k):
    words = Q.lds_words(k)
    assert words * 4 <= 160 * 1024
    assert words * 4 > 80 * 1024  # one CTA per CU


def test_select_ref_ties_to_lower_block():
    rng = np.random.default_rng(3)
    blocks = 600
    s = rng.random(blocks).astype(np.float32)
    s[:] = np.float32(1.0)  # every key tied: the 512 lowest blocks
    assert (R.select_ref(s, blocks) == np.arange(512)).all()
    s = rng.random(blocks).astype(np.float32)
    keys = s.view(np.uint32).astype(np.int64) >> 15
    sel = R.select_ref(s, blocks)
    kth = np.sort(keys)[::-1][511]
    assert len(sel) == 512 and (np.diff(sel) > 0).all()
    assert keys[sel].min() >= kth and (keys > kth).sum() <= 512


def test_positions():
    assert (R.positions(130, None) == np.arange(131)).all()
    sel = np.arange(512) * 2
    p = R.positions(4100, sel)  # blocks = 1025 > 512, tail = 4101 % 4 = 1
    assert len(p) == 512 * 4 + 1 and p[-1] == 1025 * 4 and p[4] == 8


def test_bf16_roundtrip_and_fp8_table():
    x = np.array([1.0, -2.5, 3.14159, 1e-3], np.float32)
    y = R.bf16_to_f32(R.f32_to_bf16(x))
    assert np.allclose(y, x, rtol=2 ** -8)
    assert R.E4M3[0x38] == 1.0 and R.E4M3[0x7E] == 448.0 and np.isnan(R.E4M3[0x7F])


@pytest.mark.parametrize("shape,k", [
    (Q.TP2, Q.QsaKnobs(batch=8, ctxcap=192)),
    (Q.TP1, Q.QsaKnobs(batch=8, ctxcap=8256, walk="phase", sel="v1", stamp=True)),
    (Q.TP2, Q.QsaKnobs(batch=128, ctxcap=32832, score="mfma", kw16=True, sel="v2", sela=True, spf=True, qloc=True,
                       gearly=True, mpad=True, kvcp="ntsc1")),
    (Q.TP1, Q.QsaKnobs(batch=64, ctxcap=32832, waves=8, pf=1, qloc=True, qlds=True, sel="nib", scan_its=4)),
])
def test_ir_lowers(shape, k):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    ll = lower_kernel_to_llvm(Q.build_qsa_kernel(shape, k), arch="gfx950")
    text = ll if isinstance(ll, str) else str(ll)
    assert "define" in text and Q.kernel_name(shape, k) in text
    # the inline-asm VMEM hazard pad (hz) is applied to every asm global_* with an SGPR base
    assert "s_nop 4" in text


def test_floor_kernel_lowers():
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    assert "q38lb_qsafloor" in str(lower_kernel_to_llvm(Q.build_floor_kernel(), arch="gfx950"))
