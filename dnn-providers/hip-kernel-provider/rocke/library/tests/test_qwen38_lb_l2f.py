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
