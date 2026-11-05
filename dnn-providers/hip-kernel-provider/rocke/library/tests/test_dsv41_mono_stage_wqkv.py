# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.dsv41_mono.stage_wqkv``: the V4.1 K1 ``wqkv_a`` stage function. CPU only (the emitted LLVM text).

The GPU proofs (in-situ bitwise A/B against ATOM's FlyDSL K1, must-fail mutants, timing) run on gfx950 in the
atom-dev container: ``hackathon-prep/dsv41/p2/p2_rig.py``. Results: ``experiments/dsv41-mono/reverse/MONO-STAGES.md``.
"""

from __future__ import annotations

import re

import pytest
from kernels.gfx950.dsv41_mono import stage_wqkv as W

FLAVOR = "llvm23"

_MFMA = re.compile(r"call <4 x float> @llvm\.amdgcn\.mfma\.scale\.f32\.16x16x128\.f8f6f4[^(]*\(([^)]*)\)")


def _ir(rows=6, **kn):
    return W.lower_wqkv_function(rows, W.WqkvKnobs(**kn), flavor=FLAVOR)


def _mfma_args(text):
    return [[a.strip() for a in m.group(1).split(",")] for m in _MFMA.finditer(text)]


def test_symbol_and_abi():
    fn = _ir(6)
    assert fn.symbol == "rocke_dsv41_wqkv_n6"
    assert [n for n, _ in fn.abi] == [n for n, _ in W.ABI] + ["lds_base"]
    assert fn.ffi_types()[:2] == ["int32", "int32"]
    assert f"define void @{fn.symbol}(" in fn.ir


@pytest.mark.parametrize("rows", [1, 6, 16])
def test_five_scaled_mfma_steps_fp8_fp8(rows):
    args = _mfma_args(_ir(rows).ir)
    assert len(args) == 5  # 40 K steps of 128 over 8 waves
    for a in args:
        # cbsz / blgp 0 (fp8 x fp8), op_sel 0 for both scales in the FlyDSL form
        assert a[3].endswith("0") and a[4].endswith("0") and a[5].endswith("0") and a[7].endswith("0")


def test_sop_picks_bytes_with_op_sel():
    args = _mfma_args(_ir(6, sop=2).ir)
    assert [a[5].split()[-1] for a in args] == ["0", "1", "2", "3", "0"]
    assert [a[7].split()[-1] for a in args] == ["0", "1", "2", "3", "0"]
    a1 = _mfma_args(_ir(6, sop=1).ir)
    assert [a[5].split()[-1] for a in a1] == ["0", "1", "2", "3", "0"]
    assert all(a[7].split()[-1] == "0" for a in a1)


def test_default_is_the_flydsl_stage_and_knobs_change_the_ir():
    base = _ir(6).ir
    assert W.WqkvKnobs().spec() == ""
    for kn in ({"pause": 0}, {"poll": "asm"}, {"w_cpol": 0}, {"sop": 1}, {"ra": 1}, {"x_first": True}):
        assert _ir(6, **kn).ir != base, kn


def test_weights_nt_and_mailbox_sc1():
    t = _ir(6).ir
    assert t.count("@llvm.amdgcn.raw.ptr.buffer.load.v4i32(") >= 10
    lines = t.splitlines()
    nt = [x for x in lines if "call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(" in x and x.rstrip().endswith("i32 2)")]
    assert len(nt) == 10  # 5 K steps x (lo, hi) nt weight loads (FlyDSL CM_NT)
    puts = [x for x in lines if "@llvm.amdgcn.raw.ptr.buffer.store.v2i32(" in x and x.rstrip().endswith("i32 16)")]
    assert len(puts) == 1  # the (payload, tag) put at sc1
    assert "s_sleep 1" in t  # the poll's pause (FlyDSL _spin)


@pytest.mark.parametrize("mut", ["put_bit", "chain_drop", "row", "scale_byte", "wave_order"])
def test_mutants_change_the_ir(mut):
    assert _ir(6, mutate=mut).ir != _ir(6).ir


def test_opsel_mutant_needs_sop():
    assert _ir(6, sop=2, mutate="opsel").ir != _ir(6, sop=2).ir
    assert _ir(6, mutate="opsel").ir == _ir(6).ir  # without sop the mutant has nothing to act on


def test_chain_drop_drops_one_mfma():
    assert len(_mfma_args(_ir(6, mutate="chain_drop").ir)) == 4


def test_knob_spec_round_trip():
    k = W.knobs_from_spec("sop=2,ra=1,pause=0,poll=asm,x_first=1")
    assert (k.sop, k.ra, k.pause, k.poll, k.x_first) == (2, 1, 0, "asm", True)
    assert W.knobs_from_spec(k.spec()) == k
    with pytest.raises(ValueError):
        W.knobs_from_spec("nope=1")
    with pytest.raises(ValueError):
        W.WqkvKnobs(sop=3)


def test_rows_bounds():
    with pytest.raises(ValueError):
        W.build_wqkv_kernel(17)
    with pytest.raises(ValueError):
        W.build_wqkv_kernel(1, W.WqkvKnobs(mutate="row"))
