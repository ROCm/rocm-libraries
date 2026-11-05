# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``kernels.gfx950.dsv41_mono.stage_moe``: the V4.1 K2b routed-MoE stage functions (W15 PORT K2-moe). CPU only.

The GPU proofs (the in-place bitwise gate against ATOM's FlyDSL K2 at every served width, must-fail mutants, the in-situ
timing and WaveScope) run on gfx950 in the atom-dev container: ``hackathon-prep/dsv41/w15/rig/moe``. Results:
``experiments/dsv41-mono/w15/PORT-K2-moe.md``.
"""

from __future__ import annotations

import re

import pytest
from kernels.gfx950.dsv41_mono import stage_moe as M

FLAVOR = "llvm23"
# the ABIs ATOM-dsv41's rk_extern_moe declares (lds_base last): a drift here breaks the link
ATOM_ABI = {
    "xq": ["t", "x", "x8r", "x8rs", "x8", "x8s"],
    "route": ["tag", "t", "bias", "logit", "route"],
    "down": ["tag", "task0", "task1", "w2", "w2s", "sw2", "sw2s", "smq", "smc", "ugf", "mid", "mids", "dst0", "dst1",
             "rank", "tab", "red", "pair", "wsl", "sxl", "sxsl", "contrib", "sdl"],
    "reduce": ["tag", "task", "own", "out"],
    "shm": ["tag", "task", "t0", "n", "w", "ws", "x8", "x8s", "x8rdy", "smid", "smq", "smc", "xl", "xsl", "red", "pair"],
}  # fmt: skip
SH6 = M.Shape(6, 384, 6, 1, False)
SH48 = M.Shape(48, 384, 6, 2, True)
DR5 = M.Shape(5, 128, 3, 1, False)


def _fn(stage, sh=SH6, nt=1, **kn):
    k = M.MoeKnobs(**kn)
    if stage == "xq":
        ker, sym = M.build_xq_kernel(16, k), M.symbol("xq", "c16")
    elif stage == "route":
        ker, sym = M.build_route_kernel(sh, k), M.symbol("route", sh.variant())
    elif stage == "down":
        ker, sym = M.build_down_kernel(sh, nt, k), M.symbol("down", sh.variant(f"t{nt}"))
    elif stage == "reduce":
        ker, sym = M.build_reduce_kernel(sh.tokens, k), M.symbol("reduce", f"n{sh.tokens}")
    else:
        ker, sym = M.build_shm_kernel(k), M.symbol("shm", "t")
    return M._lower(ker, sym, flavor=FLAVOR)


@pytest.mark.parametrize("stage", ["xq", "route", "down", "reduce", "shm"])
def test_abi_matches_atom(stage):
    fn = _fn(stage)
    assert [n for n, _ in fn.abi] == ATOM_ABI[stage] + ["lds_base"]
    assert [n for n, _ in M.ABIS[stage]] == ATOM_ABI[stage]
    assert f"define void @{fn.symbol}(" in fn.ir


@pytest.mark.parametrize("sh", [SH6, SH48, DR5])
def test_down_mfma_count(sh):
    # per (task, slot): 9 K steps of A fp4 x B fp8; the shared expert's fp8 x fp8 steps (2 a wave, the 2nd masked)
    ir = _fn("down", sh).ir
    fp4 = len(re.findall(r"mfma\.scale\.f32\.16x16x128\.f8f6f4\.v4i32\.v8i32\([^)]*, i32 4, i32 0,", ir))
    fp8 = len(re.findall(r"call <4 x float> @llvm.amdgcn.mfma\.scale\.f32\.16x16x128\.f8f6f4\.v8i32\.v8i32\(", ir))
    assert fp4 % 9 == 0 and fp4 > 0
    assert fp8 == 2 * len(M.token_tiles(sh.tokens))  # down_shared: 2 steps a wave a token tile


def test_route_table_offsets_follow_atom_rule():
    # RouteShape: expert_slot0 = 3 picks + 1 + max_u, order0 = + experts, first0 = + picks, mask0 = + max_u + 1
    sh = SH48
    assert sh.expert_slot0 == 3 * sh.picks + 1 + sh.max_u
    assert sh.order0 - sh.expert_slot0 == sh.experts
    assert sh.first0 - sh.order0 == sh.picks
    assert sh.mask0 - sh.first0 == sh.max_u + 1
    assert sh.tile_slot0 - sh.tile_first0 == sh.max_u + 1


def test_default_is_atom_and_knobs_change_the_ir():
    assert M.MoeKnobs().spec() == ""
    base = _fn("down").ir
    for kn in ({"sop": 1}, {"r_max": 1}, {"sh_first": 0}, {"wfirst": 0}, {"pslot": 1}, {"smap": 1}, {"chains": 1},
               {"pause": 0}):  # fmt: skip
        assert _fn("down", **kn).ir != base, kn
    # NaN-propagating max / min (ATOM's arith.maximumf) where the stage has them: route, xq (down has none)
    for st in ("route", "xq"):
        assert _fn(st, nan="num").ir != _fn(st).ir
    assert M.MoeKnobs(chains=1).tolerance() and not M.MoeKnobs(pslot=1).tolerance()


@pytest.mark.parametrize("bad", [{"nan": "x"}, {"r_max": 3}, {"chains": 4}, {"sop": 2}, {"pslot": 2}])
def test_bad_knobs_raise(bad):
    with pytest.raises(ValueError):
        M.MoeKnobs(**bad)


@pytest.mark.parametrize("stage,mut", [(s, m) for s, ms in M.MUTANTS.items() for m in ms])
def test_mutants_change_the_ir(stage, mut):
    sh = SH6
    assert _fn(stage, sh, mutate=mut).ir != _fn(stage, sh).ir


def test_standalone_builds_take_the_workgroup_id():
    with M.standalone():
        k = M.build_down_kernel(SH6, 1)
    assert [p.name.lstrip("%") for p in k.params][-1] == "tabg"
    with M.standalone():
        with pytest.raises(ValueError):
            M.build_down_kernel(SH6, 2)
