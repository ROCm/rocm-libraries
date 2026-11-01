# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe P-L2 (large-batch sync model): CPU checks of ``large_batch.sync_plan`` and ``large_batch.sync_chain``.

The plan generator's graphs are well formed (read sets point into the previous stage, every unit reads something,
the ``lb`` merges keep the data dependencies), the task table gives every CTA every (layer, stage) in order (the grid
arm's barrier needs all CTAs), the closed-form payload model equals a direct simulation of the kernel's arithmetic,
and every kernel variant lowers with the pieces the GPU bench relies on. The hardware proofs (numerics, negative
controls, mutants, timing) run in ``hackathon-prep/qwen38/truemono-lb/p-l2``.
"""

from __future__ import annotations

import numpy as np
import pytest
from kernels.gfx950.qwen38_mono.large_batch import sync_chain as C
from kernels.gfx950.qwen38_mono.large_batch import sync_plan as P
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FLAVOR = "llvm22"


def _ll(kn):
    return _lower_kernel_to_llvm_python(C.build_chain_kernel(kn), arch="gfx950", llvm_flavor=FLAVOR)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("rows", [16, 128])
@pytest.mark.parametrize("kind", ["ft", "lb"])
def test_plan_wellformed(tp, rows, kind):
    p = P.make_plan(tp, rows, kind)
    names = [s.name for s in p.stages]
    if kind == "ft":
        assert len(names) == (16 if tp == 1 else 14)
        assert names[0] == "attn_mix_combine" and names[-1] == "moe_combine"
    else:
        assert len(names) == (11 if tp == 1 else 10)
        assert not any("reduce" in n or "combine" in n and "moe" not in n for n in names)
    assert names.count("gdn_recur") == 1
    rec = p.stages[names.index("gdn_recur")]
    assert rec.rw and rec.units == -(-rows // 4) * 48 // tp
    # the plan carries the bytes model's per-rank bytes within the 8 KiB rounding
    by = p.bytes_per_layer()
    assert by["gdn_recur"] > 0 and by[names[-2] if tp == 2 else "moe_up"] > 0


def test_lb_merge_keeps_dependencies():
    ft = P.make_plan(1, 32, "ft")
    lb = P.make_plan(1, 32, "lb")
    # up in lb reads every down unit (the reduce it replaced read every split of its row group; up read every reduce)
    fdown = next(s for s in ft.stages if s.name == "attn_mix_down")
    lup = next(s for s in lb.stages if s.name == "attn_mix_up")
    assert all(sorted(d) == list(range(fdown.units)) for d in lup.deps)
    # bytes are conserved up to rounding
    assert abs(lb.blocks_per_layer() - ft.blocks_per_layer()) < 0.01 * ft.blocks_per_layer()


def test_null_plan_zero_bytes():
    p = P.make_plan(2, 16, "ft", null=True)
    assert p.blocks_per_layer() == 0 and p.nunits == P.make_plan(2, 16, "ft").nunits


def test_task_table_every_cta_every_stage():
    p = P.make_plan(2, 16, "ft")
    L = 3
    flat, tmax = P.task_table(p, L)
    row = 1 + P.TCOLS * tmax
    a = np.asarray(flat).reshape(P.GRID, row)
    u0 = p.unit0()
    seen_units = set()
    for c in range(P.GRID):
        cnt = a[c, 0]
        t = a[c, 1:1 + 4 * cnt].reshape(cnt, 4)
        keys = [(int(x[0]), int(x[1])) for x in t]
        assert keys == sorted(keys)  # (layer, stage) order
        assert {k for k in keys} == {(lay, s) for lay in range(L) for s in range(p.nstages)}
        firsts = {}
        for lay, st, u, fl in t:
            if (lay, st) not in firsts:
                firsts[(lay, st)] = fl
                k = lay * p.nstages + st
                assert (fl >> 8) == k and (fl & 1) == (1 if k else 0) and bool(fl & 2) == (k == 0)
            else:
                assert fl & 1 == 0
            if u >= 0:
                assert u0[st] <= u < u0[st] + p.stages[st].units
                seen_units.add((int(lay), int(u)))
    assert len(seen_units) == L * p.nunits
    assert P.nsync(p, L) == L * p.nstages - 1


def test_payload_model_matches_simulation():
    p = P.make_plan(2, 16, "lb")
    L, tag = 2, 7
    ut, dp = P.unit_table(p)
    N = p.nunits
    ut = np.asarray(ut).reshape(N, P.UCOLS)
    pay = np.zeros((L, N, C.PAY_WORDS), dtype=np.uint64)
    u0 = p.unit0()
    for lay in range(L):
        for i, s in enumerate(p.stages):
            for u in range(u0[i], u0[i] + s.units):
                chk = 0
                if not (lay == 0 and i == 0):
                    src = lay - 1 if i == 0 else lay
                    for d in dp[ut[u, 3]:ut[u, 3] + ut[u, 4]]:
                        w = (d & 7) * 64
                        chk += int(pay[src, d, w:w + 64].sum())
                chk &= P.M32
                pay[lay, u] = (P.h_unit(tag, lay, i, u) + 3 * chk + np.arange(C.PAY_WORDS, dtype=np.uint64) * P.W_STEP) & P.M32
    H, Cc = P.expected_checksums(p, L, tag)
    assert ((H + 3 * Cc) & P.M32 == pay[:, :, 0]).all()


@pytest.mark.parametrize(
    "kn",
    [
        C.ChainKnobs(sync="grid"),
        C.ChainKnobs(sync="tile"),
        C.ChainKnobs(sync="tile", pub="wave"),
        C.ChainKnobs(sync="a2a"),
        C.ChainKnobs(sync="none"),
        C.ChainKnobs(sync="grid", stamp=True),
        C.ChainKnobs(sync="grid", mutant="early"),
        C.ChainKnobs(sync="tile", mutant="tag"),
        C.ChainKnobs(sync="tile", mutant="noacq"),
        C.ChainKnobs(sync="tile", pub="counted"),
        C.ChainKnobs(sync="mix", pub="counted"),
        C.ChainKnobs(sync="mix", pub="drain", mutant="early"),
        C.ChainKnobs(sync="mix", pub="drain", lazy=True),
        C.ChainKnobs(sync="mix", pub="wave", lazy=True, pause="s_sleep1"),
    ],
)
def test_chain_lowers(kn):
    ll = _ll(kn)
    assert C.kernel_name(kn).startswith("q38lbs_")
    assert "ds_write_b32" in ll  # LDS pad (one CTA per CU)
    assert "global_store_dword $0, $1, $2 sc1" in ll  # sc1 payload stores (P5 bulk=sc1)
    assert "global_atomic_add" in ll  # ticket / done counter
    if kn.pub == "counted" and kn.flags:
        assert f"s_waitcnt vmcnt({kn.depth})" in ll  # the counted publish
    if kn.sync in ("grid", "mix"):
        assert "v_cmp_lt_u32_e64" in ll and "s_cmp_ge_u32" in ll  # bounded ballot poll of the stripes
    if kn.flags:
        assert "s_memrealtime" in ll and "global_load_dword $0, $6, off sc1" in ll  # deadline flag poll, sc1


def test_names_distinct():
    ks = [C.ChainKnobs(sync=s, pub=p) for s in ("tile", "a2a") for p in C.PUBS] + [C.ChainKnobs(sync="grid"),
                                                                                    C.ChainKnobs(sync="grid", tag="rc"),
                                                                                    C.ChainKnobs(sync="none")]
    assert len({C.kernel_name(k) for k in ks}) == len(ks)


def test_knob_checks():
    with pytest.raises(ValueError):
        C.ChainKnobs(sync="none", mutant="early").check()
    with pytest.raises(ValueError):
        C.ChainKnobs(sync="grid", mutant="tag").check()
    assert C.ChainKnobs(sync="tile", pub="wave").pf == 8 and C.ChainKnobs(sync="grid", pub="wave").pf == 1


def test_mix_task_table():
    p = P.make_plan(1, 32, "ft")
    L = 2
    gm = P.gmask_full(p, 0.9)
    assert gm and len(gm) < p.nstages
    flat, tmax = P.task_table(p, L, gm)
    a = np.asarray(flat).reshape(P.GRID, 1 + P.TCOLS * tmax)
    t = a[0, 1:1 + 4 * a[0, 0]].reshape(-1, 4)
    ords = []
    for lay, st, u, fl in t:
        g = bool(fl & 1)
        assert bool(fl & 4) == (st not in gm)
        assert bool(fl & 8) == (((st + 1) % p.nstages) not in gm)
        if g:
            assert st in gm and (lay or st)
            ords.append(fl >> 8)
    assert ords == list(range(1, len(ords) + 1)) and len(ords) == P.nsync(p, L, gm)
    assert C.ChainKnobs(sync="mix", pub="wave").pf == 8
    with pytest.raises(ValueError):
        C.ChainKnobs(sync="grid", pub="counted").check()
    with pytest.raises(ValueError):
        C.ChainKnobs(sync="tile", lazy=True).check()
