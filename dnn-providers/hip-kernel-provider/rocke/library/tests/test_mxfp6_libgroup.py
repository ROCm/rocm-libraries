# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the grouped (MoE) GEMM path of the MXFP6 tile library: ``Mxfp6GemmSpec.group_table`` (validation,
launch predicates, kernel arguments, the IR's table load and its must-fail mutants) and ``builders.gfx950.mxfp6_gemm.
libgroup`` (routing, group layout, host verification: a host-emulated correct grouped output passes, the ``gt_zero`` /
``gt_swap`` emulations and a stale padding row fail)."""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest
from builders.gfx950.mxfp6_gemm import libgroup as LG
from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import layout as L


def test_routes_and_layout():
    assert LG.route("uniform:4:3") == [3, 3, 3, 3]
    assert LG.route("list:5,0,7") == [5, 0, 7]
    ms = LG.route("topk:16:40:4:1.2:7")
    assert len(ms) == 16 and sum(ms) == 160 and max(ms) <= 40  # each token: 4 distinct experts
    assert LG.route("topk:16:40:4:1.2:7") == ms  # seeded
    offs, mpad, tab = LG.group_layout([5, 0, 70, 64, 1, 130], 64)
    assert offs == [0, 64, 64, 192, 256, 320] and mpad == 512
    assert tab.tolist() == [0, 2, 2, 3, 4, 5, 5, 5] and tab.dtype == np.int32  # the empty group owns no tile row
    st = LG.route_stats([5, 0, 70, 64, 1, 130], 64)
    assert st["Mp"] == 192 and st["nonzero"] == 5 and st["tile_rows"] == 8


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_group_table_spec(fam):
    sg, sp = LG.base_spec("64x64x256", fam, "e2m3", 8192, "family", {})
    assert sg.group_table and not sp.group_table
    assert G.validate_spec(sg)[0]
    assert not G.validate_spec(replace(sg, runtime_shape=False))[0]
    assert not G.validate_spec(replace(sp, mutant="gt_zero"))[0]  # group mutants need a group-table spec
    assert G.runtime_problem_ok(sg, 512, 4096, 8192, 1)[0]
    assert not G.runtime_problem_ok(sg, 512, 4096, 8192, 2)[0]  # the groups come from the table: batch 1
    assert not G.runtime_problem_ok(sg, 500, 4096, 8192, 1)[0]  # padded groups: M a multiple of tile_m
    for sched in ("dp", "persistent"):
        s = replace(sg, schedule=sched)
        ka = G.kernargs(s, 1, 2, 3, 4, 5, 512, 4096, 8192, 1, stride_b=100, stride_sb=10, gtab=0xABC)
        lda = sg.tile_m * 6 // 8 if fam == "am" else 0  # am: one [K][tile_m] slab per tile row
        base = G.runtime_kargs(replace(s, group_table=False), (1, 2, 3, 4, 5), 512, 4096, 8192, 1, lda=lda,
                               strides=G.packed_strides(s, 512, 4096, 8192)[:1] + (100,)
                               + G.packed_strides(s, 512, 4096, 8192)[2:3] + (10,)
                               + G.packed_strides(s, 512, 4096, 8192)[4:])
        assert ka[:-8] == base and struct.unpack("<Q", ka[-8:]) == (0xABC,)  # gtab last, everything else unchanged
        with pytest.raises(ValueError):
            G.kernargs(s, 1, 2, 3, 4, 5, 512, 4096, 8192, 1)  # a group-table launch needs the table
    mus = LG.mutants_of(G, sg, 6)
    assert "gt_zero" in mus and "gt_swap" in mus and "batch_swap" not in mus
    assert "gt_swap" not in LG.mutants_of(G, sg, 5)  # odd group count: group ^ 1 may not exist


def test_group_table_ir():
    from rocke.core.ir_serialize import serialize

    sg, _ = LG.base_spec("64x64x256", "ak", "e2m3", 8192, "family", {})
    txt = serialize(G.build_mxfp6_gemm(sg))
    assert "gtab" in txt and txt.count("s_load_dword") == 1
    assert "s_load_dword" not in serialize(G.build_mxfp6_gemm(replace(sg, mutant="gt_zero")))
    assert "s_load_dword" not in serialize(G.build_mxfp6_gemm(replace(sg, group_table=False)))
    assert G.kernel_name(sg) != G.kernel_name(replace(sg, group_table=False))


def test_host_verify_catches_group_errors(tmp_path):
    from builders.gfx950.mxfp6_gemm import libbench as LB

    ms = [5, 0, 70, 64, 1, 130]
    data = LB.Data(tmp_path, 42, 1)
    sg, sp = LG.base_spec("64x64x256", "ak", "e2m3", 2048, "family", {})
    gd = LG.GroupData(data, L, G, sg, sp, ms, 256, 2048, "int", 3)

    def emulate(sel, grouped=True):
        rows = gd.mpad if grouped else len(ms) * gd.mp
        out = np.zeros((rows, 256), np.uint16)
        for e, m in enumerate(ms):
            if not m:
                continue
            o = gd.offs[e] if grouped else e * gd.mp
            a = gd.a_grp if grouped else gd.a_pad[e]
            a0 = gd.offs[e] if grouped else 0
            op = type(a)(rows=m, k=2048, codes=a.codes[a0:a0 + m], scales=a.scales[a0:a0 + m], fmt="e2m3")
            rc, _ = L.reference_f64(op, gd.src[sel(e) % 3][1])
            out[o:o + m] = L.f32_to_bf16_bits(rc.astype(np.float32))
        return out

    assert gd.check(L, emulate(lambda e: e), True)["pass"]
    assert gd.check(L, emulate(lambda e: e, False), False)["pass"]
    assert not gd.check(L, emulate(lambda e: 0), True)["pass"]  # gt_zero
    assert not gd.check(L, emulate(lambda e: e ^ 1), True)["pass"]  # gt_swap
    bad = emulate(lambda e: e)
    bad[gd.offs[0] + 10] = 0x3F80  # a padding row of group 0 written (must stay +-0)
    r = gd.check(L, bad, True)
    assert not r["pass"] and r["pad_nonzero"] > 0


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_dispatch_fits_the_problem(fam):
    """SB-1: the shipped work group (WG_MULT) shrinks to the problem when it does not fit (MT1 > G_WG_MT1 here, so not
    the G regime); problems it fits are dispatched exactly as before (config_spec + schedule)."""
    from kernels.gfx950.mxfp6_gemm import library as LIB

    if fam == "am":  # 128x96 am: WG_MULT (2, 2), measured native-best on G (G_WG_NONE): the SB-1 path
        g0, g1 = LIB.WG_MULT[(128, 96, 256, fam)]
        s = LIB.dispatch(128, 96, 256, fam, "e2m3", 96, 4096, 128)  # M = 96 < the WG's g1 x 96 tokens
        assert (s.tile_m, s.tile_n) == (96, 128 * g0) and g1 == 2
        assert G.runtime_problem_ok(s, 96, 4096, 8192, 128)[0]
    fit = LIB.dispatch(64, 64, 256, fam, "e2m3", 2048, 2048, 4)
    assert fit == LIB.config_spec(64, 64, 256, fam, "e2m3", schedule=fit.schedule, epi_lds=fit.epi_lds,
                                  **({k: getattr(fit, k) for k in LIB.PERSIST_KNOBS.get((64, 64, 256, fam), {})}))


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_dispatch_g_regime(fam):
    """G / MoE regime (one token tile row per entry, MT1 <= 64): a work group of up to G_WG_ROWS weight rows along N
    (the activation tile read once for g0 N tiles), data-parallel, with the measured per-config knobs."""
    from kernels.gfx950.mxfp6_gemm import library as LIB

    s = LIB.dispatch(64, 64, 256, fam, "e2m3", 64, 4096, 64)
    assert (s.tile_m, s.tile_n, s.schedule, s.edge_tiles) == (64, 512, "dp", True)
    assert s.b_direct == LIB.G_WG[(64, 64, 256, fam)][1].get("b_direct", False)
    assert G.runtime_problem_ok(s, 64, 4096, 8192, 64)[0]
    assert LIB.dispatch(64, 64, 256, fam, "e2m3", 64, 256, 64).tile_n == 256  # capped by N
    assert LIB.g_wg(64, 64, 256, fam, 128, 4096) is None  # two token tile rows: not the G regime
    assert LIB.g_wg(256, 256, 256, fam, 256, 4096) is None  # MT1 > G_WG_MT1 and not in G_WG
    assert LIB.g_wg(128, 128, 256, fam, 128, 4096) == LIB.G_WG[(128, 128, 256, fam)]  # measured entry
    assert LIB.g_wg(512, 64, 256, fam, 64, 4096) is None  # no wider WG within G_WG_ROWS


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_grouped_spec(fam):
    """library.grouped_spec: the G work group when every group fits one token tile row, else the shipped WG along
    N with g1 = 1; group_table, group_m 1, edge tiles; it builds and runs its own packed problem."""
    from kernels.gfx950.mxfp6_gemm import library as LIB

    dec = LIB.grouped_spec(256, 32, 256, fam, "e2m3", 4096, 16, k_target=7168)
    assert (dec.tile_m, dec.tile_n, dec.group_table, dec.group_m) == (32, 512, True, 1)
    pre = LIB.grouped_spec(256, 128, 256, fam, "e2m3", 4096, 565, k_target=7168)
    assert pre.tile_m == 128 and pre.group_table and G.validate_spec(pre)[0]
    assert G.runtime_problem_ok(pre, 128 * 1156, 4096, 7168, 1)[0]
    _, pad = LG.base_spec("256x32x256", fam, "e2m3", 7168, "grouped", {}, n=4096, max_tokens=16)
    assert not pad.group_table and pad.tile_n == 512
