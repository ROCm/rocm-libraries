# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the MXFP6 GEMM split-K / stream-K units (``splitk="fused"``, kernels/gfx950/mxfp6_gemm/splitk.py):
host plan, runtime predicates, kernel-argument layout, validation, IR of the split builds, the order set."""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
import pytest

from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import library as LIB
from kernels.gfx950.mxfp6_gemm import splitk as SK


def _sk(mt0=256, mt1=32, fam="ak", k_target=2048, **over):
    return replace(LIB.family_spec(mt0, mt1, 256, fam, k_target=k_target), splitk="fused", **over)


def test_default_off_and_fields():
    sp = G.Mxfp6GemmSpec()
    assert sp.splitk == "none" and sp.sk_split == 1 and not sp.sk_hybrid and not sp.m_mask
    assert sp.sk_release == "fence"
    for f in ("splitk", "sk_split", "sk_hybrid", "sk_release", "m_mask"):
        assert G.KNOBS[f][0] == getattr(sp, f)


def test_kernel_name_ignores_host_plan_fields():
    sp = _sk(sk_split=2)
    assert G.kernel_name(sp) == G.kernel_name(replace(sp, sk_split=8)) == G.kernel_name(replace(sp, sk_hybrid=True))
    assert G.kernel_name(sp) != G.kernel_name(replace(sp, splitk="none"))


@pytest.mark.parametrize("s", [1, 2, 4, 8])
def test_plan_pure_split(s):
    sp = _sk(sk_split=s)
    m, n, k = 32, 8192, 8192
    p = SK.plan(sp, m, n, k)
    T = (m // sp.tile_m) * (n // sp.tile_n)
    assert p["T"] == T
    if s == 1:
        assert (p["D"], p["s"], p["units"]) == (T, 1, T)
    else:
        assert (p["D"], p["s"], p["c"], p["units"]) == (0, s, k // 128 // s, T * s)
        assert p["ws_bytes"] == T * s * sp.tile_m * sp.tile_n * 4
        assert p["cnt_bytes"] == T * sp.waves_m * sp.waves_n * 4
    assert G.grid_for(sp, m, n, 1) == ((p["units"], 1, 1), (64 * sp.waves_m * sp.waves_n, 1, 1))


def test_plan_hybrid_quantizes_the_last_wave():
    sp = _sk(256, 256, k_target=8192, sk_split=4, sk_hybrid=True)
    W = 256 * SK.occupancy_wgs(sp)
    # 320 tiles: one full wave DP, the 64 left split 4 ways
    p = SK.plan(sp, 256 * 16, 256 * 20, 8192)
    assert p["T"] == 320 and p["D"] == 320 - 320 % W and p["units"] == p["D"] + (320 - p["D"]) * 4
    # a whole number of waves: all data-parallel (no split)
    p = SK.plan(sp, 256 * 16, 256 * 16, 8192)
    assert p["D"] == p["T"] and p["s"] == 1 and p["ws_bytes"] == 0
    # fewer tiles than a wave: all split
    p = SK.plan(sp, 256 * 4, 256 * 4, 8192)
    assert p["D"] == 0 and p["units"] == 64


def test_problem_ok_split_residue():
    sp = _sk(k_target=2048, sk_split=4)
    U, H, L, T0 = G.loop_plan(sp)
    ok, why = G.runtime_problem_ok(sp, 32, 8192, 4 * 2048)
    assert ok, why
    assert not G.runtime_problem_ok(sp, 32, 8192, 4 * 2048 + 128)[0]  # 4 does not split K / 128 evenly / residue
    # the split length must be in the build's residue class: K / s = T0 + j U
    bad_k = 4 * 128 * (T0 + U) + 4 * 128 if U > 1 else None
    if bad_k is not None:
        assert not G.runtime_problem_ok(sp, 32, 8192, bad_k)[0]
    assert not G.runtime_problem_ok(sp, 31, 8192, 8192)[0]  # M not a tile multiple without m_mask / edge
    assert G.runtime_problem_ok(replace(sp, m_mask=True), 1, 8192, 8192)[0]  # decode: any M with m_mask
    assert G.runtime_problem_ok(replace(sp, m_mask=True), 77, 8192, 8192)[0]


def test_kernargs_tail_layout():
    sp = _sk(sk_split=4)
    m, n, k = 32, 8192, 8192
    ka = G.kernargs(sp, 1, 2, 3, 4, 5, m, n, k, ws=0x1000, cnt=0x2000)
    base = G.kernargs(replace(sp, splitk="none", sk_split=1), 1, 2, 3, 4, 5, m, n, 2048 + 0 * k)  # dp layout length
    assert len(ka) == len(base) + 8 + 8 + 8 == 128  # one kernarg round trip: two s_load_dwordx16, one wait
    ws, cnt, D, sc = struct.unpack("<QQii", ka[-24:])
    p = SK.plan(sp, m, n, k)
    assert (ws, cnt, D, sc & 0xFFFF, sc >> 16) == (0x1000, 0x2000, p["D"], p["s"], p["c"])
    with pytest.raises(ValueError):
        G.kernargs(sp, 1, 2, 3, 4, 5, m, n, k)  # split units need ws / cnt
    assert G.descriptor(sp)["params"][-4:] == ["ws", "cnt", "sk_dp", "sk_sc"]


def test_validate_rules():
    ok = lambda s: G.validate_spec(s)[0]  # noqa: E731
    assert ok(_sk())
    assert not ok(replace(_sk(), runtime_shape=False))
    assert not ok(replace(_sk(), schedule="persistent"))
    assert not ok(replace(_sk(), rt_tail="switch"))
    assert not ok(replace(_sk(), probe="mfma_only"))
    assert not ok(replace(_sk(), sk_release="xx"))
    assert not ok(replace(LIB.family_spec(256, 32, 256, "ak"), m_mask=True))  # m_mask needs the fused path
    assert not ok(replace(LIB.family_spec(256, 32, 256, "ak"), mutant="sk_drop"))
    assert set(SK.SK_SHOULD_FAIL) <= set(SK.SK_MUTANTS)
    for mu in SK.SK_MUTANTS:
        assert ok(_sk(sk_split=3, mutant=mu))
        assert not ok(_sk(sk_split=1, mutant=mu))
        # two partials commute: reordering them is not a mutant
        assert ok(_sk(sk_split=2, mutant=mu)) == (mu not in ("sk_order", "sk_arrival"))
    assert ok(replace(_sk(256, 64, "am"), m_mask=True))
    # line-ring A (line2m: tile_m a multiple of 128) has no exact-range descriptor
    l2 = replace(LIB.family_spec(256, 128, 256, "ak", k_target=2048), splitk="fused")
    if l2.canon_ring == "line2m":
        assert ok(l2) and not ok(replace(l2, m_mask=True))
        assert SK.granule(l2) % 8 == 0


@pytest.mark.parametrize("fam,tile", [("ak", (256, 32)), ("am", (256, 64)), ("ak", (256, 256)), ("am", (128, 128))])
def test_split_builds(fam, tile):
    for over in ({}, {"m_mask": True}, {"sk_release": "wt"}, *({"sk_split": 4, "mutant": mu} for mu in SK.SK_MUTANTS)):
        sp = _sk(*tile, fam=fam, **over)
        assert G.validate_spec(sp)[0], G.validate_spec(sp)[1]
        k = G.build_mxfp6_gemm(sp)
        names = [p.name for p in k.params]
        assert names[-4:] == ["ws", "cnt", "sk_dp", "sk_sc"]


def test_non_split_ir_unchanged_by_the_fields():
    """The split-K fields at their defaults leave the IR of a library build unchanged (the full byte-identity sweep
    over every preset / family / config / dispatch spec is mxfp6-gemm-roofline/sk/ident.py)."""
    from rocke.core.ir_serialize import serialize

    sp = LIB.config_spec(128, 128, 256, "ak")
    a = serialize(G.build_mxfp6_gemm(sp))
    b = serialize(G.build_mxfp6_gemm(replace(sp, sk_split=4, sk_hybrid=True)))  # host-only fields, splitk none
    assert a == b


def test_order_set_partials_and_fold():
    from builders.gfx950.mxfp6_gemm.skbench import order_operands
    from kernels.gfx950.mxfp6_gemm import layout as L

    m, n, k, s = 16, 12, 3 * 1024, 3
    A, B, want = order_operands(m, n, k, s)
    av, bv = L.decode_operand(A), L.decode_operand(B)
    kc = k // s
    parts = np.stack([av[:1, j * kc:(j + 1) * kc] @ bv[:, j * kc:(j + 1) * kc].T for j in range(s)], -1)[0]
    acc = parts[:, 0].astype(np.float32)
    for j in range(1, s):
        acc = (acc + parts[:, j].astype(np.float32)).astype(np.float32)
    assert np.array_equal(acc, want)
    # every column's exact sum is 1; the split-order fold keeps it only when the 1 comes after both large partials
    assert np.allclose(parts.sum(-1), 1.0)
    # the reverse order (mutant sk_order) changes some columns
    rev = parts[:, s - 1].astype(np.float32)
    for j in range(s - 2, -1, -1):
        rev = (rev + parts[:, j].astype(np.float32)).astype(np.float32)
    assert not np.array_equal(rev, want)
    # own partial first (mutant sk_arrival) changes some columns for a reducer j >= 2; j = 1 only swaps the first two
    # addends (fp32 addition commutes), the same bits
    for j in range(1, s):
        alt = parts[:, j].astype(np.float32)
        for jj in range(s):
            if jj != j:
                alt = (alt + parts[:, jj].astype(np.float32)).astype(np.float32)
        assert np.array_equal(alt, want) == (j == 1)


def test_select_short_grids_split_and_full_grids_stay():
    sp = LIB.family_spec(256, 32, 256, "ak", k_target=None)
    s = SK.select(sp, 1, 8192, 8192)  # decode: 32 tiles, M < tile_m
    assert s.splitk == "fused" and s.m_mask and s.sk_split > 1 and G.runtime_problem_ok(s, 1, 8192, 8192)[0]
    big = LIB.family_spec(256, 256, 256, "ak", k_target=8192)
    assert SK.select(big, 16384, 16384, 8192) is big  # 4096 tiles: the dispatch kernel unchanged
    few = SK.select(big, 1024, 1024, 8192)  # 16 tiles
    assert few.splitk == "fused" and few.sk_split > 1 and G.runtime_problem_ok(few, 1024, 1024, 8192)[0]
    assert G.validate_spec(few)[0]


@pytest.mark.parametrize("args", [(256, 32, 256, "ak", 1, 8192, 8192, 1), (256, 32, 256, "ak", 77, 4096, 4096, 1),
                                  (128, 128, 256, "am", 512, 512, 8192, 1), (64, 64, 256, "ak", 256, 512, 4096, 3)])
def test_select_config_runs_its_problem(args):
    mt0, mt1, du, fam, m, n, k, b = args
    sp = SK.select_config(mt0, mt1, du, fam, "e2m3", m, n, k, b)
    assert G.validate_spec(sp)[0], G.validate_spec(sp)[1]
    assert G.runtime_problem_ok(sp, m, n, k, b)[0], G.runtime_problem_ok(sp, m, n, k, b)[1]
    if m < mt1:
        assert sp.m_mask and sp.tile_m == mt1  # the native tile (no WG_MULT row block past M)


def test_libbench_hooks():
    from builders.gfx950.mxfp6_gemm import libbench as LB
    from builders.gfx950.mxfp6_gemm import libtiles as LT

    d = LB.cell_spec(LT.Tile.parse("256x32x256"), "ak", "e2m3", LT.Problem("dec", "256x32x256", 1, 1, 8192, 8192), {}, "select")
    assert d["splitk"] == "fused" and d["m_mask"]
    po, _ = LB._probe_over(d)
    assert po is None or (po["splitk"] == "none" and not po["m_mask"])
    mus = LB._lib_mutants("ak")
    assert set(SK.SK_MUTANTS) - set(SK.SK_SHOULD_FAIL) <= set(mus) and not set(SK.SK_SHOULD_FAIL) & set(mus)


@pytest.mark.parametrize("prob", [(256, 32, 256, "ak", 16, 7808, 3072), (128, 32, 256, "ak", 4, 14976, 4608),
                                  (32, 32, 512, "ak", 2, 9728, 10752), (256, 32, 256, "ak", 1, 23104, 7680),
                                  (512, 32, 256, "ak", 3, 57344, 8192), (256, 64, 256, "am", 16, 7808, 3072)])
def test_dispatch_serves_decode(prob):
    """library.dispatch on M < MT1 (decode): the lane's m_mask split-K / unit grid, runnable on its problem."""
    mt0, mt1, du, fam, m, n, k = prob
    sp = LIB.dispatch(mt0, mt1, du, fam, "e2m3", m, n, 1, k_target=k)
    assert sp.splitk == "fused" and sp.m_mask
    assert G.validate_spec(sp)[0] and G.runtime_problem_ok(sp, m, n, k)[0]
    if sp.sk_split > 1:
        assert sp.sk_static and sp.sk_release == "wt"


def test_owner_mode_rules():
    """sk_fix="owner": split s - 1 polls its peers' flags -- only with the static reducer, never with a mutant that
    would leave a flag down (the owner would spin), and only when every unit is co-resident (problem_ok)."""
    ok = lambda s: G.validate_spec(s)[0]  # noqa: E731
    ow = _sk(sk_split=4, sk_static=True, sk_fix="owner", sk_release="wt")
    assert ok(ow) and not ok(replace(ow, sk_static=False)) and not ok(replace(ow, sk_fix="xx"))
    for mu in ("sk_noflag", "sk_noreset", "sk_arrival"):
        assert not ok(replace(ow, mutant=mu))
    for mu in ("sk_drop", "sk_order"):
        assert ok(replace(ow, mutant=mu))
    # 32 tiles x 4 = 128 units: co-resident; 7 x 32 tiles x 4 = 896 > 256 x 1 slots: refused
    assert G.runtime_problem_ok(ow, 32, 8192, 8192)[0]
    assert not G.runtime_problem_ok(ow, 32, 7 * 8192, 8192)[0]
    # flags per (split tile, split, wave)
    p = SK.plan(ow, 32, 8192, 8192)
    assert p["cnt_bytes"] == p["T"] * 4 * ow.waves_m * ow.waves_n * 4
    names = [x.name for x in G.build_mxfp6_gemm(ow).params]
    assert names[-4:] == ["ws", "cnt", "sk_dp", "sk_sc"]


@pytest.mark.parametrize("rel", ["fence", "wt", "wt1"])
def test_release_modes_build(rel):
    sp = _sk(sk_split=4, sk_static=True, sk_release=rel)
    assert G.validate_spec(sp)[0]
    G.build_mxfp6_gemm(sp)


def test_am_any_m_routes_to_masked_units():
    """am (M-contiguous A) with M % 16 != 0 or M < tile_m: the M-padded producer layout (ceil16(M) codes per K row,
    lda / batch stride from gemm.mcontig_rows, device image zero-padded by layout.pack_operand) and the fused path's
    m_mask units -- library.dispatch routes every such problem there, so select_candidates is never empty."""
    from kernels.gfx950.mxfp6_gemm import layout as L

    assert [G.mcontig_rows(m) for m in (1, 15, 16, 17, 33, 48)] == [16, 16, 16, 32, 48, 48]
    rng = np.random.default_rng(0)
    for rows in (1, 7, 20, 32):
        op = L.Operand(rows, 128, rng.integers(0, 256, (rows, 96), dtype=np.uint8),
                       rng.integers(0, 256, (rows, 4), dtype=np.uint8))
        img = L.pack_operand(op, "mcontig", 64)
        assert img.nbytes == 128 * G.mcontig_rows(rows) * 6 // 8
        mc = L.mcontig_codes(L.pad_rows(op, G.mcontig_rows(rows))).reshape(-1)
        assert np.array_equal(img, mc)
        if rows % 16 == 0:
            assert np.array_equal(img, L.mcontig_codes(op).reshape(-1))
    for m, n, k in ((1, 4096, 4096), (8, 4096, 4096), (17, 4096, 4096), (33, 4096, 4096), (100, 4096, 2048),
                    (1000, 8192, 4096), (5, 4096, 4128), (2050, 4096, 4096)):
        c = LIB.select_candidates(m, n, k, 1, "am", top=2)
        assert c, (m, n, k)
        for _, _, sp in c:
            assert G.validate_spec(sp)[0] and G.runtime_problem_ok(sp, m, n, k)[0]
            if m % 16 or m < sp.tile_m:
                assert sp.splitk == "fused" and sp.m_mask
            ka = G.runtime_kargs(sp, (0, 0, 0, 0, 0), m, n, k, ws=16, cnt=16)
            assert struct.unpack_from("<i", ka, 52)[0] == G.mcontig_rows(m) * 6 // 8  # lda
            assert G.packed_strides(sp, m, n, k)[0] == k * G.mcontig_rows(m) * 6 // 8
    # ak keeps its own rule (canonical A: any M with edge tiles / m_mask, no padding)
    sp = LIB.dispatch(128, 32, 256, "ak", "e2m3", 17, 4096, k_target=4096)
    assert G.packed_strides(sp, 17, 4096, 4096)[0] == 17 * 4096 * 6 // 8


def test_kargs_packed_vs_wide():
    """sk_kargs "packed" (default): the split kernel's arguments end at 128 bytes (s | c << 16 in sk_sc), so the
    prologue fetches them behind one wait; "wide" (A/B only) keeps s and c in two dwords (132 bytes)."""
    sp = _sk(sk_split=4)
    m, n, k = 32, 8192, 8192
    p = SK.plan(sp, m, n, k)
    ka = G.kernargs(sp, 1, 2, 3, 4, 5, m, n, k, ws=16, cnt=32)
    kw = G.kernargs(replace(sp, sk_kargs="wide"), 1, 2, 3, 4, 5, m, n, k, ws=16, cnt=32)
    assert len(ka) == 128 and len(kw) == 132 and ka[:-4] == kw[:-8]
    assert struct.unpack("<ii", kw[-8:]) == (p["s"], p["c"])
    assert struct.unpack("<i", ka[-4:])[0] == p["s"] | p["c"] << SK.SC_SHIFT
    assert G.validate_spec(replace(sp, sk_kargs="wide"))[0] and not G.validate_spec(replace(sp, sk_kargs="x"))[0]
    names = [x.name for x in G.build_mxfp6_gemm(replace(sp, sk_kargs="wide")).params]
    assert names[-5:] == ["ws", "cnt", "sk_dp", "sk_ns", "sk_kc"]
    # c >= 2^16 K tiles per split: refused by the packed layout only
    big = 128 * (1 << 16)
    one = _sk(sk_split=1)
    assert not G.runtime_problem_ok(one, 32, 8192, big)[0]


@pytest.mark.parametrize("fam,tile", [("ak", (256, 32)), ("am", (128, 64))])
def test_mask_mutants(fam, tile):
    """mask_over / mask_under (C rows < M + 1 / < M - 1 stored): m_mask builds only; they change the IR."""
    ok = lambda s: G.validate_spec(s)[0]  # noqa: E731
    for s_ in (1, 4):
        sp = _sk(*tile, fam=fam, sk_split=s_, m_mask=True)
        base = G.build_mxfp6_gemm(sp)
        for mu in SK.MASK_MUTANTS:
            assert ok(replace(sp, mutant=mu)) and not ok(replace(sp, m_mask=False, mutant=mu))
            assert str(G.build_mxfp6_gemm(replace(sp, mutant=mu))) != str(base)


def test_static_single_split_is_data_parallel_only():
    """sk_static with sk_split 1: every unit a whole tile -- no unit-map division, no fix-up (no workspace atomics)
    in the IR; same kernel arguments; the split mutants do not apply, the mask mutants do."""
    ok = lambda s: G.validate_spec(s)[0]  # noqa: E731
    sp = _sk(sk_split=1, m_mask=True)
    sx = replace(sp, sk_static=True)
    assert ok(sx) and G.runtime_problem_ok(sx, 5, 8192, 2048)[0]
    ir0, ir1 = str(G.build_mxfp6_gemm(sp)), str(G.build_mxfp6_gemm(sx))
    assert "atomic" in ir0 and "atomic" not in ir1 and len(ir1) < len(ir0)
    assert [x.name for x in G.build_mxfp6_gemm(sx).params][-4:] == ["ws", "cnt", "sk_dp", "sk_sc"]
    assert G.kernel_name(sx) != G.kernel_name(sp)
    for mu in SK.SK_MUTANTS:
        assert not ok(replace(sx, mutant=mu))
    for mu in SK.MASK_MUTANTS:
        assert ok(replace(sx, mutant=mu))
    p = SK.plan(sx, 5, 8192, 2048)
    assert p["D"] == p["T"] and p["s"] == 1
