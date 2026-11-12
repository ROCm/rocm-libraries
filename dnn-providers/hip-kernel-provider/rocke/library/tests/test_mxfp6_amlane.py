# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the MXFP6 ``am`` lane additions (no GPU): the M-contiguous -> canonical transpose kernel
(``transpose.py``: host model, a numpy emulation of the kernel's index math with its must-fail mutants, specs,
``pick``), the ``mc_dma`` knob (b96 LDS DMA of the mcontig A), ``library.am_plan`` and libbench's ``via``.
The GPU checks (byte-exact transposes, end-to-end GEMMs, mutants) are in ``mxfp6-gemm-roofline/amlane/h``."""

from dataclasses import replace

import numpy as np
import pytest

from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import library as LB
from kernels.gfx950.mxfp6_gemm import transpose as T


def _op(rows, k, seed=3):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 64, (rows, k), dtype=np.uint8)
    return L.Operand(rows, k, L.pack_codes(codes), rng.integers(100, 140, (rows, k // 32), dtype=np.uint8))


def test_reference_inverts_mcontig_codes():
    for rows, k in ((32, 128), (96, 256), (16, 64)):
        op = _op(rows, k)
        mc = L.mcontig_codes(op)
        assert np.array_equal(T.reference(mc.reshape(-1), rows, k), op.codes)
    for rows, k in ((1, 64), (3, 128), (21, 64), (45, 128)):  # any M: the M-padded producer layout (ceil16(M) codes)
        op = _op(rows, k)
        mc = L.pack_operand(op, "mcontig", 16)
        assert mc.size == T.in_bytes(rows, k)
        assert np.array_equal(T.reference(mc, rows, k), op.codes)


def test_scale_reference():
    s = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)  # [K/32][M]
    assert np.array_equal(T.scale_reference(s.reshape(-1), 6, 128), s.T)


def _emulate(spec: T.TrSpec, mc: np.ndarray, m: int, k: int) -> np.ndarray:
    """numpy model of ``build_transpose``'s data movement (one entry): row-contiguous 16-byte pieces -> dword
    scatter into the swizzled 16-byte slots -> ``ds_read_b96_tr_b6`` per 16-lane group -> 48-byte row stores
    (row48) / the staged row image (the same bytes). Applies the spec's mutant."""
    tm, tk, nblk = spec.tm, spec.tk, spec.tm // 16
    a2, sh = T.mc_swizzle(nblk)
    msk = (1 << a2) - 1
    lda, ldd = T.padded_m(m) * 3 // 4, k * 3 // 4
    src = np.concatenate([np.asarray(mc, np.uint8).reshape(-1), np.zeros(64, np.uint8)])
    out = np.full(m * ldd, 0xA5, np.uint8)
    nq = tk // 64
    for m0 in range(0, m, tm):
        col0 = m0 * 3 // 4
        for k0 in range(0, k, tk):
            slots = np.zeros((tk * nblk, 16), np.uint8)
            P = 3 * tm // 64
            for kr in range(tk):
                for cc in range(P):
                    g0 = (k0 + kr) * lda + col0 + 16 * cc
                    piece = src[g0:g0 + 16] if g0 + 16 <= len(src) else np.zeros(16, np.uint8)
                    if g0 >= T.in_bytes(m, k):
                        piece = np.zeros(16, np.uint8)
                    for e in range(4):
                        d = 4 * cc + e
                        blk, word = d // 3, d % 3
                        sw = (kr >> sh) & msk if a2 else 0
                        slots[kr * nblk + (blk ^ sw), 4 * word:4 * word + 4] = piece[4 * e:4 * e + 4]
            kgs = [0, 2, 1, 3] if spec.mutant == "kgswap" else [0, 1, 2, 3]
            for bq in range(nblk // 4):
                for q in range(nq):
                    for g in range(4):
                        blk = 4 * bq + g
                        for u in range(4):
                            kg = 4 * q + kgs[u]
                            rows96 = np.zeros((16, 12), np.uint8)
                            for j in range(16):
                                swr = 0 if spec.mutant == "noswz" else ((j >> sh) & msk if a2 else 0)
                                rows96[j] = slots[(16 * kg + j) * nblk + (blk ^ swr), :12]
                            tr = L.tr_b6_model(rows96)  # lane t: row 16 blk + t, K values 16 kg' .. + 15
                            for t in range(16):
                                row = m0 + 16 * blk + t + (1 if spec.mutant == "rowoff" else 0)
                                if row >= m:
                                    continue
                                o = row * ldd + (k0 + 64 * q) * 3 // 4 + 12 * u
                                out[o:o + 12] = tr[t]
    return out


@pytest.mark.parametrize("spec", [T.TrSpec(tm=64, tk=64, kc=64, waves=1), T.TrSpec(tm=128, tk=128, kc=256),
                                  T.TrSpec(tm=64, tk=128, kc=128, waves=2)])
@pytest.mark.parametrize("dm", [16, 11, 0])
def test_emulated_kernel_matches_reference_and_mutants_fail(spec, dm):
    assert T.validate(spec)[0]
    m, k = (2 * spec.tm - dm if dm else 3), 2 * spec.tk  # a partial last M range; M % 16 != 0; M < 16
    op = _op(m, k, seed=5)
    mc = L.pack_operand(op, "mcontig", 16)
    ref = T.reference(mc, m, k).reshape(-1)
    assert np.array_equal(_emulate(spec, mc, m, k), ref)
    for mu in ("noswz", "kgswap", "rowoff"):
        if mu == "noswz" and T.mc_swizzle(spec.tm // 16)[0] == 0:
            continue
        assert not np.array_equal(_emulate(replace(spec, mutant=mu), mc, m, k), ref), mu


def test_specs_build_and_names_unique():
    names = set()
    for s in T.variants() + [replace(T.TrSpec(), scales="kmajor")] + [replace(T.TrSpec(), mutant=x) for x in T.MUTANTS]:
        ok, why = T.validate(s)
        assert ok, why
        k = T.build_transpose(s)
        assert k.name == T.kernel_name(s)
        names.add(k.name)
        assert T.lds_bytes(s) <= 160 * 1024
    assert len(names) == len(T.variants()) + 1 + len(T.MUTANTS)
    assert not T.validate(T.TrSpec(tm=96))[0] and not T.validate(T.TrSpec(tk=96))[0]
    assert not T.validate(T.TrSpec(tm=256, tk=256, store="staged"))[0]  # LDS


def test_kargs_and_grid():
    s = T.TrSpec(tm=256, tk=256, kc=1024)
    assert T.grid(s, 2048, 8192, 3) == ((8, 8, 3), (256, 1, 1))
    ka = T.kargs(s, 1, 2, 2048, 8192)
    assert len(ka) == 40
    import struct

    ka2 = T.kargs(s, 1, 2, 2041, 8192)  # any M: input entries of ceil16(M) codes per K row, output of M rows
    assert struct.unpack_from("<qq", ka2, 24) == (2048 * 8192 * 6 // 8, 2041 * 8192 * 6 // 8)
    with pytest.raises(ValueError):
        T.kargs(s, 1, 2, 0, 8192)
    with pytest.raises(ValueError):
        T.kargs(s, 1, 2, 2048, 8192 + 64)
    assert len(T.kargs(replace(s, scales="kmajor"), 1, 2, 2048, 8192, 3, 4)) == 72
    assert T.traffic_bytes(2048, 8192, 4) == 2 * 4 * 2048 * 8192 * 6 // 8


def test_pick():
    for m, k, b in ((2048, 8192, 64), (2048, 8192, 1), (32, 8192, 1), (640, 2880, 128), (1625 // 16 * 16, 2048, 256)):
        s = T.pick(m, k, b)
        assert T.validate(s)[0] and k % s.tk == 0, (m, k, b, s)
    big = T.pick(2048, 8192, 64)
    assert (big.tm, big.tk, big.kc, big.store, big.ring) == (256, 256, 1024, "staged", 1)
    assert -(-2048 // T.pick(2048, 8192, 4).tm) * -(-8192 // T.pick(2048, 8192, 4).kc) * 4 >= T.PICK_MIN_WGS


def test_mc_dma_knob():
    sp = replace(LB.family_spec(256, 256, 256, "am"), mc_dma="x4")  # the am family default is LB.AM_MC_DMA
    assert LB.family_spec(256, 256, 256, "am").mc_dma == LB.AM_MC_DMA
    assert G.kernel_name(sp) == G.kernel_name(replace(sp, mc_dma="x4"))
    x3 = replace(sp, mc_dma="x3")
    assert G.validate_spec(x3)[0]
    assert not G.validate_spec(replace(LB.family_spec(256, 256, 256, "ak"), mc_dma="x3"))[0]
    assert not G.validate_spec(replace(sp, mc_dma="x2"))[0]
    from rocke.core.ir_serialize import serialize

    a = serialize(G.build_mxfp6_gemm(sp, arch="gfx950"))
    b = serialize(G.build_mxfp6_gemm(x3, arch="gfx950"))
    assert a.count("dwords = i:3") == 0 and b.count("dwords = i:3") > 0
    # only the A DTL widths differ (same instruction count, same stage image)
    assert a.count("tile.async_buffer_load_lds_addr") == b.count("tile.async_buffer_load_lds_addr")


def test_am_plan():
    keep = dict(LB.AM_MODEL)
    try:
        LB.AM_MODEL.clear()
        p = LB.am_plan(256, 256, 256, "e2m3", 2048, 2048, 8192, 64)
        assert p.path == "native" and p.gemm == LB.dispatch(256, 256, 256, "am", "e2m3", 2048, 2048, 64)
        t = LB.am_plan(256, 256, 256, "e2m3", 2048, 28672, 8192, 1, path="transpose")
        assert t.path == "transpose" and t.gemm.a_layout == "canonical" and t.transpose is not None
        assert t.workspace_bytes(2048, 8192, 1) == 2048 * 8192 * 6 // 8
        LB.AM_MODEL[(256, 256, 256)] = (1.2, 1.25, 600.0, 6e9)
        assert LB.am_plan(256, 256, 256, "e2m3", 2048, 28672, 8192, 1).path == "transpose"
        assert LB.am_plan(256, 256, 256, "e2m3", 2048, 512, 8192, 1).path == "native"
        # any M: the transpose reads the M-padded input; K must divide by the transpose's K step
        odd = LB.am_plan(256, 256, 256, "e2m3", 2041, 28672, 8192, 1, path="transpose")
        assert odd.path == "transpose" and odd.workspace_bytes(2041, 8192, 1) == 2041 * 8192 * 6 // 8
        with pytest.raises(ValueError):
            LB.am_plan(256, 256, 256, "e2m3", 2048, 28672, 8192 + 32, 1, path="transpose")
    finally:
        LB.AM_MODEL.clear()
        LB.AM_MODEL.update(keep)


def test_libbench_via():
    from builders.gfx950.mxfp6_gemm import libbench as LBB
    from builders.gfx950.mxfp6_gemm import libtiles as LT

    tile = LT.Tile(256, 256, 256)
    p = LT.Problem("X", tile.name, 64, 2048, 2048, 8192)
    fam, pre = LBB._via(tile, LBB.Config("c", "256x256x256", "am", via="transpose"), "e2m3", p)
    assert fam == "ak" and pre["tm"] == T.pick(2048, 8192, 64).tm
    keep = dict(LB.AM_MODEL)
    try:
        LB.AM_MODEL.clear()
        assert LBB._via(tile, LBB.Config("c", "256x256x256", "am", via="plan"), "e2m3", p) == ("am", None)
    finally:
        LB.AM_MODEL.update(keep)
    with pytest.raises(ValueError):
        LBB._via(tile, LBB.Config("c", "256x256x256", "ak", via="transpose"), "e2m3", p)


def test_shipped_model_is_sane():
    """The fitted AM_MODEL: every library config of an entry, positive terms, and the measured crossovers (native am
    on the 256x256 S problem, the transpose path at N = 28672 and on the tiny 64x64x512 S problem; never for a config
    whose ak is slower than its am)."""
    lib = set(LB.LIBRARY)
    for key, (r3, r4, kappa, f0) in LB.AM_MODEL.items():
        assert key in lib and r3 > 0 and r4 > 0 and kappa > 0 and f0 > 0, key
    if (256, 256, 256) in LB.AM_MODEL:
        assert LB.am_plan(256, 256, 256, "e2m3", 2048, 2048, 8192, 64).path == "native"
        assert LB.am_plan(256, 256, 256, "e2m3", 2048, 28672, 8192, 1).path == "transpose"
    if (64, 64, 512) in LB.AM_MODEL:
        assert LB.am_plan(64, 64, 512, "e2m3", 2048, 2048, 8192, 4).path == "transpose"
    for key, (r3, r4, _, _) in LB.AM_MODEL.items():
        if max(r3, r4) < 1.0:
            assert LB.am_predicted(*key, 2048, 1 << 20, 8192, 1) < 1.0


def test_mc_copy_knob():
    """mc_copy late / fence: the second-half tr_b6 copy as three explicit v_mov_b32 at the fragment's MFMA (am lane,
    mid-tile core loop); auto is the compiler's alignment copy. The must-fail mutant swaps two copied dwords."""
    am = replace(LB.dispatch(128, 224, 256, "am", "e2m3", 1792, 2048, 32, k_target=8192), mc_copy="auto")
    late, fence = replace(am, mc_copy="late"), replace(am, mc_copy="fence")
    assert G.validate_spec(late)[0] and G.validate_spec(fence)[0]
    assert len({G.kernel_name(am), G.kernel_name(late), G.kernel_name(fence)}) == 3
    assert not G.validate_spec(replace(am, mc_copy="x"))[0]
    assert not G.validate_spec(replace(LB.family_spec(256, 256, 256, "ak"), mc_copy="late"))[0]
    assert not G.validate_spec(replace(am, mutant="mc_copyswap"))[0]  # nothing to swap on the auto copy
    assert G.validate_spec(replace(late, mutant="mc_copyswap"))[0]
    from rocke.core.ir_serialize import serialize

    texts = {nm: serialize(G.build_mxfp6_gemm(s, arch="gfx950")) for nm, s in
             (("auto", am), ("late", late), ("fence", fence), ("mut", replace(late, mutant="mc_copyswap")))}
    mov = {nm: t.count("v_mov_b32 $0, $1") for nm, t in texts.items()}  # (auto has one of its own elsewhere)
    n = mov["late"] - mov["auto"]
    assert n > 0 and n % 3 == 0 and mov["fence"] == mov["late"]
    assert texts["fence"].count("~{memory}") > texts["late"].count("~{memory}")
    assert texts["late"].count("ds_read_b96_tr_b6") == texts["auto"].count("ds_read_b96_tr_b6")
    assert texts["mut"] != texts["late"] and mov["mut"] == mov["late"]


def test_am_transpose_covers_problems_without_native_am():
    """select / select_plan rank the transpose + ak pair for a config whose native am kernel refuses the problem;
    predict_seconds then is the ak kernel's plus the transpose's."""
    from kernels.gfx950.mxfp6_gemm import gemm as G2

    m, n, k, b = 1185, 835, 2816, 1
    cfg = (256, 256, 256)
    native = LB._am_native_ok(*cfg, "e2m3", m, n, k, b, {})
    t = LB.predict_seconds(*cfg, "am", m, n, k, b)
    if not native:
        assert t == pytest.approx(LB.predict_seconds(*cfg, "ak", m, n, k, b) + LB.am_transpose_seconds(m, k, b))
        assert LB.am_plan(*cfg, "e2m3", m, n, k, b).path == "transpose"
    assert LB.am_transpose_seconds(m, k, b) > LB.SB3_LAUNCH_S
    c = LB.select_candidates(m, n, k, b, "am", top=100)
    assert c and all(G2.runtime_problem_ok(s, m, n, k, b)[0] for _, _, s in c)


def test_am_mc_copy_table():
    """AM_MC_COPY configs ship mc_copy (promoted and STRUCT paths); ak, other configs and explicit overrides keep
    theirs."""
    for (mt0, mt1, du), mc in LB.AM_MC_COPY.items():
        sp = LB.dispatch(mt0, mt1, du, "am", "e2m3", 2048, 2048, 32, k_target=8192)
        assert sp.mc_copy == mc and G.validate_spec(sp)[0], (mt0, mt1, du)
        assert LB.dispatch(mt0, mt1, du, "ak", "e2m3", 2048, 2048, 32, k_target=8192).mc_copy == "auto"
        assert LB.config_spec(mt0, mt1, du, "am", mc_copy="auto").mc_copy == "auto"
    assert LB.dispatch(160, 256, 256, "am", "e2m3", 2048, 2560, 32, k_target=8192).mc_copy == "auto"


def test_am_plan_masked_regime_matches_measurement():
    """In the masked native regime am_plan's auto path is the measured winner on every AM_PLAN_MEASURED cell; other
    masked problems follow the M >= 16, K >= AM_MASKED_TR_K rule."""
    for (mt0, mt1, du, b, m, n, k), r in LB.AM_PLAN_MEASURED.items():
        want = "transpose" if r >= LB.AM_MARGIN else "native"
        assert LB.am_plan(mt0, mt1, du, "e2m3", m, n, k, b).path == want, (mt0, mt1, du, b, m, n, k)
    # unmeasured masked problems: the rule
    assert LB.am_plan(64, 64, 256, "e2m3", 645, 7168, 8192, 1).path == "transpose"
    assert LB.am_plan(64, 64, 256, "e2m3", 645, 7168, 4096, 1).path == "native"
    assert LB.am_plan(64, 64, 256, "e2m3", 5, 7168, 16384, 1).path == "native"
