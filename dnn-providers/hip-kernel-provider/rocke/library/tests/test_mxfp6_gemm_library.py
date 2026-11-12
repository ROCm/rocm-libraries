# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the MXFP6 GEMM tile-config library structures (gemm.py library fields, library.py, layout.py).

Host models of the device data paths the GPU suite (``test_mxfp6_gemm_library_gpu.py``) runs: per-wave scale
images, the M-contiguous activation's DMA + transposing-read path and its LDS bank conflicts, the uneven DTL splits,
the depth_u barrier schedule, runtime shapes (vmcnt model invariance over K), the grouped CTA order and the
batched kernel arguments.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import gemm as G
from kernels.gfx950.mxfp6_gemm import layout as L
from kernels.gfx950.mxfp6_gemm import library as LB

LIB = [(mt0, mt1, du) for mt0, mt1, du in LB.LIBRARY]


# ----------------------------------------------------------------- the library and its family defaults


def test_library_is_the_user_list():
    assert len(LIB) == 70 and len(set(LIB)) == 70
    assert sum(du == 512 for _, _, du in LIB) == 13
    assert all(mt0 % 32 == 0 and mt1 % 32 == 0 for mt0, mt1, _ in LIB)


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_every_config_has_a_valid_family_default(fam):
    for mt0, mt1, du in LIB:
        s = LB.family_spec(mt0, mt1, du, fam)
        assert G.validate_spec(s)[0], (mt0, mt1, du, G.validate_spec(s))
        assert (s.tile_n, s.tile_m) == (mt0, mt1)  # MT0 = weight rows (tile_n), MT1 = tokens (tile_m)
        assert G.operand_layout_of(s, "A") == ("mcontig" if fam == "am" else "canonical")
        assert G.lds_bytes(s) * s.occupancy <= 160 * 1024
        U, H, L_, T = G.loop_plan(s)
        assert U % (du // 128) == 0  # whole DepthU periods per loop iteration
        ok, why = G.runtime_problem_ok(s, 16 * mt1, 16 * mt0, 8192, 4)
        assert ok, why


def test_transposes_map_to_families():
    assert LB.TRANSPOSES == {"NN": "ak", "TN": "ak", "NT": "am", "TT": "am"}


def test_presets_are_not_library_specs():
    for nm in G.PRESETS:
        assert not G._library_mode(G.preset(nm)), nm


# ----------------------------------------------------------------- per-wave scale images


@pytest.mark.parametrize("R,Wr", [(32, 32), (64, 32), (96, 48), (128, 64), (256, 128), (160, 80), (288, 144),
                                  (512, 128), (384, 192), (224, 112)])
def test_wtile_image_holds_each_lanes_mfma_scales(R, Wr):
    rng = np.random.default_rng(R * 1000 + Wr)
    rows, kb = 2 * R, 12
    sc = rng.integers(0, 255, size=(rows, kb), dtype=np.uint8)
    if (Wr // 16) > 16:
        pytest.skip("more than 16 MFMA tiles per wave side")
    img = L.pack_scales_wtile(sc, R, Wr).view(np.uint8)
    D = L.wtile_dwords(Wr)
    assert D == G.wtile_dwords(Wr)
    W, nq, ns = R // Wr, rows // R, kb // 4
    assert img.size == ns * nq * W * 64 * D * 4
    for s in range(ns):
        for q in range(nq):
            for w in range(W):
                for lane in range(64):
                    for f in range(Wr // 16):
                        d, i = divmod(f, 4)
                        byte = ((((s * nq + q) * W + w) * 64 + lane) * D + d) * 4 + i
                        assert img[byte] == sc[R * q + Wr * w + 16 * f + lane % 16, 4 * s + lane // 16]


def test_wtile_equals_the_tile_layout_for_128_row_waves():
    sc = np.random.default_rng(3).integers(0, 255, size=(512, 16), dtype=np.uint8)
    assert np.array_equal(L.pack_scales_wtile(sc, 256, 128), L.pack_scales_tile(sc, "16x16x128"))


def test_wtile_kernel_image_size_and_dtl_chunks():
    for mt0, mt1, du in LIB:
        s = LB.family_spec(mt0, mt1, du)
        for X, rows, ws in (("A", s.tile_m, s.waves_m), ("B", s.tile_n, s.waves_n)):
            img = G.scale_img_bytes(s, X)
            assert img == ws * 64 * 4 * G.wtile_dwords(rows // ws) and img % 256 == 0
            sl = G.scale_layout_of(s, X)
            assert sl == f"wtile{rows}x{rows // ws}"


@pytest.mark.parametrize("n,nw", [(1, 4), (3, 4), (5, 4), (8, 4), (2, 8), (9, 8), (6, 1), (12, 2)])
def test_dtl_chunks_cover_every_chunk_with_equal_counts(n, nw):
    ch = G.dtl_chunks(n, nw)
    assert len({len(c) for c in ch}) == 1  # every wave issues the same number (one static vmcnt model)
    assert sorted({x for c in ch for x in c}) == list(range(n))
    assert all(0 <= x < n for c in ch for x in c)


# ----------------------------------------------------------------- M-contiguous activation (am family)


tr_b6 = L.tr_b6_model  # the model the GPU suite pins against the hardware (test_tr_b6_semantics)


def _mc_lds_image(spec, glob: np.ndarray, lda: int, m0: int, k0: int) -> np.ndarray:
    """The LDS stage image the kernel's DMAs write for K tile k0 / 128 of the CTA tile at row m0 (``_setup_mcontig``
    lane math): chunk c lane L -> slot 64 c + L = (k, b'), 16 bytes from global k lda + 3 m0 / 4 + 12 (b' ^ s(k))."""
    nblk = spec.tile_m // 16
    a, sh = G.mc_swizzle(nblk)
    out = np.zeros(128 * nblk * 16, np.uint8)
    glob = np.concatenate([glob, np.zeros(16, np.uint8)])  # past the entry: the exact-range descriptor reads 0
    for c in range(2 * nblk):
        for lane in range(64):
            s = 64 * c + lane
            k, bp = divmod(s, nblk)
            b = bp ^ ((k >> sh) & ((1 << a) - 1)) if a else bp
            src = (k0 + k) * lda + 3 * m0 // 4 + 12 * b
            out[16 * s:16 * s + 16] = glob[src:src + 16]
    return out


def _mc_read_addresses(spec, wm: int, f: int, h: int):
    """Byte address (within the stage) of every lane of fragment f's read h (``_setup_mcontig`` / ``_mc_read``)."""
    nblk = spec.tile_m // 16
    a, sh = G.mc_swizzle(nblk)
    wb0 = wm * (spec.tile_m // spec.waves_m) // 16
    E = 1 << a
    e, q = f % E, f // E
    out = []
    for lane in range(64):
        j, g = lane % 16, lane // 16
        sw = (j >> sh) & (E - 1)
        lane_part = (32 * g + j) * 16 * nblk
        out.append(lane_part + 16 * ((wb0 + e) ^ sw) + 16 * E * q + 256 * nblk * h)
    return out


@pytest.mark.parametrize("tm,wm", [(32, 1), (64, 1), (64, 2), (96, 1), (128, 2), (160, 2), (192, 4), (224, 2),
                                   (256, 2), (288, 2), (320, 1), (384, 4), (448, 1), (512, 4)])
def test_mcontig_dma_and_transposing_reads_give_the_mfma_fragments(tm, wm):
    """Global [K][M] codes -> the DMA's LDS image -> the transposing reads: lane l of fragment f holds row
    16 (wb0 + f) + l % 16's K values 32 (l / 16) .. + 31 (the 16x16x128 MFMA operand), for every wave row."""
    if (tm // wm) % 16 or tm // wm // 16 > 16:
        pytest.skip("not a wave layout")
    spec = replace(G.Mxfp6GemmSpec(), tile_m=tm, waves_m=wm, waves_n=1, a_layout="mcontig")
    M, K = 2 * tm, 256
    op = L.generate(M, K, L.DIST_UNIFORM_CODES, seed=tm, tag=1)
    codes = L.unpack_codes(op.codes, K)
    glob = L.pack_operand(op, "mcontig", tm)
    lda = M * 6 // 8
    for m0 in (0, tm):
        for k0 in (0, 128):
            img = _mc_lds_image(spec, glob, lda, m0, k0)
            for w in range(wm):
                for f in range(tm // wm // 16):
                    halves = []
                    for h in range(2):
                        ad = _mc_read_addresses(spec, w, f, h)
                        assert all(x % 16 == 0 for x in ad)  # the instruction's 128-bit alignment
                        res = np.zeros((64, 12), np.uint8)
                        for g in range(4):
                            rows = np.stack([img[ad[16 * g + j]:ad[16 * g + j] + 12] for j in range(16)])
                            res[16 * g:16 * g + 16] = tr_b6(rows).reshape(16, 12)
                        halves.append(L.unpack_codes(res, 16))
                    got = np.concatenate(halves, axis=1)  # [lane][32 K values]
                    wb0 = w * (tm // wm) // 16
                    for lane in range(64):
                        row = m0 + 16 * (wb0 + f) + lane % 16
                        kk = k0 + 32 * (lane // 16)
                        assert np.array_equal(got[lane], codes[row, kk:kk + 32]), (tm, wm, m0, k0, w, f, lane)


@pytest.mark.parametrize("mt0,mt1,du", LIB)
def test_mcontig_transposing_reads_are_bank_conflict_free(mt0, mt1, du):
    """Every 16-lane group of every tr_b6 read (one 256-byte bank pass of 16 x 16 bytes) hits 16 distinct 16-byte
    bank slots, for every wave and fragment of the family default."""
    s = LB.family_spec(mt0, mt1, du, "am")
    for w in range(s.waves_m):
        for f in range(s.tile_m // s.waves_m // 16):
            for h in range(2):
                ad = _mc_read_addresses(s, w, f, h)
                for g in range(4):
                    slots = {(ad[16 * g + j] // 16) % 16 for j in range(16)}
                    assert len(slots) == 16, (mt0, mt1, w, f, h, g)


def test_mcontig_swizzle_rule():
    for nblk in range(2, 65, 2):
        a, sh = G.mc_swizzle(nblk)
        assert nblk % (1 << a) == 0 and (a == 4 or nblk % (2 << a)) and sh == 4 - a
        for b in range(nblk):
            for k0 in range(0, 128, 16):
                sl = {((k0 + j) * nblk + (b ^ (((k0 + j) >> sh) & ((1 << a) - 1)))) % 16 for j in range(16)}
                assert len(sl) == 16
                assert all(b ^ (((k0 + j) >> sh) & ((1 << a) - 1)) < nblk for j in range(16))


def test_mcontig_codes_layout():
    op = L.generate(64, 256, L.DIST_UNIFORM_CODES, seed=5, tag=1)
    mc = L.mcontig_codes(op)
    assert mc.shape == (256, 48)
    assert np.array_equal(L.unpack_codes(mc, 64).T, L.unpack_codes(op.codes, 256))
    img = L.pack_operand(op, "mcontig", 64)
    assert img.size == 256 * 48 and np.array_equal(img, mc.reshape(-1))  # no tail pad: the kernel's range ends there


# ----------------------------------------------------------------- schedule / vmcnt model


def _lib(**over):
    base = dict(LB._COMMON, tile_m=64, tile_n=128, waves_m=2, waves_n=2, stages=4, a_layout="canonical")
    base.update(over)
    s = replace(G.Mxfp6GemmSpec(), **base)
    if s.runtime_shape:
        s = replace(s, k=G.library_k_ref(s))
    return s


LIB_SPECS = {
    "rt-s4": {},
    "rt-s3-unroll2": {"stages": 3, "unroll": 2},
    "rt-du256-s4": {"depth_u": 256},
    "rt-du256-s6": {"depth_u": 256, "stages": 6},
    "rt-du512-s8": {"depth_u": 512, "stages": 8, "tile_m": 32, "tile_n": 32, "waves_m": 1, "waves_n": 1},
    "rt-mcontig": {"a_layout": "mcontig"},
    "rt-mcontig-du256": {"a_layout": "mcontig", "depth_u": 256},
    "rt-jit-8w": {"tile_m": 128, "tile_n": 256, "waves_m": 2, "waves_n": 4, "frag_mode": "jit", "acq_slot": -1},
    "rt-uneven-dtl": {"tile_m": 32, "tile_n": 32, "waves_m": 1, "waves_n": 1, "stages": 4},
    "rt-vgpr-scales": {"scale_src": "vgpr"},
    "static-batch": {"runtime_shape": False, "batch": 3, "m": 512, "n": 512, "k": 4096},
    "rt-switch": {"rt_tail": "switch"},
    "rt-line2m": {**LB.LINE2M, "tile_m": 256, "tile_n": 256, "stages": 3},
    "rt-deep-ring": {"stages": 8, "tile_m": 64, "tile_n": 64},
}


@pytest.mark.parametrize("name", list(LIB_SPECS))
def test_library_vm_issue_order_matches_the_vmcnt_model(name):
    spec = _lib(**LIB_SPECS[name])
    assert G.validate_spec(spec)[0], G.validate_spec(spec)
    bd = G._Builder(spec)
    bd.build()
    model = {}
    for tile, _slot, kind, _tgt in bd._vm_events():
        model.setdefault(tile, []).append(kind)
    got = {}
    for tag, kind in bd.vm_trace:
        got.setdefault(tag, []).append(kind)
    assert got.get(None, []) == model.get(-1, [])
    for tag, kinds in got.items():
        if isinstance(tag, int):
            # (switch tail: every residue's tail tiles carry their static index; they match the T + r model)
            if spec.rt_tail != "switch" or not spec.runtime_shape:
                assert kinds == model.get(tag, []), (tag, kinds, model.get(tag))
    counts = bd._acquire_counts()
    assert all(0 <= v < 64 for v in counts.values())


@pytest.mark.parametrize("name", ["rt-s4", "rt-s3-unroll2", "rt-du256-s4", "rt-du512-s8", "rt-mcontig", "rt-jit-8w",
                                  "rt-line2m", "rt-deep-ring"])
def test_runtime_k_keeps_the_vmcnt_model(name):
    """A runtime-shape build runs the reference schedule with more loop iterations: for K = k + 128 U j the model's
    loop waits (strictest per body position) and its tail waits (relative to the end) equal the reference's."""
    spec = _lib(**LIB_SPECS[name])
    U, H, L_, T = G.loop_plan(spec)

    def waits(s):
        bd = G._Builder(s)
        c = bd._acquire_counts()
        u, h, l_, t = G.loop_plan(s)
        body = {p: min(c[h + j * u + p + bd.acq_off] for j in range(l_) if h + j * u + p + bd.acq_off < t)
                for p in range(u)}
        tail = [c.get(i + bd.acq_off, None) for i in range(h + l_ * u, t)]
        return body, tail

    ref = waits(spec)
    for j in (1, 3):
        big = waits(replace(spec, k=spec.k + 128 * U * j))
        assert big == ref, (j, big, ref)


def test_depth_u_acquires_once_per_period():
    spec = _lib(depth_u=256)
    bd = G._Builder(spec)
    bd.build()
    from rocke.core.ir_serialize import serialize

    k = G.build_mxfp6_gemm(spec)
    txt = serialize(k)
    n_bar = txt.count("tile.s_barrier")
    k1 = G.build_mxfp6_gemm(replace(spec, depth_u=128))
    assert n_bar < serialize(k1).count("tile.s_barrier")


@pytest.mark.parametrize("over", [{"depth_u": 256, "stages": 3}, {"depth_u": 512, "stages": 6}, {"depth_u": 384},
                                  {"runtime_shape": True, "cta_order": "xcd_block"}, {"rt_tail": "both"},
                                  {"mutant": "batch_swap", "runtime_shape": False, "m": 512, "n": 512, "k": 4096},
                                  {"mutant": "mc_noswz"}, {"occupancy": 9}, {"scale_layout": "wtile", "atom": "32x32x64"}])
def test_library_rules_refuse(over):
    assert not G.validate_spec(_lib(**over) if "k" in over else replace(_lib(), **over))[0]


def test_runtime_problem_predicates():
    s = _lib()
    U, H, L_, T = G.loop_plan(s)
    assert G.runtime_problem_ok(s, 640, 1280, s.k, 1)[0]
    assert G.runtime_problem_ok(s, 640, 1280, s.k + 128 * U * 5, 7)[0]
    assert not G.runtime_problem_ok(s, 640 + 32, 1280, s.k, 1)[0]
    assert not G.runtime_problem_ok(s, 640, 1280, s.k + 128, 1)[0] or U == 1
    assert not G.runtime_problem_ok(s, 640, 1280, s.k - 128 * U, 1)[0]
    sw = replace(s, rt_tail="switch")
    assert G.runtime_problem_ok(sw, 640, 1280, s.k + 128, 1)[0]


@pytest.mark.parametrize("gm,gn,G_", [(4, 8, 8), (16, 16, 8), (5, 3, 2), (7, 9, 4), (32, 32, 16), (1, 64, 8), (3, 8, 1)])
def test_grouped_cta_order_is_a_permutation(gm, gn, G_):
    s = replace(_lib(), group_m=G_)
    seen = {G.tile_of_wg(s, wg, gm * s.tile_m, gn * s.tile_n) for wg in range(gm * gn)}
    assert seen == {(i, j) for i in range(gm) for j in range(gn)}


def test_grouped_order_gives_each_xcd_a_contiguous_tile_range():
    s = _lib()
    gm, gn = 16, 32
    for x in range(8):
        ts = sorted(G.tile_of_wg(s, wg, gm * s.tile_m, gn * s.tile_n)[0] for wg in range(x, gm * gn, 8))
        assert max(ts) - min(ts) <= gm // 8 + s.group_m  # an XCD walks a band of tile rows


def test_kernargs_contract():
    """``kernargs`` (the harness contract): static builds take the 5 pointers for their own shape only; runtime
    builds the shape, packed leading dimensions and the packed per-entry strides when none are given."""
    st = replace(_lib(runtime_shape=False, batch=2, m=512, n=512, k=4096))
    assert G.kernargs(st, 1, 2, 3, 4, 5, 512, 512, 4096, 2) == __import__("struct").pack("<QQQQQ", 1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        G.kernargs(st, 1, 2, 3, 4, 5, 512, 512, 8192, 2)
    rt_ = _lib()
    ka = G.kernargs(rt_, 1, 2, 3, 4, 5, 640, 1280, rt_.k, 3)
    assert ka == G.runtime_kargs(rt_, (1, 2, 3, 4, 5), 640, 1280, rt_.k, 3)
    assert G.kernargs(rt_, 1, 2, 3, 4, 5, 640, 1280, rt_.k, 3, stride_c=1 << 33) != ka


def test_mfma_only_probe_builds_for_library_specs():
    for fam in ("ak", "am"):
        s = LB.family_spec(128, 128, 256, fam, probe="mfma_only", scale_src="vgpr")
        assert G.validate_spec(s)[0], G.validate_spec(s)
        G.build_mxfp6_gemm(s)


def test_runtime_kargs_and_strides():
    s = _lib(a_layout="mcontig")
    ka = G.runtime_kargs(s, (1, 2, 3, 4, 5), 640, 1280, 8192, 2)
    assert len(ka) == 5 * 8 + 6 * 4 + 5 * 8
    st = G.packed_strides(s, 640, 1280, 8192)
    assert st[0] == 640 * 8192 * 6 // 8 and st[1] == 1280 * 8192 * 6 // 8 and st[4] == 640 * 1280 * 2
    a_sc = np.zeros((640, 8192 // 32), np.uint8)
    assert st[2] == L.pack_scales_for(a_sc, s.atom, G.scale_layout_of(s, "A")).nbytes


def test_library_ir_uses_the_runtime_arguments_and_batch_axis():
    from rocke.core.ir_serialize import serialize

    txt = serialize(G.build_mxfp6_gemm(_lib(a_layout="mcontig")))
    for p in ("%M :", "%N :", "%K :", "%lda :", "%ldb :", "%ldc :", "%bstride_a :", "%bstride_c :"):
        assert p in txt, p
    assert 'axis = s:"z"' in txt
    assert txt.count("ds_read_b96_tr_b6") >= 2


def test_library_fields_are_inert_at_their_defaults():
    """The tile-library fields at their defaults leave every preset's IR as it was (the full corpus check of 1400
    specs is mxfp6-gemm-roofline/tiles/ir_ident.py; here: the defaults themselves and a preset spot check)."""
    d = G.Mxfp6GemmSpec()
    assert (d.runtime_shape, d.batch, d.depth_u, d.cta_order, d.occupancy, d.rt_tail) == (
        False, 1, 128, "xcd_block", 1, "fixed")
    assert not G._library_mode(d)
    for nm in ("best_mixed", "best_v6", "best_canonical"):
        s = G.preset(nm)
        assert G.kernel_name(s) == G.kernel_name(replace(s, runtime_shape=False, batch=1, depth_u=128))


def test_kbug1_batch_swap_refuses_odd_batches():
    """K-BUG-1 (harness lib-base1): the batch_swap mutant reads entry z ^ 1's A, past the buffer at batch 1 (the
    32x32x512 S problem is the one batch-1 cell): the host contract refuses it instead of faulting."""
    s = LB.family_spec(32, 32, 512, "ak", mutant="batch_swap")
    assert not G.runtime_problem_ok(s, 2048, 2048, 8192, 1)[0]
    assert not G.runtime_problem_ok(s, 2048, 2048, 8192, 3)[0]
    assert G.runtime_problem_ok(s, 2048, 2048, 8192, 2)[0]
    with pytest.raises(ValueError):
        G.kernargs(s, 1, 2, 3, 4, 5, 2048, 2048, 8192, 1)
    assert G.runtime_problem_ok(LB.family_spec(32, 32, 512, "ak"), 2048, 2048, 8192, 1)[0]


@pytest.mark.parametrize("mt0,mt1,du", [c for c in LIB if LB.line2m_ok(c[0], c[1])])
def test_line2m_family_variant_matches_the_vmcnt_model(mt0, mt1, du):
    """ring="line2m" family variants (every eligible config but 64x512, whose two rings exceed the LDS): valid, the
    VM issue order equals the model, the release after the ring reads and before the last MFMA."""
    try:
        spec = LB.family_spec(mt0, mt1, du, "ak", ring="line2m")
    except ValueError as e:
        assert (mt0, mt1) == (64, 512), e
        return
    assert spec.canon_ring == "line2m" and G.validate_spec(spec)[0]
    bd = G._Builder(spec)
    bd.build()
    model = {}
    for tile, _slot, kind, _tgt in bd._vm_events():
        model.setdefault(tile, []).append(kind)
    got = {}
    for tag, kind in bd.vm_trace:
        got.setdefault(tag, []).append(kind)
    assert got.get(None, []) == model.get(-1, [])
    for tag, kinds in got.items():
        if isinstance(tag, int):
            assert kinds == model.get(tag, []), (tag, kinds)
    assert G.last_ring_read(spec) < spec.line_release < G.mfmas_per_tile(spec) - 1


# ------------------------------------------------------------------ persistent schedule (work-unit walk)
@pytest.mark.parametrize("units", [1, 5, 8, 12, 16, 24, 40, 64, 100, 1024, 4096])
@pytest.mark.parametrize("ngrid", [1, 3, 8, 16, 24, 40, 256, 2048])
def test_persistent_walk_partitions_units(units, ngrid):
    """Every work unit exactly once over the persistent grid (multiples of 8 when the units are), each XCD's CTAs
    on a contiguous unit range when the units are a multiple of 8; persist_drop loses units."""
    if units % 8 == 0 and ngrid % 8:
        pytest.skip("host contract: ngrid a multiple of 8 when the units are")
    walk = G.persistent_units(units, ngrid)
    flat = sorted(u for us in walk for u in us)
    assert flat == list(range(units))
    if units % 8 == 0:
        c = units // 8
        for w, us in enumerate(walk):
            assert all(u // c == w % 8 for u in us), (w, us)
    dropped = sorted(u for us in G.persistent_units(units, ngrid, drop=True) for u in us)
    assert len(dropped) < units


@pytest.mark.parametrize("cfg", [(128, 128, 256), (256, 256, 256), (64, 64, 256), (32, 32, 512), (256, 288, 256)])
@pytest.mark.parametrize("fam", ["ak", "am"])
def test_persistent_spec_grid_kargs(cfg, fam):
    spec = LB.family_spec(*cfg, fam, schedule="persistent")
    assert G.validate_spec(spec)[0]
    G.build_mxfp6_gemm(spec, arch="gfx950")
    m, n = 64 * spec.tile_m, 64 * spec.tile_n
    (g, gy, gz), _ = G.grid_for(spec, m, n, 2)
    units = 64 * 64 * 2
    assert gy == gz == 1 and g % 8 == 0 and 8 <= g <= units
    ptrs = (1, 2, 3, 4, 5)
    ka = G.runtime_kargs(spec, ptrs, m, n, spec.k, 2)
    base = G.runtime_kargs(replace(spec, schedule="dp"), ptrs, m, n, spec.k, 2)
    assert ka[:len(base)] == base and len(ka) == len(base) + 8
    assert np.frombuffer(ka[len(base):], np.int32).tolist() == [2, g]
    with pytest.raises(ValueError):
        G.runtime_kargs(spec, ptrs, m, n, spec.k, 2, ngrid=12)
    assert G.validate_spec(replace(spec, mutant="persist_drop"))[0]
    assert not G.validate_spec(replace(spec, schedule="dp", mutant="persist_drop"))[0]
    assert not G.validate_spec(replace(spec, runtime_shape=False))[0]


# ------------------------------------------------------------------ wtile period scale loads (wtile_period)
@pytest.mark.parametrize("R,Wr,P", [(64, 32, 4), (128, 64, 2), (32, 16, 4), (256, 128, 2), (128, 128, 1), (64, 64, 4)])
def test_wtile_period_packer_index(R, Wr, P):
    """``pack_scales_wtile(period=P)``: dword ((((s / P nq + q) W + w) 64 + l) P + s % P) D + d of the period image
    is dword (((s nq + q) W + w) 64 + l) D + d of the per-tile image."""
    D = L.wtile_dwords(Wr)
    if D * P > 4:
        pytest.skip("P D > 4 is not a layout")
    rows, nkb = 2 * R, 4 * 8
    sc = np.random.default_rng(3).integers(0, 255, size=(rows, nkb), dtype=np.uint8)
    base = L.pack_scales_wtile(sc, R, Wr)
    per = L.pack_scales_wtile(sc, R, Wr, P)
    assert per.size == base.size
    nq, W, ns = rows // R, R // Wr, nkb // 4
    b5 = base.reshape(ns, nq, W, 64, D)
    p6 = per.reshape(ns // P, nq, W, 64, P, D)
    for s in range(ns):
        assert np.array_equal(p6[s // P, :, :, :, s % P, :], b5[s])
    name = f"wtile{R}x{Wr}" + (f"p{P}" if P > 1 else "")
    assert np.array_equal(L.pack_scales_for(sc, "16x16x128", name), per)


@pytest.mark.parametrize("cfg", [(64, 64, 256), (32, 128, 256), (128, 32, 256), (32, 32, 512), (128, 128, 256),
                                 (256, 256, 256)])
def test_wtile_period_specs(cfg):
    for P in (2, 4):
        s = LB.family_spec(*cfg, "ak", scale_src="vgpr", wtile_period=P) if _period_ok(cfg, P) else None
        if s is None:
            with pytest.raises(ValueError):
                LB.family_spec(*cfg, "ak", scale_src="vgpr", wtile_period=P)
            continue
        assert G.validate_spec(s)[0]
        assert G.scale_layout_of(s, "A").endswith(f"p{P}")
        U = G.loop_plan(s)[0]
        assert U % P == 0 and (s.k // 128) % P == 0
        G.build_mxfp6_gemm(s, arch="gfx950")
        assert G.validate_spec(replace(s, mutant="wper_rot"))[0]
        assert not G.validate_spec(replace(s, scale_src="lds"))[0]
    s1 = LB.family_spec(*cfg, "ak", scale_src="vgpr")
    assert not G.validate_spec(replace(s1, mutant="wper_rot"))[0]


def _period_ok(cfg, P):
    s = LB.family_spec(*cfg, "ak", scale_src="vgpr")
    D = max(G.wtile_dwords(s.tile_m // s.waves_m), G.wtile_dwords(s.tile_n // s.waves_n))
    return D * P <= 4


# ------------------------------------------------------------------ in-tree ports: magic divisors, kernarg preload
def test_magic_u31_exact():
    rng = np.random.default_rng(5)
    ds = list(range(1, 4097)) + [2 ** j + o for j in range(12, 31) for o in (-1, 0, 1) if 2 ** j + o < 2 ** 31] \
        + [int(x) for x in rng.integers(1, 2 ** 31 - 1, size=2000)]
    top = 2 ** 31 - 1
    for d in ds:
        m, s = G.magic_u31(d)
        assert 0 < m < 2 ** 32 and 0 <= s <= 31
        ns = [0, 1, d - 1, d, d + 1, top, top - 1, top - d] + [int(x) for x in rng.integers(0, top, size=8)]
        for n in ns:
            if 0 <= n <= top:
                assert ((2 * n * m) >> 32) >> s == n // d, (d, n)


@pytest.mark.parametrize("over", [{}, {"cta_order": "linear"}, {"schedule": "persistent"}])
def test_magic_div_kargs(over):
    s = LB.family_spec(128, 128, 256, "ak", magic_div=True, **over)
    G.build_mxfp6_gemm(s, arch="gfx950")
    m, n = 5 * 128, 3 * 128
    ka = G.runtime_kargs(s, (1, 2, 3, 4, 5), m, n, s.k, 2)
    base = G.runtime_kargs(replace(s, magic_div=False), (1, 2, 3, 4, 5), m, n, s.k, 2)
    assert ka[:len(base)] == base and len(ka) == len(base) + 24
    vals = np.frombuffer(ka[len(base):], np.int32).astype(np.int64) % 2 ** 32
    ds = G.magic_divisors(s, m, n)
    assert [int(v) for v in vals] == [v for d in ds for v in G.magic_u31(d)]
    gm, gn = 5, 3
    assert ds == ((gn if s.cta_order == "linear" else s.group_m * gn), gm % s.group_m or s.group_m, gm * gn)
    assert not G.validate_spec(replace(s, runtime_shape=False))[0]


@pytest.mark.parametrize("cfg", [(128, 128, 256), (256, 256, 256), (64, 64, 256)])
def test_kernarg_preload_marks_params(cfg):
    s = LB.family_spec(*cfg, "ak", kernarg_preload=True)
    k = G.build_mxfp6_gemm(s, arch="gfx950")
    assert all(p.attrs.get("inreg") for p in k.params)
    k0 = G.build_mxfp6_gemm(replace(s, kernarg_preload=False), arch="gfx950")
    assert not any(p.attrs.get("inreg") for p in k0.params)


# ------------------------------------------------------------------ raw canonical scales in VGPRs (scale_raw)
def test_raw_transpose_model_is_wtile():
    """The transpose of the raw row words gives the wtile op_sel dword: lane (g, r) loads raw row 16 q + r of a wave
    side (q = lane group) -> after the transpose its byte i = scale[16 i + r, block g]."""
    rng = np.random.default_rng(9)
    sc = rng.integers(0, 255, size=(64, 4 * 8), dtype=np.uint8)  # one 64-row wave side, 8 K tiles
    want = L.pack_scales_wtile(sc, 64, 64).reshape(8, 64)  # [K tile][lane], D = 1
    for t in range(8):
        words = np.array([sc[16 * (l // 16) + l % 16, 4 * t:4 * t + 4].view(np.uint32)[0] for l in range(64)],
                         np.uint32)
        assert np.array_equal(L.raw_transpose_model(words), want[t])


@pytest.mark.parametrize("cfg", [(64, 64, 256), (128, 128, 256), (256, 256, 256), (32, 128, 256), (32, 32, 512)])
@pytest.mark.parametrize("fam", ["ak", "am"])
def test_scale_raw_specs(cfg, fam):
    s = LB.family_spec(*cfg, fam, scale_src="vgpr", scale_raw=True)
    assert G.validate_spec(s)[0] and G.scale_layout_of(s, "A") == "raw" and G.scale_layout_of(s, "B") == "raw"
    assert G.loop_plan(s)[0] % 4 == 0
    G.build_mxfp6_gemm(s, arch="gfx950")
    assert not G.validate_spec(replace(s, scale_src="lds"))[0]
    assert G.scale_image_bytes(s, "A", 256, 1024) == 256 * 1024 // 32


# ------------------------------------------------------------------ partial tiles (edge_tiles: shifted last tile)
@pytest.mark.parametrize("rows,R", [(200, 64), (130, 128), (48, 32), (96, 32), (300, 256)])
def test_edge_packers_shift_last_block(rows, R):
    op = L.generate(rows, 512, L.DIST_GAUSS, seed=4, tag=1)
    keep = rows - rows % R
    idx = np.concatenate([np.arange(keep), np.arange(rows - R, rows)]) if rows % R else np.arange(rows)
    ref = L.Operand(len(idx), op.k, op.codes[idx], op.scales[idx], op.fmt)
    assert np.array_equal(L.pack_operand(op, "c0c1", R), L.pack_operand(ref, "c0c1", R))
    for Wr in (16, 32):
        if R % Wr == 0:
            assert np.array_equal(L.pack_scales_wtile(op.scales, R, Wr), L.pack_scales_wtile(ref.scales, R, Wr))


@pytest.mark.parametrize("fam", ["ak", "am"])
def test_edge_predicates(fam):
    s = LB.family_spec(128, 64, 256, fam, edge_tiles=True)
    k = s.k
    assert G.runtime_problem_ok(s, 64 * 3 + 16, 128 * 2 + 48, k)[0]
    assert not G.runtime_problem_ok(s, 32, 256, k)[0]  # M < tile_m
    assert G.runtime_problem_ok(s, 64 * 3 + 4, 256, k)[0] == (fam == "ak")  # M-contiguous: M % 16
    assert G.n_tiles(s, "A", 64 * 3 + 16) == 4 and G.n_tiles(s, "B", 128 * 2 + 48) == 3
    assert G.grid_for(s, 64 * 3 + 16, 128 * 2 + 48, 2)[0] == (12, 1, 2)
    assert not G.validate_spec(replace(s, runtime_shape=False))[0]
    G.build_mxfp6_gemm(s, arch="gfx950")


# ------------------------------------------------------------------ scale_dtl split (bubble-loop knob, default off)
@pytest.mark.parametrize("cfg", [(64, 64, 256, "ak"), (64, 64, 256, "am"), (128, 32, 256, "ak"), (512, 32, 256, "ak"),
                                 (64, 256, 256, "ak"), (32, 32, 512, "ak")])
def test_scale_dtl_split_issues_each_chunk_once_first_in_the_refill(cfg):
    """``scale_dtl="split"``: every 256-byte chunk of both wtile scale images is issued by exactly one wave
    (``wsc_split``, no repeated last chunk), and the split DMAs ("sdtl", outside the static vmcnt model like x4w)
    lead every tile's refill; the rest of the VM issue order equals the model."""
    s = LB.family_spec(*cfg, scale_dtl="split", scale_src="lds")  # (some family defaults load VGPR scales)
    assert G.validate_spec(s)[0]
    bd = G._Builder(s)
    bd.build()
    nch = {X: G.scale_img_bytes(s, X) // 256 for X in ("A", "B")}
    assert sorted((X, off) for X, off, _w in bd.wsc_split) == sorted(
        (X, 256 * q) for X in ("A", "B") for q in range(nch[X]))
    nw = s.waves_m * s.waves_n
    assert all(0 <= w < nw for _X, _o, w in bd.wsc_split)
    model, got = {}, {}
    for tile, _slot, kind, _tgt in bd._vm_events():
        model.setdefault(tile, []).append(kind)
    for tag, kind in bd.vm_trace:
        got.setdefault(tag, []).append(kind)
    n = len(bd.wsc_split)
    for tag, kinds in got.items():
        if isinstance(tag, int) and "sdtl" in kinds:
            assert kinds[:n] == ["sdtl"] * n and "sdtl" not in kinds[n:], (tag, kinds)
        if isinstance(tag, int) or tag is None:
            assert [k for k in kinds if k != "sdtl"] == model.get(-1 if tag is None else tag, []), tag
    # dword DTLs and the per-wave count are gone from the model; the default build is unchanged
    assert G._n_dtl(s) < G._n_dtl(replace(s, scale_dtl="dword"))


def test_scale_dtl_split_needs_lds_wtile_scales():
    s = LB.family_spec(64, 64, 256, "ak")
    assert not G.validate_spec(replace(s, scale_dtl="split", scale_src="vgpr"))[0]
    assert not G.validate_spec(replace(G.preset("best_mixed"), scale_dtl="split"))[0]  # tile layout: x4w / dword only


def test_config_spec_every_config():
    """config_spec: every library config x family is a valid edge_tiles spec whose tile is the WG_MULT multiple of
    the config (native when absent), and builds."""
    n = 0
    for (mt0, mt1, du) in LB.LIBRARY:
        for fam in ("ak", "am"):
            s = LB.config_spec(mt0, mt1, du, fam, promoted=False)
            g0, g1 = LB.WG_MULT.get((mt0, mt1, du, fam), (1, 1))
            assert (s.tile_n, s.tile_m) == (mt0 * g0, mt1 * g1) and s.edge_tiles and G.validate_spec(s)[0]
            n += (g0, g1) != (1, 1)
    assert n == len(LB.WG_MULT)
    G.build_mxfp6_gemm(LB.config_spec(32, 128, 256, "ak"), arch="gfx950")


def test_promoted_specs_ship_on_multi_entry_problems_only():
    """PROMOTED: every entry is a library config whose spec is valid, builds, and covers whole MT0 x MT1 tiles;
    dispatch ships it on multi-entry problems (its own schedule) and keeps the family path on batch 1, K tails and
    explicit knobs; config_spec(promoted=False) is the family path."""
    for (mt0, mt1, du, fam), d in LB.PROMOTED.items():
        assert (mt0, mt1, du) in LB.LIBRARY and fam in ("ak", "am") and "dlevers" in d
        s = LB.config_spec(mt0, mt1, du, fam)
        assert G.validate_spec(s)[0] and s.edge_tiles and s.runtime_shape
        assert s.tile_n % mt0 == 0 and s.tile_m % mt1 == 0, (mt0, mt1, s.tile_n, s.tile_m)
        m, n = 16 * s.tile_m, 16 * s.tile_n  # >= 512 tiles: not the stream-K lane's short-grid route
        assert LB._dispatch(mt0, mt1, du, fam, "e2m3", m, n, 4, k_target=8192) == s
        assert LB.dispatch(mt0, mt1, du, fam, "e2m3", m, n, 4, k_target=8192) == (
            replace(s, epi_lds=True) if (mt0, mt1, du, fam) in LB.EPI_LDS and G.validate_spec(
                replace(s, epi_lds=True))[0] else s)
        fam_path = LB._dispatch(mt0, mt1, du, fam, "e2m3", m, n, 1, k_target=8192)
        assert fam_path == LB.config_spec(mt0, mt1, du, fam, promoted=False, schedule="dp", k_target=8192)
        assert LB.dispatch(mt0, mt1, du, fam, "e2m3", m, n, 4, k_target=8192 - 32).k_tail
        assert LB.config_spec(mt0, mt1, du, fam, stages=2) == LB.config_spec(mt0, mt1, du, fam, promoted=False,
                                                                              stages=2)
    for (mt0, mt1, du, fam), d in list(LB.PROMOTED.items())[:3]:
        G.build_mxfp6_gemm(LB.config_spec(mt0, mt1, du, fam), arch="gfx950")


def test_promoted_mechanism_with_a_synthetic_entry(monkeypatch):
    """The PROMOTED path itself (independent of the table's contents): a complete spec dict + dlevers delta becomes
    a 16 x 16-tile reference spec with library_k_ref, ships at batch >= 2 only, and an invalid entry raises."""
    key = (128, 128, 256, "ak")
    from dataclasses import asdict

    base = asdict(LB.config_spec(*key, promoted=False, schedule="persistent", read_every=3))
    dflt = asdict(G.Mxfp6GemmSpec())
    entry = {f: v for f, v in base.items() if v != dflt[f] and f not in ("m", "n", "k")}
    entry["dlevers"] = "-m0fill"
    monkeypatch.setitem(LB.PROMOTED, key, entry)
    s = LB.dispatch(*key, "e2m3", 4096, 4096, 8, k_target=8192)
    assert (s.tile_m, s.tile_n, s.m, s.n, s.schedule, s.dlevers) == (base["tile_m"], base["tile_n"],
                                                                     16 * base["tile_m"], 16 * base["tile_n"],
                                                                     "persistent", "-m0fill")
    assert s.k == G.library_k_ref(s, k_target=8192) and G.validate_spec(s)[0]
    assert LB.dispatch(*key, "e2m3", 4096, 4096, 1, k_target=8192).schedule == "dp"
    assert LB.dispatch(*key, "e2m3", 4096, 4096, 1, k_target=8192) != s
    monkeypatch.setitem(LB.PROMOTED, key, {**entry, "edge_tiles": True, "runtime_shape": False})
    with pytest.raises(ValueError):
        LB.config_spec(*key)


def test_dlevers_kernel_attribute_is_default_off_and_only_adds_the_attribute():
    """``dlevers`` (default off): the IR carries exactly one more kernel attribute, the kernel name is unchanged, and
    an empty lever string emits nothing (rustgemm crate-up: the object equals the ROCKE_DIRECT_LEVERS build)."""
    from rocke.core.ir_serialize import serialize

    for s, lev in [(LB.family_spec(128, 128, 256, "ak"), LB.LEVERS), (G.preset("best_v6"), G.PRESET_LEVERS["best_v6"])]:
        on = LB.self_levered(s, lev)
        assert on.dlevers == lev and G.kernel_name(on) == G.kernel_name(s)
        a, b = serialize(G.build_mxfp6_gemm(s)).splitlines(), serialize(G.build_mxfp6_gemm(on)).splitlines()
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        assert len(a) == len(b) and len(diff) == 1 and f'dlevers = s:"{lev}"' in diff[0][1]
        assert diff[0][1].replace(f'dlevers = s:"{lev}", ', "") == diff[0][0]
        assert serialize(G.build_mxfp6_gemm(G.with_dlevers(s, " , "))).splitlines() == a
    assert G.with_dlevers(G.preset("best_v6"), " scalesep, m0fill ").dlevers == "scalesep,m0fill"


def test_dispatch_schedule_gate():
    """dispatch: persistent on multi-entry problems except PERSIST_EXCLUDE (PROMOTED: its own schedule and tile),
    data-parallel on one entry."""
    for (mt0, mt1, du) in LB.LIBRARY:
        for fam in ("ak", "am"):
            s1 = LB.dispatch(mt0, mt1, du, fam, "e2m3", 4096, 4096, 1)
            s8 = LB.dispatch(mt0, mt1, du, fam, "e2m3", 2048, 2048, 8)
            assert s1.schedule == "dp" and G.validate_spec(s8)[0]
            key = (mt0, mt1, du, fam)
            if key in LB.PROMOTED:
                assert s8.schedule == LB.PROMOTED[key].get("schedule", "dp")
                continue
            dp_ = key in LB.PERSIST_EXCLUDE or key in LB.PERSIST_SPILL
            assert s8.schedule == ("dp" if dp_ else "persistent")
            assert (s8.tile_m, s8.tile_n) == (s1.tile_m, s1.tile_n)


@pytest.mark.parametrize("gm,gn", [(16, 8), (5, 3), (4, 64), (64, 4), (7, 9)])
def test_mfast_cta_order(gm, gn):
    """cta_order mfast: a permutation of the tiles; with the XCD chunking, consecutive tiles of an XCD walk M within
    one weight panel (N tile)."""
    s = LB.family_spec(32, 128, 256, "ak", cta_order="mfast")
    tiles = [G.tile_of_wg(s, wg, gm * s.tile_m, gn * s.tile_n) for wg in range(gm * gn)]
    assert sorted(tiles) == [(a, c) for a in range(gm) for c in range(gn)]
    if gm * gn % 8 == 0:
        chunk = gm * gn // 8
        for x in range(8):
            seq = [G.tile_of_wg(s, wg, gm * s.tile_m, gn * s.tile_n) for wg in range(x, gm * gn, 8)]
            assert [t[0] + gm * t[1] for t in seq] == list(range(x * chunk, (x + 1) * chunk))
    G.build_mxfp6_gemm(s, arch="gfx950")


@pytest.mark.parametrize("prob", [(16384, 8192, 8192, 1), (2048, 2048, 8192, 64), (128, 7168, 2048, 8),
                                  (1024, 1024, 4096, 1), (4096, 2880, 2944, 1), (4096, 2880, 2880, 1)])
@pytest.mark.parametrize("fam", ["ak", "am"])
def test_select_returns_a_runnable_kernel(prob, fam):
    """SB-3: library.select picks a config whose dispatch kernel validates and accepts the problem; the model is
    monotone in the rate (a config's predicted time scales with 1 / EFF_S)."""
    m, n, k, b = prob
    cfg, spec = LB.select(m, n, k, b, fam)
    assert G.validate_spec(spec)[0] and G.runtime_problem_ok(spec, m, n, k, b)[0]
    t = LB.predict_seconds(*cfg, fam, m, n, k, b)
    assert t > 0


def test_sb3_model_terms():
    """SB-3 model: compute time scales with 1 / rate; HBM-bound problems sit on the compulsory-bytes floor (plus the
    launch); short K pays the per-WG fill; candidates come fastest first and select / select_plan agree with them."""
    sp = LB.dispatch(128, 128, 256, "ak", "e2m3", 2048, 2048, 16, k_target=8192)
    t1 = LB._sb3_kernel_seconds(sp, 1000, 2048, 2048, 8192, 16)
    t2 = LB._sb3_kernel_seconds(sp, 2000, 2048, 2048, 8192, 16)
    assert 1.9 < (t1 - LB.SB3_LAUNCH_S) / (t2 - LB.SB3_LAUNCH_S) < 2.1
    m, n, k, b = 128, 4096, 8192, 128  # G-like: weights streamed once, HBM-bound
    floor = b * ((m + n) * k * (0.75 + 1 / 32) + 2 * m * n) / LB.SB3_HBM_BPS + LB.SB3_LAUNCH_S
    c = LB.select_candidates(m, n, k, b, "ak", top=3)
    assert c[0][0] >= floor and c[0][0] < floor * 1.05
    assert [x[0] for x in c] == sorted(x[0] for x in c)
    assert LB.select(m, n, k, b, "ak")[0] == c[0][1]
    # the K fill: at a quarter of the reference K, the rate per FLOP drops by the fill share
    kf = lambda kt: (LB.SB3_KT_REF + LB.SB3_FILL_TILES) / LB.SB3_KT_REF * kt / (kt + LB.SB3_FILL_TILES)
    tk8 = LB._sb3_kernel_seconds(sp, 3000, 2048, 2048, 8192, 16) - LB.SB3_LAUNCH_S
    tk2 = LB._sb3_kernel_seconds(sp, 3000, 2048, 2048, 2048, 16) - LB.SB3_LAUNCH_S
    assert abs(tk2 / tk8 - 0.25 * kf(64) / kf(16)) < 1e-3
    # a big square problem picks a big work group; a decode problem a 32-token one
    big = LB.select(8192, 8192, 8192, 1, "ak")[1]
    assert big.tile_m * big.tile_n >= 128 * 256
    assert LB.select(16, 4096, 4096, 1, "ak")[0][1] == 32


def test_select_plan_routes_am_through_am_plan():
    """select_plan: ak -> a native plan of the selected kernel; am -> am_plan of the selected config (transpose + ak
    where the model takes it); a transpose path whose ak kernel refuses the problem falls back to native am."""
    cfg, pl = LB.select_plan(2048, 2048, 8192, 16, "ak")
    assert pl.path == "native" and pl.gemm == LB.select(2048, 2048, 8192, 16, "ak")[1]
    for prob in ((8192, 8192, 8192, 1), (128, 4096, 8192, 128), (2048, 2048, 8192, 16), (8192, 8192, 1024, 1)):
        cfg, pl = LB.select_plan(*prob, family="am")
        assert pl.path in ("native", "transpose")
        if pl.path == "transpose":
            assert pl.gemm.a_layout == "canonical" and LB.am_predicted(*cfg, *prob) >= LB.AM_MARGIN
        else:
            assert pl.gemm.a_layout == "mcontig"
    assert any(LB.select_plan(*p, family="am")[1].path == "transpose" for p in ((8192, 8192, 8192, 1),
                                                                                (128, 4096, 8192, 128)))


def test_b_aux_sets_the_direct_b_loads_cache_policy_only_when_explicit():
    """b_direct + an explicit ``b_aux`` (MoE lane R1): the direct B VGPR loads carry the cache policy (``aux``), every
    other line unchanged but the kernel name (a hash of the spec); -1 (the default) and 0 leave the body byte-identical, and
    ``dtl_aux`` does not reach them."""
    import re

    from rocke.core.ir_serialize import serialize

    def body(spec):
        return [ln for ln in serialize(G.build_mxfp6_gemm(spec)).splitlines() if not ln.startswith("kernel @")]

    s = LB.config_spec(512, 32, 256, "ak")
    assert s.b_direct and s.b_aux == -1
    base = body(s)
    assert not any("tile.buffer_load_vN" in ln and "aux = i:" in ln for ln in base)
    assert body(replace(s, b_aux=0)) == base
    dtl = body(replace(s, dtl_aux=2))
    assert not any("tile.buffer_load_vN" in ln and "aux = i:" in ln for ln in dtl)
    for aux in (1, 2, 3):
        on = body(replace(s, b_aux=aux))
        diff = [(x, y) for x, y in zip(base, on) if x != y]
        assert len(on) == len(base) and diff, aux
        # two loads (dwordx4 + dwordx2) per direct B fragment, and nothing else changes
        assert len(diff) % 2 == 0 and {("i:4" in x, "i:2" in x) for x, _ in diff} == {(True, False), (False, True)}
        for x, y in diff:
            assert "tile.buffer_load_vN" in y and f"aux = i:{aux}" in y, y
            assert re.sub(r"aux = i:\d+, ", "", y) == x, (x, y)


def test_no_gc_build_restores_the_collector_and_keeps_the_ir():
    """build_mxfp6_gemm / build_ir_text pause the cyclic GC (build speed) and restore its state, also on a refused
    spec; the IR text equals serialize(build) with the GC running."""
    import gc

    from rocke.core.ir_serialize import serialize

    spec = LB.config_spec(64, 64, 256, "ak")
    assert gc.isenabled()
    text = G.build_ir_text(spec)
    assert gc.isenabled() and text == serialize(G.build_mxfp6_gemm(spec, arch="gfx950"))
    gc.disable()
    try:
        G.build_mxfp6_gemm(spec, arch="gfx950")
        assert not gc.isenabled()  # a disabled collector stays disabled
    finally:
        gc.enable()
    with pytest.raises(ZeroDivisionError):
        with G.no_gc():
            raise ZeroDivisionError
    assert gc.isenabled()


def test_epi_lds_dispatch_table():
    """EPI_LDS: dispatch adds epi_lds to the kernel it picks for the listed configs (when that kernel takes it), never
    to a split-K unit's base or an explicit override; every listed config takes it on its S problem."""
    from builders.gfx950.mxfp6_gemm import libtiles as LT

    for key in LB.EPI_LDS:
        mt0, mt1, du, fam = key
        p = LT.problems(LT.Tile(mt0, mt1, du))[0]
        s = LB.dispatch(mt0, mt1, du, fam, "e2m3", p.m, p.n, p.batch, k_target=p.k)
        assert s.epi_lds and G.validate_spec(s)[0], key
        assert not LB.dispatch(mt0, mt1, du, fam, "e2m3", p.m, p.n, p.batch, k_target=p.k, epi_lds=False).epi_lds
    other = next(k for k in LB.STRUCT if k not in LB.EPI_LDS)
    p = LT.problems(LT.Tile(*other[:3]))[0]
    assert not LB.dispatch(*other[:3], other[3], "e2m3", p.m, p.n, p.batch, k_target=p.k).epi_lds


def test_a_fill_c0c1_dma_map_builds_the_c0c1_image():
    """a_fill c0c1: the DMA lane map of _setup_afill (c0: dwordx4 lane l of chunk q from row 16 q + l % 16 at byte
    24 (l / 16); c1: dword lane j of DMA p from row 16 (p / 2) + (j / 2) % 16 at byte 48 (p % 2) + 24 (j / 32) + 16 +
    4 (j % 2)) applied to canonical rows reproduces layout.pack_operand_c0c1's stage image of every 128-K tile."""
    import numpy as np

    from kernels.gfx950.mxfp6_gemm import layout as L

    R, K = 64, 384
    op = L.generate(R, K, L.DIST_UNIFORM_CODES, seed=7, tag=3)
    rows = np.ascontiguousarray(op.codes).reshape(-1)
    rbytes = K * 6 // 8
    ref = L.pack_operand_c0c1(op, R).reshape(K // 128, R * 96)
    for t in range(K // 128):
        img = np.zeros(R * 96, np.uint8)
        for q in range(R // 16):
            for l in range(64):
                src = (16 * q + l % 16) * rbytes + 96 * t + 24 * (l // 16)
                img[1024 * q + 16 * l:1024 * q + 16 * l + 16] = rows[src:src + 16]
        for p in range(R // 8):
            for j in range(64):
                src = (16 * (p // 2) + (j // 2) % 16) * rbytes + 96 * t + 48 * (p % 2) + 24 * (j // 32) + 16 + 4 * (j % 2)
                d = 64 * R + 256 * p + 4 * j
                img[d:d + 4] = rows[src:src + 4]
        assert np.array_equal(img, ref[t]), t
    s = LB.family_spec(256, 256, 256, "ak", ring="stage", a_fill="c0c1")
    assert G.validate_spec(s)[0] and s.a_layout == "canonical"
    G.build_mxfp6_gemm(s, arch="gfx950")
    assert not G.validate_spec(replace(LB.family_spec(256, 256, 256, "ak"), a_fill="c0c1"))[0]  # line2m: refused
