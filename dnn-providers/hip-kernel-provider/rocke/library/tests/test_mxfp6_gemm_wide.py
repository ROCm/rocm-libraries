# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the wide-tile MXFP6 GEMM (``gemm_wide``): host images, tile map, the static VM / barrier model,
builds of every knob value and the mutants (no GPU)."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest
from kernels.gfx950.mxfp6_gemm import gemm_wide as W
from kernels.gfx950.mxfp6_gemm import layout as L


def _op(rows, k=512, seed=3, tag=1):
    return L.generate(rows, k, L.DIST_GAUSS, seed=seed, tag=tag)


# ----------------------------------------------------------------- geometry / map


@pytest.mark.parametrize("n,tile_n,want", [(4096, 384, [384] * 10 + [256]), (4096, 256, [256] * 16),
                                           (6144, 384, [384] * 16), (640, 384, [384, 256]), (1024, 384, [384] * 2 + [256])])
def test_col_widths(n, tile_n, want):
    assert W.col_widths(replace(W.WideGemmSpec(), n=n, tile_n=tile_n)) == want


@pytest.mark.parametrize("m,n,xmap", [(4096, 4096, "band"), (4096, 6144, "band"), (4096, 4096, "none"),
                                      (2048, 640, "band"), (512, 1024, "none")])
def test_tile_map_is_a_bijection_and_balanced(m, n, xmap):
    spec = replace(W.WideGemmSpec(), m=m, n=n, xcd_map=xmap)
    (g, _, _), _ = W.grid_of(spec)
    tiles = [W.tile_of_wg(spec, wg) for wg in range(g)]
    gm, gc = m // 256, len(W.col_widths(spec))
    assert sorted(tiles) == [(r, c) for r in range(gm) for c in range(gc)]
    if xmap == "band":
        per = {}
        for wg, t in enumerate(tiles):
            per.setdefault(wg % 8, []).append(t)
        rows = {x: {r for r, _ in v} for x, v in per.items()}
        assert all(len(v) == g // 8 for v in per.values())
        assert all(len(r) <= gm // 4 for r in rows.values())  # an XCD stays inside its row band


def test_band_map_4096_matches_v5_4x8_on_a_square_grid():
    from kernels.gfx950.mxfp6_gemm import gemm as G

    spec = replace(W.WideGemmSpec(), tile_n=256)
    v5 = G.preset("best")
    assert [W.tile_of_wg(spec, wg) for wg in range(256)] == [G.tile_of_wg(v5, wg) for wg in range(256)]


# ----------------------------------------------------------------- host images


def test_pack_scales_b_matches_definition():
    widths = [384, 256]
    b = _op(640, k=1024)
    img = W.pack_scales_b(b.scales, widths).view(np.uint8)
    T = 1024 // 128
    r0 = 0
    for c, w in enumerate(widths):
        hg = w // 2
        for s in (0, T - 1):
            for g in range(2):
                for lane in (0, 17, 63):
                    for d in range(W.SB_DWORDS):
                        for i in range(4):
                            byte = (((((c * T + s) * 2 + g) * 64 + lane) * W.SB_DWORDS + d) * 4 + i)
                            if 64 * d < hg:
                                row = r0 + g * hg + 64 * d + 16 * i + lane % 16
                                assert img[byte] == b.scales[row, 4 * s + lane // 16]
                            else:
                                assert img[byte] == 0
        r0 += w
    assert img.size == len(widths) * T * 2048


def test_pack_b_operand_blocks_are_c0c1_images():
    widths = [384, 256]
    b = _op(640, k=256)
    img = W.pack_b_operand(b, widths)
    assert img.size == 640 * 256 * 6 // 8
    first = L.pack_operand_c0c1(L.Operand(384, 256, b.codes[:384], b.scales[:384], b.fmt), 384)
    second = L.pack_operand_c0c1(L.Operand(256, 256, b.codes[384:], b.scales[384:], b.fmt), 256)
    assert np.array_equal(img[:first.size], first) and np.array_equal(img[first.size:], second)


def test_c0c1_fragment_read_model_recovers_codes():
    """The kernel's fragment address (c0 at 16 o, c1 at 64 R + 8 o, o = 64 (wrb + f) + lane) returns the 24-byte
    lane chunk (row 16 (wrb + f) + lane % 16, K group lane // 16) of a 384-row stage image."""
    R = 384
    b = _op(R, k=128)
    img = L.pack_operand_c0c1(b, R)
    for wrb, f, lane in ((0, 0, 0), (12, 11, 63), (5, 3, 17)):
        o = 64 * (wrb + f) + lane
        got = np.concatenate([img[16 * o:16 * o + 16], img[64 * R + 8 * o:64 * R + 8 * o + 8]])
        row, kg = 16 * (wrb + f) + lane % 16, lane // 16
        assert np.array_equal(got, b.codes[row, 24 * kg:24 * kg + 24])


# ----------------------------------------------------------------- static schedule model


def _specs():
    out = []
    for tile_n, sa, sb, h, rel, rd, ad, dd, be, ad_, ae, bd, ul in itertools.product(
            (384, 256), (3, 2), (2, 3), (1, 2), (True, False), (1, 4), (0, -4), (1, 0), ("acq", "end"),
            ("spread", "late"), ("acq", "end"), ("release", "spread"), (1, 2)):
        s = replace(W.WideGemmSpec(), tile_n=tile_n, stages_a=sa, stages_b=sb, head_rows=h, release=rel,
                    release_delay=rd, acq_delay=ad, dtl_delay=dd, b_dtl_end=be, a_dtl=ad_, a_dtl_end=ae, b_dtl=bd,
                    unroll=ul, k=8192)
        ok, _ = W.validate_spec(s)
        if ok:
            out.append(s)
    return out


SPECS = _specs()


def test_some_specs_are_legal():
    assert len(SPECS) > 30


@pytest.mark.parametrize("spec", SPECS[::97], ids=lambda s: W.kernel_name(s)[-12:])
def test_schedule_invariants(spec):
    """For every fill (X, u): (1) it is issued after a barrier that follows every wave's consumption of the reads of
    (X, u - S_X) (the slot it overwrites), (2) the acquire of tile u (Q_{u-1}) waits for it (in-order VM completion),
    (3) every read of (X, u) follows Q_{u-1}."""
    T = spec.k // 128
    for fn in sorted({w // 32 for w in W.col_widths(spec)}):
        pp = W.path_plan(spec, fn)
        ev, counts = W.vm_events(pp, T)
        consumed = {"B": pp.h * fn - 1, "A": W.unit_starts(fn, pp.h)[-1]}
        S = {"A": spec.stages_a, "B": spec.stages_b}
        bars = [pp.q] + ([pp.r] if pp.r >= 0 else [])
        for t, slot, X, u in ev:
            if t < 0:
                assert u < S[X]  # prologue fills only empty slots
                continue
            old = u - S[X]
            # the latest barrier before this fill: of tile t before its slot, else the last one of tile t - 1
            prior = [(t, s) for s in bars if s < slot] or [(t - 1, max(bars))]
            bt, bs = max(prior)
            assert old < 0 or (bt, bs) >= (old, consumed[X]), (X, u, t, slot)
        # (2): position of the acquire of u: count events before it; the fill's index must be <= pos - 1 - count
        n_pro = sum(1 for e in ev if e[0] == -1)
        for u in range(T):
            if u == 0:
                pos = n_pro
            else:
                pos = next((i for i, e in enumerate(ev) if (e[0] > u - 1) or (e[0] == u - 1 and e[1] >= pp.q)), len(ev))
            last = max(i for i, e in enumerate(ev) if e[3] <= u)
            assert last <= pos - 1 - counts[u]
        # (3): reads of (X, u + 1) in tile u happen at / after the acquire slot
        assert pp.a0_next >= pp.q
        assert all(x >= pp.q for x in range(pp.n_mfma - fn, pp.n_mfma))


@pytest.mark.parametrize("fn", [8, 12])
@pytest.mark.parametrize("h", [1, 2])
def test_mfma_order_covers_the_tile_and_chains_tiles(fn, h):
    """Every (row, column) once per tile; the last row frees the B columns in the order the next tile's first unit
    uses them (so B(t+1)[c] gets the same lead for every c); the A rows a unit needs are read one unit earlier."""
    for parity in (0, 1):
        o = W.mfma_order(fn, parity, h)
        assert sorted(o) == [(mi, c) for mi in range(W.FM) for c in range(fn)]
        assert [mi for mi, _ in o[-fn:]] == [W.FM - 1] * fn
        nxt = W.mfma_order(fn, 1 - parity, h)
        first = []
        for _, c in nxt:
            if c not in first:
                first.append(c)
        assert [c for _, c in o[-fn:]] == first
    st = W.unit_starts(fn, h)
    assert st[0] == 0 and st[1] == h * fn and st[-1] == (W.FM - 1) * fn


@pytest.mark.parametrize("h", [1, 2])
def test_a_reads_precede_use_with_bounded_buffers(h):
    spec = replace(W.WideGemmSpec(), head_rows=h)
    pp = W.path_plan(spec, 12)
    o = W.mfma_order(12, 0, h)
    first = {}
    last = {}
    for x, (mi, _) in enumerate(o):
        first.setdefault(mi, x)
        last[mi] = x
    for mi, slot in pp.a_read.items():
        assert slot < first[mi] - 1  # at least a few MFMAs of lead
    # live A fragments after the reads of slot x (a read in pre[x] is live from x; a fragment whose last use is
    # MFMA x is dead after it) <= h + 1
    born = {mi: pp.a_read.get(mi, -10) for mi in range(W.FM)}
    for x in range(pp.n_mfma):
        live = sum(1 for mi in range(W.FM) if born[mi] <= x < last[mi])
        nxt = h if x >= pp.a0_next else 0  # A(t + 1)[0..h-1] read at a0_next
        assert live + nxt <= h + 1, (x, live, nxt)


def test_default_plan_numbers():
    spec = W.WideGemmSpec()
    pp = W.path_plan(spec, 12)
    assert (pp.q, pp.r, pp.d, pp.U, pp.h) == (84, 12, {"B": 2, "A": 3}, 6, 1)
    assert W.path_plan(replace(spec, a_dtl="spread", head_rows=2), 12).d == {"B": 2, "A": 2}
    assert W.path_plan(replace(spec, head_rows=2), 12).r == 24
    assert pp.n_dtl("A") == 6 + 1 and pp.n_dtl("B") == 9 + 2
    assert W.lds_bytes(spec) <= 160 * 1024


def test_illegal_specs_are_refused():
    for over in ({"tile_n": 320}, {"stages_a": 4}, {"stages_b": 3}, {"n": 4000}, {"acq_delay": -90},
                 {"release_delay": 80}, {"stages_a": 2, "release": False, "acq_delay": -8, "a_dtl": "late"}):
        ok, why = W.validate_spec(replace(W.WideGemmSpec(), **over))
        assert not ok, over


# ----------------------------------------------------------------- builds


def _vm_kinds_per_tile(builder_trace):
    out = {}
    for tag, kind in builder_trace:
        out.setdefault(tag, []).append(kind)
    return out


@pytest.mark.parametrize("over", [{}, {"tile_n": 256}, {"stages_a": 2}, {"release": False}, {"unroll": 2},
                                  {"dtl_imm": False}, {"xcd_map": "none"}, {"probe": "mfma_only"}, {"setprio": 1},
                                  {"n": 6144}, {"acq_delay": -4, "release_delay": 4}, {"b_dtl_end": "end"},
                                  {"sc_exact": False},
                                  {"tile_n": 256, "stages_b": 3, "release": False, "b_dtl": "spread"}])
def test_builds(over):
    spec = replace(W.WideGemmSpec(), k=8192, **over)
    k = W.build_mxfp6_gemm_wide(spec)
    assert k.name == W.kernel_name(spec)


def test_emitted_vm_order_matches_the_model():
    """The DTLs the builder emits per loop / static tile = the model's per-tile fill list (kinds and count)."""
    spec = replace(W.WideGemmSpec(), k=8192, n=6144)  # one path (fn 12)
    bld = W._Builder(spec)
    bld.build()
    per = _vm_kinds_per_tile(bld.vm_trace)
    pp = W.path_plan(spec, 12)
    want = [f"{X}{kind}" for _, X, kind, _ in pp.dtl]
    for p in range(pp.U):
        assert per[("loop", p)] == want
    T = spec.k // 128
    for t in range(T):
        if t in per:
            exp = [f"{X}{kind}" for _, X, kind, _ in pp.dtl if t + pp.d[X] < T]
            assert per[t] == exp


@pytest.mark.parametrize("mutant", ["scale_sel", "k_order", "drop_kstep", "wrong_fmt"])
def test_mutants_change_the_kernel(mutant):
    from rocke.core.ir_serialize import serialize

    base = serialize(W.build_mxfp6_gemm_wide(replace(W.WideGemmSpec(), k=8192)))
    mut = serialize(W.build_mxfp6_gemm_wide(replace(W.WideGemmSpec(), k=8192, mutant=mutant)))
    strip = lambda s: "\n".join(x for x in s.splitlines() if "rocke_gemm" not in x)  # noqa: E731
    assert strip(base) != strip(mut)


# ----------------------------------------------------------------- stream-K roles


def test_stream_k_roles_cover_every_tile_once():
    """Every wide tile has one head (K [0, k_head)) and one tail segment (K [k_head, K)); rows 0-7 are the tails'
    first segments, rows 8-15 their second; every narrow tile one full-K CTA; heads fill XCDs 0-4, tails 5, 6 and
    half of 7, so every XCD runs one kind of K phase (apart from XCD 7's narrow half)."""
    spec = replace(W.WideGemmSpec(), stream_k=True)
    assert W.grid_of(spec)[0] == (256, 1, 1)
    heads, tails, narrow = {}, {}, {}
    for wg in range(256):
        role, tiles = W.sk_role(wg)
        x = wg % 8
        if role == "head":
            assert x <= 4
            heads[tiles[0]] = wg
        elif role == "tail":
            assert x >= 5
            t1, t2 = tiles
            assert t1[0] < 8 and t2 == (t1[0] + 8, t1[1])
            tails[t1] = tails[t2] = wg
        else:
            assert x == 7
            narrow[tiles[0]] = wg
    wide = {(r, c) for r in range(16) for c in range(10)}
    assert set(heads) == wide and set(tails) == wide
    assert set(narrow) == {(r, 10) for r in range(16)}
    # L2 blocks: each head XCD covers <= 8 rows x 4 columns or 16 rows x 2 columns
    for x in range(5):
        ts = [t for t, wg in heads.items() if wg % 8 == x]
        assert len(ts) == 32 and len({r for r, _ in ts}) * len({c for _, c in ts}) == 32


def test_stream_k_validation_and_workspace():
    spec = replace(W.WideGemmSpec(), stream_k=True)
    assert W.validate_spec(spec)[0]
    assert W.k_head_of(spec) == 341
    assert W.workspace_bytes(spec) == (160 * 4 * 96 * 1024, 640)
    for over in ({"n": 6144}, {"tile_n": 256}, {"k_phase": 3}, {"k_head": 509}):
        assert not W.validate_spec(replace(spec, **over))[0], over


def test_stream_k_builds_with_handoff_ops():
    from rocke.core.ir_serialize import serialize

    text = serialize(W.build_mxfp6_gemm_wide(replace(W.WideGemmSpec(), stream_k=True, k=8192)))
    assert text.count("memref.global_flag_store") >= 4 and text.count("memref.global_flag_wait_eq") >= 2
    assert "memref.fence" in text


def test_stamps_probe_is_stream_k_only_and_builds():
    from rocke.core.ir_serialize import serialize

    assert not W.validate_spec(replace(W.WideGemmSpec(), probe="stamps"))[0]
    spec = replace(W.WideGemmSpec(), stream_k=True, probe="stamps", k=8192)
    assert W.validate_spec(spec)[0]
    assert W.workspace_bytes(spec)[1] == W.STAMP_OFF + 256 * W.STAMP_BYTES
    assert serialize(W.build_mxfp6_gemm_wide(spec)).count("tile.memrealtime") >= 9
