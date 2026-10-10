# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CTA work decode of the backward main kernels (``_attention_bwd_worklist``).

The emitters run on the typed host evaluator (the exact integer formulas the
kernel executes):

* every load-balance order (natural, reverse, mirror pairing, odd and even
  tile counts, a single tile) visits each kv tile of a (unit, batch) exactly
  once over the grid, in the documented order; the decode without pairing
  (the main kernel's) is one kv tile per CTA;
* the head unit / split decode covers every ``y`` of ``units * g_split``;
* direct mode maps a unit to its kv head and whole group; atomic mode maps
  every (unit, split) to disjoint query-head ranges that cover all ``h_q``
  heads, each sharing one K head ``h / gk`` and one V head ``h / gv``;
* the XCD remap equals the chunked chiplet transform of ``helpers/grid.py``,
  is a bijection of each batch plane, keeps the per-XCD dispatch order, puts
  every kv tile of a head unit on one XCD when the chunk is a multiple of
  grid x and ``xcd_n`` divides the plane, and is the identity at ``xcd_n = 0``;
* the emitters also build on the IR builder (CTA-uniform integer ops only).
"""

from __future__ import annotations

import math

import pytest
from rocke.core.ir import I32, IRBuilder

from kernels.common._attention_bwd_worklist import (
    LB_INTERLEAVE,
    LB_MIRROR,
    LB_NATURAL,
    LB_REVERSE,
    XcdRemap,
    decode_cta,
    head_range,
    main_grid_x,
    xcd_remap_id,
)

from ._attention_bwd_int_eval import IntEvalBuilder, i32


def _tiles(n_kv_tiles, lb_order, pair, *, y=0, g_split=None, interleave=False):
    """kv tiles each CTA x visits, through the emitted decode (``pair=None``:
    the decode without mirror pairing, as the main kernel emits it;
    ``interleave``: the decode of an ``lb_interleave`` main kernel)."""
    b = IntEvalBuilder()
    out = []
    for x in range(main_grid_x(n_kv_tiles, pair or 0)):
        w = decode_cta(
            b,
            x=i32(b, x),
            y=i32(b, y),
            z=i32(b, 0),
            n_kv_tiles=i32(b, n_kv_tiles),
            lb_order=i32(b, lb_order),
            pair=None if pair is None else i32(b, pair),
            g_split=None if g_split is None else i32(b, g_split),
            interleave=interleave,
        )
        assert w.n_pass.v in (1, 2)
        out.append([w.kv_tile(b, i32(b, p)).v for p in range(w.n_pass.v)])
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17])
@pytest.mark.parametrize(
    "lb_order,pair",
    [(LB_NATURAL, 0), (LB_REVERSE, 0), (LB_MIRROR, 1)],
    ids=["natural", "reverse", "mirror"],
)
def test_every_order_visits_each_kv_tile_once(n, lb_order, pair):
    per_cta = _tiles(n, lb_order, pair)
    flat = [t for ts in per_cta for t in ts]
    assert sorted(flat) == list(range(n))
    if lb_order == LB_NATURAL:
        assert per_cta == [[t] for t in range(n)]
    elif lb_order == LB_REVERSE:
        assert per_cta == [[n - 1 - t] for t in range(n)]
    else:
        assert len(per_cta) == math.ceil(n / 2)
        for t, ts in enumerate(per_cta):
            assert ts == ([t] if t == n - 1 - t else [t, n - 1 - t])


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_decode_without_pairing_is_one_tile_per_cta(n):
    # the main kernel's decode: natural or reverse, one kv tile per CTA
    assert _tiles(n, LB_NATURAL, None) == _tiles(n, LB_NATURAL, 0)
    assert _tiles(n, LB_REVERSE, None) == _tiles(n, LB_REVERSE, 0)
    assert all(len(ts) == 1 for ts in _tiles(n, LB_REVERSE, None))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17])
@pytest.mark.parametrize("pair", [None, 0])
def test_interleave_visits_each_kv_tile_once(n, pair):
    per_cta = _tiles(n, LB_INTERLEAVE, pair, interleave=True)
    assert all(len(ts) == 1 for ts in per_cta)
    want = [t // 2 if t % 2 == 0 else n - 1 - t // 2 for t in range(n)]
    assert [ts[0] for ts in per_cta] == want  # 0, n-1, 1, n-2, ...
    assert sorted(want) == list(range(n))
    # the switch leaves the other orders as they were
    for lb in (LB_NATURAL, LB_REVERSE):
        assert _tiles(n, lb, pair, interleave=True) == _tiles(n, lb, pair)
    # without the switch the value decodes as natural (the plan sets it)
    assert _tiles(n, LB_INTERLEAVE, pair) == _tiles(n, LB_NATURAL, pair)


def test_pairing_takes_precedence_over_interleave():
    assert _tiles(6, LB_INTERLEAVE, 1, interleave=True) == [[0, 5], [1, 4], [2, 3]]


def test_pairing_takes_precedence_over_the_order():
    # pair = 1 always visits (t, n - 1 - t), whatever lb_order says
    for lb in (LB_NATURAL, LB_REVERSE, LB_MIRROR):
        assert _tiles(6, lb, 1) == [[0, 5], [1, 4], [2, 3]]


def test_main_grid_x():
    assert [main_grid_x(n, 0) for n in range(5)] == [0, 1, 2, 3, 4]
    assert [main_grid_x(n, 1) for n in range(5)] == [0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        main_grid_x(-1, 0)


@pytest.mark.parametrize("units,g_split", [(1, 1), (3, 1), (4, 2), (2, 4), (5, 3)])
def test_unit_and_split_decode(units, g_split):
    b = IntEvalBuilder()
    seen = set()
    for y in range(units * g_split):
        w = decode_cta(
            b, x=i32(b, 0), y=i32(b, y), z=i32(b, 7), n_kv_tiles=i32(b, 1),
            lb_order=i32(b, 0), pair=i32(b, 0), g_split=i32(b, g_split),
        )  # fmt: skip
        assert 0 <= w.split.v < g_split and w.batch.v == 7
        seen.add((w.unit.v, w.split.v))
    assert seen == {(u, s) for u in range(units) for s in range(g_split)}
    # without a split the unit is y
    w = decode_cta(
        b, x=i32(b, 0), y=i32(b, 5), z=i32(b, 0), n_kv_tiles=i32(b, 1),
        lb_order=i32(b, 0), pair=i32(b, 0),
    )  # fmt: skip
    assert (w.unit.v, w.split.v) == (5, 0)


def _heads(h_q, h_k, h_v, g_split):
    """Query heads of every (unit, split), through the emitted head map."""
    gk, gv = h_q // h_k, h_q // h_v
    atomic = h_k != h_v or g_split > 1
    group = math.gcd(gk, gv) if atomic else gk
    units = h_q // group
    b = IntEvalBuilder()
    out = []
    for unit in range(units):
        for split in range(g_split):
            hr = head_range(
                b, unit=i32(b, unit), split=i32(b, split), group=i32(b, group),
                gk=i32(b, gk), gv=i32(b, gv),
                g_split=i32(b, g_split) if atomic else None,
            )  # fmt: skip
            heads = list(range(hr.h0.v, hr.h0.v + hr.n_heads.v))
            out.append((heads, hr.hk.v, hr.hv.v))
            if not atomic:
                break
    return out, gk, gv


@pytest.mark.parametrize(
    "h_q,h_k,h_v,g_split",
    [
        (8, 8, 8, 1),  # MHA, direct
        (8, 2, 2, 1),  # GQA, direct
        (8, 1, 1, 1),  # MQA, direct
        (8, 4, 2, 1),  # h_k != h_v
        (8, 2, 8, 1),  # K fewer than V
        (12, 4, 6, 1),  # gk = 3, gv = 2: G = 1
        (8, 2, 2, 2),  # GQA, split
        (8, 1, 1, 4),  # MQA, split
        (16, 4, 2, 2),  # h_k != h_v, split
    ],
)
def test_head_ranges_cover_every_query_head_once(h_q, h_k, h_v, g_split):
    out, gk, gv = _heads(h_q, h_k, h_v, g_split)
    flat = [h for heads, _hk, _hv in out for h in heads]
    assert sorted(flat) == list(range(h_q))
    for heads, hk, hv in out:
        assert heads and all(h // gk == hk and h // gv == hv for h in heads)


def test_direct_mode_unit_is_the_kv_head():
    out, gk, _ = _heads(8, 2, 2, 1)
    assert [(hs[0], len(hs), hk) for hs, hk, _hv in out] == [(0, 4, 0), (4, 4, 1)]


def test_emitters_build_on_the_ir_builder():
    b = IRBuilder("worklist_probe")
    a = {n: b.param(n, I32) for n in ("n", "lb", "pair", "gs", "G", "gk", "gv")}
    w = decode_cta(
        b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
        n_kv_tiles=a["n"], lb_order=a["lb"], pair=a["pair"], g_split=a["gs"],
    )  # fmt: skip
    kt = w.kv_tile(b, b.const_i32(1))
    hr = head_range(
        b, unit=w.unit, split=w.split, group=a["G"], gk=a["gk"], gv=a["gv"],
        g_split=a["gs"],
    )  # fmt: skip
    for v in (kt, w.n_pass, hr.h0, hr.n_heads, hr.hk, hr.hv):
        assert v.type == I32
    names = {op.name for op in b.kernel.body.ops}
    assert not any(n.startswith(("memref.", "tile.mma", "scf.")) for n in names)


# --------------------------------------------------------------------------- XCD remap
def _chiplet_reference(wgid, num_wgs, n, chunk):
    """The chunked chiplet transform (helpers/grid.py), in plain Python."""
    block = n * chunk
    if wgid >= (num_wgs // block) * block:
        return wgid
    local = wgid // n
    return (local // chunk) * block + (wgid % n) * chunk + local % chunk


@pytest.mark.parametrize("num_wgs", [1, 7, 16, 40, 64, 100, 513])
@pytest.mark.parametrize("n,chunk", [(8, 1), (8, 4), (2, 3), (4, 16), (8, 64)])
def test_xcd_remap_id_is_the_chiplet_transform(num_wgs, n, chunk):
    b = IntEvalBuilder()
    got = [
        xcd_remap_id(
            b, i32(b, L), num_wgs=i32(b, num_wgs), n=i32(b, n), chunk=i32(b, chunk)
        ).v
        for L in range(num_wgs)
    ]
    assert got == [_chiplet_reference(L, num_wgs, n, chunk) for L in range(num_wgs)]
    assert sorted(got) == list(range(num_wgs))  # a bijection
    for x in range(n):  # the i-th CTA of an XCD gets an increasing logical id
        ids = [got[L] for L in range(x, num_wgs, n)]
        assert ids == sorted(ids)


def test_xcd_remap_id_matches_the_platform_helper():
    from rocke.helpers.grid import chiplet_transform_chunked_dynamic

    b = IntEvalBuilder()
    for num_wgs in (24, 100, 512):
        for L in range(num_wgs):
            ref = chiplet_transform_chunked_dynamic(
                b, i32(b, L), num_wgs=i32(b, num_wgs), num_xcds=4, chunk_size=6
            )
            ours = xcd_remap_id(
                b, i32(b, L), num_wgs=i32(b, num_wgs), n=i32(b, 4), chunk=i32(b, 6)
            )
            assert ours.v == ref.v


@pytest.mark.parametrize("n,chunk", [(0, 0), (0, 64), (8, 0)])
def test_xcd_remap_id_off_is_the_identity(n, chunk):
    b = IntEvalBuilder()
    for L in range(50):
        out = xcd_remap_id(
            b, i32(b, L), num_wgs=i32(b, 50), n=i32(b, n), chunk=i32(b, chunk)
        )
        assert out.v == L


def _remapped(gx, gy, n, chunk, *, g_split=None, lb_order=LB_NATURAL, z=0):
    """(kv tile, y) each CTA of plane z decodes with the remap, by hardware id."""
    b = IntEvalBuilder()
    out = {}
    for y in range(gy):
        for x in range(gx):
            w = decode_cta(
                b, x=i32(b, x), y=i32(b, y), z=i32(b, z), n_kv_tiles=i32(b, gx),
                lb_order=i32(b, lb_order),
                g_split=None if g_split is None else i32(b, g_split),
                xcd=XcdRemap(n=i32(b, n), chunk=i32(b, chunk), gy=i32(b, gy)),
            )  # fmt: skip
            assert w.batch.v == z
            unit_y = w.unit.v if g_split is None else w.unit.v * g_split + w.split.v
            out[x + gx * y] = (w.first_kt.v, unit_y)
    return out


@pytest.mark.parametrize(
    "gx,gy,n,chunk",
    [
        (3, 8, 8, 3),
        (3, 8, 8, 6),
        (4, 16, 8, 64),
        (5, 3, 8, 5),
        (2, 6, 4, 8),
        (7, 2, 2, 7),
    ],
)
@pytest.mark.parametrize("lb_order", [LB_NATURAL, LB_REVERSE])
def test_remapped_decode_visits_every_tile_of_the_plane_once(
    gx, gy, n, chunk, lb_order
):
    got = _remapped(gx, gy, n, chunk, lb_order=lb_order)
    assert sorted(got.values()) == [(t, y) for t in range(gx) for y in range(gy)]


@pytest.mark.parametrize("gx,gy,mult", [(3, 8, 1), (3, 16, 2), (5, 8, 1), (2, 32, 4)])
def test_remap_keeps_a_head_unit_on_one_xcd(gx, gy, mult):
    # chunk a multiple of gx and n * chunk dividing the plane: every kv tile of
    # a unit runs on the XCD the hardware gives its CTAs (L % n)
    n, chunk = 8, mult * gx
    assert (gx * gy) % (n * chunk) == 0
    for z in range(3):
        got = _remapped(gx, gy, n, chunk, z=z)
        xcd_of_unit = {}
        for L_plane, (_t, y) in got.items():
            L = L_plane + gx * gy * z
            xcd_of_unit.setdefault(y, set()).add(L % n)
        assert all(len(s) == 1 for s in xcd_of_unit.values())


def test_remap_equals_the_whole_grid_transform_when_the_block_divides_the_plane():
    gx, gy, B, n, chunk = 3, 16, 3, 8, 6
    for z in range(B):
        got = _remapped(gx, gy, n, chunk, z=z)
        for L_plane, (t, y) in got.items():
            L = L_plane + gx * gy * z
            assert _chiplet_reference(L, gx * gy * B, n, chunk) == t + gx * (y + gy * z)


def test_remap_with_xcd_n_zero_is_the_plain_decode():
    for gx, gy in ((3, 4), (1, 5)):
        got = _remapped(gx, gy, 0, 0)
        assert got == {x + gx * y: (x, y) for y in range(gy) for x in range(gx)}


@pytest.mark.parametrize("units,g_split", [(4, 2), (3, 1)])
def test_remapped_decode_with_a_head_split(units, g_split):
    gx, gy = 3, units * g_split
    got = _remapped(gx, gy, 2, gx, g_split=g_split)
    assert sorted(got.values()) == [(t, y) for t in range(gx) for y in range(gy)]


def test_no_remap_emits_no_remap_ir():
    def ops(xcd):
        b = IRBuilder("worklist_probe")
        a = {n: b.param(n, I32) for n in ("n", "lb", "xn", "xc", "gy")}
        remap = XcdRemap(n=a["xn"], chunk=a["xc"], gy=a["gy"]) if xcd else None
        decode_cta(
            b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
            n_kv_tiles=a["n"], lb_order=a["lb"], xcd=remap,
        )  # fmt: skip
        return [op.name for op in b.kernel.body.ops]

    plain, remapped = ops(False), ops(True)
    assert len(remapped) > len(plain)
    assert not any(n.startswith(("memref.", "tile.mma", "scf.")) for n in remapped)
