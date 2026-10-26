# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensor GETT on MN-layout bodies (``TensorGettSpec.mn_tile``) and K padding.

CPU-only. The address arithmetic of the hook (per-lane DTL offsets, the
permuted 8- and 16-byte epilogues) is run on a small SIMT model of the IR
builder: every IR value is a list of 64 lane values, buffer stores honour the
resource range check, and accumulator / C elements are symbolic. The tests
then check that every output element of a tile is written exactly once, at
its own D offset, with its own accumulator (and C) element, and that each DTL
lane reads the operand elements its LDS slot holds.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from rocke.instances.common import gemm_mn_operand as mno
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from rocke.instances.common.tensor import gett as G
from rocke.instances.common.tensor.gett_plan import (
    k_split,
    k_split_padded,
    mixed_offset,
    plan_gett,
)

W = 64


def _wrap(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


class V:
    """One IR value: 64 lane values (python ints or symbolic tuples)."""

    def __init__(self, lanes):
        self.l = list(lanes)


def _lanes(x):
    return x.l if isinstance(x, V) else [x] * W


class SimtOps:
    """The IRBuilder subset GettHooks uses, on 64 lanes."""

    def __init__(self):
        self.stores = []  # (lane, byte offset, elements, rsrc)
        self.loads = 0
        self.mask = [True] * W
        self.nt_stores = 0

    def _map(self, f, *xs):
        cols = [_lanes(x) for x in xs]
        return V(f(*vals) for vals in zip(*cols))

    def const_i32(self, v):
        return V([_wrap(int(v))] * W)

    def add(self, a, b):
        return self._map(lambda x, y: _wrap(x + y), a, b)

    def add_nuw(self, a, b):
        return self._map(lambda x, y: _wrap((x & 0xFFFFFFFF) + (y & 0xFFFFFFFF)), a, b)

    def sub(self, a, b):
        return self._map(lambda x, y: _wrap(x - y), a, b)

    def mul(self, a, b):
        return self._map(lambda x, y: _wrap(x * y), a, b)

    def div(self, a, b):
        return self._map(lambda x, y: int(x / y), a, b)

    def mod(self, a, b):
        return self._map(lambda x, y: x - int(x / y) * y, a, b)

    def cmp_lt(self, a, b):
        return self._map(lambda x, y: x < y, a, b)

    def cmp_eq(self, a, b):
        return self._map(lambda x, y: x == y, a, b)

    def land(self, a, b):
        return self._map(lambda x, y: bool(x and y), a, b)

    def select(self, c, a, b):
        return self._map(lambda cc, x, y: x if cc else y, c, a, b)

    def optimization_barrier(self, v):
        return v

    def to_sgpr_u32(self, v):
        return v

    # data (symbolic)
    def fmul(self, a, x):
        return self._map(lambda s, y: ("mul", s, y), a, x)

    def fma(self, a, b_, c):
        return self._map(lambda s, y, z: ("fma", s, y, z), a, b_, c)

    def cast_to_f32(self, x):
        return x

    def cast_f32_to(self, x, _dt):
        return self._map(lambda y: ("h", y), x)

    def bitcast(self, x, _t):
        return x

    def vec_extract(self, v, e):
        return self._map(lambda t: t[e], v)

    def vec_pack(self, xs, _dt):
        return V(tuple(t) for t in zip(*[_lanes(x) for x in xs]))

    def vec_bitcast(self, v, t):
        n = t.count
        out = []
        for tup in v.l:
            flat = []
            for x in tup:
                flat.extend(x if isinstance(x, list) else [x])
            k = len(flat) // n
            out.append(
                tuple(flat[i * k : (i + 1) * k] if k > 1 else flat[i] for i in range(n))
            )
        return V(out)

    def permlane32_swap(self, lo, hi):
        # lanes 32..63 of ``lo`` <-> lanes 0..31 of ``hi``
        lo2 = lo.l[:32] + hi.l[:32]
        hi2 = lo.l[32:] + hi.l[32:]
        return V(lo2), V(hi2)

    def warp_shuffle_xor(self, v, m):
        return V(v.l[i ^ m] for i in range(W))

    def buffer_rsrc(self, ptr, n):
        return (ptr, _lanes(n)[0] & 0xFFFFFFFF)

    def _flat(self, vec):
        out = []
        for x in vec if isinstance(vec, tuple) else (vec,):
            out.extend(x if isinstance(x, list) else [x])
        return tuple(out)

    def scf_if(self, cond):
        ops = self

        class _If:
            def __enter__(self_):
                self_.saved = ops.mask
                ops.mask = [m and bool(c) for m, c in zip(ops.mask, _lanes(cond))]

            def __exit__(self_, *exc):
                ops.mask = self_.saved
                return False

        return _If()

    def global_store_vN(self, ptr, idx, vec, n, align=None, temporal_hint=None):
        self.nt_stores += 1
        for ln in range(W):
            if not self.mask[ln]:
                continue
            o = _lanes(idx)[ln]
            assert 0 <= o < (1 << 31)
            self.stores.append((ln, 2 * o, self._flat(_lanes(vec)[ln]), ptr))

    def _store(self, rs, off, vec, nbytes):
        for ln in range(W):
            if not self.mask[ln]:
                continue
            o = _lanes(off)[ln] & 0xFFFFFFFF
            if o + nbytes > rs[1]:
                continue  # range check: dropped
            self.stores.append((ln, o, self._flat(_lanes(vec)[ln]), rs[0]))

    def buffer_store_vN_f16(self, rs, off, _c0, vec, ndw):
        self._store(rs, off, vec, 4 * ndw)

    buffer_store_vN_bf16 = buffer_store_vN_f16

    def buffer_store_f16(self, rs, off, _c0, x):
        self._store(rs, off, x, 2)

    buffer_store_bf16 = buffer_store_f16

    def buffer_load_vN(self, rs, off, _c0, _dt, n):
        self.loads += 1
        return V(
            tuple(("C", (o & 0xFFFFFFFF) // 2 + i) for i in range(n))
            for o in _lanes(off)
        )

    def buffer_load(self, rs, off, _c0, _dt):
        self.loads += 1
        return V(("C", (o & 0xFFFFFFFF) // 2) for o in _lanes(off))


# ------------------------------------------------------------------------------------------ problems
_P3 = (
    ["K3", "K2", "K1", "K0", "M3", "M2", "M1", "M0"],
    ["K3", "K2", "K1", "K0", "N3", "N2", "N1", "N0"],
    ["N3", "M3", "N2", "M2", "N1", "M1", "N0", "M0"],
    {
        "K0": 8,
        "K1": 8,
        "K2": 6,
        "K3": 4,
        "M0": 16,
        "M1": 12,
        "M2": 10,
        "M3": 8,
        "N0": 14,
        "N1": 10,
        "N2": 12,
        "N3": 8,
    },
)
# P2-style: D canonical (M fastest), operands interleaved with 32 / 24-element runs
_P2 = (
    ["K2", "M2", "K1", "M1", "K0", "M0"],
    ["K2", "N2", "K1", "N1", "K0", "N0"],
    ["N2", "N1", "N0", "M2", "M1", "M0"],
    {
        "K0": 16,
        "K1": 12,
        "K2": 8,
        "M0": 32,
        "M1": 20,
        "M2": 12,
        "N0": 24,
        "N1": 20,
        "N2": 14,
    },
)
# K extents 24 x 24: not separable at tile_k 32 without padding
_KPAD = (
    ["K1", "M1", "K0", "M0"],
    ["K1", "N1", "K0", "N0"],
    ["N1", "M1", "N0", "M0"],
    {"K0": 24, "K1": 24, "M0": 48, "M1": 32, "N0": 40, "N1": 36},
)
# ragged canonical problem whose D spans more than 2 GiB (the wide-D epilogue)
_WIDE = (("K", "M"), ("K", "N"), ("N", "M"), {"M": 38400, "N": 30240, "K": 3072})

_B3_TILE = TileSpec(
    tile_m=256,
    tile_n=256,
    tile_k=32,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)
_B3_TRAIT = TraitSpec(
    direct_to_lds=True,
    dtl_prefetch=True,
    dtl_stages=3,
    dtl_frag_prefetch=True,
    acc_agpr=True,
    mfma_interleave=True,
    ilv_merge=True,
    ilv_barrier_after=2,
    interwave_setprio=True,
    lds_swizzle=True,
    lds_swizzle_r=1,
    dtl_hoist=True,
    chiplet_swizzle=True,
    chiplet_wgm=4,
    chiplet_chunk_size=64,
)
_SEL_TILE = TileSpec(
    tile_m=128,
    tile_n=128,
    tile_k=64,
    warp_m=2,
    warp_n=4,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)
_SEL_TRAIT = TraitSpec(
    direct_to_lds=True,
    dtl_prefetch=True,
    dtl_stages=2,
    lds_swizzle=True,
    interwave_setprio=True,
    dtl_hoist=True,
    chiplet_swizzle=True,
    chiplet_chunk_size=32,
    epilogue="cshuffle",
    pad_m=True,
    pad_n=True,
    dtl_tight_rsrc=True,
    pad_fast_tile=True,
)
BODIES = {"b3": (_B3_TILE, _B3_TRAIT), "sel128": (_SEL_TILE, _SEL_TRAIT)}


def _spec(prob, body="b3", **kn):
    a, b_, d, ext = prob
    tile, trait = BODIES[body]
    return G.TensorGettSpec.make(
        a,
        b_,
        d,
        ext,
        knobs=G.GettKnobs(**kn),
        tile_k=tile.tile_k,
        mn_tile=tile,
        mn_trait=trait,
    )


def _hooks(spec):
    body = spec.body()
    h = G.GettHooks(spec.plan(), spec.knobs, body, dtype=spec.dtype)
    return h, body


# ------------------------------------------------------------------------------------------ DTL geometry
@pytest.mark.parametrize(
    "tile_mn,tile_k,warp_ext,bs",
    [(256, 32, 128, 256), (128, 64, 64, 512), (256, 64, 128, 256), (128, 64, 32, 512)],
)
def test_dtl_lane_k_mn_matches_the_layout(tile_mn, tile_k, warp_ext, bs):
    lay = mno.MnLayout(
        tile_mn=tile_mn, tile_k=tile_k, block_size=bs, wave_size=64, warp_ext=warp_ext
    )
    ops = mno.IntOps()
    for tid in range(bs):
        got = mno.emit_dtl_lane_k_mn(ops, lay, tid)
        assert got == [lay.dtl_lane(p, tid) for p in range(lay.n_pass)]


@pytest.mark.parametrize("prob", [_P3, _P2, _KPAD], ids=["P3", "P2", "kpad"])
@pytest.mark.parametrize("body", ["b3", "sel128"])
def test_mn_dtl_offsets_read_what_each_lds_slot_holds(prob, body):
    spec = _spec(prob, body)
    assert G.is_valid_spec(spec) == (True, "ok")
    h, bdy = _hooks(spec)
    pl = spec.plan()
    b = SimtOps()
    nthr = bdy.block_size
    for nm in ("A", "B"):
        op = pl.ops[nm]
        lay = h.lay[nm]
        tile_mn = lay.tile_mn
        last = (op.rows - 1) // tile_mn * tile_mn
        for mn0 in sorted({0, tile_mn, last}):
            seen = set()
            for w in range(nthr // W):
                tid = V(range(w * W, w * W + W))
                h.tid = tid
                h._bind_mn(b)
                offs = h._tile_voffs_mn(b, nm, b.const_i32(mn0))
                for p, v in enumerate(offs):
                    for ln in range(W):
                        k, mn = lay.dtl_lane(p, w * W + ln)
                        row = mn0 + mn
                        kreal = True
                        if pl.k_pad is not None:
                            mode, real, _ = pl.k_pad
                            div = 1
                            for r, m, _mult in pl.k_local:
                                if m == mode:
                                    kreal = (k // div) % r < real
                                    break
                                div *= r
                        if row >= op.rows or not kreal:
                            assert v.l[ln] == G.OOB
                            continue
                        exp = mixed_offset(row, op.row_digits) + pl.klocal(k, nm)
                        assert v.l[ln] == 2 * exp
                        # 8 consecutive rows are 8 contiguous elements
                        for j in range(8):
                            assert (
                                mixed_offset(row + j, op.row_digits)
                                == mixed_offset(row, op.row_digits) + j
                            )
                        seen.add((k, mn))
            # every (k, mn) slot of the tile with a real row / K index is loaded
            want = {
                (k, mn)
                for k in range(pl.tile_k)
                for mn in range(0, tile_mn, 8)
                if mn0 + mn < op.rows
            }
            if pl.k_pad is None:
                assert seen == want


# ------------------------------------------------------------------------------------------ epilogues
def _acc_symbol(lane, wm, wn, mi, ni, e, mfA, sub_perm, wmn):
    q = lane // 16
    rq = mno.QUAD_PERM[q] if sub_perm else q
    return (
        "acc",
        wm * wmn[0] + 16 * mi + 4 * rq + e,
        wn * wmn[1] + 16 * ni + lane % 16,
    )


def _run_epilogue(spec, bm, bn):
    h, body = _hooks(spec)
    pl = spec.plan()
    t = body.tile
    mfA, mfB = h.mf["A"], h.mf["B"]
    wmn = (h.wmn["A"], h.wmn["B"])
    b = SimtOps()
    h.p = {"alpha": "alpha", "beta": "beta", "Cin": "Cin"}
    lane = V(range(W))
    for wm, wn in itertools.product(range(t.warp_m), range(t.warp_n)):
        vals = [
            V(
                tuple(
                    _acc_symbol(ln, wm, wn, mi, ni, e, mfA, h.a_sub_perm, wmn)
                    for e in range(4)
                )
                for ln in range(W)
            )
            for mi in range(mfA)
            for ni in range(mfB)
        ]
        h.epilogue(
            b,
            vals,
            b.const_i32(bm),
            b.const_i32(bn),
            C="D",
            warp_m_idx=b.const_i32(wm),
            warp_n_idx=b.const_i32(wn),
            lane=lane,
            acc_out=lambda v: v,
        )
    return h, pl, body, b


def _check_epilogue(spec, bm, bn):
    h, pl, body, b = _run_epilogue(spec, bm, bn)
    t = body.tile
    written = {}
    for _ln, off, elems, ptr in b.stores:
        assert ptr == "D"
        for j, el in enumerate(elems):
            written.setdefault(off + 2 * j, []).append(el)
    expect = {}
    for r in range(t.tile_m):
        for c in range(t.tile_n):
            m, n = bm + r, bn + c
            if m >= pl.M or n >= pl.N:
                continue
            o = 2 * (
                mixed_offset(m, pl.d_row_digits) + mixed_offset(n, pl.d_col_digits)
            )
            acc = ("mul", "alpha", ("acc", r, c))
            if spec.bilinear:
                acc = ("fma", "beta", ("C", o // 2), acc)
            expect[o] = ("h", acc)
    assert set(written) == set(expect), (len(written), len(expect))
    for o, els in written.items():
        assert els == [expect[o]], (o, els, expect[o])
    return b


@pytest.mark.parametrize("prob", [_P3, _P2, _KPAD], ids=["P3", "P2", "kpad"])
@pytest.mark.parametrize("body", ["b3", "sel128"])
@pytest.mark.parametrize("v16", [False, True, "nt"])
@pytest.mark.parametrize("bilinear", [False, True])
def test_mn_epilogue_writes_every_output_once(prob, body, v16, bilinear):
    spec = _spec(
        prob,
        body,
        epi_v16=bool(v16),
        store_nt=v16 == "nt",
        bilinear=bilinear,
        c_prefetch=2 if bilinear else 0,
    )
    ok, why = G.is_valid_spec(spec)
    assert ok, why
    pl = spec.plan()
    t = spec.body().tile
    last_m = (pl.M - 1) // t.tile_m * t.tile_m
    last_n = (pl.N - 1) // t.tile_n * t.tile_n
    for bm, bn in {(0, 0), (last_m, last_n), (0, last_n)}:
        b = _check_epilogue(spec, bm, bn)
        if v16:
            # one 16-byte store per lane and atom pair
            assert all(len(e) == 8 for _l, _o, e, _p in b.stores)


@pytest.mark.parametrize("nt", [False, True])
def test_wide_d_ragged_tile_masks_the_summed_offset(nt):
    spec = _spec(_WIDE, "b3", epi_v16=True, store_nt=nt, bilinear=nt)
    h, _ = _hooks(spec)
    assert h.wide_d()
    pl = spec.plan()
    assert pl.N % 256 and pl.M % 256 == 0
    # the last (ragged) column tile: columns past N must be dropped, never land inside D
    _check_epilogue(spec, (pl.M // 256 - 1) * 256, (pl.N - 1) // 256 * 256)


def test_narrow_sentinel_on_wide_d_would_write_inside_d(monkeypatch):
    """Mutation guard for the model: with the wide-D path disabled, the ragged
    tile writes a dropped column back inside D (the defect)."""
    spec = _spec(_WIDE, "b3", epi_v16=True)
    monkeypatch.setattr(G.GettHooks, "wide_d", lambda self: False)
    pl = spec.plan()
    with pytest.raises(AssertionError):
        _check_epilogue(spec, (pl.M // 256 - 1) * 256, (pl.N - 1) // 256 * 256)


def test_c_prefetch_issues_the_same_loads():
    s0 = _spec(_P3, "b3", epi_v16=True, bilinear=True)
    s2 = dataclasses.replace(s0, knobs=dataclasses.replace(s0.knobs, c_prefetch=2))
    b0 = _check_epilogue(s0, 0, 0)
    b2 = _check_epilogue(s2, 0, 0)
    assert b0.loads == b2.loads


# ------------------------------------------------------------------------------------------ validity
def test_mn_body_drops_ragged_and_cshuffle_levers():
    spec = _spec(_P3, "sel128")
    body = spec.body()
    assert body.trait.epilogue == "default"
    assert not (body.trait.pad_m or body.trait.pad_n or body.trait.dtl_tight_rsrc)
    assert body.data.layout == "CRR"
    assert G.mn_body_errors(body) == []


@pytest.mark.parametrize(
    "lever",
    [
        dict(split_k=2),
        dict(lds_row_interleave=True),
        dict(dtl_k_tail=1),
        dict(mfma_swap_ab=True),
        dict(dtl_stages=0),
        dict(streamk=True),
        dict(k_stagger=2),
        dict(dtl_addr_spread=True),
    ],
)
def test_mn_body_refuses_levers_it_does_not_model(lever):
    spec = _spec(_P3, "b3")
    bad = dataclasses.replace(
        spec, mn_trait=dataclasses.replace(spec.mn_trait, **lever)
    )
    ok, why = G.is_valid_spec(bad)
    assert not ok, lever


def test_mn_body_needs_16_byte_runs_and_tile_k_match():
    a = ["K2", "M2", "K1", "M1", "K0", "M0"]
    ext = dict(_P2[3], M0=10)
    p = G.TensorGettSpec.make(
        a, _P2[1], _P2[2], ext, tile_k=32, mn_tile=_B3_TILE, mn_trait=_B3_TRAIT
    )
    ok, why = G.is_valid_spec(p)
    assert not ok and "16-byte" in why
    assert G.is_valid_spec(dataclasses.replace(p, mn_tile=None, mn_trait=None))[0]
    q = _spec(_P3, "b3")
    assert not G.is_valid_spec(dataclasses.replace(q, tile_k=64))[0]
    assert not G.is_valid_spec(dataclasses.replace(q, dtl_stages=2))[0]
    assert not G.is_valid_spec(dataclasses.replace(q, mn_tile=None))[0]


def test_v16_needs_8_element_d_runs():
    # D's fastest run of M0 = 4 elements
    a, b_, d, ext = _P3
    p = _spec((a, b_, d, dict(ext, M0=4)), "b3", epi_v16=True)
    ok, why = G.is_valid_spec(p)
    assert not ok and "epi_v16" in why
    assert G.is_valid_spec(dataclasses.replace(p, knobs=G.GettKnobs())) == (True, "ok")


def test_ring_body_kernel_names_are_unchanged_by_the_new_fields():
    import hashlib
    import json

    a, b_, d, ext = _P3
    s = G.TensorGettSpec.make(a, b_, d, ext)
    dd = dataclasses.asdict(s)
    for f in ("mn_tile", "mn_trait", "kc_tile", "kc_trait"):
        dd.pop(f)
    for f in ("epi_v16", "c_prefetch", "store_nt"):
        dd["knobs"].pop(f)
    key = json.dumps(dd, sort_keys=True)
    assert (
        s.kernel_name()
        == f"tensor_gett_f16_{hashlib.sha1(key.encode()).hexdigest()[:12]}"
    )
    assert s.kernel_name() != _spec(_P3, "b3").kernel_name()


# ------------------------------------------------------------------------------------------ K padding
def test_k_split_padded_pads_one_in_tile_mode():
    ext = {"K0": 24, "K1": 24}
    assert k_split(["K0", "K1"], ext, 32) is None
    local, tile, pad = k_split_padded(["K0", "K1"], ext, 32)
    assert pad == ("K0", 24, 32)
    assert local == [(32, "K0", 1)] and tile == [(24, "K1", 1)]
    # 6 x 10 at 32: padding the inner 6 to 8 leaves 10 unsplittable by 4;
    # with 10 inner, 10 -> 16 and 6 splits as 2 x 3
    assert k_split_padded(["K0", "K1"], {"K0": 6, "K1": 10}, 32) is None
    local, tile, pad = k_split_padded(["K1", "K0"], {"K0": 6, "K1": 10}, 32)
    assert pad == ("K1", 10, 16)
    assert local == [(16, "K1", 1), (2, "K0", 1)] and tile == [(3, "K0", 2)]


def test_plan_pads_only_when_needed():
    a, b_, d, ext = _KPAD
    pl = plan_gett(a, b_, d, ext, tile_k=32)
    assert pl.k_pad is not None and pl.K == 32 * 24
    a, b_, d, ext = _P3
    assert plan_gett(a, b_, d, ext, tile_k=32).k_pad is None


@pytest.mark.parametrize("v16", [False, True])
def test_kpad_mn_kernel_builds(v16):
    spec = _spec(_KPAD, "sel128", epi_v16=v16)
    assert G.is_valid_spec(spec) == (True, "ok")
    kd = G.build_tensor_gett(spec)
    assert kd.name.startswith(spec.kernel_name())


def test_kpad_ring_kernel_masks_padded_k_rows():
    a, b_, d, ext = _KPAD
    spec = G.TensorGettSpec.make(a, b_, d, ext)
    assert G.is_valid_spec(spec) == (True, "ok")
    h = G.GettHooks(spec.plan(), spec.knobs, spec.body())
    assert not h.lane_split("A") and not h.lane_split("B")
    b = SimtOps()
    pl = spec.plan()
    for nm in ("A", "B"):
        for w in range(4):
            h.tid = V(range(w * W, w * W + W))
            offs = h.tile_voffs(b, nm, b.const_i32(0))
            for p, v in enumerate(offs):
                for ln in range(W):
                    k, row = h._lane_geom(nm, p, w * W + ln)
                    kreal = k % 32 < 24 if pl.k_local[0][1] == pl.k_pad[0] else True
                    if not kreal:
                        assert v.l[ln] == G.OOB
                    elif row < pl.ops[nm].rows:
                        assert v.l[ln] == 2 * (
                            mixed_offset(row, pl.ops[nm].row_digits) + pl.klocal(k, nm)
                        )


# ------------------------------------------------------------------------------------------ IR
def test_v16_ir_uses_one_permlane_per_dword():
    sp = _spec(_P3, "b3", epi_v16=True)
    kd = G.build_tensor_gett(sp)
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    ll = lower_kernel_to_llvm(kd, llvm_flavor="llvm22", arch="gfx950")
    h, body = _hooks(sp)
    pairs = h.mf["A"] // 2 * h.mf["B"]

    def calls(t):
        return sum(
            1 for ln in t.splitlines() if "call" in ln and "permlane32.swap" in ln
        )

    assert calls(ll) == 2 * pairs
    bl = dataclasses.replace(sp, knobs=dataclasses.replace(sp.knobs, bilinear=True))
    ll = lower_kernel_to_llvm(
        G.build_tensor_gett(bl), llvm_flavor="llvm22", arch="gfx950"
    )
    assert calls(ll) == 4 * pairs


def test_store_nt_uses_streaming_global_stores():
    sp = _spec(_P3, "b3", epi_v16=True, store_nt=True)
    b = _check_epilogue(sp, 0, 0)
    h, body = _hooks(sp)
    assert (
        b.nt_stores == h.mf["A"] // 2 * h.mf["B"] * body.tile.warp_m * body.tile.warp_n
    )
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    ll = lower_kernel_to_llvm(
        G.build_tensor_gett(sp), llvm_flavor="llvm22", arch="gfx950"
    )
    assert "!nontemporal" in ll
    assert not G.is_valid_spec(_spec(_P3, "b3", store_nt=True))[0]


# ------------------------------------------------------------------------------------------ K-contiguous bodies
_RVE_TILE = TileSpec(
    tile_m=256,
    tile_n=256,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_k=1,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=32,
)
_RVE_TRAIT = TraitSpec(
    direct_to_lds=True,
    dtl_prefetch=True,
    dtl_stages=2,
    dtl_frag_prefetch=True,
    acc_agpr=True,
    mfma_interleave=True,
    dtl_hoist=True,
    lds_row_interleave=True,
    rilv_pad=32,
    hb_sched=True,
    hb_read_every=2,
    rilv_vec_epilogue=True,
    rve_store_nt=True,
    dtl_soff0=True,
    kloop_unroll=4,
    mfma_serpentine=True,
    chiplet_swizzle=True,
    chiplet_wgm=16,
    chiplet_chunk_size=8,
)
_RING_TRAIT = dataclasses.replace(_B3_TRAIT, pad_n=True, dtl_tight_rsrc=True)
# a P2-style problem whose M and N are tile multiples (the vector epilogue has
# no bounds path): D canonical
_P2A = (
    ["K2", "M2", "K1", "M1", "K0", "M0"],
    ["K2", "N2", "K1", "N1", "K0", "N0"],
    ["N2", "N1", "N0", "M2", "M1", "M0"],
    {
        "K0": 16,
        "K1": 12,
        "K2": 8,
        "M0": 32,
        "M1": 16,
        "M2": 16,
        "N0": 16,
        "N1": 16,
        "N2": 8,
    },
)
# P1-style: D permuted, column-side run of 16
_P1A = (
    ["K1", "M1", "K0", "M0"],
    ["K1", "N1", "K0", "N0"],
    ["M1", "N1", "M0", "N0"],
    {"K0": 24, "K1": 32, "M0": 32, "M1": 16, "N0": 16, "N1": 32},
)


def _kc_spec(prob, tile, trait, **kn):
    a, b_, d, ext = prob
    return G.TensorGettSpec.make(
        a, b_, d, ext, knobs=G.GettKnobs(**kn), kc_tile=tile, kc_trait=trait
    )


def _check_rve(spec, bm, bn):
    h, body = _hooks(spec)
    pl = spec.plan()
    t = body.tile
    mfA, mfB = h.mf["A"], h.mf["B"]
    b = SimtOps()
    h.p = {"alpha": "alpha", "beta": "beta", "Cin": "Cin"}
    lane = V(range(W))
    q = [ln // 16 for ln in range(W)]
    cn = [ln % 16 for ln in range(W)]
    for wm, wn in itertools.product(range(t.warp_m), range(t.warp_n)):
        wmo, wno = wm * h.wmn["A"], wn * h.wmn["B"]
        # accumulator (mi, ni) slot v of lane (q, c): C[wm + (4q + v) mfA + mi, wn + c mfB + ni]
        accf = [
            V(
                tuple(
                    ("acc", wmo + (4 * q[ln] + v) * mfA + mi, wno + cn[ln] * mfB + ni)
                    for v in range(4)
                )
                for ln in range(W)
            )
            for mi in range(mfA)
            for ni in range(mfB)
        ]
        row0 = V(bm + wmo + q[ln] * 4 * mfA for ln in range(W))
        col0 = V(bn + wno + cn[ln] * mfB for ln in range(W))
        h.epilogue_rve(
            b,
            accf=accf,
            row0=row0,
            col0=col0,
            mfmas_m=mfA,
            mfmas_n=mfB,
            c_per_lane=4,
            C="D",
            M=None,
            N=None,
            block_m_off=bm,
            block_n_off=bn,
            warp_n_idx=wn,
            lane=lane,
        )
    written = {}
    for _ln, off, elems, ptr in b.stores:
        assert ptr == "D"
        for j, el in enumerate(elems):
            written.setdefault(off + 2 * j, []).append(el)
    expect = {}
    for r in range(t.tile_m):
        for c in range(t.tile_n):
            m, n = bm + r, bn + c
            if m >= pl.M or n >= pl.N:
                continue
            o = 2 * (
                mixed_offset(m, pl.d_row_digits) + mixed_offset(n, pl.d_col_digits)
            )
            acc = ("mul", "alpha", ("acc", r, c))
            if spec.bilinear:
                acc = ("fma", "beta", ("C", o // 2), acc)
            expect[o] = ("h", acc)
    assert set(written) == set(expect), (len(written), len(expect))
    for o, els in written.items():
        assert els == [expect[o]], (o, els, expect[o])


@pytest.mark.parametrize("prob", [_P2A, _P1A], ids=["P2", "P1"])
@pytest.mark.parametrize("bilinear", [False, True])
@pytest.mark.parametrize("cpf", [0, 3])
def test_kc_rve_epilogue_writes_every_output_once(prob, bilinear, cpf):
    spec = _kc_spec(
        prob,
        _RVE_TILE,
        _RVE_TRAIT,
        bilinear=bilinear,
        c_prefetch=cpf if bilinear else 0,
    )
    ok, why = G.is_valid_spec(spec)
    assert ok, why
    pl = spec.plan()
    assert pl.kc and pl.vec_dc8
    # the vector epilogue's 8 columns per lane: GEMM B owns D's fastest mode
    d_fast = min(pl.modes_d, key=lambda x: pl.strides_d[x])
    assert d_fast in pl.ops["B"].rows_order
    for bm, bn in {(0, 0), (pl.M - 256, pl.N - 256)}:
        _check_rve(spec, bm, bn)


@pytest.mark.parametrize("bilinear", [False, True])
def test_kc_default_epilogue_body_on_ragged_shape(bilinear):
    a, b_, d, ext = _P2
    spec = _kc_spec(_P2, _B3_TILE, _RING_TRAIT, bilinear=bilinear)
    ok, why = G.is_valid_spec(spec)
    assert ok, why
    pl = spec.plan()
    d_fast = min(pl.modes_d, key=lambda x: pl.strides_d[x])
    assert d_fast in pl.ops["A"].rows_order
    last_m = (pl.M - 1) // 256 * 256
    last_n = (pl.N - 1) // 256 * 256
    for bm, bn in {(0, 0), (last_m, last_n)}:
        _check_epilogue(spec, bm, bn)


def test_kc_operand_modes_are_rows_then_k():
    spec = _kc_spec(_P2A, _RVE_TILE, _RVE_TRAIT)
    pl = spec.plan()
    ma, mb = G.kc_operand_modes(spec)
    k_outer = tuple(reversed(pl.k_order))
    assert ma == tuple(pl.ops["A"].rows_order) + k_outer
    assert mb == tuple(pl.ops["B"].rows_order) + k_outer
    assert set(k_outer) == {"K0", "K1", "K2"}


def test_kc_body_rules():
    # ragged M / N on the vector epilogue body: refused by the launch rules
    ok, why = G.is_valid_spec(_kc_spec(_P2, _RVE_TILE, _RVE_TRAIT))
    assert not ok
    # MN-body knobs are refused
    assert not G.is_valid_spec(_kc_spec(_P2A, _RVE_TILE, _RVE_TRAIT, epi_v16=True))[0]
    # split-K / swapped accumulators are refused
    for lever in (dict(split_k=2), dict(mfma_swap_ab=True)):
        bad = _kc_spec(_P2A, _RVE_TILE, dataclasses.replace(_RVE_TRAIT, **lever))
        assert not G.is_valid_spec(bad)[0], lever
    # the body keeps its own operand bounds levers, drops the epilogue ones
    body = _kc_spec(_P2, _B3_TILE, _RING_TRAIT).body()
    assert (
        body.trait.pad_n and body.trait.dtl_tight_rsrc and not body.trait.rve_store_nt
    )


@pytest.mark.parametrize("prob", [_P2A, _P1A], ids=["P2", "P1"])
def test_kc_kernel_builds_and_uses_the_hook(prob):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    spec = _kc_spec(prob, _RVE_TILE, _RVE_TRAIT, bilinear=True)
    kd = G.build_tensor_gett(spec)
    names = [p.name for p in kd.params]
    assert names[:7] == ["A", "B", "C", "M", "N", "K", "alpha"] and names[7:] == [
        "Cin",
        "beta",
    ]
    ll = lower_kernel_to_llvm(kd, llvm_flavor="llvm22", arch="gfx950")
    # every D store is a 16-byte buffer store of the hook (no direct global stores)
    assert ll.count("raw.ptr.buffer.store.v4i32(") >= 2


# ------------------------------------------------------------------------------------------ mutation-guided
def test_spec_mn_layouts_reads_the_gett_quad_permutation():
    """gemm_universal derives the MN layouts from the body spec (with the hook
    attached): A's transpose reads use QUAD_PERM exactly when the GETT asks
    for the 16-byte epilogue."""
    for v16 in (False, True):
        spec = _spec(_P3, "b3", epi_v16=v16)
        body = spec.body()
        hooks = G.GettHooks(spec.plan(), spec.knobs, body)
        object.__setattr__(body, "_gett", hooks)
        la, lb = mno.spec_mn_layouts(body)
        assert la.sub_perm is v16
        assert lb.sub_perm is False


def test_kc_rve_epilogue_masks_ragged_columns():
    # vector-epilogue body with the padded-bounds levers; N (the columns,
    # D's fastest side) is not a tile multiple
    trait = dataclasses.replace(
        _RVE_TRAIT,
        pad_n=True,
        dtl_tight_rsrc=True,
        rve_pad=2,
        rve_store_nt=False,
        dtl_soff0=False,
    )
    a, b_, d, ext = _P2A
    # GEMM columns = the M modes (D's fastest side): 40 x 12 x 15 = 7200
    spec = _kc_spec((a, b_, d, dict(ext, M0=40, M1=12, M2=15)), _RVE_TILE, trait)
    pl = spec.plan()
    assert pl.N % 256, pl.N
    ok, why = G.is_valid_spec(spec)
    assert ok, why
    _check_rve(spec, 0, (pl.N - 1) // 256 * 256)


def test_kc_rve_needs_8_element_column_runs():
    # D's fastest run on the column side is 4 elements: no vector epilogue
    a, b_, d, ext = _P1A
    spec = _kc_spec((a, b_, d, dict(ext, N0=4, N1=64)), _RVE_TILE, _RVE_TRAIT)
    assert not spec.plan().vec_dc8
    ok, why = G.is_valid_spec(spec)
    assert not ok and "8 consecutive columns" in why


def test_pack_order_runs_to_a_multiple_of_8():
    from rocke.dispatch.families.tensor_gett import _pack_order

    modes = ("K1", "M1", "K0", "M0")
    ext = {"K0": 3, "K1": 5, "M0": 12, "M1": 2}
    # M0 alone (12) is not a multiple of 8: M1 joins the run (24)
    assert _pack_order(modes, {"K0", "K1"}, ext) == ("K1", "K0", "M1", "M0")
    ext8 = dict(ext, M0=16)
    assert _pack_order(modes, {"K0", "K1"}, ext8) == modes


def test_pack_order_keeps_outer_free_modes_outside():
    from rocke.dispatch.families.tensor_gett import _pack_order

    modes = ("M2", "K1", "M1", "K0", "M0")
    ext = {"K0": 3, "K1": 5, "M0": 12, "M1": 2, "M2": 3}
    # only K0 (inside the M1 x M0 = 24 run) moves; M2 and K1 stay outside
    assert _pack_order(modes, {"K0", "K1"}, ext) == ("M2", "K1", "K0", "M1", "M0")


def test_kc_vec_dc8_needs_8_contiguous_columns_in_strided_d():
    # D's column-side fastest mode has 4 elements and the next one a padded
    # stride of 8: the 8 GEMM columns of a vector store are not contiguous
    a, b_, d, ext = _P1A
    ext = dict(ext, N0=4, N1=64)
    st = {"N0": 1, "N1": 8, "M0": 8 * 64, "M1": 8 * 64 * 32}
    spec = G.TensorGettSpec.make(
        a, b_, d, ext, strides_d=st, kc_tile=_RVE_TILE, kc_trait=_RVE_TRAIT
    )
    pl = spec.plan()
    assert not pl.vec_dc8
    assert not G.is_valid_spec(spec)[0]
