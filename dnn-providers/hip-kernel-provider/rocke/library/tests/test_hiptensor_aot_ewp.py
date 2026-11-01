# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tuned tiled elementwise kernel (``kernels.common.hiptensor_aot.ewp``) and its host rules
(``ewp_host``).

CPU: spec validity, kernel names and ABI, the runs / launch arguments of layouts (refusals of
negative strides, unaligned strides, too many batch modes, offsets past 2^31), the vector widths
below 16 bytes and the ragged classes (run growth, tile counts, run lengths), the palettes of the
packs (every class valid, names unique, LDS within the target's), the tuned problems (every entry
serves its problem with its class), the choice rule (the ragged classes only where no other class
serves), and the C++ engine's copy of the table (``gen_ewp_table.py --check``).

GPU: classes against the pack's generic elementwise kernel (``.ew``) on the same problem, bit for
bit (a NaN equals a NaN of any sign and payload), and on integer data also against an
independent float64 torch reference: transposing permutations, binary and trinary operations
(inputs transposed or laid out like D, both modes; extents and strides that are not multiples of
16 bytes, runs that do not hold whole tiles), every data-type row, (1) integer data with the
exact operators (the fast path and the operator table), (2) NaN, infinities, signed zeros,
overflowing and tiny values with every basic and binary operator, (3) a zero scale (that input,
all NaN, is not read) and (4) a NaN scale (counts as zero). In place (C == D, C laid out like
D): the ragged classes whose C is read like D, exact against a reference computed from copies.
Streaming classes run on gfx942 / gfx950 only.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rocke.core.arch import ArchTarget

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot import ewp_host as H
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew
from kernels.common.hiptensor_aot.ewp import (
    LDS_BYTES,
    SLOTS_B,
    EwpSpec,
    build_ewp,
    default_swizzle,
    ewp_kinds,
    ewp_plan,
    ewp_runs,
    vector_widths,
)
from kernels.common.hiptensor_aot.host import canonicalize, ew_args


def colmajor(modes, ext):
    s, acc = {}, 1
    for m in modes:
        s[m] = acc
        acc *= ext[m]
    return s, acc


def problem(arity, ma, md, ext, mb=None, mc=None):
    """hipTensor's canonical problem of packed column-major tensors: A in ``ma`` order, B in
    ``mb`` (default ``ma``), C in ``mc`` (default ``md``), D in ``md`` order."""
    cols = [colmajor(ma, ext)[0]]
    if arity == 3:
        cols.append(colmajor(mb or ma, ext)[0])
    if arity >= 2:
        cols.append(colmajor(mc or md, ext)[0])
    cols.append(colmajor(md, ext)[0])
    return canonicalize([(ext[m], [c[m] for c in cols]) for m in md], len(cols) - 1)


NAMES = {1: ["A", "D"], 2: ["A", "C", "D"], 3: ["A", "B", "C", "D"]}


def row(arity, dt):
    return {1: (dt, None, None, dt), 2: (dt, None, dt, dt), 3: (dt, dt, dt, dt)}[arity]


def spec(arity, dt, **kw):
    a, b, c, d = row(arity, dt)
    return EwpSpec(arity=arity, a=a, b=b, c=c, d=d, **kw)


# ------------------------------------------------------------------------------------------ CPU
def test_spec_checks():
    spec(2, "f16", kinds="TV").check()
    bad = [
        dict(kinds="VV"),  # no transposed input
        dict(kinds="T"),  # one letter for two inputs
        dict(kinds="TV", to=48),  # not a power of two
        dict(kinds="TV", to=4),  # mode t: below the vector (8 f16)
        dict(kinds="TV", tw=2),  # mode t with tw
        dict(kinds="TV", block=32),
        dict(kinds="TV", nt=4),
        dict(kinds="TV", swz=16, swsh=2),  # not a bijection
        dict(kinds="TV", walk="x"),
        dict(kinds="TV", tpc=3),
        dict(kinds="TV", ilp=3),
        dict(kinds="TV", to=256, tt=256, block=64),  # phase-1 registers
        dict(kinds="TT", to=256, tt=256, block=512),  # two 128 KiB LDS tiles
        dict(kinds="TV", vb=3),  # not a vector width
        dict(kinds="TV", vb=32),
        dict(kinds="TV", rag=2),
        dict(kinds="TV", vb=2, nt=1),  # one-element vectors: no streaming loads
        dict(kinds="TV", vb=4, to=1, tt=4),  # mode t: below the vector (2 f16)
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            spec(2, "f16", **kw).check()
    with pytest.raises(ValueError):
        EwpSpec(arity=2, a="f16", c="f32", d="f16", kinds="TV").check()  # element sizes
    with pytest.raises(ValueError):
        spec(2, "f32", kinds="TV", vb=2).check()  # below the element size
    for vb, v in ((16, 8), (8, 4), (4, 2), (2, 1)):
        s = spec(2, "f16", kinds="TV", vb=vb, rag=1, nt=2)
        s.check()
        assert s.vec == v and s.chunks() == s.tile_bytes() // vb
    # the LDS budget without a target: the largest LDS of the pack targets (the catalog's)
    assert LDS_BYTES == max(
        ArchTarget.from_gfx(a).lds_capacity_bytes for a in C.PACK_TARGETS
    )
    assert SLOTS_B == 8
    # with a target, its LDS: two 64 KiB f16 tiles fit gfx950, not gfx942
    big = spec(2, "f16", kinds="TT", to=256, tt=128, block=512)
    big.check("gfx950")
    with pytest.raises(ValueError):
        big.check("gfx942")
    # the vector widths: powers of two up to the catalog's widest vector access
    assert vector_widths() == tuple(
        w for w in (2, 4, 8, 16, 32, 64) if w <= C.vector_bytes()
    )


def test_kernel_abi_and_names():
    s = spec(
        3,
        "f32",
        kinds="TTV",
        to=64,
        tt=32,
        block=128,
        nt=3,
        swz=16,
        swsh=5,
        walk="t",
        tpc=2,
        ilp=4,
    )
    k = build_ewp(s)
    names = [p.name for p in k.params]
    assert names[:12] == [
        "A",
        "B",
        "C",
        "D",
        "alpha",
        "beta",
        "gamma",
        "opA",
        "opB",
        "opC",
        "opAB",
        "opABC",
    ]
    assert names[12:27] == [
        "eo",
        "om",
        "os",
        "et",
        "tm",
        "ts",
        "nw",
        "wm",
        "ws",
        "nf",
        "fm",
        "fs",
        "ntl",
        "nb",
        "ystep",
    ]
    assert names[27:35] == [
        "sAo0",
        "sAo1",
        "sBo0",
        "sBo1",
        "sCo0",
        "sCo1",
        "sDo0",
        "sDo1",
    ]
    assert "be7" in names and "bm6" in names and "bm7" not in names
    assert (
        k.name
        == s.kernel_name()
        == ("hiptensor_aot_ewp3_f32_f32_f32_f32_TTV_t_o64_t32_b128_nt3_z16s5_wt_x2_g4")
    )
    assert spec(1, "f16", kinds="T", mode="a", to=4, tt=16, tw=8).kernel_name() == (
        "hiptensor_aot_ewp1_f16_f16_T_a_o4_t16_w8_b256"
    )
    # ragged classes: the run lengths after ystep (mode a: the shared mode's extent too)
    rt = spec(2, "f16", kinds="TV", vb=4, rag=1)
    na = [p.name for p in build_ewp(rt).params]
    assert na[na.index("ystep") + 1 : na.index("ystep") + 3] == ["lo", "lt"]
    assert "lw" not in na and rt.kernel_name().endswith("_b256_v4_r")
    ra = spec(1, "f64", kinds="T", mode="a", to=4, tt=16, tw=16, vb=8, rag=1, nt=2)
    na = [p.name for p in build_ewp(ra).params]
    assert na[na.index("ystep") + 1 : na.index("ystep") + 4] == ["lo", "lt", "lw"]
    assert ra.kernel_name() == "hiptensor_aot_ewp1_f64_f64_T_a_o4_t16_w16_b256_nt2_v8_r"
    nr = [p.name for p in build_ewp(replace(rt, rag=0)).params]
    assert "lo" not in nr and "lt" not in nr


def test_runs_and_plan():
    ext = dict(a=64, b=256, c=112, d=112)
    # NCHW-style permutation of 128-byte atoms (mode a): a shared, o = cd, t = b
    dims = problem(1, "abcd", "acdb", ext)
    assert (
        ewp_kinds(NAMES[1], dims, "a") == "T" and ewp_kinds(NAMES[1], dims, "t") == "V"
    )
    s = spec(1, "f16", kinds="T", mode="a", to=4, tt=16, tw=8)
    o, t, w = ewp_runs(s, dims)
    assert (
        dims[w][0] == 64
        and [dims[i][0] for i in o] == [12544]
        and [dims[i][0] for i in t] == [256]
    )
    vals, grid = ewp_plan(s, dims)
    assert vals["ntl"] == 1 * (12544 // 4) * (256 // 16) and grid == (vals["ntl"], 1, 1)
    # full reversal, binary (C like D): o = d (+c), t = a (+b)
    dims = problem(2, "abcd", "dcba", ext)
    assert ewp_kinds(NAMES[2], dims, "t") == "TV"
    s = spec(2, "f16", kinds="TV", to=128, tt=64)
    o, t, w = ewp_runs(s, dims)
    assert (
        w is None
        and [dims[i][0] for i in o] == [112, 112]
        and [dims[i][0] for i in t] == [64]
    )
    vals, grid = ewp_plan(s, dims)
    assert (
        vals["eo"] == 112
        and vals["sDo0"] == 1
        and vals["sDo1"] == 112
        and vals["sAt0"] == 1
    )
    assert vals["nb"] == 256 and grid[1] == 256
    # a run grows only by the mode contiguous after it in its own tensor: the 128 x 128 class
    # cannot place b in both runs, the 256 x 64 class can (o = c, b)
    dims = problem(1, "abcd", "cbad", ext)
    assert ewp_runs(spec(1, "f16", kinds="T", to=128, tt=128), dims) is None
    o, t, w = ewp_runs(spec(1, "f16", kinds="T", to=256, tt=64), dims)
    assert [dims[i][0] for i in o] == [112, 256] and [dims[i][0] for i in t] == [64]


def test_ragged_runs_and_plan():
    # hipTensor's rank-5 binary bench layout: D = (c, e, b, a, d), A = C = (a, b, c, d, e)
    ext = dict(a=48, b=40, c=50, d=46, e=50)
    dims = problem(2, "abcde", "cebad", ext, None, "abcde")
    assert ewp_kinds(NAMES[2], dims, "t") == "TT"
    # 50 elements along D's stride-1 mode: no 16-byte (nor 8-byte) f16 vectors
    for vb in (16, 8):
        assert ewp_plan(spec(2, "f16", kinds="TT", vb=vb, rag=1), dims) is None
    s = spec(2, "f16", kinds="TT", vb=4, rag=1)
    o, t, w = ewp_runs(s, dims)
    # ragged runs grow by the next contiguous mode: o = c (+e), t = a (+b)
    assert w is None
    assert [dims[i][0] for i in o] == [50, 50] and [dims[i][0] for i in t] == [48, 40]
    vals, grid = ewp_plan(s, dims)
    assert vals["lo"] == 2500 and vals["lt"] == 1920 and "lw" not in vals
    assert vals["ntl"] == -(-2500 // 64) * (1920 // 64) and vals["nb"] == 46
    # the non-ragged class of the same knobs needs whole tiles
    assert ewp_plan(replace(s, rag=0), dims) is None
    # D = (f, b, a, c, d, e), the inputs (a, b, ...): both 16-element stride-1 runs continue in
    # b, so one run takes it; the two orders waste as much (the 16-element run fills half a
    # 32-element tile). One or two inputs: D's run takes b; three: the inputs' run.
    ext = dict(a=16, b=32, c=4, d=4, e=2, f=16)
    for arity in (1, 2, 3):
        dims = problem(arity, "abcdef", "fbacde", ext, None, "abcdef")
        kinds = "T" * arity
        assert ewp_kinds(NAMES[arity], dims, "t") == kinds
        s = spec(arity, "f16", kinds=kinds, to=32, tt=32, block=64, rag=1)
        o, t, w = ewp_runs(s, dims)
        lo = 1
        for i in o:
            lo *= dims[i][0]
        lt = 1
        for i in t:
            lt *= dims[i][0]
        assert (lo, lt) == ((512, 16) if arity < 3 else (16, 512)), (arity, lo, lt)
    # a shared stride-1 mode of an odd extent (mode a, one-element vectors)
    ext = dict(a=121, b=120, c=110, d=130)
    dims = problem(2, "abcd", "adcb", ext, None, "abcd")
    assert ewp_kinds(NAMES[2], dims, "a") == "TT"
    for vb in (16, 8, 4):
        cls = spec(
            2, "f16", kinds="TT", mode="a", to=4, tt=16, tw=128 // vb, vb=vb, rag=1
        )
        assert ewp_plan(cls, dims) is None
    s = spec(2, "f16", kinds="TT", mode="a", to=4, tt=16, tw=64, vb=2, rag=1, nt=2)
    vals, grid = ewp_plan(s, dims)
    assert vals["lw"] == 121 and vals["nw"] == 2
    # o = d (130), t = b (+c: contiguous after b in A)
    assert vals["lo"] == 130 and vals["lt"] == 120 * 110
    assert vals["ntl"] == 2 * -(-130 // 4) * -(-13200 // 16)
    # a tile row shorter than 128 bytes of a longer shared mode: refused
    assert ewp_plan(replace(s, tw=32), dims) is None
    # wider vectors for the transposed inputs (tb): A's 48-element stride-1 mode takes 16-byte
    # loads while D's 50-element mode takes pairs
    ext = dict(a=48, b=40, c=50, d=46, e=50)
    dims = problem(2, "abcde", "cebad", ext, None, "abcde")
    u = spec(2, "f16", kinds="TT", vb=4, tb=16, rag=1)
    assert u.vect == 8 and u.vec == 2 and u.p1_units() == (64 // 2) * (64 // 8)
    assert u.kernel_name().endswith("_v4_r_u16")
    vals, grid = ewp_plan(u, dims)
    assert vals == ewp_plan(spec(2, "f16", kinds="TT", vb=4, rag=1), dims)[0]
    # A's stride-1 run of 6 elements: no 16-byte loads along it
    ext = dict(a=6, b=40, c=50, d=46)
    assert ewp_plan(u, problem(2, "abcd", "cdba", ext, None, "abcd")) is None
    with pytest.raises(ValueError):
        spec(
            1, "f16", kinds="T", mode="a", to=4, tt=16, tw=8, tb=16
        ).check()  # mode t only
    with pytest.raises(ValueError):
        spec(2, "f32", kinds="TT", vb=8, tb=2).check()  # below the element size


def test_plan_refusals():
    s = spec(2, "f16", kinds="TV", to=32, tt=32, block=64)
    ext = dict(a=32, b=32)
    dims = problem(2, "ab", "ba", ext)
    assert ewp_plan(s, dims) is not None
    neg = [(e, [st[0], -st[1], st[2]]) for e, st in dims]
    assert ewp_plan(s, neg) is None
    assert ewp_plan(s, [(32, [1, 32, 32]), (32, [104, 1, 1])]) is not None
    assert ewp_plan(s, [(32, [1, 32, 32]), (32, [99, 1, 1])]) is None  # A's stride % 8
    assert ewp_plan(s, [(32, [1, 32, 32]), (32, [1 << 27, 1, 1])]) is None  # 2^31
    many = [(32, [1, 32, 32]), (32, [32, 1, 1])] + [
        (2, [1024 << i, 1024 << i, 1024 << i]) for i in range(SLOTS_B + 1)
    ]
    assert ewp_plan(s, many) is None
    assert ewp_plan(s, many[: 2 + SLOTS_B]) is not None
    assert (
        ewp_plan(s, [(24, [1, 24, 24]), (32, [24, 1, 1])]) is None
    )  # 24 % 32 (no run)


def test_palettes():
    for arch in H.ARCHES:
        cls = H.pack_classes(arch)
        assert cls, arch
        names = [c.kernel_name() for c in cls]
        assert len(set(names)) == len(names)
        for c in cls:
            c.check()
            assert ArchTarget.from_gfx(arch).fits_lds(
                c.kinds.count("T") * c.tile_bytes()
            )
            assert c.swz != 1  # names do not tell swizzle 0 from 1
            # no streaming loads for one-element vectors and the ragged mode a classes
            assert c.nt == (2 if c.vec == 1 or (c.rag and c.mode == "a") else 3)
        # the ragged mode a classes (rows in bytes): vectors of several elements in rows of
        # 256 then 128 bytes (8 x 8), one-element vectors in rows of 64 elements (2 x 32) then 128
        # bytes (8 x 8), every vector width in rows of 64 bytes (8 x 16), one-element vectors in
        # rows of 64 elements of 8 x 8 (the fallback) unless that is 128 bytes; none for 8-byte
        # elements
        for es in (2, 4, 8):
            ra = [
                (t["tw"] * t["vb"], t["to"], t["tt"], t["vb"])
                for t in H.RAG_TEMPLATES[es]
                if t["mode"] == "a"
            ]
            wide = [vb for vb in (16, 8, 4) if vb > es]
            want = (
                [(256, 8, 8, vb) for vb in wide]
                + [(128, 8, 8, vb) for vb in wide]
                + [(64 * es, 2, 32, es), (128, 8, 8, es)]
                + [(64, 8, 16, vb) for vb in wide + [es]]
                + ([(64 * es, 8, 8, es)] if 64 * es != 128 else [])
            )
            assert ra == (want if es < 8 else []), es
    assert H.pack_classes("gfx90a") == []


def test_tuned_entries_serve_their_problems():
    for arch in H.ARCHES:
        cls = H.pack_classes(arch)
        for key, c in H.TUNED[arch]:
            s = EwpSpec(**c)
            assert s in cls
            arity, dts, modes = key.split("|")
            dims = [
                (int(m.split(":")[0]), [int(x) for x in m.split(":")[1].split(",")])
                for m in modes.split(";")
            ]
            dt = [x or None for x in dts.split(",")]
            assert H.problem_key(int(arity), dt, dims) == key
            got = H.choose(arch, int(arity), dt, dims, cls)
            assert got is not None and got[0] == s and got[2] == "tuned", key
        routes = H.routes(arch)
        assert len(routes) == 2 * len({k for k, _c in H.TUNED[arch]})


def test_choice_rule():
    cls = H.pack_classes("gfx950")
    ext = dict(a=64, b=256, c=112, d=112)
    dims = problem(1, "abcd", "acdb", ext)
    s, (vals, grid), src = H.choose("gfx950", 1, row(1, "bf16"), dims, cls) or (
        None,
        (0, 0),
        "",
    )
    assert s is None  # no bf16 permutation row
    s, (vals, grid), src = H.choose("gfx950", 1, row(1, "f32"), dims, cls)
    assert (
        s.mode == "a" and src in ("rule", "tuned") and grid[0] * grid[1] >= H.MIN_CTAS
    )
    # few CTAs: the class with the most (of the classes with whole tiles: the ragged ones only
    # where none of those serves)
    small = problem(2, "ab", "ba", dict(a=128, b=128))
    s, (vals, grid), src = H.choose("gfx950", 2, row(2, "f16"), small, cls)
    assert src == "rule" and s.rag == 0
    n = grid[0] * grid[1] * grid[2]
    for c in cls:
        if (c.arity, c.a, c.d, c.kinds, c.rag) == (
            2,
            "f16",
            "f16",
            "TV",
            0,
        ) and c.mode == "t":
            pl = ewp_plan(c, small)
            if pl is not None:
                assert pl[1][0] * pl[1][1] <= n
    # an input laid out neither like D nor fastest along a mode: no class
    weird = [(16, [2, 1, 1]), (16, [32, 16, 16]), (2, [1, 256, 256])]
    assert H.choose("gfx950", 2, row(2, "f16"), weird, cls) is None
    # the ragged classes serve what the others cannot, with the widest vector that fits
    ext = dict(a=48, b=40, c=50, d=46, e=50)
    for arity, mc, dt, vb in (
        (2, "abcde", "f16", 4),
        (1, None, "f16", 4),
        (3, "abcde", "f16", 4),
        (2, "abcde", "f32", 8),
        (2, "abcde", "f64", 16),
    ):
        dims = problem(arity, "abcde", "cebad", ext, None, mc)
        got = H.choose("gfx950", arity, row(arity, dt), dims, cls)
        assert got is not None and got[0].rag == 1 and got[0].vb == vb, (arity, dt, got)
    dims = problem(2, "abcd", "adcb", dict(a=121, b=120, c=110, d=130), None, "abcd")
    got = H.choose("gfx950", 2, row(2, "f16"), dims, cls)
    assert got[0].rag == 1 and got[0].vb == 2 and got[0].mode == "a"
    # D's odd stride-1 mode, A's of 696 elements: single elements on D's side, 16-byte
    # loads of the transposed inputs (tb)
    dims = problem(2, "abc", "bac", dict(a=696, b=585, c=40), None, "abc")
    got = H.choose("gfx950", 2, row(2, "f16"), dims, cls)
    assert got[0].rag == 1 and got[0].vb == 2 and got[0].tb == 16 and got[0].mode == "t"
    # a shared mode of 16-byte multiples (40 f16): no ragged class of mode a (the generic
    # kernel's vector form serves it); no other class either
    dims = problem(2, "abcd", "adcb", dict(a=40, b=6, c=5, d=7), None, "abcd")
    assert H.choose("gfx950", 2, row(2, "f16"), dims, cls) is None
    dims = problem(2, "abcd", "adcb", dict(a=42, b=6, c=5, d=7), None, "abcd")
    got = H.choose("gfx950", 2, row(2, "f16"), dims, cls)
    assert got[0].rag == 1 and got[0].mode == "a" and got[0].vb == 4
    # mode a: 121 f16 elements (one-element vectors) take rows of 64 elements of 2 x 32
    # positions; 120 f16 elements (8-byte vectors) rows of 256 bytes of 8 x 8 positions
    dims = problem(2, "abcd", "adcb", dict(a=121, b=120, c=110, d=130), None, "abcd")
    got = H.choose("gfx950", 2, row(2, "f16"), dims, cls)
    assert (got[0].tw * got[0].vb, got[0].to, got[0].tt, got[0].nt) == (128, 2, 32, 2)
    assert got[0].vec == 1
    dims = problem(2, "abcd", "adcb", dict(a=124, b=120, c=110, d=130), None, "abcd")
    got = H.choose("gfx950", 2, row(2, "f16"), dims, cls)
    assert (got[0].tw * got[0].vb, got[0].to, got[0].tt, got[0].vb) == (256, 8, 8, 8)
    # 23 f16 elements: rows of 128 bytes would span 2.8 times the shared mode (more than
    # twice): the 64-byte rows (8 x 16 positions)
    dims = problem(1, "abcd", "adcb", dict(a=23, b=48, c=48, d=48))
    got = H.choose("gfx950", 1, row(1, "f16"), dims, cls)
    assert got[0].rag == 1 and got[0].mode == "a"
    assert (got[0].tw * got[0].vb, got[0].to, got[0].tt) == (64, 8, 16)
    # 37 elements (one-element vectors) and short runs: no ragged class within twice the
    # elements; the 8 x 8 class of 64-element rows within three times (f16: 128 bytes, f32: 256)
    dims = problem(1, "abcd", "adcb", dict(a=37, b=9, c=5, d=33))
    for dt in ("f16", "f32"):
        got = H.choose("gfx950", 1, row(1, dt), dims, cls)
        assert (
            got[0].mode,
            got[0].to,
            got[0].tt,
            got[0].tw * got[0].vec,
            got[0].vec,
        ) == (
            "a",
            8,
            8,
            64,
            1,
        ), dt
        useful, work = H._rag_work(got[0], got[1][0])
        assert 2 * useful < work <= 3 * useful
    # 8-byte elements: no ragged class of mode a (the generic kernel serves them)
    dims = problem(2, "abcd", "adcb", dict(a=121, b=120, c=110, d=130), None, "abcd")
    assert H.choose("gfx950", 2, row(2, "f64"), dims, cls) is None
    # never where a class with whole tiles serves
    for arch in H.ARCHES:
        ac = H.pack_classes(arch)
        for arity, dims in (
            (1, problem(1, "abcd", "acdb", dict(a=64, b=256, c=112, d=112))),
            (2, problem(2, "abcd", "dcba", dict(a=64, b=256, c=112, d=112))),
        ):
            for dt in ("f16", "f32"):
                got = H.choose(arch, arity, row(arity, dt), dims, ac)
                assert got is not None and got[0].rag == 0


def test_engine_table_is_current():
    table = (
        Path(__file__).resolve().parents[2]
        / "platform"
        / "cpp"
        / "instances"
        / "common"
        / "hiptensor_aot_ewp_table.inc"
    )
    if not table.is_file():
        pytest.skip("no C++ engine sources here (the installed test tree)")
    tool = (
        Path(__file__).resolve().parents[1]
        / "builders"
        / "hiptensor"
        / "gen_ewp_table.py"
    )
    r = subprocess.run(
        [sys.executable, str(tool), "--check"], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------ GPU
torch = pytest.importorskip("torch")

TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
_BITS = {
    "f16": torch.int16,
    "bf16": torch.int16,
    "f32": torch.int32,
    "f64": torch.int64,
}
_SPECIALS = [math.nan, math.inf, -math.inf, -0.0, 0.0, 7.0e4, -3.0e38, 1.0e-30, 0.75]
_LAUNCHERS = {}


def _gpu():
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only runners
        pytest.skip("needs a GPU")
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _launch(kernel, vals, grid, block):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if kernel.name not in _LAUNCHERS:
        art = compile_kernel(
            kernel, arch=_gpu(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _LAUNCHERS[kernel.name] = (
            KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig),
            [p.name for p in kernel.params],
        )
    launcher, names = _LAUNCHERS[kernel.name]
    launcher(
        {n: vals[n] for n in names}, config=LaunchConfig(grid=grid, block=(block, 1, 1))
    )
    torch.cuda.synchronize()


def _binary(a, c, op):
    return {
        C.OP_ADD: lambda: a + c,
        C.OP_MUL: lambda: a * c,
        C.OP_MAX: lambda: torch.maximum(a, c),
        C.OP_MIN: lambda: torch.minimum(a, c),
    }[op]()


def _unary(x, op):
    return {
        C.OP_IDENTITY: lambda t: t,
        C.OP_CONJ: lambda t: t,
        C.OP_NEG: torch.neg,
        C.OP_ABS: torch.abs,
        C.OP_RELU: torch.relu,
    }[op](x)


def _bits(t, dt):
    t = t.cpu().contiguous()
    return t.view(_BITS[dt]), torch.isnan(t)


# (arity, A modes, D modes, B modes, C modes, extents): transposing layouts of every kind mix
LAYOUTS = [
    (1, "ab", "ba", None, None, dict(a=128, b=192)),
    (1, "abcd", "acdb", None, None, dict(a=32, b=32, c=6, d=8)),  # shared mode (atoms)
    (1, "abcd", "cbad", None, None, dict(a=64, b=8, c=128, d=3)),
    (2, "abc", "cba", None, None, dict(a=64, b=3, c=128)),  # TV
    (2, "abc", "cba", None, "abc", dict(a=64, b=3, c=128)),  # TT
    (2, "abcdef", "fedcba", None, None, dict(a=8, b=8, c=3, d=2, e=4, f=16)),
    (2, "abc", "acb", None, "acb", dict(a=16, b=32, c=32)),  # shared mode, TV
    (2, "abc", "cba", None, "cab", dict(a=64, b=8, c=128)),  # C transposed too
    (3, "abc", "cba", None, None, dict(a=64, b=3, c=128)),  # TTV
    (3, "abc", "cba", "cba", None, dict(a=64, b=3, c=128)),  # TVV
    (3, "abcd", "dcba", "abcd", "abcd", dict(a=64, b=3, c=2, d=128)),  # TTT
    # extents / strides that are not multiples of 16 bytes, runs without whole tiles (ragged
    # classes; the rank-5 bench layout of hipTensor's elementwise tests at a small size)
    (
        2,
        "abcde",
        "cebad",
        None,
        "abcde",
        dict(a=6, b=5, c=10, d=3, e=10),
    ),  # TT, 4-byte f16
    (1, "abcde", "cebad", None, None, dict(a=6, b=5, c=10, d=3, e=10)),
    (3, "abcde", "cebad", "abcde", "abcde", dict(a=6, b=5, c=10, d=3, e=10)),  # TTT
    (2, "abcd", "adcb", None, "abcd", dict(a=11, b=6, c=5, d=7)),  # shared odd mode (a)
    (
        1,
        "abcde",
        "acdbe",
        None,
        None,
        dict(a=12, b=11, c=10, d=11, e=3),
    ),  # shared 12 (a)
    (
        2,
        "abc",
        "cba",
        None,
        None,
        dict(a=72, b=3, c=40),
    ),  # 16-byte vectors, ragged runs
]
DTYPES = {
    1: ("f16", "f32", "f64"),
    2: ("f16", "bf16", "f32", "f64"),
    3: ("f16", "f32", "f64"),
}


def _cases():
    out = []
    for i, (arity, ma, md, mb, mc, ext) in enumerate(LAYOUTS):
        for dt in DTYPES[arity]:
            out.append((i, dt))
    return out


def _classes_for(arity, dt, dims, arch):
    """On the pack targets, the rule's pick over the device's palette plus every other palette
    class of the row that serves the layout (at most 3); everywhere the knob-coverage specs that
    serve it (small tiles, no streaming bits off the pack targets)."""
    cls = H.pack_classes(arch) if arch in H.ARCHES else []
    out = []
    got = H.choose(arch, arity, row(arity, dt), dims, cls) if cls else None
    if got is not None:
        out.append(got[0])
    for c in cls:
        if len(out) >= 4:
            break
        if (c.arity, c.d) == (arity, dt) and c not in out:
            k = ewp_kinds(NAMES[arity], dims, c.mode)
            if (
                c.kinds == k
                and ewp_plan(c, dims) is not None
                and (c.a, c.b, c.c) == row(arity, dt)[:3]
            ):
                out.append(c)
    extra, ragged = [], []
    es = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}[dt]
    for mode in ("t", "a"):
        k = ewp_kinds(NAMES[arity], dims, mode)
        if not k or "T" not in k:
            continue
        v = 16 // es
        tiles = (v, 2 * v, 4 * v) if mode == "t" else (1, 2, 4)
        # mode a: the whole shared mode per tile row (16-byte pieces of it only for 16-byte atoms)
        unit = next((e for e, st in dims if all(x == 1 for x in st)), v)
        tw = max(1, unit // v) if mode == "a" else 1
        for to in tiles:
            for tt in tiles:
                s = spec(
                    arity,
                    dt,
                    kinds=k,
                    mode=mode,
                    to=to,
                    tt=tt,
                    tw=tw,
                    block=64,
                    walk="t",
                    tpc=2,
                    ilp=1,
                )
                z, sh = default_swizzle(s)
                s = replace(s, swz=z, swsh=sh)
                try:
                    s.check()
                except ValueError:
                    continue
                if ewp_plan(s, dims) is not None:
                    extra.append(s)
        # ragged classes of every vector width that serves the layout (small tiles)
        for vb in (16, 8, 4, 2):
            if vb < es:
                continue
            vr = vb // es
            to, tt, rw = (
                (max(vr, 8), max(vr, 8), 1) if mode == "t" else (2, 4, 128 // vb)
            )
            s = spec(
                arity,
                dt,
                kinds=k,
                mode=mode,
                to=to,
                tt=tt,
                tw=max(1, rw),
                block=64,
                nt=0 if vr == 1 else 3,
                walk="t" if vb in (8, 2) else "o",
                tpc=2 if vb in (16, 2) else 1,
                ilp=1,
                vb=vb,
                rag=1,
            )
            z, sh = default_swizzle(s)
            s = replace(s, swz=z, swsh=sh)
            try:
                s.check()
            except ValueError:
                continue
            if ewp_plan(s, dims) is not None:
                ragged.append(s)
        # one-element vectors on D's side, 16-byte loads of the transposed inputs (tb)
        if mode == "t" and es < 16:
            s = spec(
                arity,
                dt,
                kinds=k,
                to=16,
                tt=16,
                block=64,
                nt=2,
                ilp=1,
                vb=es,
                rag=1,
                tb=16,
            )
            z, sh = default_swizzle(s)
            s = replace(s, swz=z, swsh=sh)
            if ewp_plan(s, dims) is not None:
                ragged.insert(0, s)
    out += extra[:2] + ragged[:3]
    if arch not in ("gfx942", "gfx950"):
        out = [replace(s, nt=0) for s in out]
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"L{c[0]}-{c[1]}")
def test_ewp_matches_generic_kernel(case):
    arch = _gpu()
    li, dt = case
    arity, ma, md, mb, mc, ext = LAYOUTS[li]
    dims = problem(arity, ma, md, ext, mb, mc)
    names = NAMES[arity]
    ins = names[:-1]
    specs = _classes_for(arity, dt, dims, arch)
    assert specs, (LAYOUTS[li], dt)
    base = EwSpec(
        arity=arity,
        a=dt,
        b=dt if arity == 3 else None,
        c=dt if arity >= 2 else None,
        d=dt,
        block=256,
    )
    n_el = math.prod(ext.values())
    rng = random.Random(1000 * li + len(dt))
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    exact_ops = [C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]
    modes = {"A": ma, "B": mb or ma, "C": mc or md, "D": md}
    for s in specs:
        vals0, grid = ewp_plan(s, dims)
        for case_ in ("fast", "reference", "special", "zero_scale", "nan_scale"):
            g = torch.Generator().manual_seed(rng.randrange(1 << 30))
            values = {}
            for n in ins:
                v = torch.randint(-6, 7, (n_el,), generator=g).double()
                if case_ == "special":
                    idx = torch.randperm(n_el, generator=g)[: max(1, n_el // 3)]
                    for k, j in enumerate(idx.tolist()):
                        v[j] = _SPECIALS[k % len(_SPECIALS)]
                values[n] = v
            if case_ == "fast":
                ops = {n: rng.choice([C.OP_IDENTITY, C.OP_CONJ]) for n in ins}
            else:
                pool = list(C.BASIC_UNARY_OPS) if case_ == "special" else exact_ops
                ops = {n: rng.choice(pool) for n in ins}
            scal = {n: rng.choice([1.0, 2.0, -1.0, 0.5]) for n in ins}
            if case_ in ("zero_scale", "nan_scale"):
                off = rng.choice(ins)
                scal[off] = 0.0 if case_ == "zero_scale" else math.nan
                values[off] = torch.full((n_el,), math.nan, dtype=torch.float64)
            dev = {n: values[n].to(TD[dt]).cuda() for n in ins}
            vals = {"op" + n: ops[n] for n in ins}
            vals["opAB"] = rng.choice(list(C.BINARY_OPS))
            vals["opABC"] = (
                vals["opAB"] if case_ == "fast" else rng.choice(list(C.BINARY_OPS))
            )
            run = dict(dev)
            run.update({sn[n]: scal[n] for n in ins})
            out_p = torch.full((n_el,), 5.0, dtype=TD[dt], device="cuda")
            out_g = out_p.clone()
            gv, ggrid, idx64 = ew_args(dims, names)
            assert not idx64
            _launch(build_ew(base), dict(gv, **vals, **run, D=out_g), ggrid, 256)
            _launch(build_ewp(s), dict(vals0, **vals, **run, D=out_p), grid, s.block)
            (gb, gnan), (pb, pnan) = _bits(out_g, dt), _bits(out_p, dt)
            diff = ((gb != pb) & ~(gnan & pnan)).nonzero().view(-1)
            assert diff.numel() == 0, (
                s.kernel_name(),
                case_,
                ops,
                vals["opAB"],
                scal,
                [
                    (i, out_g.cpu()[i].item(), out_p.cpu()[i].item())
                    for i in diff[:4].tolist()
                ],
            )
            if case_ == "special":
                continue

            def term(n):
                if not scal[n] == scal[n] or scal[n] == 0.0:
                    return torch.zeros(n_el, dtype=torch.float64)
                x = values[n].to(TD[dt]).double()
                src = list(reversed(modes[n]))
                dst = list(reversed(md))
                x = x.view(*[ext[m] for m in src]).permute(*[src.index(m) for m in dst])
                return scal[n] * _unary(x.reshape(-1), ops[n])

            if arity == 1:
                ref = term("A")
            elif arity == 2:
                ref = _binary(term("A"), term("C"), vals["opAB"])
            else:
                ref = _binary(
                    _binary(term("A"), term("B"), vals["opAB"]),
                    term("C"),
                    vals["opABC"],
                )
            want = ref.to(TD[dt])
            assert torch.equal(out_p.cpu(), want), (s.kernel_name(), case_, ops)


# In place: hipTensor allows C == D for binary and trinary operations when C has D's layout (so C
# is of kind V). Ragged classes clamp the units past the end of a run onto owned vectors; the
# store of such a unit, or of a later ilp group of the owning thread, must not land before the
# owner read C there (the operator would apply twice). (arity, A modes, D modes, B modes,
# extents); C is laid out like D and is D's buffer.
IN_PLACE_LAYOUTS = [
    (2, "ab", "ba", None, dict(a=4096, b=70)),  # D = A^T + C, 70-element rows
    (2, "ab", "ba", None, dict(a=70, b=4096)),
    (3, "abc", "cba", None, dict(a=40, b=3, c=72)),  # TTV
    (3, "abc", "cba", "cba", dict(a=40, b=3, c=72)),  # TVV
    (2, "abcde", "cebad", None, dict(a=6, b=5, c=10, d=3, e=10)),
    (2, "abcd", "adcb", None, dict(a=11, b=6, c=5, d=7)),  # mode a, odd shared mode
    (2, "abcde", "acdbe", None, dict(a=12, b=11, c=10, d=11, e=3)),  # mode a
]


def _in_place_cases():
    out = []
    for i, (arity, *_rest) in enumerate(IN_PLACE_LAYOUTS):
        for dt in ("f16", "f32", "f64"):
            out.append((i, dt))
    return out


def _in_place_classes(arity, dt, dims, arch):
    """Ragged classes whose C is of kind V that serve the layout: the rule's pick, other
    classes of the device's palette (at most 4 in all) and the small knob-coverage classes of
    :func:`_classes_for` (one-wave blocks, one unit per ilp group)."""
    cls = H.pack_classes(arch) if arch in H.ARCHES else []
    out = []
    got = H.choose(arch, arity, row(arity, dt), dims, cls) if cls else None
    if got is not None and got[0].rag and got[0].kind("C") == "V":
        out.append(got[0])
    for c in cls:
        if len(out) >= 4:
            break
        if (
            c.rag
            and (c.arity, c.d) == (arity, dt)
            and (c.a, c.b, c.c) == row(arity, dt)[:3]
            and c.kind("C") == "V"
            and c not in out
            and c.kinds == ewp_kinds(NAMES[arity], dims, c.mode)
            and ewp_plan(c, dims) is not None
        ):
            out.append(c)
    for s in _classes_for(arity, dt, dims, arch):
        if s.rag and s.kind("C") == "V" and s not in out:
            out.append(s)
    if arch not in ("gfx942", "gfx950"):
        out = [replace(s, nt=0) for s in out]
    return out


@pytest.mark.parametrize("case", _in_place_cases(), ids=lambda c: f"P{c[0]}-{c[1]}")
def test_ewp_in_place(case):
    """C == D: every ragged class that may run in place, exact against a float64 reference
    computed from copies of the inputs (integer data, the exact operators)."""
    arch = _gpu()
    li, dt = case
    arity, ma, md, mb, ext = IN_PLACE_LAYOUTS[li]
    dims = problem(arity, ma, md, ext, mb, None)
    names = NAMES[arity]
    ins = names[:-1]
    specs = _in_place_classes(arity, dt, dims, arch)
    if not specs:
        pytest.skip("no ragged class with C of kind V serves this layout")
    n_el = math.prod(ext.values())
    rng = random.Random(7000 + 10 * li + len(dt))
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    exact_ops = [C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]
    modes = {"A": ma, "B": mb or ma, "C": md}
    for s in specs:
        vals0, grid = ewp_plan(s, dims)
        for fast in (True, False):
            g = torch.Generator().manual_seed(rng.randrange(1 << 30))
            values = {
                n: torch.randint(-6, 7, (n_el,), generator=g).double() for n in ins
            }
            if fast:
                ops = {n: C.OP_IDENTITY for n in ins}
            else:
                ops = {n: rng.choice(exact_ops) for n in ins}
            scal = {n: rng.choice([1.0, 2.0, -1.0]) for n in ins}
            vals = {"op" + n: ops[n] for n in ins}
            vals["opAB"] = rng.choice(list(C.BINARY_OPS))
            vals["opABC"] = vals["opAB"] if fast else rng.choice(list(C.BINARY_OPS))
            run = {n: values[n].to(TD[dt]).cuda() for n in ins}
            run.update({sn[n]: scal[n] for n in ins})
            out = run["C"]  # D is C's buffer
            _launch(build_ewp(s), dict(vals0, **vals, **run, D=out), grid, s.block)

            def term(n):
                x = values[n].to(TD[dt]).double()
                src = list(reversed(modes[n]))
                dst = list(reversed(md))
                x = x.view(*[ext[m] for m in src]).permute(*[src.index(m) for m in dst])
                return scal[n] * _unary(x.reshape(-1), ops[n])

            if arity == 2:
                ref = _binary(term("A"), term("C"), vals["opAB"])
            else:
                ref = _binary(
                    _binary(term("A"), term("B"), vals["opAB"]),
                    term("C"),
                    vals["opABC"],
                )
            want = ref.to(TD[dt])
            bad = (out.cpu() != want).nonzero().view(-1)
            assert bad.numel() == 0, (
                s.kernel_name(),
                fast,
                ops,
                vals["opAB"],
                scal,
                bad.numel(),
                [(i, out.cpu()[i].item(), want[i].item()) for i in bad[:4].tolist()],
            )
