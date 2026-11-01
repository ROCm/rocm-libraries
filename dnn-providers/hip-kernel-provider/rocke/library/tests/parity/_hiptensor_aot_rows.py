# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/_hiptensor_aot_rows.py -- the hiptensor_aot kernel rows of
# hipTensor's kernel pack (the descriptor tables of hipTensor's pack builder),
# shared by the hiptensor_aot_* parity emitters. The C emitters enumerate the
# same rows in the same order.
_REAL = ("f16", "bf16", "f32", "f64")

EW_ROWS_1 = [
    ("f16", "f16"),
    ("f16", "f32"),
    ("bf16", "bf16"),
    ("bf16", "f32"),
    ("f32", "f32"),
    ("f32", "f16"),
    ("f32", "bf16"),
    ("f64", "f64"),
    ("f16", "f64"),
    ("bf16", "f64"),
    ("f32", "f64"),
]
EW_ROWS_2 = [
    ("f16", "f16", "f16"),
    ("f16", "f16", "f32"),
    ("bf16", "bf16", "bf16"),
    ("bf16", "bf16", "f32"),
    ("f32", "f32", "f32"),
    ("f32", "f16", "f16"),
    ("f32", "bf16", "bf16"),
    ("f64", "f64", "f64"),
    ("f64", "f16", "f16"),
    ("f64", "bf16", "bf16"),
    ("f64", "f32", "f32"),
]
EW_ROWS_3 = [
    ("f16", "f16", "f16", "f16"),
    ("f16", "f16", "f16", "f32"),
    ("bf16", "bf16", "bf16", "bf16"),
    ("bf16", "bf16", "bf16", "f32"),
    ("f32", "f32", "f32", "f32"),
    ("f64", "f64", "f64", "f64"),
]
EW_ROWS_C1 = [("f32", "f32"), ("f64", "f64"), ("f32", "f64")]
EW_ROWS_C2 = [("f32", "f32", "f32"), ("f64", "f64", "f64"), ("f64", "f32", "f32")]
EW_ROWS_1_MIXED = [
    (a, d) for d in ("f16", "bf16") for a in _REAL if (a, d) not in EW_ROWS_1
]
EW_ROWS_2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in _REAL
    for d in _REAL
    if (t, c, d) not in EW_ROWS_2
]
EW_ROWS_C2_MIXED = [
    (t, c, d)
    for t in ("f32", "f64")
    for c in ("f32", "f64")
    for d in ("f32", "f64")
    if (t, c, d) not in EW_ROWS_C2
]
REDUCE_ROWS = [("f16", "f16"), ("bf16", "bf16"), ("f32", "f32"), ("f64", "f64")]


def ew_params(idx64, mixed=True):
    """The EwSpec arguments of the generic elementwise rows (the pack's order)."""
    rows1 = EW_ROWS_1 + (EW_ROWS_1_MIXED if mixed else [])
    rows2 = EW_ROWS_2 + (EW_ROWS_2_MIXED if mixed else [])
    rowsc2 = EW_ROWS_C2 + (EW_ROWS_C2_MIXED if mixed else [])
    out = []
    for a, d in rows1:
        out.append(dict(arity=1, a=a, d=d, idx64=idx64))
    for a, c, d in rows2:
        out.append(dict(arity=2, a=a, c=c, d=d, idx64=idx64))
    for a, bb, c, d in EW_ROWS_3:
        out.append(dict(arity=3, a=a, b=bb, c=c, d=d, idx64=idx64))
    for a, d in EW_ROWS_C1:
        out.append(dict(arity=1, a=a, d=d, cplx=True, idx64=idx64))
    for a, c, d in rowsc2:
        out.append(dict(arity=2, a=a, c=c, d=d, cplx=True, idx64=idx64))
    out += [dict(p, math=True) for p in out if not p.get("cplx")]
    return out


def ew_rows():
    return [p for idx64 in (False, True) for p in ew_params(idx64)]


def reduce_rows():
    out = []
    for idx64 in (False, True):
        for math in (False, True):
            for a, d in REDUCE_ROWS:
                for mode in ("row", "col", "final"):
                    out.append(dict(mode=mode, a=a, d=d, idx64=idx64, math=math))
    return out


def plain_rows():
    """The real rows of the API's elementwise operations (32-bit offsets, no math)."""
    return [
        p
        for p in ew_params(False, mixed=False)
        if not p.get("cplx") and not p.get("math")
    ]


def ewt_rows():
    """Every plain row, then its folded class (several small planes per tile)."""
    keys = ("arity", "a", "b", "c", "d")
    out = []
    for p in plain_rows():
        for fold in (False, True):
            q = {k: v for k, v in p.items() if k in keys}
            if fold:
                q["fold"] = True
            out.append(q)
    return out


def ewv_rows():
    """(base, vec, unit) of the pack's vector class kernels."""
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    out = []
    for p in plain_rows():
        for c in jit_classes([EwSpec(**p)]):
            if c.vec > 1:
                out.append((p, c.vec, tuple(c.unit)))
    return out


def jit_rows():
    """(base, vec, unit) of every JIT class of the generic rows (32- and 64-bit offsets, the
    mixed-type rows and the math rows included: the class set of hipTensor's JIT)."""
    from kernels.common.hiptensor_aot.ew import EwSpec
    from kernels.common.hiptensor_aot.jit_ew import jit_classes

    out = []
    for idx64 in (False, True):
        for p in ew_params(idx64):
            for c in jit_classes([EwSpec(**p)]):
                out.append((p, c.vec, tuple(c.unit)))
    return out


def geometry(names, seed):
    """Distinct geometry values for ``names`` (the C emitters draw the same values from the
    same 64-bit LCG)."""
    state = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    out = {}
    for n in names:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out[n] = 1000 + (state >> 33) % ((1 << 30) - 1000)
    return out


#: The geometry points of the JIT emitters, with the target each point is lowered for.
GEOMETRY_POINTS = ("gfx950", "gfx942", "gfx90a")
#: Point 2: the edges of the constants' types, taken in turn by the axes.
_I32_EDGES = (-(1 << 31), -1, (1 << 31) - 1, 0, 1)
_I64_EDGES = (1 << 31, (1 << 32) + 7, (1 << 40) + 3, (1 << 63) - 1, -(1 << 31) - 1)


def geometry_point(axes, seed, point):
    """Geometry values for ``axes`` (``(name, is_i64)`` pairs) at one of three points:

    0. :func:`geometry`: distinct values in [1000, 2^30);
    1. 0, 1, 2 in turn: unit extents, zero (broadcast) and unit strides, small magic numbers;
    2. the edges of the constant types, each sixth axis the negative of a drawn value (an i32
       axis: a magic number stored as its i32 bits) or 2^31 plus it (an i64 axis): i32 axes
       INT32_MIN, -1, INT32_MAX, 0, 1; i64 axes 2^31, 2^32 + 7, 2^40 + 3, INT64_MAX,
       -2^31 - 1 (the strides and offsets of tensors past 2^31 elements).

    The C emitters compute the same values."""
    names = [n for n, _ in axes]
    drawn = geometry(names, seed)
    if point == 0:
        return drawn
    out = {}
    for j, (n, is64) in enumerate(axes):
        if point == 1:
            out[n] = j % 3
        elif j % 6 == 5:
            out[n] = (1 << 31) + drawn[n] if is64 else -drawn[n]
        else:
            out[n] = (_I64_EDGES if is64 else _I32_EDGES)[j % 6]
    return out


# The tuned reduction classes (kernels.common.hiptensor_aot.rreduce): a fixed set covering the
# builder's paths (both modes, every operator and data-type row, slot counts, lanes below / at /
# above the wave, thread rows, tiles per block, vector widths, unroll, streaming loads, opA,
# 64-bit offsets, wave32, mixed compute / output types), then every class of the packs of
# hipTensor's targets (rreduce_host.pack_classes), then the finalize classes. Fields: mode, a, d,
# op, nr, block, vec, lanes, unroll, rows, nt, una, idx64, wave.
RREDUCE_FIXED = [
    ("row", "f32", "f32", "add", 1, 256, 4, 64, 2, 0, 0, 0, 0, 64),
    ("row", "f16", "f16", "max", 1, 512, 8, 512, 1, 0, 1, 0, 0, 64),
    ("row", "bf16", "bf16", "min", 2, 256, 8, 8, 8, 0, 1, 0, 0, 64),
    ("row", "f64", "f64", "mul", 3, 256, 2, 64, 4, 0, 1, 0, 0, 64),
    ("row", "f32", "f32", "max", 4, 128, 1, 1, 1, 0, 0, 0, 0, 64),
    ("row", "f16", "f16", "add", 8, 1024, 2, 16, 4, 0, 0, 1, 0, 64),
    ("row", "f32", "f32", "add", 1, 256, 4, 256, 2, 0, 0, 0, 1, 64),
    ("row", "f64", "f64", "max", 2, 64, 2, 32, 2, 0, 0, 1, 1, 64),
    ("row", "f16", "f16", "min", 1, 64, 8, 4, 2, 0, 0, 0, 0, 32),
    ("row", "bf16", "bf16", "mul", 2, 128, 4, 64, 1, 0, 0, 0, 0, 32),
    ("row", "f16", "f64", "max", 1, 256, 8, 64, 2, 0, 0, 0, 0, 64),
    ("col", "f32", "f32", "add", 1, 256, 4, 32, 2, 0, 0, 0, 0, 64),
    ("col", "f16", "f16", "max", 1, 1024, 4, 32, 2, 8, 1, 0, 0, 64),
    ("col", "bf16", "bf16", "min", 2, 1024, 8, 8, 1, 16, 1, 0, 0, 64),
    ("col", "f64", "f64", "mul", 3, 256, 2, 2, 4, 8, 1, 0, 0, 64),
    ("col", "f32", "f32", "max", 2, 512, 2, 16, 1, 0, 0, 0, 0, 64),
    ("col", "f16", "f16", "add", 4, 64, 1, 64, 1, 0, 0, 0, 0, 64),
    ("col", "f64", "f64", "add", 8, 128, 1, 1, 8, 0, 0, 1, 0, 64),
    ("col", "f32", "f32", "min", 1, 256, 4, 64, 2, 1, 0, 0, 1, 64),
    ("col", "bf16", "bf16", "max", 1, 256, 2, 16, 2, 4, 0, 1, 1, 64),
    ("col", "f16", "f16", "mul", 2, 128, 8, 8, 2, 0, 0, 0, 0, 32),
    ("col", "f32", "f32", "add", 1, 256, 4, 128, 2, 0, 0, 0, 0, 64),
    ("col", "f64", "f16", "add", 1, 256, 2, 32, 2, 0, 0, 0, 0, 64),
]
RREDUCE_FINAL_FIXED = [
    # a, d, op, lanes, idx64, block, wave
    ("f16", "f64", "max", 8, 0, 128, 64),
    ("f32", "f32", "min", 16, 0, 64, 32),
    ("f64", "f16", "add", 2, 1, 256, 64),
]
RREDUCE_FIELDS = (
    "mode",
    "a",
    "d",
    "op",
    "nr",
    "block",
    "vec",
    "lanes",
    "unroll",
    "rows",
    "nt",
    "una",
    "idx64",
    "wave",
)
#: the targets whose packs carry the classes (hipTensor's HIPTENSOR_ROCKE_SUPPORTED_TARGETS)
RREDUCE_PACK_ARCHES = ("gfx90a", "gfx942", "gfx950")


def rreduce_rows():
    """``(kind, fields, arch)``: kind "class" (RreduceSpec fields) or "final"
    (RreduceFinalSpec fields); the arch the config is lowered for. A spec already listed (the
    fixed set, or another target's pack) is not listed again."""
    from dataclasses import asdict

    from kernels.common.hiptensor_aot import rreduce_host

    out = []
    for r in RREDUCE_FIXED:
        f = dict(zip(RREDUCE_FIELDS, r))
        for k in ("nt", "una", "idx64"):
            f[k] = bool(f[k])
        out.append(("class", f, "gfx950" if f["wave"] == 64 else "gfx1151"))
    seen = {tuple(sorted(f.items())) for k, f, _ in out if k == "class"}
    for arch in RREDUCE_PACK_ARCHES:
        # a class several targets share (the same spec) is one config, lowered for the first
        for s in rreduce_host.pack_classes(arch):
            f = asdict(s)
            key = tuple(sorted(f.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(("class", f, arch))
    for a, d, op, lanes, idx64, block, wave in RREDUCE_FINAL_FIXED:
        f = dict(
            a=a, d=d, op=op, lanes=lanes, idx64=bool(idx64), block=block, wave=wave
        )
        out.append(("final", f, "gfx950" if wave == 64 else "gfx1151"))
    for s in rreduce_host.final_classes("gfx950"):
        out.append(("final", asdict(s), "gfx950"))
    return out


def ewf_rows():
    """(row, nt) of the flat kernels: every plain row (the API's real 32-bit rows) without and
    with streaming accesses (the pack has the streaming classes on gfx942 / gfx950)."""
    return [(p, nt) for nt in (False, True) for p in plain_rows()]


#: Knob coverage of the tuned tiled elementwise kernel (ewp): every mode, kind mix, swizzle,
#: walk, tiles per CTA, phase-2 grouping, streaming bits and element size, beside the classes of
#: the packs. The C emitter lists the same specs in the same order.
EWP_COVER = [
    dict(
        arity=1,
        a="f16",
        d="f16",
        kinds="T",
        mode="t",
        to=64,
        tt=64,
        block=256,
        nt=3,
        swz=16,
        swsh=6,
    ),
    dict(arity=1, a="f32", d="f32", kinds="T", mode="t", to=32, tt=32, block=128),
    dict(
        arity=1,
        a="f64",
        d="f64",
        kinds="T",
        mode="t",
        to=16,
        tt=16,
        block=64,
        nt=1,
        walk="t",
    ),
    dict(
        arity=1,
        a="f16",
        d="f16",
        kinds="T",
        mode="a",
        to=8,
        tt=8,
        tw=8,
        block=256,
        nt=3,
        swz=2,
        swsh=6,
    ),
    dict(
        arity=2,
        a="f16",
        c="f16",
        d="f16",
        kinds="TV",
        mode="t",
        to=64,
        tt=64,
        block=128,
        nt=3,
        swz=16,
        swsh=6,
        walk="t",
        tpc=2,
        ilp=1,
    ),
    dict(
        arity=2,
        a="bf16",
        c="bf16",
        d="bf16",
        kinds="VT",
        mode="t",
        to=32,
        tt=64,
        block=64,
        nt=2,
        ilp=2,
    ),
    dict(
        arity=2,
        a="f32",
        c="f32",
        d="f32",
        kinds="TT",
        mode="t",
        to=32,
        tt=32,
        block=256,
        tpc=4,
    ),
    dict(
        arity=2,
        a="f64",
        c="f64",
        d="f64",
        kinds="TV",
        mode="a",
        to=4,
        tt=8,
        tw=1,
        block=64,
    ),
    dict(
        arity=3,
        a="f16",
        b="f16",
        c="f16",
        d="f16",
        kinds="TTV",
        mode="t",
        to=64,
        tt=64,
        block=128,
        nt=3,
        swz=16,
        swsh=6,
        ilp=4,
    ),
    dict(
        arity=3,
        a="f32",
        b="f32",
        c="f32",
        d="f32",
        kinds="VVT",
        mode="t",
        to=16,
        tt=32,
        block=64,
    ),
    dict(
        arity=3,
        a="f32",
        b="f32",
        c="f32",
        d="f32",
        kinds="TVT",
        mode="a",
        to=2,
        tt=4,
        tw=4,
        block=128,
        nt=3,
        tpc=8,
        ilp=8,
    ),
    dict(
        arity=2,
        a="bf16",
        c="bf16",
        d="bf16",
        kinds="TV",
        mode="t",
        to=256,
        tt=64,
        block=256,
        nt=3,
        swz=16,
        swsh=8,
        ilp=4,
    ),
    # vector widths below 16 bytes and ragged classes (EwpSpec.vb / rag)
    dict(
        arity=2,
        a="f16",
        c="f16",
        d="f16",
        kinds="TV",
        mode="t",
        to=16,
        tt=16,
        block=64,
        swz=16,
        swsh=4,
        walk="t",
        tpc=2,
        ilp=1,
        vb=4,
        rag=1,
    ),
    dict(
        arity=3,
        a="f32",
        b="f32",
        c="f32",
        d="f32",
        kinds="TVT",
        mode="a",
        to=2,
        tt=4,
        tw=8,
        block=64,
        nt=1,
        walk="t",
        tpc=2,
        vb=8,
        rag=1,
    ),
    dict(
        arity=1,
        a="f64",
        d="f64",
        kinds="T",
        mode="t",
        to=16,
        tt=16,
        block=64,
        nt=2,
        swz=16,
        swsh=4,
        vb=8,
        rag=1,
    ),
    dict(
        arity=2,
        a="bf16",
        c="bf16",
        d="bf16",
        kinds="VT",
        mode="t",
        to=32,
        tt=32,
        block=64,
        nt=3,
        swz=32,
        swsh=5,
        ilp=2,
        vb=8,
    ),
    dict(
        arity=2,
        a="f16",
        c="f16",
        d="f16",
        kinds="TT",
        mode="a",
        to=4,
        tt=4,
        tw=16,
        block=64,
        nt=2,
        swz=4,
        swsh=6,
        vb=2,
        rag=1,
    ),
    # transposed inputs read with wider vectors than D's (EwpSpec.tb)
    dict(
        arity=3,
        a="f16",
        b="f16",
        c="f16",
        d="f16",
        kinds="TVT",
        mode="t",
        to=16,
        tt=32,
        block=64,
        nt=2,
        walk="t",
        tpc=2,
        ilp=1,
        vb=2,
        rag=1,
        tb=8,
    ),
    dict(
        arity=2,
        a="f32",
        c="f32",
        d="f32",
        kinds="TT",
        mode="t",
        to=16,
        tt=16,
        block=64,
        nt=3,
        ilp=2,
        vb=8,
        tb=16,
    ),
]


def ewp_rows():
    """The ewp specs of the parity family: :data:`EWP_COVER`, then the classes of the gfx950 and
    gfx942 packs (``ewp_host.pack_classes``) not listed before (by kernel name)."""
    from dataclasses import asdict

    from kernels.common.hiptensor_aot import ewp_host
    from kernels.common.hiptensor_aot.ewp import EwpSpec

    out, seen = [], set()
    for d in EWP_COVER:
        n = EwpSpec(**d).kernel_name()
        if n not in seen:
            seen.add(n)
            out.append(dict(d))
    for arch in ("gfx950", "gfx942"):
        for s in ewp_host.pack_classes(arch):
            n = s.kernel_name()
            if n not in seen:
                seen.add(n)
                out.append(asdict(s))
    return out
