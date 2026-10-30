# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tuned tiled elementwise kernel (``kernels.common.hiptensor_aot.ewp``) and its host rules
(``ewp_host``).

CPU: spec validity, kernel names and ABI, the runs / launch arguments of layouts (refusals of
negative strides, unaligned strides, too many batch modes, offsets past 2^31), the palettes of the
packs (every class valid, names unique, LDS within the target's), the tuned problems (every entry
serves its problem with its class), the choice rule, and the C++ engine's copy of the table
(``gen_ewp_table.py --check``).

GPU: classes against the pack's generic elementwise kernel (``.ew``) on the same problem, bit for
bit (a NaN equals a NaN of any sign and payload), and on integer data also against an
independent float64 torch reference: transposing permutations, binary and trinary operations
(inputs transposed or laid out like D, both modes), every data-type row, (1) integer data with the
exact operators (the fast path and the operator table), (2) NaN, infinities, signed zeros,
overflowing and tiny values with every basic and binary operator, (3) a zero scale (that input,
all NaN, is not read) and (4) a NaN scale (counts as zero). Streaming classes run on gfx942 /
gfx950 only.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

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
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            spec(2, "f16", **kw).check()
    with pytest.raises(ValueError):
        EwpSpec(arity=2, a="f16", c="f32", d="f16", kinds="TV").check()  # element sizes
    assert LDS_BYTES == 163840 and SLOTS_B == 8


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
            assert c.kinds.count("T") * c.tile_bytes() <= H.LDS_LIMIT[arch]
            assert c.swz != 1  # names do not tell swizzle 0 from 1
            assert c.nt == 3
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
    # few CTAs: the class with the most
    small = problem(2, "ab", "ba", dict(a=128, b=128))
    s, (vals, grid), src = H.choose("gfx950", 2, row(2, "f16"), small, cls)
    assert src == "rule"
    n = grid[0] * grid[1] * grid[2]
    for c in cls:
        if (c.arity, c.a, c.d, c.kinds) == (2, "f16", "f16", "TV") and c.mode == "t":
            pl = ewp_plan(c, small)
            if pl is not None:
                assert pl[1][0] * pl[1][1] <= n
    # an input laid out neither like D nor fastest along a mode: no class
    weird = [(16, [2, 1, 1]), (16, [32, 16, 16]), (2, [1, 256, 256])]
    assert H.choose("gfx950", 2, row(2, "f16"), weird, cls) is None


def test_engine_table_is_current():
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
    extra = []
    for mode in ("t", "a"):
        k = ewp_kinds(NAMES[arity], dims, mode)
        if not k or "T" not in k:
            continue
        v = 16 // {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}[dt]
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
    out += extra[:2]
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
