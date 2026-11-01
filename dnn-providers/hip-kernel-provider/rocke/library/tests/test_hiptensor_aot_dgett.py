# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Numerics of the direct GETT kernel (``kernels.common.hiptensor_aot.dgett``): contractions read
and written in place, no workspace.

Every case is checked against an independent float64 (complex128) reference computed with torch
on the CPU from the problem's own strided views. Integer-valued operands make every product and
sum exact in the accumulator, so those results are compared bit for bit with the reference
rounded once to D's type; random operands are compared against the error bound of the
programmer's guide, ``2 (K + 4) u_c |alpha| sum |a b| + 2 u_d |d| + 4 u_c |beta c|``. Launch
arguments come from :mod:`kernels.common.hiptensor_aot.gett_host`, the Python statement of
hipTensor's C++ planner. The CPU tests check the K plans against a brute-force walk of the K
space.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.dgett import (
    DT_CODES,
    RND_BF16,
    RND_F16,
    RND_NONE,
    SLOTS_KL,
    SLOTS_KT,
    DgettSpec,
    build_dgett,
    dgett_config,
)
from kernels.common.hiptensor_aot.gett_host import (
    contraction_groups,
    dgett_plan,
    k_plan,
)

# ------------------------------------------------------------------------------------------------
# CPU: the K plans walk exactly the K space
# ------------------------------------------------------------------------------------------------


def _walk(kp, tk):
    """Every (A offset, B offset) the plan reads, from its digits."""
    out = []

    def mixed(x, digits, which):
        off = 0
        for r, sa, sb in digits:
            d = x % r
            x //= r
            off += d * (sa if which == 0 else sb)
        return off, x

    for kt in range(kp["nk"]):
        rest = kt
        first = None
        offs_t = [0, 0]
        for i, (e, sa, sb) in enumerate(kp["tile"]):
            d = rest % e
            rest //= e
            if i == 0:
                first = d
            offs_t[0] += d * sa
            offs_t[1] += d * sb
        for kl in range(tk):
            x = kl
            ok = True
            offs = [offs_t[0], offs_t[1]]
            for s, (r, sa, sb) in enumerate(kp["local"]):
                d = x % r
                x //= r
                if kp["pad"] is not None and kp["pad"][0] == s and d >= kp["pad"][1]:
                    ok = False
                offs[0] += d * sa
                offs[1] += d * sb
            if kp["krl"] is not None and kl + tk * first >= kp["krl"]:
                ok = False
            if ok:
                out.append(tuple(offs))
    return out


K_CASES = [
    ([(64, 1, 1)], 32),
    ([(24, 1, 1), (24, 24, 24)], 32),
    ([(8, 1, 1), (4, 8, 8), (3, 32, 32)], 32),
    ([(1000, 1, 3)], 32),
    ([(33, 1, 1)], 32),
    ([(5, 1, 1)], 16),
    ([(2, 1, 6), (2, 2, 3), (3, 4, 1)], 16),
    ([(7, 1, 7), (7, 7, 1)], 8),
    ([], 32),
    ([(2, 1, 1)] * 0 + [(2, 1 << i, 1 << (5 - i)) for i in range(6)], 16),
]


@pytest.mark.parametrize("kmodes,tk", K_CASES)
def test_k_plan_walks_the_k_space(kmodes, tk):
    modes = [(e, {"A": sa, "B": sb}) for e, sa, sb in kmodes]
    kp = k_plan(modes, tk, SLOTS_KL, SLOTS_KT)
    got = sorted(_walk(kp, tk))
    want = []
    exts = [e for e, _ in modes] or [1]
    for idx in itertools.product(*[range(e) for e in exts]):
        oa = sum(i * st["A"] for i, (_e, st) in zip(idx, modes)) if modes else 0
        ob = sum(i * st["B"] for i, (_e, st) in zip(idx, modes)) if modes else 0
        want.append((oa, ob))
    assert got == sorted(want)
    assert len(kp["local"]) <= SLOTS_KL and len(kp["tile"]) <= SLOTS_KT


def test_k_plan_prefers_an_exact_split():
    kp = k_plan([(24, {"A": 1, "B": 1}), (32, {"A": 24, "B": 24})], 32, 6, 8)
    # the mode of 32 innermost splits exactly (no padding)
    assert kp["pad"] is None and kp["krl"] is None
    assert kp["nk"] * 32 == 24 * 32


def test_specs_build():
    for comp, cx in (
        ("f16", 0),
        ("bf16", 0),
        ("f32", 0),
        ("f64", 0),
        ("f32", 1),
        ("f64", 1),
    ):
        for af, bf in itertools.product("kr", "kr"):
            sp = DgettSpec(comp, cplx=bool(cx), a_fast=af, b_fast=bf)
            k = build_dgett(sp, arch="gfx950")
            names = [p.name for p in k.params]
            assert names[:4] == ["A", "B", "C", "D"]
            assert len(set(names)) == len(names)


# ------------------------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
GPU = torch.cuda.is_available()
gpu = pytest.mark.skipif(not GPU, reason="needs a GPU")

TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
CTD = {"f32": torch.complex64, "f64": torch.complex128}
U = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24, "f64": 2.0**-53}
TINY = {"f16": 2.0**-25, "bf16": 2.0**-134, "f32": 2.0**-150, "f64": 0.0}
_CACHE = {}


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _run(kernel, vals, grid):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if kernel.name not in _CACHE:
        art = compile_kernel(
            kernel, arch=_arch(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _CACHE[kernel.name] = KernelLauncher(
            hsaco=art.hsaco, kernel_name=kernel.name, signature=sig
        )
    L = _CACHE[kernel.name]
    L(
        {p.name: vals[p.name] for p in kernel.params},
        config=LaunchConfig(grid=grid, block=(256, 1, 1)),
    )
    torch.cuda.synchronize()


def _tensor(shape, dtype, rng, cplx=False, kind="int", pad=0, perm=None):
    """(CPU float64 / complex128 values, device view with permuted padded strides, strides)."""
    n = len(shape)
    perm = list(range(n)) if perm is None else perm
    stor = [shape[p] for p in perm] if n else []
    if n:
        stor[-1] += pad
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    if kind == "int":
        base = torch.randint(-3, 4, stor or [1], generator=g).double()
        if cplx:
            base = torch.complex(
                base, torch.randint(-3, 4, stor or [1], generator=g).double()
            )
    else:
        base = torch.rand(stor or [1], generator=g, dtype=torch.float64) * 2 - 1
        if cplx:
            base = torch.complex(
                base, torch.rand(stor or [1], generator=g, dtype=torch.float64) * 2 - 1
            )
    tdt = CTD[dtype] if cplx else TD[dtype]
    dev = base.to(tdt).cuda()
    if not n:
        return (
            dev.cpu().reshape([]).to(torch.complex128 if cplx else torch.float64),
            dev.reshape([]),
            [],
        )
    view = dev[..., : shape[perm[-1]]] if pad else dev
    view = view.permute([perm.index(i) for i in range(n)])
    host = view.cpu().to(torch.complex128 if cplx else torch.float64)
    return host, view, list(view.stride())


def _op_ref(x, op):
    f = {
        C.OP_IDENTITY: lambda t: t,
        C.OP_CONJ: lambda t: t.conj() if t.is_complex() else t,
        C.OP_SQRT: torch.sqrt,
        C.OP_RELU: torch.relu,
        C.OP_RCP: torch.reciprocal,
        C.OP_SIGMOID: torch.sigmoid,
        C.OP_TANH: torch.tanh,
        C.OP_EXP: torch.exp,
        C.OP_LOG: torch.log,
        C.OP_ABS: torch.abs,
        C.OP_NEG: torch.neg,
        C.OP_SIN: torch.sin,
        C.OP_ATAN: torch.atan,
        C.OP_FLOOR: torch.floor,
    }[op]
    return f(x).resolve_conj() if x.is_complex() else f(x)


def _round(x, dt):
    """float64 values rounded once (to nearest even) to ``dt``: power-of-two scaling and
    ``torch.round`` (half to even) are exact in f64."""
    if dt == "f64":
        return x
    if dt == "f32":
        return x.to(torch.float32).double()
    p, emin, emax = {"f16": (11, -14, 15), "bf16": (8, -126, 127)}[dt]
    _m, e = torch.frexp(x)
    q = torch.clamp(e - 1, min=emin) - (p - 1)
    out = torch.round(torch.ldexp(x, -q)) * torch.pow(2.0, q.double())
    big = (2 - 2.0 ** (1 - p)) * 2.0**emax
    out = torch.where(
        out.abs() > big, torch.copysign(torch.full_like(out, math.inf), x), out
    )
    return torch.where(x.isfinite(), out, x)


def _letters(modes):
    return "".join(chr(ord("a") + m) for m in modes)


def run_case(
    rng,
    comp,
    *,
    dts=("f16", "f16", "f16", "f16"),
    cplx=False,
    ext=None,
    modes=None,
    bilinear=True,
    ops=(C.OP_IDENTITY, C.OP_IDENTITY, C.OP_IDENTITY),
    alpha=1.0,
    beta=1.0,
    kind="int",
    rnd=RND_NONE,
    math_ops=False,
    in_place=False,
    specials=None,
    idx64=False,
):
    """Builds a problem, runs it on the dgett kernel, checks it. ``modes``: dict A/B/D -> mode
    lists (column-major order irrelevant: the storage orders are random permutations).
    """
    da, db, dc, dd = dts
    tens = {}
    for t, dt in zip("ABCD", dts):
        ms = modes["D" if t == "C" else t]
        shape = [ext[m] for m in ms]
        perm = list(range(len(ms)))
        rng.shuffle(perm)
        pad = rng.choice([0, 0, 1, 3])
        tens[t] = (ms,) + _tensor(shape, dt, rng, cplx, kind, pad, perm)
    if in_place:
        tens["C"] = tens["D"]
    if specials:
        specials(tens)
    groups = contraction_groups(
        {t: tens[t][0] for t in "ABCD"},
        ext,
        {t: tens[t][3] for t in "ABCD"},
    )
    cfg = dgett_config(DgettSpec(comp, cplx=cplx))
    es = {
        t: (2 if dt in ("f16", "bf16") else 4 if dt == "f32" else 8)
        * (2 if cplx else 1)
        for t, dt in zip("ABCD", dts)
    }
    plan = dgett_plan(groups, es, cfg.tk, cfg.tm, cfg.tn)
    spec = DgettSpec(
        comp,
        cplx=cplx,
        a_fast=plan["a_fast"],
        b_fast=plan["b_fast"],
        math=math_ops,
        idx64=idx64 or plan["idx64"],
    )
    k = build_dgett(spec, arch=_arch())
    vals = dict(plan["vals"])
    ka, kb = ("B", "A") if plan["swapped"] else ("A", "B")
    oa, ob = (ops[1], ops[0]) if plan["swapped"] else (ops[0], ops[1])
    vals.update(
        A=tens[ka][2],
        B=tens[kb][2],
        C=tens["C"][2],
        D=tens["D"][2],
        dtA=DT_CODES[dts["AB".index(ka)]],
        dtB=DT_CODES[dts["AB".index(kb)]],
        dtC=DT_CODES[dc],
        dtD=DT_CODES[dd],
        opA=oa,
        opB=ob,
        opC=ops[2],
        rnd=rnd,
    )
    b_eff = beta if bilinear else 0.0
    if cplx:
        a_c, b_c = complex(alpha), complex(b_eff)
        vals.update(
            alpha_re=a_c.real, alpha_im=a_c.imag, beta_re=b_c.real, beta_im=b_c.imag
        )
    else:
        vals.update(alpha=float(alpha), beta=float(b_eff))
    d_before = tens["D"][1].clone()
    _run(k, vals, plan["grid"])
    got = tens["D"][2].cpu().to(torch.complex128 if cplx else torch.float64)

    # reference
    def operand(t, op, rd):
        x = _op_ref(tens[t][1], op)
        if rd:
            x = _round(x, rd)
        return x

    rd = {RND_NONE: None, RND_F16: "f16", RND_BF16: "bf16"}[rnd]
    if comp in ("f16", "bf16") and not cplx:
        rd = comp
    A = operand("A", ops[0], rd)
    B = operand("B", ops[1], rd)
    eq = f"{_letters(tens['A'][0])},{_letters(tens['B'][0])}->{_letters(tens['D'][0])}"
    S = torch.einsum(eq, A, B)
    Sabs = torch.einsum(eq, A.abs(), B.abs())
    kext = (
        math.prod(ext[m] for m in tens["A"][0] if m not in tens["D"][0])
        if tens["A"][0]
        else 1
    )
    if alpha == 0:
        ref = torch.zeros_like(S)
        sab = torch.zeros_like(Sabs)
    else:
        ref = alpha * S
        sab = abs(alpha) * Sabs
    cterm = torch.zeros_like(Sabs)
    if bilinear and beta != 0:
        cv = _op_ref(d_before if in_place else tens["C"][1], ops[2])
        ref = ref + beta * cv
        cterm = abs(beta) * cv.abs()
    u_acc = U["f64" if comp == "f64" else "f32"]
    want = _round(ref.real, dd) if not cplx else ref
    if kind == "int" and not cplx:
        assert torch.equal(got.isnan(), want.isnan())
        fin = ~want.isnan()
        assert torch.equal(got[fin], want[fin]), (got[fin] - want[fin]).abs().max()
        return
    bound = (
        2 * (kext + 4) * u_acc * sab
        + 2 * U[dd] * ref.abs()
        + 4 * u_acc * cterm
        + TINY[dd]
    )
    if any(
        o not in (C.OP_IDENTITY, C.OP_CONJ, C.OP_NEG, C.OP_ABS, C.OP_RELU, C.OP_FLOOR)
        for o in ops[:2]
    ):
        bound = bound + 16 * U["f32"] * sab
    if ops[2] not in (
        C.OP_IDENTITY,
        C.OP_CONJ,
        C.OP_NEG,
        C.OP_ABS,
        C.OP_RELU,
        C.OP_FLOOR,
    ):
        bound = bound + 16 * U["f64"] * cterm
    err = (got - ref).abs()
    bad = err > bound
    assert not bad.any(), (err[bad][:5], bound[bad][:5], int(bad.sum()))


def _shape(rng, rank_m, rank_n, rank_k, rank_l=0, big=False):
    ms = list(range(rank_m + rank_n + rank_k + rank_l))
    pick = [1, 2, 3, 5, 7, 8, 13, 16, 24, 33] if not big else [17, 32, 48, 64, 96, 130]
    ext = {m: rng.choice(pick) for m in ms}
    mm = ms[:rank_m]
    nn = ms[rank_m : rank_m + rank_n]
    kk = ms[rank_m + rank_n : rank_m + rank_n + rank_k]
    ll = ms[rank_m + rank_n + rank_k :]
    vol = lambda g: math.prod(ext[m] for m in g)  # noqa: E731

    def shrink(g):
        m = max(g, key=lambda x: ext[x])
        ext[m] = max(1, ext[m] // 2 + rng.choice([0, 1]))

    # keep the problem small enough for a float64 reference on the CPU
    while vol(mm + nn + ll) > 1 << 17:
        shrink(mm + nn + ll)
    while kk and vol(kk) > 1 << 11:
        shrink(kk)
    while kk and vol(mm + kk + ll) > 1 << 19:
        shrink(mm + kk)
    while kk and vol(nn + kk + ll) > 1 << 19:
        shrink(nn + kk)
    A = mm + kk + ll
    B = nn + kk + ll
    D = mm + nn + ll
    for x in (A, B, D):
        rng.shuffle(x)
    return ext, {"A": A, "B": B, "D": D}


def _comps():
    out = ["f16", "bf16", "f64"]
    if GPU and _arch() != "gfx90a":
        out.append("f32")
    return out


CASES = []
_rng = random.Random(4242)
for comp in ("f16", "bf16", "f32", "f64"):
    for rm, rn, rk in ((1, 1, 1), (2, 1, 2), (2, 2, 3), (3, 3, 2), (1, 2, 4)):
        CASES.append((comp, rm, rn, rk, _rng.random() < 0.5, _rng.randrange(1 << 20)))


@gpu
@pytest.mark.parametrize("comp,rm,rn,rk,bilinear,seed", CASES)
def test_dgett_exact(comp, rm, rn, rk, bilinear, seed):
    if comp not in _comps():
        pytest.skip(f"no {comp} atom on {_arch()}")
    rng = random.Random(seed)
    ext, modes = _shape(rng, rm, rn, rk, big=True)
    dt = {"f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64"}[comp]
    run_case(
        rng,
        comp,
        dts=(dt, dt, dt, dt),
        ext=ext,
        modes=modes,
        bilinear=bilinear,
        alpha=rng.choice([1.0, -2.0, 0.5]),
        beta=rng.choice([1.0, 0.25, -1.0]),
        ops=(
            rng.choice([C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]),
            rng.choice([C.OP_IDENTITY, C.OP_NEG]),
            rng.choice([C.OP_IDENTITY, C.OP_ABS]),
        ),
    )


RANDOM_CASES = []
for comp in ("f16", "bf16", "f32", "f64"):
    for i in range(3):
        RANDOM_CASES.append((comp, _rng.randrange(1 << 20)))


@gpu
@pytest.mark.parametrize("comp,seed", RANDOM_CASES)
def test_dgett_random_within_bound(comp, seed):
    if comp not in _comps():
        pytest.skip(f"no {comp} atom on {_arch()}")
    rng = random.Random(seed)
    ext, modes = _shape(rng, 2, 2, 2, rank_l=rng.choice([0, 1]), big=True)
    dt = {"f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64"}[comp]
    run_case(
        rng,
        comp,
        dts=(dt, dt, dt, dt),
        ext=ext,
        modes=modes,
        kind="rand",
        alpha=1.5,
        beta=-0.75,
    )


MIXED = [
    # comp, A, B, C, D
    ("f16", "f32", "bf16", "f64", "f16"),
    ("bf16", "f16", "f32", "bf16", "f32"),
    ("f32", "f16", "bf16", "f16", "f64"),
    ("f32", "bf16", "f32", "f32", "bf16"),
    ("f64", "f16", "f64", "f32", "f16"),
    ("f64", "f32", "bf16", "f64", "f64"),
]


@gpu
@pytest.mark.parametrize("row", MIXED)
def test_dgett_mixed_types(row):
    comp, *dts = row
    if comp not in _comps():
        pytest.skip(f"no {comp} atom on {_arch()}")
    rng = random.Random(hash(row) & 0xFFFF)
    ext, modes = _shape(rng, 2, 1, 2, big=True)
    run_case(
        rng,
        comp,
        dts=tuple(dts),
        ext=ext,
        modes=modes,
        kind="rand",
        alpha=-1.25,
        beta=0.5,
    )


@gpu
@pytest.mark.parametrize("rnd,dt", [(RND_F16, "f16"), (RND_BF16, "bf16")])
def test_dgett_f64_kernel_rounds_operands_once(rnd, dt):
    rng = random.Random(77 + rnd)
    ext, modes = _shape(rng, 1, 2, 2, big=True)
    run_case(
        rng,
        "f64",
        dts=("f64", "f32", "f32", "f32"),
        ext=ext,
        modes=modes,
        kind="rand",
        rnd=rnd,
    )


@gpu
@pytest.mark.parametrize("comp", ["f32", "f64"])
@pytest.mark.parametrize(
    "conj",
    [
        (C.OP_IDENTITY, C.OP_IDENTITY, C.OP_IDENTITY),
        (C.OP_CONJ, C.OP_IDENTITY, C.OP_CONJ),
        (C.OP_IDENTITY, C.OP_CONJ, C.OP_IDENTITY),
    ],
)
def test_dgett_complex(comp, conj):
    if comp not in _comps():
        pytest.skip(f"no {comp} atom on {_arch()}")
    rng = random.Random(len(comp) * 31 + conj[0] * 7 + conj[1])
    ext, modes = _shape(rng, 2, 1, 2, big=True)
    run_case(
        rng,
        comp,
        dts=("f32", "f32", "f32", "f32"),
        cplx=True,
        ext=ext,
        modes=modes,
        ops=conj,
        kind="rand",
        alpha=complex(1.5, -0.5),
        beta=complex(0.25, 2.0),
    )


@gpu
def test_dgett_complex_f64_from_f32_operands():
    rng = random.Random(9)
    ext, modes = _shape(rng, 1, 2, 3, big=True)
    run_case(
        rng,
        "f64",
        dts=("f32", "f64", "f64", "f32"),
        cplx=True,
        ext=ext,
        modes=modes,
        kind="rand",
        alpha=complex(0.0, 1.0),
        beta=complex(2.0, 0.0),
    )


@gpu
@pytest.mark.parametrize("op", [C.OP_SQRT, C.OP_SIGMOID, C.OP_EXP, C.OP_TANH])
@pytest.mark.parametrize("comp", ["f16", "f32", "f64"])
def test_dgett_operand_operators(op, comp):
    if comp not in _comps():
        pytest.skip(f"no {comp} atom on {_arch()}")
    rng = random.Random(op * 13 + len(comp))
    ext, modes = _shape(rng, 2, 1, 2, big=True)
    dt = {"f16": "f16", "f32": "f32", "f64": "f64"}[comp]

    def positive(tens):
        tens["A"][2].abs_()
        tens["A"][1].abs_()

    run_case(
        rng,
        comp,
        dts=(dt, dt, dt, dt),
        ext=ext,
        modes=modes,
        kind="rand",
        ops=(op, C.OP_IDENTITY, op if op != C.OP_SQRT else C.OP_IDENTITY),
        specials=positive,
    )


@gpu
@pytest.mark.parametrize("op", [C.OP_SIN, C.OP_ATAN, C.OP_FLOOR])
def test_dgett_math_operators(op):
    comp = "f32" if "f32" in _comps() else "f64"
    rng = random.Random(op)
    ext, modes = _shape(rng, 1, 2, 2, big=True)
    run_case(
        rng,
        comp,
        dts=("f32", "f32", "f32", "f32"),
        ext=ext,
        modes=modes,
        kind="rand",
        ops=(op, op, op),
        math_ops=True,
    )


@gpu
@pytest.mark.parametrize("comp", ["f16", "f64"])
def test_dgett_zero_scales_and_specials(comp):
    """alpha == 0: A and B are not used (a NaN or Inf in them does not reach D); beta == 0: C
    is not read (a NaN in C does not reach D); NaN / Inf in A reach exactly their outputs.
    """
    rng = random.Random(5)
    ext, modes = _shape(rng, 1, 1, 1, big=True)
    dt = "f16" if comp == "f16" else "f64"

    def put(t, i, v):
        idx = []
        for n in reversed(t.shape):
            idx.append(i % n)
            i //= n
        t[tuple(reversed(idx))] = v

    def poison(tens):
        put(tens["A"][2], 0, float("nan"))
        put(tens["B"][2], 1, float("inf"))
        put(tens["C"][2], 2, float("nan"))
        tens["A"][1].copy_(tens["A"][2].cpu().double())
        tens["B"][1].copy_(tens["B"][2].cpu().double())
        tens["C"][1].copy_(tens["C"][2].cpu().double())

    # alpha 0: D = beta * C exactly, the NaN of C included
    run_case(
        rng,
        comp,
        dts=(dt,) * 4,
        ext=ext,
        modes=modes,
        alpha=0.0,
        beta=1.0,
        specials=poison,
    )
    # beta 0: C not read
    run_case(
        rng,
        comp,
        dts=(dt,) * 4,
        ext=ext,
        modes=modes,
        alpha=1.0,
        beta=0.0,
        specials=poison,
    )
    # both: NaN / Inf of A and B reach their outputs, C's NaN its own
    run_case(
        rng,
        comp,
        dts=(dt,) * 4,
        ext=ext,
        modes=modes,
        alpha=1.0,
        beta=1.0,
        specials=poison,
    )


@gpu
def test_dgett_in_place():
    rng = random.Random(11)
    ext, modes = _shape(rng, 2, 2, 1, big=True)
    run_case(
        rng, "f16", dts=("f16",) * 4, ext=ext, modes=modes, in_place=True, beta=2.0
    )


@gpu
def test_dgett_batch_and_outer_product():
    rng = random.Random(12)
    ext, modes = _shape(rng, 1, 2, 0, rank_l=2, big=True)
    run_case(rng, "f64", dts=("f64",) * 4, ext=ext, modes=modes)
    ext, modes = _shape(rng, 2, 1, 1, rank_l=1, big=True)
    run_case(rng, "f16", dts=("f16",) * 4, ext=ext, modes=modes)


@gpu
@pytest.mark.parametrize("comp", ["f16", "f64"])
def test_dgett_64bit_offsets(comp):
    """The 64-bit-offset kernels on small problems (the same plan, 64-bit address arithmetic)."""
    rng = random.Random(13)
    ext, modes = _shape(rng, 2, 2, 2, big=True)
    run_case(rng, comp, dts=(comp,) * 4, ext=ext, modes=modes, idx64=True)
    run_case(
        rng,
        "f64",
        dts=("f32",) * 4,
        cplx=True,
        ext=ext,
        modes=modes,
        idx64=True,
        kind="rand",
    )


@gpu
def test_dgett_small_extents_pad_k():
    """K extents that do not split at the K tile (padded in-tile digit, ragged innermost mode)."""
    rng = random.Random(14)
    for kext in ([3, 5], [33], [24, 24], [7, 2, 3]):
        ms = list(range(2 + len(kext)))
        ext = {0: 37, 1: 29}
        ext.update({2 + i: e for i, e in enumerate(kext)})
        A = [0] + ms[2:]
        B = [1] + ms[2:]
        D = [0, 1]
        rng.shuffle(A)
        rng.shuffle(B)
        run_case(rng, "f16", dts=("f16",) * 4, ext=ext, modes={"A": A, "B": B, "D": D})
        run_case(rng, "f64", dts=("f64",) * 4, ext=ext, modes={"A": A, "B": B, "D": D})
