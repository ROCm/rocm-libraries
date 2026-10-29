# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics of the tuned reduction classes (``kernels.common.hiptensor_aot.rreduce``).

Every case is checked against an independent float64 reference computed with torch on the CPU
(sums, products, maxima and minima over permuted views). Integer-valued inputs (and power-of-two
scales) make every partial result exact in the compute type, so results are compared bit for
bit after the one rounding to D's type. NaN and infinity propagation, zero scales (a tensor whose
scale is zero is not read: NaN data must not reach D), unary operators on A and C, splits with
compute-type partials and 64-bit offsets are covered. Launch arguments come from
:mod:`kernels.common.hiptensor_aot.rreduce_host`, the Python statement of hipTensor's C++
planner.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover - CPU-only runners
    pytest.skip("needs a GPU", allow_module_level=True)

from kernels.common.hiptensor_aot import common as C  # noqa: E402
from kernels.common.hiptensor_aot import rreduce_host as H  # noqa: E402
from kernels.common.hiptensor_aot.host import canonicalize  # noqa: E402
from kernels.common.hiptensor_aot.rreduce import (  # noqa: E402
    build_rreduce,
    build_rreduce_final,
)

TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
OPCODE = {"add": C.OP_ADD, "mul": C.OP_MUL, "max": C.OP_MAX, "min": C.OP_MIN}
_CACHE = {}


def _arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _wave() -> int:
    return 32 if _arch().startswith("gfx1") else 64


def _launcher(kernel):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    if kernel.name not in _CACHE:
        art = compile_kernel(
            kernel, arch=_arch(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "") or ""}
            for p in kernel.params
        ]
        _CACHE[kernel.name] = (
            KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig),
            kernel,
        )
    return _CACHE[kernel.name]


def _run(kernel, vals, grid, block):
    from rocke.runtime.launcher import LaunchConfig

    L, k = _launcher(kernel)
    L(
        {p.name: vals[p.name] for p in k.params},
        config=LaunchConfig(grid=grid, block=block),
    )
    torch.cuda.synchronize()


def _strided(shape, dtype, rng, perm, pad, values):
    """A tensor with ``shape`` stored in the mode order ``perm`` (innermost last), ``pad``
    elements of padding on its innermost storage row, entries drawn from ``values``. Returns
    (cpu view, device view, strides)."""
    n = len(shape)
    stor = [shape[p] for p in perm]
    stor[-1] += pad
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    pick = torch.randint(0, len(values), stor, generator=g)
    base = torch.tensor(values, dtype=torch.float64)[pick].to(TD[dtype])
    view = base[..., : shape[perm[-1]]] if pad else base
    view = view.permute([perm.index(i) for i in range(n)])
    dev = base.cuda()
    dview = dev[..., : shape[perm[-1]]] if pad else dev
    dview = dview.permute([perm.index(i) for i in range(n)])
    return view, dview, list(dview.stride())


def _reduce_ref(x, red, op):
    if not red:
        return x
    if op == "add":
        return x.sum(dim=red)
    if op == "max":
        return x.amax(dim=red)
    if op == "min":
        return x.amin(dim=red)
    r = x
    for dd in sorted(red, reverse=True):
        r = r.prod(dim=dd)
    return r


def _unary_ref(x, code):
    return {
        C.OP_IDENTITY: lambda t: t,
        C.OP_ABS: torch.abs,
        C.OP_NEG: torch.neg,
        C.OP_RELU: torch.relu,
    }[code](x)


def _problem(shape, kept, sa, sd):
    """hipTensor's canonical kept / reduced modes of a problem."""
    kd = [(shape[k], [sa[k], sd[j]]) for j, k in enumerate(kept)]
    rd = [(shape[r], [sa[r]]) for r in range(len(shape)) if r not in kept]
    kd = canonicalize(kd, 0)
    rd = canonicalize(rd, 0)
    return [(e, s[0], s[1]) for e, s in kd], [(e, s[0]) for e, s in rd]


def _launch(plan, A, Cd, Dd, alpha, beta, opa, opc, W=None):
    vals = dict(plan.args)
    cdt = torch.float64 if plan.spec.cdt == "f64" else torch.float32
    if W is None:
        W = torch.full(
            (max(1, plan.workspace // (8 if cdt == torch.float64 else 4)),),
            float("nan"),
            dtype=cdt,
            device="cuda",
        )
    vals.update(A=A, C=Cd, D=Dd, W=W, alpha=alpha, beta=beta, opA=opa, opC=opc)
    _run(build_rreduce(plan.spec), vals, plan.grid, plan.block)
    if plan.final is not None:
        fv = dict(plan.final_args)
        fv.update(W=W, C=Cd, D=Dd, alpha=alpha, beta=beta, opC=opc)
        _run(
            build_rreduce_final(plan.final),
            fv,
            plan.final_grid,
            (plan.final.block, 1, 1),
        )


def _knob_sets(kept, red, a, wave, rng):
    """Valid knob sets of the problem: a spread over lanes, vec, unroll, rows and splits."""
    mode = H.mode_of(kept, red)
    out = []
    mv = H.max_vec(kept, red, a, mode)
    vecs = sorted({1, mv})
    if mode == "row":
        lane_opts = (1, 4, wave, 2 * wave)
        rows_opts = (0,)
    else:
        lane_opts = (1, 8, wave)
        rows_opts = (0, 2, 16)
    for block, lanes, vec, unroll, rows in itertools.product(
        (wave * 2, 256), lane_opts, vecs, (1, 4), rows_opts
    ):
        for split in (1, 2, 7):
            k = H.Knobs(mode, block, vec, lanes, unroll, split, rows)
            if H.check_knobs(kept, red, a, k, wave=wave, arch=_arch()) is None:
                out.append(k)
    rng.shuffle(out)
    return out[:6]


DTS = (("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("f64", "f64"))
CASES = []
_rng0 = random.Random(2026)
for (a, d), op in itertools.product(DTS, ("add", "mul", "max", "min")):
    for _ in range(4):
        CASES.append((a, d, op, _rng0.randrange(1 << 20)))


@pytest.mark.parametrize("a,d,op,seed", CASES)
def test_rreduce_exact(a, d, op, seed):
    """Random ranks, extents, storage orders, padding and kept sets; every knob set the rules
    admit gives the reference bit for bit."""
    rng = random.Random(seed)
    wave = _wave()
    rank = rng.choice([1, 2, 3, 4, 5, 6])
    shape = [rng.choice([2, 3, 5, 8, 16, 24, 40, 64]) for _ in range(rank)]
    kept = sorted(rng.sample(range(rank), rng.randrange(0, rank)))
    red = [i for i in range(rank) if i not in kept]
    # at most 2^20 elements (and 256 reduced elements for products: no overflow of f32)
    while math.prod(shape) > 1 << 20:
        i = max(range(rank), key=lambda j: shape[j])
        shape[i] = max(2, shape[i] // 2)
    while op == "mul" and math.prod(shape[r] for r in red) > 256:
        i = max(red, key=lambda j: shape[j])
        shape[i] = max(2, shape[i] // 2)
    if op == "mul":
        values = [1.0, -1.0, 1.0, 1.0, -1.0, 2.0, 0.5]
    else:
        values = [float(v) for v in range(-3, 4)]
    perm = list(range(rank))
    rng.shuffle(perm)
    pad = rng.choice([0, 0, 3])
    A, Ad, sa = _strided(shape, a, rng, perm, pad, values)
    dshape = [shape[k] for k in kept] or [1]
    dperm = list(range(len(dshape)))
    rng.shuffle(dperm)
    Cm, Cd, sc = _strided(dshape, d, rng, dperm, 0, [float(v) for v in range(-3, 4)])
    kd, rd = _problem(shape, kept, sa, sc if kept else [])
    alpha = rng.choice([1.0, 2.0, 0.5, -1.0])
    beta = rng.choice([0.0, 1.0, -2.0])
    ref = alpha * _reduce_ref(A.double(), red, op).reshape(dshape) + beta * Cm.double()
    want = ref.float().to(TD[d]) if d != "f64" else ref
    for k in _knob_sets(kd, rd, a, wave, rng):
        p = H.plan(kd, rd, a, d, op, k, wave=wave, arch=_arch())
        Dd = torch.empty_strided(Cd.shape, Cd.stride(), dtype=TD[d], device="cuda")
        Dd.fill_(float("nan"))
        _launch(p, Ad, Cd, Dd, alpha, beta, C.OP_IDENTITY, C.OP_IDENTITY)
        got = Dd.cpu()
        assert torch.equal(got, want), (
            p.spec.kernel_name(),
            k,
            (got.double() - want.double()).abs().max(),
        )


@pytest.mark.parametrize(
    "op,mode", list(itertools.product(("max", "min", "add", "mul"), ("row", "col")))
)
def test_rreduce_nan_inf(op, mode):
    """NaN reaches exactly the outputs whose reduction saw one (MAX / MIN propagate NaN); an
    infinity behaves as in IEEE arithmetic."""
    rng = random.Random(7)
    wave = _wave()
    # torch layouts: the last mode is innermost (row: reduced, col: kept)
    shape = [33, 48] if mode == "row" else [40, 64]
    kept = [0] if mode == "row" else [1]
    red = [1] if mode == "row" else [0]
    x = torch.tensor([float(v) for v in range(-3, 4)], dtype=torch.float64)[
        torch.randint(0, 7, shape, generator=torch.Generator().manual_seed(3))
    ]
    nan_out = [2, 5, 17]
    for o in nan_out:
        if mode == "row":
            x[o, rng.randrange(shape[1])] = float("nan")
        else:
            x[rng.randrange(shape[0]), o] = float("nan")
    inf_out = 9
    if mode == "row":
        x[inf_out, 3] = float("inf")
    else:
        x[3, inf_out] = float("inf")
    A = x.float()
    Ad = A.cuda()
    sa = list(Ad.stride())
    dshape = [shape[kept[0]]]
    Dd = torch.full(dshape, 7.0, dtype=torch.float32, device="cuda")
    kd, rd = _problem(shape, kept, sa, list(Dd.stride()))
    assert H.mode_of(kd, rd) == mode
    ref = _reduce_ref(x, red, op)
    for k in _knob_sets(kd, rd, "f32", wave, rng):
        p = H.plan(kd, rd, "f32", "f32", op, k, wave=wave, arch=_arch())
        Dd.fill_(7.0)
        _launch(p, Ad, Dd, Dd, 1.0, 0.0, C.OP_IDENTITY, C.OP_IDENTITY)
        got = Dd.cpu().double()
        assert torch.equal(got.isnan(), ref.isnan()), (k, got, ref)
        fin = ~ref.isnan()
        assert torch.equal(got[fin], ref[fin]), k


@pytest.mark.parametrize("dt", ["f32", "f16", "f64"])
@pytest.mark.parametrize("mode", ["row", "col"])
def test_rreduce_zero_scales(dt, mode):
    """alpha == 0: A (all NaN) is not read; beta == 0: C (all NaN) is not read; both zero: D = 0."""
    wave = _wave()
    shape = [24, 40] if mode == "row" else [24, 64]
    kept = [0] if mode == "row" else [1]
    Ad = torch.full(shape, float("nan"), dtype=TD[dt], device="cuda")
    dshape = [shape[kept[0]]]
    Cd = torch.arange(dshape[0], dtype=TD[dt], device="cuda") - 5
    Dd = torch.empty_like(Cd)
    kd, rd = _problem(shape, kept, list(Ad.stride()), list(Cd.stride()))
    rng = random.Random(5)
    for op in ("add", "max"):
        for k in _knob_sets(kd, rd, dt, wave, rng)[:3]:
            p = H.plan(kd, rd, dt, dt, op, k, wave=wave, arch=_arch())
            Dd.fill_(float("nan"))
            _launch(p, Ad, Cd, Dd, 0.0, -2.0, C.OP_IDENTITY, C.OP_NEG)
            assert torch.equal(Dd.cpu().double(), 2.0 * Cd.cpu().double()), k
            Dd.fill_(float("nan"))
            _launch(p, Ad, Cd, Dd, 0.0, 0.0, C.OP_IDENTITY, C.OP_IDENTITY)
            assert torch.equal(
                Dd.cpu().double(), torch.zeros(dshape, dtype=torch.float64)
            ), k
    # beta == 0 with a NaN C, finite A
    A2 = (torch.arange(math.prod(shape), dtype=torch.float64) % 5 - 2).reshape(shape)
    A2d = A2.to(TD[dt]).cuda()
    Cn = torch.full(dshape, float("nan"), dtype=TD[dt], device="cuda")
    ref = 2.0 * _reduce_ref(A2, [1 - kept[0]], "add")
    for k in _knob_sets(kd, rd, dt, wave, rng)[:3]:
        p = H.plan(kd, rd, dt, dt, "add", k, wave=wave, arch=_arch())
        Dd.fill_(float("nan"))
        _launch(p, A2d, Cn, Dd, 2.0, 0.0, C.OP_IDENTITY, C.OP_IDENTITY)
        assert torch.equal(Dd.cpu().double(), ref.to(TD[dt]).double()), k


@pytest.mark.parametrize("opa", [C.OP_ABS, C.OP_NEG, C.OP_RELU])
@pytest.mark.parametrize("op", ["add", "max"])
def test_rreduce_unary_a(opa, op):
    """The ``una`` classes apply opA to every element before the reduction (opC to C)."""
    rng = random.Random(opa * 7 + len(op))
    wave = _wave()
    for shape, kept in (([96, 17], [1]), ([64, 30], [0]), ([8, 12, 20], [0, 2])):
        A, Ad, sa = _strided(
            shape,
            "f32",
            rng,
            list(range(len(shape))),
            0,
            [float(v) for v in range(-3, 4)],
        )
        dshape = [shape[k] for k in kept]
        Cm, Cd, sc = _strided(
            dshape, "f32", rng, list(range(len(dshape))), 0, [1.0, -2.0]
        )
        kd, rd = _problem(shape, kept, sa, sc)
        red = [i for i in range(len(shape)) if i not in kept]
        ref = _reduce_ref(_unary_ref(A.double(), opa), red, op) - _unary_ref(
            Cm.double(), C.OP_NEG
        )
        for k in _knob_sets(kd, rd, "f32", wave, rng)[:3]:
            p = H.plan(kd, rd, "f32", "f32", op, k, una=True, wave=wave, arch=_arch())
            Dd = torch.full(dshape, float("nan"), device="cuda")
            _launch(p, Ad, Cd, Dd, 1.0, -1.0, opa, C.OP_NEG)
            assert torch.equal(Dd.cpu().double(), ref), (k, opa)


def test_rreduce_split_partials_are_compute_type():
    """Split partials stay in the compute type: an f16 sum whose partials are not f16 values
    still rounds once."""
    wave = _wave()
    n = 4096
    # 1 + many tiny values: every partial is 2^-12 * count, not an f16 multiple near 1
    x = torch.full([n], 2.0**-12, dtype=torch.float64)
    x[0] = 1.0
    Ad = x.half().cuda()
    Dd = torch.zeros([1], dtype=torch.float16, device="cuda")
    kd, rd = _problem([n], [], list(Ad.stride()), [])
    ref = x.half().double().sum()
    for split in (2, 8, 32):
        k = H.Knobs("row", 256, 8 if wave == 64 else 4, wave, 1, split)
        if H.check_knobs(kd, rd, "f16", k, wave=wave, arch=_arch()):
            continue
        p = H.plan(kd, rd, "f16", "f16", "add", k, wave=wave, arch=_arch())
        assert p.final is not None
        _launch(p, Ad, Dd, Dd, 1.0, 0.0, C.OP_IDENTITY, C.OP_IDENTITY)
        assert Dd.cpu().double().item() == ref.half().double().item(), split


def test_rreduce_forced_idx64():
    """The 64-bit-offset classes give the same results (plans with i64 offsets forced)."""
    from dataclasses import replace

    rng = random.Random(11)
    wave = _wave()
    shape = [24, 40, 16]
    for kept in ([1], [0, 2], [2]):
        A, Ad, sa = _strided(
            shape, "f32", rng, [2, 0, 1], 2, [float(v) for v in range(-3, 4)]
        )
        dshape = [shape[k] for k in kept]
        Dd = torch.zeros(dshape, device="cuda")
        kd, rd = _problem(shape, kept, sa, list(Dd.stride()))
        red = [i for i in range(3) if i not in kept]
        ref = _reduce_ref(A.double(), red, "add")
        for k in _knob_sets(kd, rd, "f32", wave, rng)[:3]:
            p = H.plan(kd, rd, "f32", "f32", "add", k, wave=wave, arch=_arch())
            spec = replace(p.spec, idx64=True)
            fin = replace(p.final, idx64=True) if p.final is not None else None
            p64 = replace(p, spec=spec, final=fin)
            Dd.fill_(float("nan"))
            _launch(p64, Ad, Dd, Dd, 1.0, 0.0, C.OP_IDENTITY, C.OP_IDENTITY)
            assert torch.equal(Dd.cpu().double(), ref), k


# The operators of the una classes and of opC that are not exact, with hipTensor's bounds in ulp
# of the compute type (f32 compute, f64 compute; the programmer's guide's table).
_OP_BOUND = {
    C.OP_SQRT: (0.5, 0.5),
    C.OP_RCP: (0.5, 0.5),
    C.OP_EXP: (1.0, 1.0),
    C.OP_LOG: (2.25, 1.0),
    C.OP_TANH: (2.0, 1.0),
    C.OP_SIGMOID: (3.0, 2.0),
}
_SPECIAL = (float("nan"), math.inf, -math.inf, 0.0, -0.0, -1.0)


def _op_ref(code, x):
    """The exact result of the operator (a 60-digit Decimal) for finite arguments and results;
    the C special values (a float) otherwise. Independent of the kernels' f32 / f64 programs.
    """
    import decimal

    ctx = decimal.Context(prec=60)
    if math.isnan(x):
        return math.nan
    if code == C.OP_SQRT:
        if x < 0:
            return math.nan
        if x == 0 or math.isinf(x):
            return x
        return ctx.sqrt(decimal.Decimal(x))
    if code == C.OP_RCP:
        if x == 0:
            return math.copysign(math.inf, x)
        if math.isinf(x):
            return math.copysign(0.0, x)
        return ctx.divide(1, decimal.Decimal(x))
    if code == C.OP_SIGMOID:
        if math.isinf(x):
            return 1.0 if x > 0 else 0.0
        return ctx.divide(1, ctx.add(1, ctx.exp(decimal.Decimal(-x))))
    if code == C.OP_TANH:
        if math.isinf(x):
            return math.copysign(1.0, x)
        if x == 0:
            return x
        e = ctx.exp(decimal.Decimal(2 * x))
        return ctx.divide(e - 1, e + 1)
    if code == C.OP_EXP:
        if math.isinf(x):
            return math.inf if x > 0 else 0.0
        return ctx.exp(decimal.Decimal(x))
    assert code == C.OP_LOG
    if x < 0:
        return math.nan
    if x == 0:
        return -math.inf
    if math.isinf(x):
        return math.inf
    return ctx.ln(decimal.Decimal(x))


def _op_arg(code, rng):
    """An argument in the operator's domain whose result keeps sums of 64 terms far from the
    f16 range and in the compute type's normal range."""
    x = rng.random()
    return {
        C.OP_SQRT: 100.0 * x,
        C.OP_RCP: rng.choice((1.0, -1.0)) * (0.25 + 7.75 * x),
        C.OP_SIGMOID: -30.0 + 60.0 * x,
        C.OP_TANH: -5.0 + 10.0 * x,
        C.OP_EXP: -4.0 + 8.0 * x,
        C.OP_LOG: 1e-3 + 100.0 * x,
    }[code]


def _combine(terms, op):
    """Sum or maximum of exact terms (Decimal) and special terms (float) with IEEE semantics:
    NaN if a term is NaN (MAX propagates NaN) or for inf - inf, an infinity if one is there.
    Returns (value, size): size = sum (or max) of the terms' magnitudes (finite ones).
    """
    import decimal

    vals = [t for t in terms if not isinstance(t, decimal.Decimal)]
    exact = [t for t in terms if isinstance(t, decimal.Decimal)]
    if any(math.isnan(t) for t in vals):
        return math.nan, 0
    infs = {t for t in vals if math.isinf(t)}
    fin = exact + [decimal.Decimal(t) for t in vals if not math.isinf(t)]
    mags = [abs(t) for t in fin]
    if op == "add":
        if len(infs) == 2:
            return math.nan, 0
        if infs:
            return infs.pop(), 0
        return sum(fin, decimal.Decimal(0)), sum(mags, decimal.Decimal(0))
    if math.inf in infs:
        return math.inf, 0
    if not fin:
        return -math.inf, 0
    return max(fin), max(mags)


@pytest.mark.parametrize("dt", ["f32", "f64", "f16", "bf16"])
@pytest.mark.parametrize("on", ["A", "C"])
@pytest.mark.parametrize("op", ["add", "max"])
@pytest.mark.parametrize("mode", ["row", "col"])
@pytest.mark.parametrize("code", sorted(_OP_BOUND))
def test_rreduce_unary_math_bounds(code, mode, op, on, dt):
    """Square root, reciprocal, sigmoid, tanh, exp and log as opA (the ``una`` classes) and as
    opC: within hipTensor's per-operator bounds (ulp of the compute type) plus the reduction's
    and the output rounding's error, against an exact (60-digit) reference; the special arguments
    (NaN, infinities, signed zeros, -1 outside the domains of sqrt and log, the poles of rcp and
    log), one per output, give the C results (NaN, infinities exact)."""
    import decimal

    rng = random.Random(code * 97 + len(mode) * 7 + len(op) + (on == "C") * 3 + len(dt))
    wave = _wave()
    # torch layouts: the last mode is innermost (row: reduced, col: kept)
    shape = [33, 48] if mode == "row" else [40, 64]
    kept = [0] if mode == "row" else [1]
    n_out, n_red = shape[kept[0]], shape[1 - kept[0]]
    f64 = dt == "f64"
    vals = torch.empty(shape, dtype=torch.float64)
    cvals = torch.empty([n_out], dtype=torch.float64)
    for o in range(n_out):
        where = rng.randrange(n_red)
        for r in range(n_red):
            v = rng.choice(range(-2, 3)) if on == "C" else _op_arg(code, rng)
            if on == "A" and o < len(_SPECIAL) and r == where:
                v = _SPECIAL[o]
            idx = (o, r) if mode == "row" else (r, o)
            vals[idx] = v
        if on == "C":
            cvals[o] = _SPECIAL[o] if o < len(_SPECIAL) else _op_arg(code, rng)
        else:
            cvals[o] = rng.choice(range(-2, 3))
    Ad = vals.to(TD[dt]).cuda()
    Cd = cvals.to(TD[dt]).cuda()
    A = Ad.cpu().double()  # the stored values
    Cm = Cd.cpu().double()
    kd, rd = _problem(shape, kept, list(Ad.stride()), list(Cd.stride()))
    assert H.mode_of(kd, rd) == mode
    beta = 1.0 if on == "C" else 0.0
    opa = code if on == "A" else C.OP_IDENTITY
    opc = code if on == "C" else C.OP_IDENTITY
    uc = 2.0**-53 if f64 else 2.0**-24
    ud = {"f16": 2.0**-11, "bf16": 2.0**-8, "f32": 2.0**-24, "f64": 2.0**-53}[dt]
    floor_d = {"f16": 2.0**-24, "bf16": 2.0**-133, "f32": 2.0**-149, "f64": 5e-324}[dt]
    bound = _OP_BOUND[code][1 if f64 else 0]
    want, tol = [], []
    for o in range(n_out):
        xs = [float(A[o, r] if mode == "row" else A[r, o]) for r in range(n_red)]
        terms = [_op_ref(code, x) if on == "A" else decimal.Decimal(x) for x in xs]
        r_val, size = _combine(terms, op)
        ct = _op_ref(code, float(Cm[o])) if on == "C" else decimal.Decimal(0)
        if isinstance(r_val, float) or isinstance(ct, float):
            w = float(r_val) + beta * float(
                ct
            )  # specials: IEEE in double is exact here
            want.append(w)
            tol.append(0.0)
            continue
        w = r_val + decimal.Decimal(beta) * ct
        op_err = 2 * bound * uc * float(abs(ct) if on == "C" else size)
        sum_err = (
            (2 if op == "max" else n_red + 2)
            * uc
            * (float(size) + beta * float(abs(ct)))
        )
        want.append(float(w))
        tol.append(op_err + sum_err + ud * abs(float(w)) + floor_d)
    want_t = torch.tensor(want, dtype=torch.float64)
    tol_t = torch.tensor(tol, dtype=torch.float64)
    nonfinite = ~torch.isfinite(want_t)
    assert int(nonfinite.sum()) >= 1  # the NaN argument at least
    for k in _knob_sets(kd, rd, dt, wave, rng)[:2]:
        p = H.plan(kd, rd, dt, dt, op, k, una=on == "A", wave=wave, arch=_arch())
        Dd = torch.full([n_out], float("nan"), dtype=TD[dt], device="cuda")
        _launch(p, Ad, Cd, Dd, 1.0, beta, opa, opc)
        got = Dd.cpu().double()
        assert torch.equal(got.isnan(), want_t.isnan()), (k, got, want_t)
        inf = torch.isinf(want_t)
        assert torch.equal(got[inf], want_t[inf]), (k, got[inf], want_t[inf])
        fin = ~nonfinite
        err = (got[fin] - want_t[fin]).abs()
        assert bool((err <= tol_t[fin]).all()), (
            p.spec.kernel_name(),
            k,
            float((err / tol_t[fin]).max()),
        )
