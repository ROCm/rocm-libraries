# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU numerics of the shape-generic hipTensor kernels (``kernels.common.hiptensor_aot``).

Every case is checked against an independent float64 reference computed with torch on the CPU
(einsum-free loops over permuted views): integer-valued inputs make sums, products, maxima and
minima exact in f32, so those results are compared bit for bit; transcendental operators are
compared with a bound of a few ulps of the output type. Launch arguments come from
:mod:`kernels.common.hiptensor_aot.host`, the Python statement of hipTensor's C++ planner.
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
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew  # noqa: E402
from kernels.common.hiptensor_aot.host import (
    canonicalize,
    ew_args,
    reduce_args,
)  # noqa: E402
from kernels.common.hiptensor_aot.reduce import ReduceSpec, build_reduce  # noqa: E402

TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}
_CACHE = {}


def _arch() -> str:
    name = torch.cuda.get_device_properties(0).gcnArchName
    return name.split(":")[0]


def _wave() -> int:
    return 32 if _arch().startswith("gfx1") else 64


def _launcher(kernel):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    if kernel.name not in _CACHE:
        art = compile_kernel(
            kernel, arch=_arch(), backend="python", capture_ir_text=False
        )
        sig = []
        for p in kernel.params:
            t = p.type
            nm = getattr(t, "name", "") or ""
            if nm.startswith("ptr<"):
                sig.append({"name": p.name, "type": nm})
            else:
                sig.append({"name": p.name, "type": nm})
        _CACHE[kernel.name] = (
            KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig),
            kernel,
        )
    return _CACHE[kernel.name]


def _run(kernel, vals, grid, block):
    from rocke.runtime.launcher import LaunchConfig

    L, k = _launcher(kernel)
    vals = {p.name: vals[p.name] for p in k.params}
    L(vals, config=LaunchConfig(grid=grid, block=block))
    torch.cuda.synchronize()


def _strided(shape, dtype, rng, perm=None, pad=0, lo=-4, hi=5):
    """A tensor with ``shape`` stored in the mode order ``perm`` (innermost last) with ``pad``
    elements of row padding, integer valued. Returns (view, storage, strides in elements).
    """
    n = len(shape)
    perm = list(range(n)) if perm is None else perm
    stor_shape = [shape[p] for p in perm]
    stor_shape[-1] += pad
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    base = torch.randint(lo, hi, stor_shape, generator=g).to(TD[dtype])
    view = base[..., : shape[perm[-1]]] if pad else base
    view = view.permute([perm.index(i) for i in range(n)])
    dev = base.cuda()
    dview = dev[..., : shape[perm[-1]]] if pad else dev
    dview = dview.permute([perm.index(i) for i in range(n)])
    return view, dview, list(dview.stride())


def _unary_ref(x, op):
    f = {
        C.OP_IDENTITY: lambda t: t,
        C.OP_SQRT: torch.sqrt,
        C.OP_RELU: torch.relu,
        C.OP_RCP: torch.reciprocal,
        C.OP_SIGMOID: torch.sigmoid,
        C.OP_TANH: torch.tanh,
        C.OP_EXP: torch.exp,
        C.OP_LOG: torch.log,
        C.OP_ABS: torch.abs,
        C.OP_NEG: torch.neg,
    }[op]
    return f(x)


def _binary_ref(a, c, op):
    return {
        C.OP_ADD: lambda: a + c,
        C.OP_MUL: lambda: a * c,
        C.OP_MAX: lambda: torch.maximum(a, c),
        C.OP_MIN: lambda: torch.minimum(a, c),
    }[op]()


def _ulp_bound(ref, dtype):
    eps = {"f16": 2.0**-10, "bf16": 2.0**-7, "f32": 2.0**-23, "f64": 2.0**-52}[dtype]
    return 4 * eps * ref.abs() + 4 * eps


EW_CASES = []
_rng = random.Random(1234)
for arity, (a, d) in itertools.product(
    (1, 2, 3),
    (("f16", "f16"), ("f32", "f32"), ("bf16", "bf16"), ("f64", "f64"), ("f16", "f32")),
):
    for rank in (1, 3, 5, 7):
        EW_CASES.append((arity, a, d, rank, _rng.randrange(1 << 20)))


@pytest.mark.parametrize("arity,a,d,rank,seed", EW_CASES)
def test_ew_strided_exact(arity, a, d, rank, seed):
    rng = random.Random(seed)
    shape = [rng.choice([1, 2, 3, 5, 8, 17]) for _ in range(rank)]
    if math.prod(shape) < 64:
        shape[0] = 64
    names = {1: ["A"], 2: ["A", "C"], 3: ["A", "B", "C"]}[arity]
    ins = {}
    for n in names:
        perm = list(range(rank))
        rng.shuffle(perm)
        ins[n] = _strided(shape, a, rng, perm, pad=rng.choice([0, 0, 3]))
    operm = list(range(rank))
    rng.shuffle(operm)
    dref = torch.zeros([shape[p] for p in operm], dtype=TD[d]).permute(
        [operm.index(i) for i in range(rank)]
    )
    ddev = torch.zeros([shape[p] for p in operm], dtype=TD[d], device="cuda").permute(
        [operm.index(i) for i in range(rank)]
    )
    dstr = list(ddev.stride())
    # exact ops only (integer data): identity / neg / abs / relu, add / mul / max / min
    ops = {n: rng.choice([C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]) for n in names}
    bops = [rng.choice(C.BINARY_OPS), rng.choice(C.BINARY_OPS)]
    scal = {n: float(rng.choice([1, 2, -1, 0.5])) for n in names}
    if arity >= 2 and rng.random() < 0.2:
        scal[names[-1]] = 0.0  # a zero scale must not read the tensor
    dims = [(shape[i], [ins[n][2][i] for n in names] + [dstr[i]]) for i in range(rank)]
    cdims = canonicalize(dims, len(names))
    vals, grid, idx64 = ew_args(cdims, names + ["D"])
    spec = EwSpec(
        arity=arity,
        a=a,
        b=a if arity == 3 else None,
        c=a if arity >= 2 else None,
        d=d,
        idx64=idx64,
    )
    k = build_ew(spec)
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    for n in names:
        vals[n] = ins[n][1]
        vals["op" + n] = ops[n]
        vals[sn[n]] = scal[n]
    vals["D"] = ddev
    vals["opAB"] = bops[0]
    vals["opABC"] = bops[1]
    _run(k, vals, grid, (256, 1, 1))

    def term(n):
        if scal[n] == 0.0:
            return torch.zeros(shape, dtype=torch.float64)
        return scal[n] * _unary_ref(ins[n][0].double(), ops[n])

    if arity == 1:
        ref = term("A")
    elif arity == 2:
        ref = _binary_ref(term("A"), term("C"), bops[0])
    else:
        ref = _binary_ref(
            _binary_ref(term("A"), term("B"), bops[0]), term("C"), bops[1]
        )
    got = ddev.cpu().double()
    want = ref.to(TD[d]).double()
    assert torch.equal(got, want), (got - want).abs().max()
    del dref


@pytest.mark.parametrize(
    "op", [C.OP_SQRT, C.OP_RCP, C.OP_SIGMOID, C.OP_TANH, C.OP_EXP, C.OP_LOG]
)
@pytest.mark.parametrize("dt", ["f32", "f64", "f16"])
def test_ew_transcendental(op, dt):
    rng = random.Random(op * 7 + len(dt))
    shape = [33, 7, 5]
    x = (torch.rand(shape, dtype=torch.float64) * 3 + 0.25).to(TD[dt])
    xd = x.cuda().permute(2, 0, 1).contiguous().permute(1, 2, 0)
    dd = torch.zeros(shape, dtype=TD[dt], device="cuda")
    dims = [(shape[i], [xd.stride()[i], dd.stride()[i]]) for i in range(3)]
    vals, grid, idx64 = ew_args(canonicalize(dims, 1), ["A", "D"])
    k = build_ew(EwSpec(arity=1, a=dt, d=dt, idx64=idx64))
    vals.update(A=xd, D=dd, opA=op, alpha=1.0)
    _run(k, vals, grid, (256, 1, 1))
    ref = _unary_ref(x.double(), op)
    got = dd.cpu().double()
    tol = _ulp_bound(ref, dt) if dt != "f64" else 1e-14 * ref.abs() + 1e-300
    if dt == "f16":
        tol = tol * 2
    assert ((got - ref).abs() <= tol).all(), (got - ref).abs().max()
    _ = rng


def test_ew_complex_pack_conj():
    shape = [6, 10, 3]
    z = torch.complex(
        torch.randint(-3, 4, shape).float(), torch.randint(-3, 4, shape).float()
    )
    zd = z.cuda().permute(1, 2, 0).contiguous().permute(2, 0, 1)
    out = torch.zeros(shape, dtype=torch.complex128, device="cuda")
    sa = list(zd.stride())
    so = list(out.stride())
    vals, grid, idx64 = ew_args(
        canonicalize([(shape[i], [sa[i], so[i]]) for i in range(3)], 1), ["A", "D"]
    )
    k = build_ew(EwSpec(arity=1, a="f32", d="f64", cplx=True, idx64=idx64))
    vals.update(
        A=torch.view_as_real(zd),
        D=torch.view_as_real(out),
        opA=C.OP_CONJ,
        alpha_re=2.0,
        alpha_im=-1.0,
    )
    vals["A"] = zd
    vals["D"] = out
    _run(k, vals, grid, (256, 1, 1))
    ref = (2 - 1j) * torch.conj(z.to(torch.complex128))
    assert torch.equal(out.cpu(), ref)


RED_CASES = []
_rng2 = random.Random(99)
for (a, d), opr in itertools.product(
    (("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("f64", "f64")), C.BINARY_OPS
):
    for _ in range(3):
        RED_CASES.append((a, d, opr, _rng2.randrange(1 << 20)))


@pytest.mark.parametrize("a,d,opr,seed", RED_CASES)
def test_reduce_exact(a, d, opr, seed):
    rng = random.Random(seed)
    rank = rng.choice([2, 3, 4, 5])
    shape = [rng.choice([2, 3, 4, 7, 16, 33]) for _ in range(rank)]
    perm = list(range(rank))
    rng.shuffle(perm)
    lo, hi = (-2, 3) if opr == C.OP_MUL else (-4, 5)
    if opr == C.OP_MUL:
        shape = [min(s, 4) for s in shape]
    A, Ad, sa = _strided(shape, a, rng, perm, lo=lo, hi=hi)
    nk = rng.randrange(0, rank)
    kept = sorted(rng.sample(range(rank), nk))
    red = [i for i in range(rank) if i not in kept]
    dshape = [shape[i] for i in kept]
    Cm, Cd, sc = _strided(dshape or [1], d, rng) if dshape else _strided([1], d, rng)
    Dd = torch.zeros(dshape or [1], dtype=TD[d], device="cuda")
    sd = list(Dd.stride())
    if dshape:
        kdims = [(shape[k], [sa[k], sc[j], sd[j]]) for j, k in enumerate(kept)]
    else:
        kdims = []
    rdims = [(shape[r], [sa[r]]) for r in red]
    kdims = canonicalize(kdims, 0)
    rdims = canonicalize(rdims, 0)
    if (
        _wave() != 64
        and rdims
        and (not kdims or abs(rdims[0][1][0]) < abs(kdims[0][1][0]))
    ):
        pytest.skip("the row kernel needs wave64")
    alpha, beta = float(rng.choice([1, 2, 0.5])), float(rng.choice([0, 1, -1]))
    for split in (None, 1):
        mode, vals, grid, fgrid, split_ = reduce_args(kdims, rdims, split=split)
        idx64 = False
        k = build_reduce(ReduceSpec(mode=mode, a=a, d=d, idx64=idx64))
        cdt = torch.float64 if "f64" in (a, d) else torch.float32
        n_out = vals["n_out"]
        W = torch.zeros(max(1, split_ * n_out), dtype=cdt, device="cuda")
        Dd.zero_()
        vals.update(
            A=Ad,
            C=Cd,
            D=Dd,
            W=W,
            alpha=alpha,
            beta=beta,
            opA=C.OP_IDENTITY,
            opC=C.OP_IDENTITY,
            opR=opr,
        )
        _run(k, vals, grid, (256, 1, 1))
        if split_ > 1:
            fk = build_reduce(ReduceSpec(mode="final", a=a, d=d, idx64=idx64))
            _run(fk, vals, fgrid, (256, 1, 1))
        x = A.double()
        if red:
            if opr == C.OP_ADD:
                r = x.sum(dim=red)
            elif opr == C.OP_MUL:
                r = x
                for dd in sorted(red, reverse=True):
                    r = r.prod(dim=dd)
            elif opr == C.OP_MAX:
                r = x.amax(dim=red)
            else:
                r = x.amin(dim=red)
        else:
            r = x
        ref = alpha * r.reshape(dshape or [1]) + beta * Cm.double().reshape(
            dshape or [1]
        )
        want = ref.to(TD[d]).double()
        got = Dd.cpu().double()
        assert torch.equal(got, want), (mode, split_, (got - want).abs().max())


@pytest.mark.parametrize("dt", ["f32", "f64", "f16"])
def test_ew_zero_scale_skips_input_and_operator(dt):
    """A tensor whose scale is zero is neither read nor transformed: NaN data and an operator
    that maps the placeholder zero to -inf (LOG) do not reach D."""
    shape = [37, 5]
    a = torch.full(shape, float("nan"), dtype=TD[dt], device="cuda")
    c = (torch.arange(37 * 5, dtype=torch.float64).reshape(shape) % 7 - 3).to(TD[dt]).cuda()
    d = torch.zeros(shape, dtype=TD[dt], device="cuda")
    dims = [(shape[i], [a.stride()[i], c.stride()[i], d.stride()[i]]) for i in range(2)]
    vals, grid, idx64 = ew_args(canonicalize(dims, 2), ["A", "C", "D"])
    k = build_ew(EwSpec(arity=2, a=dt, c=dt, d=dt, idx64=idx64))
    vals.update(A=a, C=c, D=d, opA=C.OP_LOG, opC=C.OP_IDENTITY, opAB=C.OP_ADD, alpha=0.0, gamma=-2.0)
    _run(k, vals, grid, (256, 1, 1))
    assert torch.equal(d.cpu().double(), (-2.0 * c.cpu().double()).to(TD[dt]).double())
