# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The problem-specialized elementwise kernel (``kernels.common.hiptensor_aot.jit_ew``).

CPU: every kernel class rolls into one parametric recipe that the recipe expander reproduces
exactly at fresh random geometry (``roll_jit_ew`` verifies against new recordings), the class set
is the documented one, and the vector-class rule (:func:`jit_class_of`) follows the layouts.

GPU: the specialized kernels against independent float64 torch references, integer-valued data
(exact): vector classes with vector and gathered inputs, every arity, mixed types, zero scales,
64-bit offset rows, and padding limits of the permute form.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.ew import EwSpec
from kernels.common.hiptensor_aot.host import canonicalize, ew_args
from kernels.common.hiptensor_aot.jit_ew import (
    EwJitSpec,
    build_ew_jit,
    jit_class_of,
    jit_classes,
    roll_jit_ew,
)

from ._hiptensor_aot_rounding import round_rne, rounding_probes

ROWS = [
    EwSpec(arity=1, a="f16", d="f16"),
    EwSpec(arity=1, a="f32", d="f16", idx64=True),
    EwSpec(arity=2, a="f16", c="f16", d="f32"),
    EwSpec(arity=2, a="f64", c="f32", d="f32"),
    EwSpec(arity=3, a="bf16", b="bf16", c="bf16", d="bf16"),
    EwSpec(arity=1, a="f32", d="f64", cplx=True),
]


@pytest.mark.parametrize("spec", jit_classes(ROWS)[::3], ids=lambda s: s.kernel_name())
def test_class_rolls(spec):
    recipe = roll_jit_ew(spec, verify=2)
    assert {s["name"] for s in recipe["spec"]} == {n for n, _ in spec.axes()}
    assert recipe["kernel_name_fmt"] == spec.kernel_name()


def test_recipe_independent_of_build_history():
    """A class's recipe depends on the class only: kernels built before it in the same process
    (a roller may roll the classes in a process pool, in any order) change nothing, so a file
    of rolled recipes is reproducible across builds."""
    from kernels.common.hiptensor_aot.ew import build_ew
    from rocke.portable_ir.src.recipe_bundle import cbor_encode

    classes = jit_classes(ROWS)
    spec = classes[0]
    first = cbor_encode(roll_jit_ew(spec, verify=1))
    for other in classes[1:4]:
        roll_jit_ew(other, verify=1)
    build_ew(ROWS[2])
    assert cbor_encode(roll_jit_ew(spec, verify=1)) == first


def test_class_set():
    classes = jit_classes(ROWS)
    keys = [c.key() for c in classes]
    assert len(keys) == len(set(keys))
    # vec 1 for every row; vector classes: every input mask (64-bit rows: all inputs vectors)
    assert len(classes) == 6 + 2 + 1 + 4 + 4 + 8
    for c in classes:
        if c.vec > 1:
            assert c.vec == 16 // C.DTYPES[c.base.d][1]
            assert not c.base.idx64 or all(c.unit)


def test_class_rule():
    base = EwSpec(arity=2, a="f16", c="f16", d="f16")
    dims = [(64, [1, 512, 1]), (9, [64, 1, 64])]
    vals, _, _ = ew_args(canonicalize(dims, 2), ["A", "C", "D"])
    ops = dict(opA=C.OP_IDENTITY, opC=C.OP_IDENTITY, opAB=C.OP_ADD)
    vals.update(ops)
    spec, geom, n = jit_class_of(base, vals)
    assert spec.vec == 8 and spec.unit == (True, False)
    assert geom["e0"] == 8 and n == vals["n_in"] // 8
    # an output whose fastest extent is not a multiple of the vector width: vec 1
    vals2, _, _ = ew_args(
        canonicalize([(12, [1, 1, 1]), (5, [12, 12, 12])], 2), ["A", "C", "D"]
    )
    vals2.update(ops)
    assert jit_class_of(base, vals2)[0].vec == 1


def test_reference_rounding_matches_numpy():
    """The exact rational reference of one f64 -> f16 rounding agrees with numpy's direct
    conversion on every probe (it is then trusted for bf16, which numpy lacks)."""
    np = pytest.importorskip("numpy")
    for v in rounding_probes("f16"):
        with np.errstate(over="ignore"):
            want = float(np.float16(v))
        got = round_rne(v, "f16")
        assert (math.isnan(want) and math.isnan(got)) or (
            got == want and math.copysign(1.0, got) == math.copysign(1.0, want)
        ), v


@pytest.mark.parametrize("d", ["f16", "bf16"])
def test_f64_to_16bit_rows_roll(d):
    """The single-rounding f64 -> f16 / bf16 store (round to odd through f32) rolls into a
    recipe that the expander reproduces, for the scalar and the vector class."""
    base = EwSpec(arity=2, a="f64", c=d, d=d)
    for spec in jit_classes([base])[:2]:
        recipe = roll_jit_ew(spec, verify=2)
        assert recipe["kernel_name_fmt"] == spec.kernel_name()


torch = pytest.importorskip("torch")
TD = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
}


def _gpu():
    if not torch.cuda.is_available():  # pragma: no cover - CPU-only runners
        pytest.skip("needs a GPU")
    name = torch.cuda.get_device_properties(0).gcnArchName
    return name.split(":")[0]


def _run(kernel, vals, grid, block=256):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    art = compile_kernel(kernel, arch=_gpu(), backend="python", capture_ir_text=False)
    sig = [{"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params]
    launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig)
    launcher(
        {p.name: vals[p.name] for p in kernel.params},
        config=LaunchConfig(grid=grid, block=(block, 1, 1)),
    )
    torch.cuda.synchronize()


def _tensor(shape, dt, rng, perm):
    n = len(shape)
    stor = [shape[p] for p in perm]
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    base = torch.randint(-6, 7, stor, generator=g).to(TD[dt])
    view = base.permute([perm.index(i) for i in range(n)])
    dev = base.cuda().permute([perm.index(i) for i in range(n)])
    return view, dev


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
        C.OP_NEG: torch.neg,
        C.OP_ABS: torch.abs,
        C.OP_RELU: torch.relu,
    }[op](x)


CASES = []
_r = random.Random(77)
for arity, (a, d), gather, idx64 in itertools.product(
    (1, 2, 3),
    (("f16", "f16"), ("f32", "f32"), ("bf16", "f32"), ("f64", "f64"), ("f32", "f16")),
    (False, True),
    (False, True),
):
    CASES.append((arity, a, d, gather, idx64, _r.randrange(1 << 20)))


@pytest.mark.parametrize("arity,a,d,gather,idx64,seed", CASES)
def test_jit_exact(arity, a, d, gather, idx64, seed):
    _gpu()
    rng = random.Random(seed)
    shape = [32, rng.choice([3, 5, 6]), rng.choice([2, 7])]
    names = {1: ["A"], 2: ["A", "C"], 3: ["A", "B", "C"]}[arity]
    ins = {}
    for i, n in enumerate(names):
        # storage order (outermost first): mode 0 innermost like D, or (the first input of a
        # gather case) mode 1 or 2 innermost
        perm = rng.choice([[2, 1, 0], [1, 2, 0]])
        if gather and i == 0:
            perm = rng.choice([[0, 2, 1], [2, 0, 1], [0, 1, 2]])
        ins[n] = _tensor(shape, a, rng, perm)
    dd = torch.zeros(shape[::-1], dtype=TD[d], device="cuda").permute(2, 1, 0)
    dims = [
        (shape[i], [ins[n][1].stride()[i] for n in names] + [dd.stride()[i]])
        for i in range(3)
    ]
    vals, _, _ = ew_args(canonicalize(dims, len(names)), names + ["D"])
    base = EwSpec(
        arity=arity,
        a=a,
        b=a if arity == 3 else None,
        c=a if arity >= 2 else None,
        d=d,
        idx64=idx64,
    )
    ops = {n: rng.choice([C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]) for n in names}
    bops = [rng.choice(C.BINARY_OPS), rng.choice(C.BINARY_OPS)]
    scal = {n: float(rng.choice([1, 2, -1, 0.5])) for n in names}
    if arity >= 2 and rng.random() < 0.3:
        scal[names[-1]] = 0.0
    for n in names:
        vals["op" + n] = ops[n]
    vals["opAB"], vals["opABC"] = bops
    spec, geom, n_in = jit_class_of(base, vals)
    if not gather:
        assert spec.vec > 1 and all(
            spec.unit
        ), "packed layouts have an all-vector class"
    k = build_ew_jit(spec, geom)
    run = {"D": dd}
    sn = {"A": "alpha", "B": "beta", "C": "gamma"}
    for n in names:
        run[n] = ins[n][1]
        run[sn[n]] = scal[n]
    _run(k, run, (-(-n_in // 256), geom["ystep"], 1))

    def term(n):
        if scal[n] == 0.0:
            return torch.zeros(shape, dtype=torch.float64)
        return scal[n] * _unary(ins[n][0].double(), ops[n])

    if arity == 1:
        ref = term("A")
    elif arity == 2:
        ref = _binary(term("A"), term("C"), bops[0])
    else:
        ref = _binary(_binary(term("A"), term("B"), bops[0]), term("C"), bops[1])
    got = dd.cpu().double()
    want = ref.to(TD[d]).double()
    assert torch.equal(got, want), (spec.kernel_name(), (got - want).abs().max())


def test_jit_permute_padding_limits():
    """The permute form writes zeros past ``lim`` (zero padding of a packed operand)."""
    _gpu()
    rng = random.Random(5)
    shape = [32, 6]
    x, xd = _tensor(shape, "f32", rng, [1, 0])  # mode 0 innermost: strides (1, 32)
    out = (
        torch.full([48, 6], 99.0, dtype=torch.float32, device="cuda")
        .t()
        .contiguous()
        .t()
    )
    dims = [(48, [1, out.stride()[0]]), (6, [32, out.stride()[1]])]
    vals, _, _ = ew_args(dims, ["A", "D"])
    vals["lim0"] = 32
    vals["opA"] = C.OP_IDENTITY
    spec, geom, n_in = jit_class_of(EwSpec(arity=1, a="f32", d="f32"), vals)
    assert spec.vec == 4
    k = build_ew_jit(spec, geom)
    _run(k, {"A": xd, "D": out, "alpha": 3.0}, (-(-n_in // 256), geom["ystep"], 1))
    want = torch.zeros([48, 6], dtype=torch.float64)
    want[:32] = 3.0 * x.double()
    assert torch.equal(out.cpu().double(), want)
    assert math.isfinite(float(out.abs().max()))


@pytest.mark.parametrize("d", ["f16", "bf16"])
def test_jit_f64_to_16bit_rounds_once(d):
    """The vector class of the F64 route's epilogue row (f64, d, d) rounds every value once to
    nearest even (values next to midpoints, subnormals, overflow, zeros, infinities, NaN).
    """
    _gpu()
    vals = rounding_probes(d)
    vals += [0.0] * (-len(vals) % 64)
    n = len(vals)
    x = torch.tensor(vals, dtype=torch.float64, device="cuda")
    c = torch.full(
        [n], float("nan"), dtype=TD[d], device="cuda"
    )  # zero scale: not read
    out = torch.zeros(n, dtype=TD[d], device="cuda")
    args, _, _ = ew_args(canonicalize([(n, [1, 1, 1])], 2), ["A", "C", "D"])
    args.update(opA=C.OP_IDENTITY, opC=C.OP_IDENTITY, opAB=C.OP_ADD)
    spec, geom, n_in = jit_class_of(EwSpec(arity=2, a="f64", c=d, d=d), args)
    assert spec.vec > 1 and all(spec.unit)
    k = build_ew_jit(spec, geom)
    _run(
        k,
        {"A": x, "C": c, "D": out, "alpha": 1.0, "gamma": 0.0},
        (-(-n_in // 256), geom["ystep"], 1),
    )
    got = out.cpu().double().tolist()
    bad = []
    for v, g in zip(vals, got):
        want = round_rne(v, d)
        # the unread C adds a zero term: a zero result is +0 (IEEE -0 + +0)
        if not (
            (math.isnan(want) and math.isnan(g))
            or (
                g == want
                and (want == 0.0 or math.copysign(1.0, g) == math.copysign(1.0, want))
            )
        ):
            bad.append((v, g, want))
    assert not bad, bad[:8]
