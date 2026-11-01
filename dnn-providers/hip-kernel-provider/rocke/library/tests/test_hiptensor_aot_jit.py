# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The problem-specialized elementwise kernel (``kernels.common.hiptensor_aot.jit_ew``).

CPU: every kernel class rolls into one parametric recipe that the recipe expander reproduces
exactly at fresh random geometry (``roll_jit_ew`` verifies against new recordings), the class set
is the documented one, and the vector-class rule (:func:`jit_class_of`) follows the layouts.

GPU: the specialized kernels against independent float64 torch references, integer-valued data
(exact): vector classes with vector and gathered inputs, every arity, mixed types, zero scales,
64-bit offset rows, and padding limits of the permute form. Every class of the mixed-type and
math rows (the classes hipTensor's JIT gained) against the pack's generic kernel, bit for bit.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew
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


def _flag_specs():
    """``(make(value), name)`` for every boolean field of the hiptensor_aot specs (and the int
    ``rag`` of the tuned tiled classes)."""
    from kernels.common.hiptensor_aot.dgett import DgettSpec
    from kernels.common.hiptensor_aot.ewf import EwfSpec
    from kernels.common.hiptensor_aot.ewp import EwpSpec
    from kernels.common.hiptensor_aot.ewt import EwtSpec
    from kernels.common.hiptensor_aot.rgett import RgettSpec
    from kernels.common.hiptensor_aot.rreduce import RreduceFinalSpec, RreduceSpec

    def raising(cls, **base):
        return lambda **kw: cls(**base, **kw).check()

    def reason(cls, **base):
        def make(**kw):
            why = cls(**base, **kw).check()
            if why is not None:
                raise ValueError(why)

        return make

    ew = raising(EwSpec, arity=1, a="f32", d="f32")
    out = [(ew, f) for f in ("cplx", "idx64", "math")]
    dg = DgettSpec(comp="f32")

    def dgett(**kw):
        import dataclasses

        dataclasses.replace(dg, **kw).check("gfx950")

    out += [(dgett, f) for f in ("cplx", "math", "idx64")]
    out.append((raising(EwfSpec, arity=1, a="f32", d="f32"), "nt"))
    out.append((raising(EwtSpec, arity=2, a="f32", c="f32", d="f32"), "fold"))
    out += [(raising(RgettSpec), f) for f in ("bilinear", "epi_pair", "wide_d")]
    out.append(
        (raising(EwpSpec, arity=2, a="f16", c="f16", d="f16", kinds="TV", vb=4), "rag")
    )
    rr = reason(RreduceSpec, mode="row", a="f32", d="f32", op="add")
    out += [(rr, f) for f in ("nt", "una", "idx64")]
    out.append((reason(RreduceFinalSpec, a="f32", d="f32", op="add"), "idx64"))
    out.append((lambda unit: EwJitSpec(ROWS[2], 4, unit).check(), "unit"))
    return out


@pytest.mark.parametrize("bad", ["false", "", None, 2.0])
def test_spec_flags_reject_non_bools(bad):
    """A boolean field (and the tuned tiled classes' ``rag``) that is not a bool is refused, not
    coerced: ``"false"`` is truthy, and a kernel built from it would not be the one asked for.
    """
    for make, field in _flag_specs():
        if field == "unit":
            with pytest.raises(ValueError):
                make((True, bad))
            make((True, False))
            continue
        with pytest.raises(ValueError):
            make(**{field: bad})
        if field == "rag":
            with pytest.raises(ValueError):
                make(rag=True)  # a bool is not the int the class's name and ABI encode
            make(rag=1)
        else:
            make(**{field: False})


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
    k = build_ew_jit(spec.with_geom(geom))
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
    k = build_ew_jit(spec.with_geom(geom))
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
    k = build_ew_jit(spec.with_geom(geom))
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


# ---------------------------------------------------------------- the widened JIT classes


def _pack_rows():
    """Every row of the pack's generic elementwise kernel (make_rocke_pack.ew_params)."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "builders"
        / "hiptensor"
        / "make_rocke_pack.py"
    )
    spec = importlib.util.spec_from_file_location("_f3_make_rocke_pack", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [p for idx64 in (False, True) for p in mod.ew_params(idx64)]
    plain = [p for idx64 in (False, True) for p in mod.ew_params(idx64, mixed=False)]
    return rows, plain


def _widened_classes():
    """The classes of the rows hipTensor's JIT gained: the mixed-type rows of contractions and
    the math rows (vec 1, every input a vector; 32- and 64-bit offsets)."""
    rows, plain = _pack_rows()
    out = []
    for p in rows:
        if p.get("math") or p not in plain:
            out.extend(jit_classes([EwSpec(**p)]))
    return out


WIDENED = _widened_classes()
_BITS = {
    "f16": torch.int16,
    "bf16": torch.int16,
    "f32": torch.int32,
    "f64": torch.int64,
}
_SPECIALS = [math.nan, math.inf, -math.inf, -0.0, 0.0, 7.0e4, -3.0e38, 1.0e-30, 0.75]
_LAUNCHERS = {}


def _launch(kernel, key, vals, grid):
    """Compile ``kernel`` once per ``key`` and launch it (block 256)."""
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    if key not in _LAUNCHERS:
        art = compile_kernel(
            kernel, arch=_gpu(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _LAUNCHERS[key] = (
            KernelLauncher(hsaco=art.hsaco, kernel_name=kernel.name, signature=sig),
            [p.name for p in kernel.params],
        )
    launcher, names = _LAUNCHERS[key]
    launcher(
        {n: vals[n] for n in names}, config=LaunchConfig(grid=grid, block=(256, 1, 1))
    )
    torch.cuda.synchronize()


def _bits(t, dt, cplx):
    """``t`` (on the host) as integers of its storage, bit for bit, and its NaN mask (per
    component)."""
    t = t.cpu().contiguous()
    if cplx:
        t = torch.view_as_real(t).contiguous()
    return t.view(_BITS[dt]), torch.isnan(t)


@pytest.mark.parametrize("spec", WIDENED, ids=lambda s: s.kernel_name())
def test_widened_class_matches_pack_kernel(spec):
    """Every class of the rows the JIT gained (mixed-type rows, math rows with vec 1, math rows
    with 64-bit offsets): a launch of the pack's generic kernel classifies to the class
    (jit_class_of, hipTensor's rule), and the specialized kernel writes the same bits as the
    generic kernel (1) on integer data with the basic operators, where both also equal an
    independent float64 reference, and (2) on data with NaN, infinities, signed zeros,
    overflowing and tiny values, with every operator of the row (math operators on the math
    rows) and a zero scale (the input and its operator are skipped). A NaN equals a NaN of any
    sign and payload."""
    _gpu()
    base = spec.base
    names = base.tensors()
    ins = names[:-1]
    rng = random.Random(sum(map(ord, spec.kernel_name())))
    vec = spec.vec
    # D column-major; vector classes: the fastest extent a multiple of the vector width and
    # the inputs not in `unit` gathered (their mode 0 is their slowest); vec 1: odd extents
    # (no vector form, also when the modes merge), often with a gathered first input
    if vec > 1:
        shape = [vec * rng.choice([2, 3, 5]), rng.choice([3, 5]), rng.choice([2, 3])]
    else:
        shape = [rng.choice([13, 21]), rng.choice([3, 5]), rng.choice([3, 5])]
    gathered = {
        n: (not spec.unit[i]) if vec > 1 else (i == 0 and rng.random() < 0.7)
        for i, n in enumerate(ins)
    }
    dts = {"A": base.a, "B": base.b, "C": base.c, "D": base.d}
    ctype = {"f32": torch.complex64, "f64": torch.complex128}

    def alloc(n, values):
        """Device tensor of `n` holding `values` (float64 / complex128, logical shape)."""
        dt = dts[n]
        td = ctype[dt] if base.cplx else TD[dt]
        perm = [1, 2, 0] if gathered.get(n) else [0, 1, 2]
        # storage: the modes in `perm` order, fastest first
        stor = values.permute(perm[::-1]).contiguous().to(td)
        return stor.cuda().permute([perm[::-1].index(i) for i in range(3)])

    def make_values(special):
        g = torch.Generator().manual_seed(rng.randrange(1 << 30))
        v = torch.randint(-6, 7, shape, generator=g).double()
        if special:
            flat = v.view(-1)
            idx = torch.randperm(flat.numel(), generator=g)[: flat.numel() // 3]
            for k, j in enumerate(idx.tolist()):
                flat[j] = _SPECIALS[k % len(_SPECIALS)]
        if base.cplx:
            w = torch.randint(-6, 7, shape, generator=g).double()
            return torch.complex(v, w)
        return v

    if base.cplx:
        ops_ref = [C.OP_IDENTITY, C.OP_CONJ]
        ops_all = ops_ref
        bops = [C.OP_ADD]
    else:
        ops_ref = [C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_RELU]
        ops_all = list(C.BASIC_UNARY_OPS) + (list(C.MATH_OPS) * 2 if base.math else [])
        bops = list(C.BINARY_OPS)

    for case in ("reference", "special", "zero_scale"):
        special = case != "reference"
        values = {n: make_values(special) for n in ins}
        dev = {n: alloc(n, values[n]) for n in ins}
        td = ctype[base.d] if base.cplx else TD[base.d]
        out_j = torch.full(shape[::-1], 5.0, dtype=td, device="cuda").permute(2, 1, 0)
        out_g = out_j.clone()
        assert out_g.stride() == out_j.stride()
        dims = [
            (shape[i], [dev[n].stride()[i] for n in ins] + [out_j.stride()[i]])
            for i in range(3)
        ]
        vals, grid, _ = ew_args(canonicalize(dims, len(ins)), names, cplx=base.cplx)
        ops = {n: rng.choice(ops_ref if case == "reference" else ops_all) for n in ins}
        for n in ins:
            vals["op" + n] = ops[n]
        vals["opAB"] = rng.choice(bops)
        vals["opABC"] = rng.choice(bops)
        scal = {n: rng.choice([1.0, 2.0, -1.0, 0.5]) for n in ins}
        if case == "zero_scale":
            scal[rng.choice(ins)] = 0.0
        got_spec, geom, n_in = jit_class_of(base, vals)
        assert (got_spec.vec, got_spec.unit) == (spec.vec, spec.unit), (
            case,
            got_spec.kernel_name(),
        )
        sn = {"A": "alpha", "B": "beta", "C": "gamma"}
        run = {}
        for n in ins:
            run[n] = dev[n]
            if base.cplx:
                im = rng.choice([0.0, 1.0, -2.0]) if scal[n] != 0.0 else 0.0
                run[sn[n] + "_re"], run[sn[n] + "_im"] = scal[n], im
                scal[n] = complex(scal[n], im)
            else:
                run[sn[n]] = scal[n]
        gkey = ("generic", base.kernel_name())
        gk = build_ew(base) if gkey not in _LAUNCHERS else None
        _launch(gk, gkey, dict(vals, D=out_g, **run), grid)
        jk = build_ew_jit(spec.with_geom(geom))
        _launch(
            jk,
            ("jit", jk.name, tuple(sorted(geom.items()))),
            dict(run, D=out_j),
            (-(-n_in // 256), geom["ystep"], 1),
        )
        (gb, gnan), (jb, jnan) = (
            _bits(out_g, base.d, base.cplx),
            _bits(out_j, base.d, base.cplx),
        )
        # a NaN is a NaN: which operand's NaN an operation of two NaN returns (its sign and
        # payload) is unspecified, and the compiler may commute the operands
        diff = ((gb != jb) & ~(gnan & jnan)).nonzero()
        where = [tuple(i) for i in diff[:4].tolist()]
        assert diff.numel() == 0, (
            case,
            ops,
            vals["opAB"],
            vals["opABC"],
            scal,
            [
                (i, out_g.cpu()[i].item(), out_j.cpu()[i].item())
                + tuple(values[n][i].item() for n in ins)
                for i in where
            ],
        )
        if case != "reference":
            continue

        def term(n):
            if scal[n] == 0:
                return torch.zeros(shape, dtype=values[n].dtype)
            x = values[n]
            if base.cplx:
                x = x.to(torch.complex128)
                x = torch.conj(x) if ops[n] == C.OP_CONJ else x
                return scal[n] * x
            return scal[n] * _unary(x.double(), ops[n])

        if base.arity == 1:
            ref = term("A")
        elif base.arity == 2:
            ref = _binary(term("A"), term("C"), vals["opAB"])
        else:
            ref = _binary(
                _binary(term("A"), term("B"), vals["opAB"]), term("C"), vals["opABC"]
            )
        want = ref.to(td)
        assert torch.equal(out_j.cpu(), want), (spec.kernel_name(), ops)
