# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The math operators of the hipTensor kernels (``common.MATH_OPS``: sin, cos, tan, sinh, cosh,
asin, acos, atan, asinh, acosh, atanh, ceil, floor) in the ``math`` rows of the generic
elementwise and reduction kernels and in their JIT form.

The operators are the f64 programs of :mod:`rocke.helpers.elementary`, whose host model
(``elementary.host``) the device reproduces bit for bit (platform tests), so every result here
is checked exactly: the host model's f64 value, rounded once to f32 for f32 compute, then the
kernel's conversion to the output type (one rounding to nearest even; the f16 / bf16 references
are numpy's conversion and the exact rational rounding of ``_hiptensor_aot_rounding``). The
inputs cover each operator's domain, its boundaries and the special values (NaN, infinities,
signed zeros, subnormals, arguments outside the domain).
"""

from __future__ import annotations

import math
import random
import re

import pytest

from kernels.common.hiptensor_aot import common as C
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew
from kernels.common.hiptensor_aot.host import canonicalize, ew_args, reduce_args
from kernels.common.hiptensor_aot.jit_ew import (
    EwJitSpec,
    build_ew_jit,
    jit_class_of,
    jit_classes,
    roll_jit_ew,
)
from kernels.common.hiptensor_aot.reduce import ReduceSpec, build_reduce
from rocke.helpers import elementary as el

from ._hiptensor_aot_rounding import round_rne

np = pytest.importorskip("numpy")

MATH = sorted(C.MATH_OPS)

# ---------------------------------------------------------------- CPU: rows and classes


def test_math_rows():
    s = EwSpec(arity=2, a="f16", c="f16", d="f16", math=True)
    assert s.kernel_name().endswith("_m")
    assert not EwSpec(arity=2, a="f16", c="f16", d="f16").kernel_name().endswith("_m")
    assert ReduceSpec("col", "f32", "f32", math=True).kernel_name().endswith("_m")
    with pytest.raises(ValueError):
        EwSpec(arity=1, a="f32", d="f32", cplx=True, math=True).check()
    with pytest.raises(ValueError):
        EwJitSpec(EwSpec(arity=1, a="f32", d="f32", math=True), 4, (False,)).check()
    EwJitSpec(EwSpec(arity=1, a="f32", d="f32", math=True), 4, (True,)).check()
    assert set(C.UNARY_OPS) == set(range(1, 39)) - {3, 4, 5, 6, 7} - set(range(13, 22))
    assert C.MATH_OPS[C.OP_SIN] == "sin" and C.MATH_OPS[C.OP_FLOOR] == "floor"


def test_math_classes_are_the_all_vector_ones():
    """A math row ships one JIT class, every input a vector (the class the JIT takes by
    default); other layouts classify to vec 1 (no recipe: the launch keeps the pack kernel).
    """
    base = EwSpec(arity=2, a="f16", c="f16", d="f16", math=True)
    classes = jit_classes([base])
    assert [(c.vec, c.unit) for c in classes] == [(8, (True, True))]
    assert classes[0].key().endswith("_m_v8_uu")
    ops = dict(opA=C.OP_SIN, opC=C.OP_IDENTITY, opAB=C.OP_ADD)
    packed, _, _ = ew_args(
        canonicalize([(64, [1, 1, 1]), (9, [64, 64, 64])], 2), ["A", "C", "D"]
    )
    packed.update(ops)
    assert jit_class_of(base, packed)[0].vec == 8
    # C gathered along the output's fastest mode: vec 1
    gathered, _, _ = ew_args(
        canonicalize([(64, [1, 9, 1]), (9, [64, 1, 64])], 2), ["A", "C", "D"]
    )
    gathered.update(ops)
    assert jit_class_of(base, gathered)[0].vec == 1
    with pytest.raises(ValueError):
        EwJitSpec(base, 8, (True, False)).check()


@pytest.mark.parametrize(
    "base",
    [
        EwSpec(arity=1, a="f32", d="f32", math=True),
        EwSpec(arity=3, a="f16", b="f16", c="f16", d="f16", math=True),
        EwSpec(arity=2, a="f64", c="f16", d="f16", math=True),
    ],
    ids=lambda s: s.kernel_name(),
)
def test_math_class_rolls(base):
    spec = jit_classes([base])[0]
    recipe = roll_jit_ew(spec, verify=2)
    assert recipe["kernel_name_fmt"] == spec.kernel_name()


def _wide_integers(o, out: set) -> set:
    if isinstance(o, bool):
        return out
    if isinstance(o, int):
        if abs(o) > 1 << 53:
            out.add(o)
    elif isinstance(o, dict):
        for v in o.values():
            _wide_integers(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _wide_integers(v, out)
    return out


# the A operand pack of a contraction on the f64 route (f32 -> f64, k contiguous)
_PACK_DIMS = [(40, [1, 1]), (3, [840, 40]), (21, [40, 120])]


def _pack_class(op, a="f32", d="f64"):
    vals, _, _ = ew_args(canonicalize(_PACK_DIMS, 1), ["A", "D"])
    vals["opA"] = op
    return jit_class_of(EwSpec(arity=1, a=a, d=d, math=True), vals)


def test_math_recipes_carry_exact_integers():
    """The C++ recipe reader holds integers as doubles: no integer of a math class's recipe
    is past 2^53 (the f64 constants' bit patterns once were, and the JIT's kernels computed
    with corrupted constants)."""
    spec, _, _ = _pack_class(C.OP_ATAN)
    assert not _wide_integers(roll_jit_ew(spec, verify=1), set())


@pytest.mark.parametrize("a,d", [("f32", "f64"), ("f16", "f16"), ("f64", "f64")])
def test_math_recipe_replay_lowers_like_the_builder(a, d):
    """The hipTensor JIT path (the recipe replayed and lowered by the C++ engine) prints the
    LLVM IR of the Python builder's kernel up to SSA value names (the replay numbers the
    values of loops with carried values differently), for every math operator (skipped
    without the portable engine library)."""
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
    from rocke.portable_ir.src import online
    from rocke.portable_ir.src.recipe_bundle import cbor_encode

    try:
        lib = online.resolve_library_path(build=False)
    except (FileNotFoundError, RuntimeError) as err:  # pragma: no cover - depends on the host
        pytest.skip(f"portable engine library: {err}")
    if lib is None:
        pytest.skip("the portable engine library is not built")
    recipe = None
    for op in MATH:
        spec, geom, _ = _pack_class(op, a, d)
        if recipe is None:
            recipe = cbor_encode(roll_jit_ew(spec, verify=1))
        cpp, _ = online.recipe_cbor_to_llvm(recipe, arch="gfx942", ints=geom)
        py = _lower_kernel_to_llvm_python(build_ew_jit(spec, geom), arch="gfx942")
        assert _ssa_free(cpp) == _ssa_free(py), (spec.key(), op)


def _ssa_free(ll: str) -> str:
    """``ll`` with its local value and label names numbered in order of appearance."""
    names: dict = {}
    return re.sub(
        r"%[A-Za-z_.0-9]+", lambda m: names.setdefault(m.group(0), f"%v{len(names)}"), ll
    )


def _vgprs(kernel, arch="gfx950"):
    """VGPR count of ``kernel`` compiled for ``arch`` (skips without COMGR)."""
    try:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
        from rocke.runtime.comgr import build_hsaco_from_llvm_ir

        blob, _ = build_hsaco_from_llvm_ir(
            _lower_kernel_to_llvm_python(kernel, arch=arch),
            isa=f"amdgcn-amd-amdhsa--{arch}",
        )
    except Exception as e:  # noqa: BLE001 - no COMGR on this host
        pytest.skip(f"no COMGR compile here: {e}")
    # the code object's msgpack metadata: ".vgpr_count" then an unsigned integer (positive
    # fixint, or 0xcc / 0xcd with one / two big-endian bytes)
    key = b".vgpr_count"
    j = blob.find(key) + len(key)
    v = blob[j]
    if v < 0x80:
        return v
    assert v in (0xCC, 0xCD), hex(v)
    return int.from_bytes(blob[j + 1 : j + (2 if v == 0xCC else 3)], "big")


def test_math_programs_are_not_hoisted_out_of_loops():
    """The math programs run inside the kernels' row loops: their constants are anchored on an
    opaque zero (``elementary.opaque_zero``), else the backend hoists them out of the loop into
    VGPRs (more than twice the registers). Generic f32 math row, and a JIT kernel of one
    operator."""
    k = build_ew(EwSpec(arity=1, a="f32", d="f32", math=True))
    assert _vgprs(k) <= 96
    base = EwSpec(arity=1, a="f32", d="f32", math=True)
    vals, _, _ = ew_args(
        canonicalize([(1000, [1, 1]), (77, [1000, 1000])], 1), ["A", "D"]
    )
    vals["opA"] = C.OP_SIN
    spec, geom, _ = jit_class_of(base, vals)
    assert _vgprs(build_ew_jit(spec, geom)) <= 48


# ---------------------------------------------------------------- references

INF, NAN = math.inf, math.nan
SPECIAL = [0.0, -0.0, INF, -INF, NAN, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1e-300, -1e-300]
SPECIAL += [5e-324, 1e300, -1e300, 710.5, -710.5, 1e5, -1e5]


def _domain(name, rng, n):
    if name in ("asin", "acos", "atanh"):
        xs = [rng.uniform(-1, 1) for _ in range(n)]
        xs += [
            rng.choice((-1, 1)) * math.ldexp(1.0, -rng.randint(1, 60))
            for _ in range(n // 4)
        ]
    elif name == "acosh":
        xs = [1 + math.ldexp(rng.uniform(1, 2), rng.randint(-40, 12)) for _ in range(n)]
    elif name in ("sinh", "cosh"):
        xs = [rng.uniform(-12, 12) for _ in range(n)]
        xs += [rng.choice((-1, 1)) * rng.uniform(80, 95) for _ in range(n // 8)]
    elif name in ("floor", "ceil"):
        xs = [rng.uniform(-50, 50) for _ in range(n)]
        xs += [rng.randint(-9, 9) + rng.choice((0.5, 0.25)) for _ in range(n // 4)]
    else:
        xs = [
            rng.choice((-1, 1)) * math.ldexp(rng.uniform(1, 2), rng.randint(-20, 20))
            for _ in range(n)
        ]
    return xs + SPECIAL


def _as(dt, values):
    """``values`` converted to ``dt`` (numpy array of that type; bf16 as the f32 values of the
    rounded bf16)."""
    v = np.array(values, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if dt == "f64":
            return v
        if dt == "f32":
            return v.astype(np.float32)
        if dt == "f16":
            return v.astype(np.float16)
        return np.array(
            [round_rne(float(np.float32(x)), "bf16") for x in v], dtype=np.float32
        )


def _cdt(*dts):
    return "f64" if "f64" in dts else "f32"


def _op(code, x: float, cdt: str) -> float:
    """hipTensor unary operator ``code`` on the compute-type value ``x`` (exact references only:
    the math operators through the host model, the exact basic operators)."""
    if code in C.MATH_OPS:
        y = el.host(C.MATH_OPS[code], x)
    elif code in (C.OP_IDENTITY, C.OP_CONJ):
        y = x
    elif code == C.OP_NEG:
        y = -x
    elif code == C.OP_ABS:
        y = abs(x)
    else:
        raise ValueError(code)
    return float(np.float32(y)) if cdt == "f32" else y


def _to_out(v: float, cdt: str, d: str) -> float:
    with np.errstate(over="ignore"):
        if d == "f64":
            return v
        if d == "f32":
            return float(np.float32(v))
        if d == "f16":
            return float(
                np.float16(np.float64(v)) if cdt == "f64" else np.float16(np.float32(v))
            )
        return round_rne(v, "bf16")


def _same(g: float, w: float) -> bool:
    if math.isnan(w) or math.isnan(g):
        return math.isnan(w) and math.isnan(g)
    return g == w and math.copysign(1.0, g) == math.copysign(1.0, w)


# ---------------------------------------------------------------- GPU

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
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


_CACHE = {}


def _run(kernel, vals, grid, block=256, key=None):
    """Compile (cached by kernel name and ``key``: a JIT kernel's name does not name its
    constants) and launch ``kernel``."""
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    ck = (kernel.name, repr(key))
    if ck not in _CACHE:
        art = compile_kernel(
            kernel, arch=_gpu(), backend="python", capture_ir_text=False
        )
        sig = [
            {"name": p.name, "type": getattr(p.type, "name", "")} for p in kernel.params
        ]
        _CACHE[ck] = KernelLauncher(
            hsaco=art.hsaco, kernel_name=kernel.name, signature=sig
        )
    _CACHE[ck](
        {p.name: vals[p.name] for p in kernel.params},
        config=LaunchConfig(grid=grid, block=(block, 1, 1)),
    )
    torch.cuda.synchronize()


def _dev(dt, arr):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return (t.to(torch.bfloat16) if dt == "bf16" else t).cuda()


ROWS1 = [
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("f32", "f32"),
    ("f64", "f64"),
    ("f16", "f32"),
    ("f32", "f16"),
    ("f32", "f64"),
    ("bf16", "f64"),
]


@pytest.mark.parametrize("a,d", ROWS1)
@pytest.mark.parametrize("op", MATH)
@pytest.mark.parametrize(
    "jit,packed",
    [(False, False), (True, False), (True, True)],
    ids=["generic", "jit_v1", "jit_vector"],
)
def test_math_operator_exact(op, a, d, jit, packed):
    """D = opA(A) through the math row (generic kernel; JIT kernel of the strided layout, vec
    1, and of a packed one, the vector class) equals the host model."""
    _gpu()
    name = C.MATH_OPS[op]
    rng = random.Random(op * 31 + len(a) + 5 * len(d) + jit + 2 * packed)
    vals_in = _domain(name, rng, 600)
    vals_in += vals_in[: (-len(vals_in)) % 64]  # a multiple of every vector width
    xs = _as(a, vals_in)
    n = xs.size
    src = xs.astype(np.float64).tolist()
    x = _dev(a, xs)
    # strided: D written with a stride of 2 (the odd elements must stay untouched)
    step = 1 if packed else 2
    dbuf = torch.full([step * n], 7.0, dtype=TD[d], device="cuda")
    dd = dbuf[::step]
    vals, grid, idx64 = ew_args(canonicalize([(n, [1, step])], 1), ["A", "D"])
    base = EwSpec(arity=1, a=a, d=d, idx64=idx64, math=True)
    vals.update(opA=op)
    if jit:
        spec, geom, n_in = jit_class_of(base, vals)
        assert spec.vec == (16 // C.DTYPES[d][1] if packed else 1)
        k = build_ew_jit(spec, geom)
        _run(
            k,
            {"A": x, "D": dd, "alpha": 1.0},
            (-(-n_in // 256), geom["ystep"], 1),
            key=sorted(geom.items()),
        )
    else:
        vals.update(A=x, D=dd, alpha=1.0)
        _run(build_ew(base), vals, grid)
    cdt = _cdt(a, d)
    got = dd.cpu().double().tolist()
    bad = []
    for v, g in zip(src, got):
        w = _to_out(_op(op, v, cdt), cdt, d)
        if not _same(g, w):
            bad.append((v, g, w))
    assert not bad, f"{name} {a}->{d}: {len(bad)} wrong: {bad[:5]}"
    if not packed:
        assert bool((dbuf[1::2].double() == 7.0).all())


@pytest.mark.parametrize("arity", [2, 3])
@pytest.mark.parametrize(
    "a,d", [("f32", "f32"), ("f16", "f16"), ("f64", "f64"), ("bf16", "f32")]
)
@pytest.mark.parametrize("jit", [False, True], ids=["generic", "jit"])
def test_math_operators_of_several_inputs(arity, a, d, jit):
    """Every input with its own operator (math ones, exact basic ones, a mix), scales, binary
    operators: one copy of the math programs serves all inputs, each with its own operator.
    """
    _gpu()
    rng = random.Random(arity * 101 + len(a) + jit)
    names = ["A", "C"] if arity == 2 else ["A", "B", "C"]
    cdt = _cdt(a, d)
    n = 768 if jit else 777  # the JIT takes the vector class of a packed multiple
    for trial in range(6):
        ops = {
            nm: rng.choice(MATH + [C.OP_IDENTITY, C.OP_NEG, C.OP_ABS, C.OP_CONJ])
            for nm in names
        }
        if trial == 0:
            ops = {nm: C.OP_IDENTITY for nm in names}  # math row, no math operator
        if trial == 1:
            ops = {nm: MATH[i % len(MATH)] for i, nm in enumerate(names)}
        data = {
            nm: _as(
                a,
                [rng.uniform(-0.9, 0.9) * 2.0 ** rng.randint(-3, 1) for _ in range(n)],
            )
            for nm in names
        }
        # in-domain inputs for every operator
        for nm in names:
            fn = C.MATH_OPS.get(ops[nm])
            if fn == "acosh":
                data[nm] = _as(a, [1 + abs(v) * 4 for v in data[nm].astype(np.float64)])
        scal = {nm: float(rng.choice([1.0, -2.0, 0.5])) for nm in names}
        bops = [rng.choice(C.BINARY_OPS), rng.choice(C.BINARY_OPS)]
        dd = torch.zeros(n, dtype=TD[d], device="cuda")
        dims = canonicalize([(n, [1] * (len(names) + 1))], len(names))
        vals, grid, idx64 = ew_args(dims, names + ["D"])
        base = EwSpec(
            arity=arity,
            a=a,
            b=a if arity == 3 else None,
            c=a,
            d=d,
            idx64=idx64,
            math=True,
        )
        sn = {"A": "alpha", "B": "beta", "C": "gamma"}
        for nm in names:
            vals["op" + nm] = ops[nm]
        vals["opAB"], vals["opABC"] = bops
        run = {"D": dd}
        for nm in names:
            run[nm] = _dev(a, data[nm])
            run[sn[nm]] = scal[nm]
        if jit:
            spec, geom, n_in = jit_class_of(base, vals)
            _run(
                build_ew_jit(spec, geom),
                run,
                (-(-n_in // 256), geom["ystep"], 1),
                key=sorted(geom.items()),
            )
        else:
            vals.update(run)
            _run(build_ew(base), vals, grid)
        ftype = np.float64 if cdt == "f64" else np.float32

        def term(nm, i):
            v = float(data[nm].astype(np.float64)[i])
            return ftype(scal[nm]) * ftype(_op(ops[nm], v, cdt))

        def binop(code, p, q):
            if code == C.OP_ADD:
                return p + q
            if code == C.OP_MUL:
                return p * q
            if math.isnan(p) or math.isnan(q):
                return ftype(p + q)
            return max(p, q) if code == C.OP_MAX else min(p, q)

        got = dd.cpu().double().tolist()
        bad = []
        with np.errstate(all="ignore"):
            for i in range(n):
                if arity == 2:
                    r = binop(bops[0], term("A", i), term("C", i))
                else:
                    r = binop(
                        bops[1],
                        binop(bops[0], term("A", i), term("B", i)),
                        term("C", i),
                    )
                w = _to_out(float(r), cdt, d)
                if not (_same(got[i], w) or (w == 0 and got[i] == 0)):
                    bad.append((i, got[i], w))
        assert not bad, (ops, bops, bad[:4])


@pytest.mark.parametrize("dt", ["f32", "f64", "f16"])
@pytest.mark.parametrize("op", C.BASIC_UNARY_OPS)
def test_basic_operators_in_math_rows_unchanged(op, dt):
    """The operators every row has give the same bits in the math row as in the plain row."""
    _gpu()
    rng = random.Random(op + len(dt))
    xs = _as(dt, [rng.uniform(-3, 3) for _ in range(500)] + SPECIAL)
    n = xs.size
    outs = []
    for math_row in (False, True):
        dd = torch.zeros(n, dtype=TD[dt], device="cuda")
        vals, grid, idx64 = ew_args(canonicalize([(n, [1, 1])], 1), ["A", "D"])
        vals.update(A=_dev(dt, xs), D=dd, opA=op, alpha=1.0)
        _run(
            build_ew(EwSpec(arity=1, a=dt, d=dt, idx64=idx64, math=math_row)),
            vals,
            grid,
        )
        outs.append(dd.cpu())
    a, b = outs[0].view(
        torch.int16 if dt == "f16" else torch.int32 if dt == "f32" else torch.int64
    ), outs[1].view(
        torch.int16 if dt == "f16" else torch.int32 if dt == "f32" else torch.int64
    )
    nan = torch.isnan(outs[0].double()) & torch.isnan(outs[1].double())
    assert bool(((a == b) | nan).all())


#: f32 / f64 results of the operators every row has at special inputs (C99 semantics; RELU
#: as hipTensor defines it, x > 0 ? x : 0)
BASIC_SPECIALS = {
    C.OP_SQRT: [
        (0.0, 0.0),
        (-0.0, -0.0),
        (INF, INF),
        (-1.0, NAN),
        (-INF, NAN),
        (NAN, NAN),
    ],
    C.OP_RCP: [(0.0, INF), (-0.0, -INF), (INF, 0.0), (-INF, -0.0), (NAN, NAN)],
    C.OP_EXP: [(0.0, 1.0), (-0.0, 1.0), (INF, INF), (-INF, 0.0), (NAN, NAN)],
    C.OP_LOG: [
        (0.0, -INF),
        (-0.0, -INF),
        (1.0, 0.0),
        (INF, INF),
        (-1.0, NAN),
        (-INF, NAN),
        (NAN, NAN),
    ],
    C.OP_TANH: [(0.0, 0.0), (-0.0, -0.0), (INF, 1.0), (-INF, -1.0), (NAN, NAN)],
    C.OP_SIGMOID: [(0.0, 0.5), (INF, 1.0), (-INF, 0.0), (NAN, NAN)],
    C.OP_RELU: [
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.5, 1.5),
        (-1.5, 0.0),
        (INF, INF),
        (-INF, 0.0),
        (NAN, 0.0),
    ],
    C.OP_ABS: [(0.0, 0.0), (-0.0, 0.0), (-INF, INF), (NAN, NAN)],
    C.OP_NEG: [(0.0, -0.0), (-0.0, 0.0), (INF, -INF), (NAN, NAN)],
    C.OP_IDENTITY: [(-0.0, -0.0), (INF, INF), (NAN, NAN)],
    C.OP_CONJ: [(-0.0, -0.0), (-INF, -INF), (NAN, NAN)],
}


@pytest.mark.parametrize("dt", ["f32", "f64"])
@pytest.mark.parametrize("op", sorted(BASIC_SPECIALS))
def test_basic_operator_specials(op, dt):
    _gpu()
    table = BASIC_SPECIALS[op]
    xs = _as(dt, [x for x, _ in table])
    n = xs.size
    dd = torch.zeros(n, dtype=TD[dt], device="cuda")
    vals, grid, idx64 = ew_args(canonicalize([(n, [1, 1])], 1), ["A", "D"])
    vals.update(A=_dev(dt, xs), D=dd, opA=op, alpha=1.0)
    _run(build_ew(EwSpec(arity=1, a=dt, d=dt, idx64=idx64, math=True)), vals, grid)
    got = dd.cpu().double().tolist()
    bad = [(x, g, w) for (x, w), g in zip(table, got) if not _same(g, w)]
    assert not bad, bad


@pytest.mark.parametrize("dt", ["f32", "f64", "f16"])
def test_zero_scale_drops_the_math_operator(dt):
    """An input with a zero scale is not read and its operator's result on the placeholder zero
    (ACOSH(0) = NaN) does not reach D."""
    _gpu()
    n = 300
    a = torch.full([n], float("nan"), dtype=TD[dt], device="cuda")
    c = torch.arange(n, dtype=torch.float64).remainder(5).sub(2).to(TD[dt]).cuda()
    d = torch.zeros(n, dtype=TD[dt], device="cuda")
    vals, grid, idx64 = ew_args(canonicalize([(n, [1, 1, 1])], 2), ["A", "C", "D"])
    vals.update(
        A=a,
        C=c,
        D=d,
        opA=C.OP_ACOSH,
        opC=C.OP_FLOOR,
        opAB=C.OP_ADD,
        alpha=0.0,
        gamma=3.0,
    )
    _run(
        build_ew(EwSpec(arity=2, a=dt, c=dt, d=dt, idx64=idx64, math=True)), vals, grid
    )
    assert torch.equal(d.cpu().double(), 3.0 * c.cpu().double())


RED_ROWS = [("f32", "f32"), ("f16", "f16"), ("bf16", "bf16"), ("f64", "f64")]


@pytest.mark.parametrize("a,d", RED_ROWS)
@pytest.mark.parametrize("layout", ["row", "col"])
@pytest.mark.parametrize("op", [C.OP_SIN, C.OP_ATAN, C.OP_COSH, C.OP_FLOOR, C.OP_ASINH])
def test_reduce_math_operator_exact(op, a, d, layout):
    """MAX reduction of opA(A) (order independent: exact against the host model), then opC on C
    in the epilogue with A unread (alpha 0); with and without split partials."""
    arch = _gpu()
    rng = random.Random(op * 13 + len(a) + len(layout))
    shape = [40, 6, 9]  # kept: mode 1; reduced: modes 0 and 2
    xs = _as(a, [rng.uniform(-4, 4) for _ in range(math.prod(shape))]).reshape(shape)
    A = _dev(a, xs)
    if layout == "col":  # kept mode innermost in memory
        A = A.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    sa = list(A.stride())
    if layout == "row" and arch.startswith("gfx1"):
        pytest.skip("the row kernel needs wave64")
    cvals = _as(d, [rng.uniform(-4, 4) for _ in range(shape[1])])
    Cd = _dev(d, cvals)
    cdt = _cdt(a, d)
    for alpha, beta, opA, opC in (
        (1.0, 0.0, op, C.OP_IDENTITY),
        (0.0, 1.0, C.OP_IDENTITY, op),
    ):
        for split in (None, 1):
            Dd = torch.zeros(shape[1], dtype=TD[d], device="cuda")
            kd = canonicalize([(shape[1], [sa[1], 1, 1])], 0)
            rd = canonicalize([(shape[0], [sa[0]]), (shape[2], [sa[2]])], 0)
            mode, vals, grid, fgrid, s_ = reduce_args(kd, rd, split=split)
            cdtt = torch.float64 if cdt == "f64" else torch.float32
            W = torch.zeros(max(1, s_ * vals["n_out"]), dtype=cdtt, device="cuda")
            vals.update(
                A=A,
                C=Cd,
                D=Dd,
                W=W,
                alpha=alpha,
                beta=beta,
                opA=opA,
                opC=opC,
                opR=C.OP_MAX,
            )
            _run(build_reduce(ReduceSpec(mode=mode, a=a, d=d, math=True)), vals, grid)
            if s_ > 1:
                _run(
                    build_reduce(ReduceSpec(mode="final", a=a, d=d, math=True)),
                    vals,
                    fgrid,
                )
            got = Dd.cpu().double().tolist()
            for j in range(shape[1]):
                if alpha:
                    ys = [
                        _op(opA, float(v), cdt)
                        for v in xs[:, j, :].astype(np.float64).ravel()
                    ]
                    r = max(ys) if not any(math.isnan(y) for y in ys) else NAN
                    w = _to_out(r, cdt, d)
                else:
                    w = _to_out(
                        _op(opC, float(cvals.astype(np.float64)[j]), cdt), cdt, d
                    )
                assert _same(got[j], w), (mode, s_, alpha, j, got[j], w)
