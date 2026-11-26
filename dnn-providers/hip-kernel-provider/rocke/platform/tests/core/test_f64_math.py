# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 math expansions (``rocke.core.f64_math``): constants, accuracy, specials,
lowering contracts and the C++ table, all CPU-only.

The accuracy checks run :func:`rocke.core.f64_math.evaluate` -- the emitted
instruction program evaluated in IEEE f64 with an exact ``fma`` -- against a
200-bit ``mpmath`` reference, so they bound exactly what both engines emit.
``tests/numeric/test_f64_math_numeric.py`` checks the device against the same
host model bit for bit.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from rocke.core import f64_math as fm
from rocke.core.ir import BF16, F16, F32, F64, IRBuilder, PtrType
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import ERROR, verify

PLATFORM = Path(__file__).resolve().parents[2]

#: Max error in ulps of the correctly rounded result, per op, measured on the
#: host model below (and on the device by the numeric test).
ULP_BOUND = {
    "math.exp": 1.0,
    "math.exp2": 1.0,
    "math.log": 1.0,
    "math.log2": 1.0,
    "math.tanh": 1.0,
    "math.sqrt": 0.5,
    "math.rsqrt": 1.0,
}


def _gen_module():
    path = PLATFORM / "tools" / "gen_f64_math.py"
    if not path.is_file():
        # the installed test tree ships neither tools/ nor the C++ table the generator checks
        pytest.skip("tools/gen_f64_math.py is not here (installed test tree)")
    spec = importlib.util.spec_from_file_location("gen_f64_math", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gen_f64_math", mod)
    spec.loader.exec_module(mod)
    return mod


def test_cpp_table_is_current():
    gen = _gen_module()
    assert gen.current_text() == gen.render(), "run tools/gen_f64_math.py"


def test_every_program_key_is_a_declared_intrinsic():
    from rocke.core.lower_llvm import _INTRINSIC_DECLS

    for prog in fm.PROGRAMS.values():
        for key in prog.keys:
            assert key in _INTRINSIC_DECLS, (prog.name, key)


def test_constants_are_the_nearest_doubles():
    ln2 = fm._ln2_fraction()
    assert fm.LN2 == float(ln2)
    assert fm.LN2_LO == float(ln2 - Fraction(fm.LN2))
    # The 33-bit split keeps k * LN2_HI33 exact for every f64 exponent.
    assert Fraction(fm.LN2_HI33).denominator <= 2**33
    assert fm.LN2_LO33 == float(ln2 - Fraction(fm.LN2_HI33))
    log2e = 1 / ln2
    assert fm.LOG2E == float(log2e)
    assert fm.LOG2E_LO == float(log2e - Fraction(fm.LOG2E))
    # SQRT_HALF is within half an ulp of sqrt(1/2).
    half_ulp = Fraction(math.ulp(fm.SQRT_HALF)) / 2
    lo, hi = Fraction(fm.SQRT_HALF) - half_ulp, Fraction(fm.SQRT_HALF) + half_ulp
    assert lo * lo < Fraction(1, 2) < hi * hi


def test_coefficients_match_their_definitions():
    table = fm.coefficient_table()
    assert table["exp"] == tuple(
        float(Fraction(1, math.factorial(k))) for k in range(2, 14)
    )
    ln2 = fm._ln2_fraction()
    assert table["exp2"] == tuple(
        float(ln2**k / math.factorial(k)) for k in range(1, 14)
    )
    assert table["log"] == tuple(float(Fraction(2, 2 * k + 1)) for k in range(1, 12))
    assert fm.EXP_C == table["exp"] and fm.EXP2_C == table["exp2"]
    assert fm.LOG_C == table["log"]


def test_program_text_is_well_formed():
    for prog in fm.PROGRAMS.values():
        assert prog.steps, prog.name
        for i, step in enumerate(prog.steps):
            assert '"' not in step.text and "\\" not in step.text
            assert all(-1 <= a < i for a in step.args), (prog.name, step)
            for j in range(len(step.args)):
                assert f"${j}" in step.text, (prog.name, step)
            assert f"${len(step.args)}" not in step.text


# ------------------------------------------------------------- accuracy


def _ulp_error(y: float, ref) -> float:
    mp = pytest.importorskip("mpmath")
    if mp.isnan(ref):
        return 0.0 if math.isnan(y) else math.inf
    if math.isnan(y):
        return math.inf
    r = float(ref)
    if math.isinf(r) or math.isinf(y):
        return 0.0 if y == r else math.inf
    e = math.frexp(abs(r))[1] if r else -1073
    ulp = 2.0 ** max(e - 53, -1074)
    return float(abs(mp.mpf(y) - ref) / ulp)


def _reference(name: str):
    mp = pytest.importorskip("mpmath")
    return {
        "math.exp": mp.exp,
        "math.exp2": lambda v: mp.power(2, v),
        "math.log": mp.log,
        "math.log2": lambda v: mp.log(v, 2),
        "math.tanh": mp.tanh,
        "math.sqrt": mp.sqrt,
        "math.rsqrt": lambda v: 1 / mp.sqrt(v),
    }[name]


def accuracy_samples(name: str, n: int, seed: int = 7):
    """Random plus edge-of-range inputs, shared with the device test."""
    rnd = random.Random(seed)
    if name == "math.exp":
        xs = [rnd.uniform(-745.2, 709.8) for _ in range(n)]
        xs += [rnd.uniform(-2.0, 2.0) for _ in range(n)]
        xs += [fm.EXP_OVERFLOW, fm.EXP_UNDERFLOW, -708.39, 1e-300, -1e-17]
    elif name == "math.exp2":
        xs = [rnd.uniform(-1080.0, 1024.0) for _ in range(n)]
        xs += [rnd.uniform(-1.0, 1.0) for _ in range(n)]
        xs += [1023.99999, -1074.5, -1075.0, -1074.0, -1022.5, 0.5, -0.5]
    elif name in ("math.log", "math.log2"):
        xs = [
            math.ldexp(rnd.uniform(0.5, 1.0), rnd.randint(-1074, 1024))
            for _ in range(n)
        ]
        xs += [rnd.uniform(0.7, 1.42) for _ in range(n)]
        xs += [1.0 + rnd.uniform(-1e-8, 1e-8) for _ in range(n // 4)]
        xs += [5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 1.0, 2.0]
        xs += [fm.SQRT_HALF, math.nextafter(fm.SQRT_HALF, 1.0), math.sqrt(2.0)]
    elif name == "math.tanh":
        xs = [rnd.uniform(-25.0, 25.0) for _ in range(n)]
        xs += [rnd.uniform(-1.0, 1.0) for _ in range(n)]
        xs += [math.ldexp(rnd.uniform(0.5, 1.0), rnd.randint(-40, 0)) for _ in range(n)]
        xs += [2.0**-27, 2.0**-28, 0.5493061443340548, 22.0, 19.06]
    else:
        xs = [
            math.ldexp(rnd.uniform(0.5, 1.0), rnd.randint(-1074, 1024))
            for _ in range(n)
        ]
    return [x for x in xs if not (name in ("math.log", "math.log2") and x <= 0)]


@pytest.mark.parametrize("name", sorted(fm.PROGRAMS))
def test_host_model_accuracy(name):
    mp = pytest.importorskip("mpmath")
    mp.mp.prec = 200
    ref = _reference(name)
    worst, at = 0.0, None
    for x in accuracy_samples(name, 400):
        err = _ulp_error(fm.evaluate(fm.PROGRAMS[name], x), ref(mp.mpf(x)))
        if err > worst:
            worst, at = err, x
    assert worst <= ULP_BOUND[name], f"{name}: {worst:.3f} ulp at {at!r}"


def _dense(lo: float, hi: float, n: int):
    return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]


def _log_uniform(lo_exp: int, hi_exp: int, n: int, seed: int):
    rnd = random.Random(seed)
    return [
        math.ldexp(rnd.uniform(0.5, 1.0), rnd.randint(lo_exp, hi_exp)) for _ in range(n)
    ]


#: Dense and targeted samples per op: the bands where the error peaks (tanh
#: near 0.2, the switch points 2^-27 and 22), subnormal results of exp / exp2
#: (the ``ldexp(1 + s, n)`` reconstruction rounds twice there), arguments
#: near 1 for log, subnormal arguments.
DENSE_SAMPLES = {
    "math.tanh": _dense(0.15, 0.3, 6000)
    + _dense(0.5, 0.6, 1000)
    + _dense(2.0**-28, 2.0**-26, 500)
    + _dense(21.5, 22.5, 500)
    + [math.nextafter(2.0**-27, 0.0), 2.0**-27, math.nextafter(22.0, 0.0), 22.0],
    "math.exp": _dense(-745.13, -708.4, 3000)
    + _dense(709.0, 709.78, 500)
    + _dense(-1e-5, 1e-5, 500),
    "math.exp2": _dense(-1074.99, -1022.0, 3000) + _dense(1023.0, 1023.999, 500),
    "math.log": _dense(1 - 2.0**-20, 1 + 2.0**-20, 1000)
    + _dense(0.7, 1.42, 2000)
    + _log_uniform(-1074, -1022, 1000, 11),
    "math.log2": _dense(1 - 2.0**-20, 1 + 2.0**-20, 1000)
    + _dense(0.7, 1.42, 2000)
    + _log_uniform(-1074, -1022, 1000, 12),
    "math.sqrt": _log_uniform(-1074, 1023, 3000, 13),
    "math.rsqrt": _log_uniform(-1074, 1023, 3000, 14),
}


@pytest.mark.parametrize("name", sorted(fm.PROGRAMS))
def test_host_model_accuracy_dense(name):
    """The bound holds on dense / targeted samples, not only on random ones."""
    mp = pytest.importorskip("mpmath")
    mp.mp.prec = 200
    ref = _reference(name)
    worst, at = 0.0, None
    for x in DENSE_SAMPLES[name]:
        err = _ulp_error(fm.evaluate(fm.PROGRAMS[name], x), ref(mp.mpf(x)))
        if err > worst:
            worst, at = err, x
    assert worst <= ULP_BOUND[name], f"{name}: {worst:.3f} ulp at {at!r}"


SPECIALS = (0.0, -0.0, math.inf, -math.inf, math.nan, -1.0, 5e-324, -5e-324)

EXPECTED_SPECIALS = {
    "math.exp": (1.0, 1.0, math.inf, 0.0, math.nan, math.exp(-1.0), 1.0, 1.0),
    "math.exp2": (1.0, 1.0, math.inf, 0.0, math.nan, 0.5, 1.0, 1.0),
    "math.log": (
        -math.inf,
        -math.inf,
        math.inf,
        math.nan,
        math.nan,
        math.nan,
        -744.4400719213812,
        math.nan,
    ),
    "math.log2": (
        -math.inf,
        -math.inf,
        math.inf,
        math.nan,
        math.nan,
        math.nan,
        -1074.0,
        math.nan,
    ),
    "math.tanh": (0.0, -0.0, 1.0, -1.0, math.nan, math.tanh(-1.0), 5e-324, -5e-324),
    "math.sqrt": (
        0.0,
        -0.0,
        math.inf,
        math.nan,
        math.nan,
        math.nan,
        math.sqrt(5e-324),
        math.nan,
    ),
    "math.rsqrt": (
        math.inf,
        -math.inf,
        0.0,
        math.nan,
        math.nan,
        math.nan,
        1.0 / math.sqrt(5e-324),
        math.nan,
    ),
}


def _same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


@pytest.mark.parametrize("name", sorted(fm.PROGRAMS))
def test_special_values(name):
    got = [fm.evaluate(fm.PROGRAMS[name], x) for x in SPECIALS]
    for x, g, want in zip(SPECIALS, got, EXPECTED_SPECIALS[name]):
        assert _same(g, want), (name, x, g, want)


# ------------------------------------------------------------- lowering


def _unary_kernel(op: str, ty):
    b = IRBuilder(f"k_{op}")
    x = b.param("x", ty)
    out = b.param("out", PtrType(ty, "global"))
    b.store(getattr(b, op)(x), out, b.const_i32(0))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("op", ["exp", "exp2", "log", "log2", "tanh", "sqrt", "rsqrt"])
@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950", "gfx1151"])
def test_f64_lowering_has_no_libcall(op, arch):
    ll = _lower_kernel_to_llvm_python(
        _unary_kernel(op, F64), arch=arch, llvm_flavor="llvm22"
    )
    for bad in (
        "@llvm.exp.f64",
        "@llvm.exp2.f64",
        "@llvm.log.f64",
        "@llvm.log2.f64",
        "@llvm.tanh.f64",
        "@llvm.amdgcn.rsq.f64",
        "@exp(",
        "@log(",
    ):
        assert bad not in ll
    prog = fm.PROGRAMS[f"math.{op}"]
    for step in prog.steps[:-1]:
        assert f"%{step.hint}." in ll


def test_f64_fma_is_always_fused():
    b = IRBuilder("fma64")
    x = b.param("x", F64)
    out = b.param("out", PtrType(F64, "global"))
    b.store(b.fma(x, x, x), out, b.const_i32(0))
    b.ret()
    ll = _lower_kernel_to_llvm_python(b.kernel, arch="gfx950", llvm_flavor="llvm22")
    assert "call double @llvm.fma.f64(" in ll and "fmuladd" not in ll


@pytest.mark.parametrize("op", ["exp", "log"])
def test_natural_f32_ops_use_llvm_intrinsics(op):
    ll = _lower_kernel_to_llvm_python(
        _unary_kernel(op, F32), arch="gfx950", llvm_flavor="llvm22"
    )
    assert f"declare float @llvm.{op}.f32(float)" in ll


@pytest.mark.parametrize("op", ["exp", "log", "tanh"])
@pytest.mark.parametrize("ty", [F16, BF16])
def test_narrow_types_are_rejected(op, ty):
    b = IRBuilder("narrow")
    x = b.param("x", ty)
    with pytest.raises(
        ValueError, match=rf"^math\.{op} requires f32 or f64 operand, got"
    ):
        getattr(b, op)(x)


@pytest.mark.parametrize("op", ["math.exp", "math.log", "math.tanh"])
def test_verifier_rejects_narrow_serialized_form(op):
    b = IRBuilder("narrow_verify")
    x = b.param("x", F16)
    b._op(op, [x], [F16], result_name_hint="v")
    errors = [d.message for d in verify(b.kernel) if d.severity == ERROR]
    assert f"{op} requires f32 or f64 operand, got f16" in errors


@pytest.mark.parametrize(
    "op,call", [("exp", "exp("), ("log", "log("), ("tanh", "tanh("), ("sqrt", "sqrt(")]
)
def test_hip_uses_double_overloads(op, call):
    hip = lower_kernel_to_hip(_unary_kernel(op, F64), arch="gfx950")
    assert f"= {call}" in hip and "(float)" not in hip.split("__global__")[-1]


def test_native_engine_matches_python():
    native = pytest.importorskip("rocke_engine")
    from rocke.core.ir_serialize import serialize

    for op in ("exp", "exp2", "log", "log2", "tanh", "sqrt", "rsqrt"):
        kernel = _unary_kernel(op, F64)
        for flavor in ("llvm20", "llvm22", "llvm23"):
            ll = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
            assert (
                native.lower_serialized_ir(
                    serialize(kernel), arch="gfx950", flavor=flavor
                )
                == ll
            )
