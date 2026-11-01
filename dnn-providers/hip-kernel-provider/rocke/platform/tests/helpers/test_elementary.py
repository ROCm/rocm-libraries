# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""f64 elementary functions (``rocke.helpers.elementary``): constants, coefficient tables,
accuracy and special values of the host model, and the lowering (CPU only).

The host model (:class:`rocke.helpers.elementary.HostMath`) evaluates exactly the operations the
emitted IR performs (IEEE f64, exact fma, the ``v_trig_preop_f64`` bit windows), so the bounds
here bound the kernels; ``tests/numeric/test_elementary_numeric.py`` checks that the device
reproduces the host model bit for bit.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from rocke.core.ir import F32, F64, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import elementary as el

PLATFORM = Path(__file__).resolve().parents[2]

#: Max error in ulps of the exact value, per function (f64): the documented bound of the
#: kernels (hipTensor's programmer's guide). The largest errors measured on 300,000 to 400,000
#: inputs per function (random over the domain, dense bands, the arguments closest to multiples
#: of pi/2) were 0.52 to 0.62 ulp.
ULP_BOUND = {
    "sin": 0.65,
    "cos": 0.65,
    "tan": 0.65,
    "sinh": 0.65,
    "cosh": 0.65,
    "asin": 0.65,
    "acos": 0.65,
    "atan": 0.65,
    "asinh": 0.65,
    "acosh": 0.65,
    "atanh": 0.65,
    "floor": 0.0,
    "ceil": 0.0,
}

#: x = m 2^e closest to a multiple of pi/2 (continued fractions of 2^e 2/pi); the first is the
#: closest double of all (|x mod pi/2| ~ 2^-60.9).
HARD_TRIG = (
    6381956970095103 * 2.0**797,
    5.379706629470344e286,
    7.161406163231498e286,
    1.241672507613542e308,
    6134899525417045.0,
    5.25764741460948e18,
    9.074492327336562e24,
    1426668733016935.2,
    1640781029691587.2,
    2.0**30,
    math.nextafter(2.0**30, 0.0),
    1.5707963267948966,
    3.141592653589793,
    4.71238898038469,
    1e22,
)


def _mp():
    return pytest.importorskip("mpmath")


def _exact(name, x):
    mp = _mp()
    with mp.workprec(256):
        try:
            v = getattr(mp, name)(mp.mpf(x))
        except (ValueError, ZeroDivisionError):
            return None
        if isinstance(v, mp.mpc) or mp.isnan(v) or mp.isinf(v):
            return None
        return v


def _ulp_error(got: float, exact) -> float:
    mp = _mp()
    if exact == 0:
        return 0.0 if got == 0 else math.inf
    if math.isinf(got) or math.isnan(got):
        return (
            0.0 if abs(exact) > 1.7976931348623157e308 and math.isinf(got) else math.inf
        )
    with mp.workprec(256):
        e = int(mp.floor(mp.log(abs(exact), 2)))
        return float(abs(mp.mpf(got) - exact) / mp.mpf(2) ** max(e - 52, -1074))


def samples(name: str, n: int, seed: int):
    """Random, every-exponent and targeted inputs of ``name`` (its domain)."""
    rnd = random.Random(seed)
    sgn = lambda: rnd.choice((-1.0, 1.0))  # noqa: E731
    xs = []
    if name in ("asin", "acos", "atanh"):
        xs += [
            sgn() * math.ldexp(rnd.uniform(1, 2), rnd.randint(-1074, -1))
            for _ in range(n)
        ]
        xs += [rnd.uniform(-1, 1) for _ in range(n)]
        xs += [sgn() * (1 - math.ldexp(1.0, -k)) for k in range(1, 54)]
        xs += [sgn() * (0.5 + d) for d in (-(2.0**-53), 0.0, 2.0**-52, 2.0**-30)]
    elif name == "acosh":
        xs += [1 + math.ldexp(rnd.uniform(1, 2), rnd.randint(-52, 5)) for _ in range(n)]
        xs += [math.ldexp(rnd.uniform(1, 2), rnd.randint(0, 1023)) for _ in range(n)]
        xs += [math.nextafter(1.0, 2.0), 2.0**28, math.nextafter(2.0**28, 3e8)]
    elif name in ("sinh", "cosh"):
        xs += [
            sgn() * math.ldexp(rnd.uniform(1, 2), rnd.randint(-1074, 9))
            for _ in range(n)
        ]
        xs += [sgn() * rnd.uniform(0.5, 2.0) for _ in range(n)]
        xs += [sgn() * rnd.uniform(705, 710.4758600739439) for _ in range(n // 4)]
        xs += [math.nextafter(1.0, 0.0), 1.0, 710.4758600739439]
    else:
        xs += [
            sgn() * math.ldexp(rnd.uniform(1, 2), rnd.randint(-1074, 1023))
            for _ in range(n)
        ]
        xs += [
            sgn() * math.ldexp(rnd.uniform(1, 2), rnd.randint(-30, 32))
            for _ in range(n)
        ]
        if name == "atan":
            for c in (7 / 16, 11 / 16, 19 / 16, 39 / 16):
                xs += [c, math.nextafter(c, 0.0), math.nextafter(c, 4.0)]
        if name == "asinh":
            xs += [2.0**-28, math.nextafter(2.0**-28, 0.0), 2.0**28, 0.25, 0.5, 2.0]
        if name in ("sin", "cos", "tan"):
            xs += list(HARD_TRIG)
        if name in ("floor", "ceil"):
            xs += [
                sgn() * (rnd.randint(0, 2**51) + rnd.choice((0.5, 0.25, 0.75)))
                for _ in range(n)
            ]
            xs += [0.5, -0.5, 2.0**52 - 0.5, -(2.0**52) + 0.5, 4503599627370497.0]
    return xs


# ------------------------------------------------------------- constants and tables


def test_constants_are_their_definitions():
    mp = _mp()
    with mp.workprec(600):

        def chunks(v, k):
            out = []
            for _ in range(k):
                h = float(v)
                out.append(h)
                v = v - mp.mpf(h)
            return out

        assert el.TWO_OVER_PI == float(2 / mp.pi)
        assert [el.PIO2_0, el.PIO2_1, el.PIO2_2] == chunks(mp.pi / 2, 3)
        assert [el.PI_H, el.PI_L] == chunks(mp.pi, 2)
        ln2 = mp.log(2)
        # a 42-bit head: n * LN2_42 is exact for |n| < 2^11
        assert Fraction(el.LN2_42).denominator <= 2**42
        assert abs(mp.mpf(el.LN2_42) - ln2) < mp.mpf(2) ** -42
        assert [el.LN2_42_1, el.LN2_42_2] == chunks(ln2 - mp.mpf(el.LN2_42), 2)
        assert el.LOG2E == float(1 / ln2)
        assert el.SQRT2 == float(mp.sqrt(2))
        for c, pair in ((0.5, el.ATAN_HALF), (1.0, el.ATAN_ONE), (1.5, el.ATAN_3HALF)):
            assert list(pair) == chunks(mp.atan(c), 2)
        assert el.ATAN_INF == (el.PIO2_0, el.PIO2_1)


#: where the coefficient generator lives: the source tree's tools/, or next to this file (the
#: installed test tree, which has no tools/ directory)
_GEN_PATHS = (
    PLATFORM / "tools" / "gen_elementary_coefficients.py",
    Path(__file__).resolve().parent / "gen_elementary_coefficients.py",
)


def _load_gen():
    """The coefficient generator module, or None when it is not there or its mpmath is missing
    (it imports mpmath at module level)."""
    path = next((p for p in _GEN_PATHS if p.is_file()), None)
    if path is None or importlib.util.find_spec("mpmath") is None:
        return None
    mod = sys.modules.get("gen_elementary_coefficients")
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "gen_elementary_coefficients", path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gen_elementary_coefficients"] = mod
        spec.loader.exec_module(mod)
    return mod


def _gen_module():
    """The generator inside a test: skips the test when the generator or mpmath is missing."""
    mod = _load_gen()
    if mod is None:
        pytest.skip("tools/gen_elementary_coefficients.py or mpmath is not available")
    return mod


def _table_params():
    gen = _load_gen()
    if gen is None:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason="tools/gen_elementary_coefficients.py or mpmath is not available"
                ),
            )
        ]
    return [t[0] for t in gen.TABLES]


@pytest.mark.parametrize("table", _table_params())
def test_coefficient_table_error_bound(table):
    """The stored double coefficients approximate their function within 2^-53.5 (relative) on
    their interval (dominated by rounding the leading coefficient to a double)."""
    _mp()
    gen = _gen_module()
    _name, f, interval, deg = next(t for t in gen.TABLES if t[0] == table)
    coefs = getattr(el, table)
    assert len(coefs) == deg + 1
    assert gen.max_rel_err_of_doubles(f, interval, coefs, n=600) <= 2.0**-53.5


def test_coefficient_intervals_cover_the_reduced_ranges():
    gen = _gen_module()
    hi = {t[0]: t[2][1] for t in gen.TABLES}
    lo = {t[0]: t[2][0] for t in gen.TABLES}
    # |r| <= pi/4 (plus the reduction's rounding), |t| <= 7/16, |x| <= 1/2, |x| < 1,
    # |r| <= ln2/2, s <= (sqrt2 - 1)/(sqrt2 + 1)
    assert hi["SIN_S"] >= (math.pi / 4 * (1 + 2.0**-40)) ** 2
    assert hi["COS_C"] >= (math.pi / 4 * (1 + 2.0**-40)) ** 2
    assert hi["ATAN_A"] >= (7 / 16) ** 2
    assert hi["ASIN_R"] >= 0.25
    assert hi["SINH_H"] >= 1.0
    assert (
        hi["EXP_P"] >= math.log(2) / 2 * (1 + 2.0**-30) and lo["EXP_P"] <= -hi["EXP_P"]
    )
    s = (math.sqrt(2) - 1) / (math.sqrt(2) + 1)
    assert hi["LOG_G"] >= s * s * (1 + 2.0**-20)


def test_coefficient_tables_are_current():
    """The generator reproduces the stored tables (Remez exchange, a few seconds)."""
    _mp()
    assert _gen_module().main(["--check"]) == 0


def test_trig_preop_reference_windows():
    """The host statement of v_trig_preop_f64: consecutive 53-bit windows of 2/pi from bit
    max(0, e - 1077), scaled by 2^128 from e = 1968; their sum is 2/pi less the leading bits
    whose products with x are multiples of 4."""
    mp = _mp()
    with mp.workprec(1600):
        two_pi = 2 / mp.pi
        for x in (
            1.0,
            2.0**30,
            2.0**60,
            1e300,
            2.0**944,
            2.0**945,
            1.7976931348623157e308,
        ):
            e = (el._bits(x) >> 52) & 0x7FF
            shift = max(0, e - 1077)
            scale = mp.mpf(2) ** (128 if e >= 1968 else 0)
            total = sum(mp.mpf(el.trig_preop_reference(x, s)) for s in range(4)) / scale
            head = mp.floor(two_pi * mp.mpf(2) ** shift) / mp.mpf(2) ** shift
            want = two_pi - head
            # the tail past 212 bits, and the rounding of a window in the subnormal range
            tol = mp.mpf(2) ** -(shift + 212) + 4 * mp.mpf(2) ** -1074 / scale
            assert abs(total - want) <= tol
            # x times the dropped head is a multiple of 4
            assert (mp.mpf(x) * head) % 4 == 0


# ------------------------------------------------------------- host model accuracy


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_host_model_accuracy(name):
    worst, at = 0.0, None
    for x in samples(name, 250, seed=sum(map(ord, name))):
        ex = _exact(name, x)
        if ex is None:
            continue
        err = _ulp_error(el.host(name, x), ex)
        if err > worst:
            worst, at = err, x
    assert worst <= ULP_BOUND[name], f"{name}: {worst:.4f} ulp at {at!r}"


INF, NAN = math.inf, math.nan
SUB = 5e-324
SPECIALS = (0.0, -0.0, INF, -INF, NAN, SUB, -SUB, 1.0, -1.0, 2.0, -2.0, 0.5)
MAXD = 1.7976931348623157e308

#: name -> results for SPECIALS (None: compare with the exact value rounded, within the bound)
EXPECTED = {
    "sin": (0.0, -0.0, NAN, NAN, NAN, SUB, -SUB, None, None, None, None, None),
    "cos": (1.0, 1.0, NAN, NAN, NAN, 1.0, 1.0, None, None, None, None, None),
    "tan": (0.0, -0.0, NAN, NAN, NAN, SUB, -SUB, None, None, None, None, None),
    "sinh": (0.0, -0.0, INF, -INF, NAN, SUB, -SUB, None, None, None, None, None),
    "cosh": (1.0, 1.0, INF, INF, NAN, 1.0, 1.0, None, None, None, None, None),
    "asin": (0.0, -0.0, NAN, NAN, NAN, SUB, -SUB, None, None, NAN, NAN, None),
    "acos": (None, None, NAN, NAN, NAN, None, None, 0.0, None, NAN, NAN, None),
    "atan": (0.0, -0.0, None, None, NAN, SUB, -SUB, None, None, None, None, None),
    "asinh": (0.0, -0.0, INF, -INF, NAN, SUB, -SUB, None, None, None, None, None),
    "acosh": (NAN, NAN, INF, NAN, NAN, NAN, NAN, 0.0, NAN, None, NAN, NAN),
    "atanh": (0.0, -0.0, NAN, NAN, NAN, SUB, -SUB, INF, -INF, NAN, NAN, None),
    "floor": (0.0, -0.0, INF, -INF, NAN, 0.0, -1.0, 1.0, -1.0, 2.0, -2.0, 0.0),
    "ceil": (0.0, -0.0, INF, -INF, NAN, 1.0, -0.0, 1.0, -1.0, 2.0, -2.0, 1.0),
}


def _same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_special_values(name):
    for x, want in zip(SPECIALS, EXPECTED[name]):
        got = el.host(name, x)
        if want is None:
            ex = _exact(name, x)
            assert ex is not None, (name, x)
            assert _ulp_error(got, ex) <= ULP_BOUND[name], (name, x, got)
        else:
            assert _same(got, want), (name, x, got, want)
    # atan of an infinity is pi/2 rounded, with the sign of the argument
    if name == "atan":
        assert el.host("atan", INF) == math.pi / 2
        assert el.host("atan", -INF) == -math.pi / 2


@pytest.mark.parametrize(
    "name,x,want",
    [
        # None: the exact value rounded, within the bound
        ("sinh", 710.4758600739439, None),
        ("sinh", 710.475860073944, INF),
        ("cosh", -710.475860073944, INF),
        ("sinh", -MAXD, -INF),
        ("cosh", MAXD, INF),
        ("asinh", MAXD, None),
        ("asinh", -MAXD, None),
        ("acosh", MAXD, None),
        ("atan", MAXD, None),
        ("asin", math.nextafter(1.0, 2.0), NAN),
        ("acos", -1.0, None),
        ("atanh", math.nextafter(1.0, 0.0), None),
        ("atanh", 2.0**-28, None),
        ("floor", -(2.0**52), -(2.0**52)),
        ("floor", -(2.0**52) + 0.5, -(2.0**52) + 0.0),
        ("ceil", 2.0**52 + 2, 2.0**52 + 2),
        ("ceil", -0.25, -0.0),
        ("floor", 0.25, 0.0),
        ("floor", -0.25, -1.0),
    ],
)
def test_edge_values(name, x, want):
    got = el.host(name, x)
    if want is None:
        ex = _exact(name, x)
        assert ex is not None, (name, x)
        assert _ulp_error(got, ex) <= ULP_BOUND[name], (name, x, got)
    else:
        assert _same(got, want), (name, x, got, want)


@pytest.mark.parametrize("name", ["sin", "cos", "tan"])
def test_hardest_trigonometric_reductions(name):
    for x in HARD_TRIG:
        for v in (x, -x):
            ex = _exact(name, v)
            err = _ulp_error(el.host(name, v), ex)
            assert err <= ULP_BOUND[name], (name, v, err)


# ------------------------------------------------------------- lowering


def _kernel(name: str, ty):
    b = IRBuilder(f"el_{name}_{ty.name}")
    xp = b.param("X", PtrType(ty, "global"), align=16)
    yp = b.param("Y", PtrType(ty, "global"), align=16)
    i = b.thread_id_x()
    x = b.global_load(xp, i, ty, align=8 if ty == F64 else 4)
    b.global_store(yp, i, el.emit(b, name, x), align=8 if ty == F64 else 4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("name", el.FUNCTIONS)
@pytest.mark.parametrize("ty", [F64, F32], ids=["f64", "f32"])
@pytest.mark.parametrize("arch", ["gfx90a", "gfx942", "gfx950"])
def test_lowering_has_no_libcall(name, ty, arch):
    ll = _lower_kernel_to_llvm_python(_kernel(name, ty), arch=arch)
    for fn in el.FUNCTIONS + ("exp", "log", "fmod", "remainder"):
        assert f"@{fn}(" not in ll and f"@llvm.{fn}." not in ll, (name, fn)
    trig = name in ("sin", "cos", "tan")
    assert ("v_trig_preop_f64" in ll) == trig


def test_narrow_types_are_refused():
    from rocke.core.ir import F16

    b = IRBuilder("el_f16")
    x = b.param("x", F16)
    with pytest.raises(ValueError):
        el.emit(b, "sin", x)


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_native_engine_lowers_the_same_text(name):
    """The C++ engine (hipTensor's JIT lowers recipes with it) prints the same LLVM IR."""
    if importlib.util.find_spec("rocke_engine") is None:
        pytest.skip("rocke_engine (the C++ engine) is not built")
    import rocke_engine as native
    from rocke.core.ir_serialize import serialize

    for ty in (F64, F32):
        k = _kernel(name, ty)
        py = _lower_kernel_to_llvm_python(k, arch="gfx950")
        cpp = native.lower_serialized_ir(serialize(k), arch="gfx950")
        assert py == cpp, name


def _integers_past_2_53(o, out: set) -> set:
    if isinstance(o, bool):
        return out
    if isinstance(o, int):
        if abs(o) > 1 << 53:
            out.add(o)
    elif isinstance(o, dict):
        for v in o.values():
            _integers_past_2_53(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _integers_past_2_53(v, out)
    return out


@pytest.mark.parametrize("name", el.FUNCTIONS)
def test_recipe_integers_are_exact_doubles(name):
    """Portable recipes (the form hipTensor's JIT replays) carry integers as doubles: every
    integer of a program is at most 2^53 in magnitude (wider i64 constants, such as the
    anchored f64 bit patterns and the sign / magnitude masks, are built from 32-bit halves).
    """
    from rocke.portable_ir.src.kerneldef_to_recipe import kerneldef_to_recipe

    for ty in (F64, F32):
        recipe = kerneldef_to_recipe(_kernel(name, ty))
        assert not _integers_past_2_53(recipe, set()), name


class _IntBuilder:
    """Evaluates the integer ops :meth:`IRMath.k64` emits (i32 constants as signed values)."""

    def const_i32(self, v):
        assert -(1 << 31) <= v < 1 << 31
        return ("i32", v)

    def const_i64(self, v):
        assert abs(v) <= 1 << 53
        return ("i64", v % (1 << 64))

    def zext(self, v, _ty):
        assert v[0] == "i32"
        return ("i64", v[1] % (1 << 32))

    def shl(self, a, s):
        return ("i64", (a[1] << s[1]) % (1 << 64))

    def lor(self, a, c):
        return ("i64", a[1] | c[1])


@pytest.mark.parametrize(
    "v",
    [0, 1, -1, 1 << 53, -(1 << 53), (1 << 53) + 1, el._ABS, el._SIGN, el._EXP_MASK]
    + [el._bits(c) for c in (el.PI_H, -el.PI_H, el.LOG2E, 0.1, -3.5e-300)],
)
def test_wide_integer_constants_are_their_value(v):
    """:meth:`IRMath.k64`: constants past 2^53 from two 32-bit halves whose combination is
    the constant (the compiler folds it back to one: the code objects are unchanged)."""
    kind, got = el.IRMath(_IntBuilder()).k64(v)
    assert kind == "i64" and got == v % (1 << 64)
