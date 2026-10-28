# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Accurate f64 elementary functions written in IR ops (sin, cos, tan, sinh, cosh, asin,
acos, atan, asinh, acosh, atanh, floor, ceil).

The AMDGPU backend has no f64 instructions or intrinsic lowerings for these functions (and the
f32 intrinsics it has are not uniformly accurate), so each one is a fixed program of correctly
rounded operations (add, sub, mul, div, fma, sqrt, compares, selects, integer bit operations on
the f64 encoding) plus the ``v_trig_preop_f64`` instruction for large trigonometric
arguments. f32 callers convert to f64, call these and round the result once to f32, which gives
the correctly rounded f32 result except when the exact value lies within about ``2^-29`` f32 ulp
of a rounding boundary.

Every function is written once against a small *math context* (:class:`IRMath` emits IR ops,
:class:`HostMath` evaluates the same steps on the host in IEEE f64 with an exact fma), so tests
bound exactly what the kernels compute. ``tests/helpers/test_elementary.py`` measures the host
model against a high-precision reference; ``tests/numeric/test_elementary_numeric.py`` checks
that the device reproduces the host model bit for bit.

Algorithms (no tables except the 2/pi bits the hardware provides; ``dd`` = double-double):

* ``sin`` / ``cos`` / ``tan``: ``|x| = k pi/2 + r``. Below ``2^30`` a three-part Cody-Waite
  reduction with fma (``pi/2`` to 159 bits; the first product is exact); from ``2^30`` a
  Payne-Hanek reduction: ``|x|`` times four 53-bit windows of ``2/pi`` chosen by its exponent
  (``v_trig_preop_f64``, 212 bits), each product split exactly, the integer part taken modulo 4
  and the fraction summed error-free to a dd (relative error below ``2^-95`` even for the
  arguments closest to a multiple of ``pi/2``). The kernels evaluate minimax polynomials on
  ``|r| <= pi/4`` with the ``-r^3/6`` (``-r^2/2``) term exact and the tail of ``r`` folded in,
  returning dd results; ``tan`` divides them in dd (``-cos/sin`` in odd quadrants).
* ``sinh`` / ``cosh``: ``|x| < 1``: ``sinh`` by its odd minimax polynomial (leading term exact);
  else ``e^|x| = 2^n m`` with ``m`` a dd (Cody-Waite with a 42-bit ``ln 2`` head, a minimax
  polynomial of ``e^r - 1 - r - r^2/2``), ``e^-|x|`` as its dd reciprocal, both halved and
  scaled by two power-of-two factors (so ``e^|x| / 2`` does not overflow before the result
  does), then summed / subtracted in dd.
* ``asin`` / ``acos``: ``|x| <= 1/2``: odd minimax polynomial; else
  ``asin(|x|) = pi/2 - 2 asin(sqrt((1 - |x|) / 2))`` with the square root as a dd, and the
  ``acos`` identities ``pi/2 - asin(x)``, ``2 asin(s)``, ``pi - 2 asin(s)`` in dd.
* ``atan``: reduction to ``|t| <= 7/16`` around ``0, 1/2, 1, 3/2, inf`` (``t`` as a dd from
  an exact numerator and a dd denominator), odd minimax polynomial, ``atan(c)`` as a dd.
* ``asinh`` / ``acosh`` / ``atanh``: ``log`` of a dd argument ``y >= 1``
  (``|x| + sqrt(x^2 + 1)``, ``x + sqrt(x^2 - 1)`` with ``x^2 -/+ 1`` exact in dd,
  ``1 + 2|x| / (1 - |x|)``); ``log`` of a dd: ``y = 2^e m``, ``m`` in ``[sqrt(1/2), sqrt(2))``,
  ``log1p(f) = f - f^2/2 + s (f^2/2 + z g(z))`` (``s = f/(2+f)``, ``z = s^2``, minimax ``g``),
  ``f^2/2`` exact, ``e ln 2`` with a 42-bit head. Large arguments use ``log(|x|) + ln 2``.
* ``floor`` / ``ceil``: the integer nearest ``|x| < 2^52`` by adding and removing ``2^52``,
  corrected by one, sign of ``x`` restored (``ceil(-0.5) = -0``).

Special values follow C: NaN in gives NaN; ``sin`` / ``cos`` / ``tan`` of an infinity are NaN;
``sinh(+-inf) = +-inf``, ``cosh(+-inf) = +inf``; ``asin`` / ``acos`` outside ``[-1, 1]``,
``acosh`` below 1 and ``atanh`` outside ``[-1, 1]`` are NaN; ``atanh(+-1) = +-inf``;
``atan(+-inf) = +-pi/2``; odd functions keep the sign of zero.

Placement: Python-only. The kernels that use it are built by the Python engine (hipTensor's
``kernels.common.hiptensor_aot`` family and the operand operators of
:mod:`rocke.instances.common.gemm_prologue`, which the C++ engine refuses), so no kernel the
C++ engine emits changes.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple, Union

from ..core.ir import F64, I32, I64, IRBuilder, Value


def _h(s: str) -> float:
    return float.fromhex(s)


# ------------------------------------------------------------------------------------------------
# Constants (tests/helpers/test_elementary.py recomputes every one from its definition)
# ------------------------------------------------------------------------------------------------

TWO_OVER_PI = _h("0x1.45f306dc9c883p-1")
# pi/2 = PIO2_0 + PIO2_1 + PIO2_2 (+ ~2^-161): consecutive nearest-double chunks
PIO2_0 = _h("0x1.921fb54442d18p+0")
PIO2_1 = _h("0x1.1a62633145c07p-54")
PIO2_2 = _h("-0x1.f1976b7ed8fbcp-110")
PI_H = _h("0x1.921fb54442d18p+1")
PI_L = _h("0x1.1a62633145c07p-53")
# ln 2 = LN2_42 + LN2_42_1 + LN2_42_2 with a 42-bit head (n * LN2_42 exact for |n| < 2^11)
LN2_42 = _h("0x1.62e42fefa38p-1")
LN2_42_1 = _h("0x1.ef35793c7673p-45")
LN2_42_2 = _h("0x1.f97b57a079a19p-103")
LOG2E = _h("0x1.71547652b82fep+0")
SQRT2 = _h("0x1.6a09e667f3bcdp+0")
# atan(c) = hi + lo for the reduction points c = 1/2, 1, 3/2, inf
ATAN_HALF = (_h("0x1.dac670561bb4fp-2"), _h("0x1.a2b7f222f65e2p-56"))
ATAN_ONE = (_h("0x1.921fb54442d18p-1"), _h("0x1.1a62633145c07p-55"))
ATAN_3HALF = (_h("0x1.f730bd281f69bp-1"), _h("0x1.007887af0cbbdp-56"))
ATAN_INF = (PIO2_0, PIO2_1)

TRIG_LARGE = 2.0**30  # Payne-Hanek reduction from here
_M52 = 1.5 * 2.0**52  # add and subtract: round an |x| < 2^51 to an integer (RNE)
_SIGN = -(1 << 63)  # f64 sign bit as a signed i64
_ABS = (1 << 63) - 1
_EXACT = 1 << 53  # integers a double holds exactly (see IRMath.k64)
_EXP_MASK = 0x7FF0000000000000
_MANT_MASK = 0x000FFFFFFFFFFFFF
_ONE_BITS = 0x3FF0000000000000

# ------------------------------------------------------------------------------------------------
# Minimax coefficients (tools/gen_elementary_coefficients.py; relative error of the double
# coefficients on the interval in the comment)
# ------------------------------------------------------------------------------------------------

# (sin(sqrt z)/sqrt z - 1)/z, z in [0, 0.6169]
SIN_S = (
    _h("-0x1.5555555555555p-3"),
    _h("0x1.1111111111110p-7"),
    _h("-0x1.a01a01a019929p-13"),
    _h("0x1.71de3a545f037p-19"),
    _h("-0x1.ae6454104d5d8p-26"),
    _h("0x1.61217d41bb89bp-33"),
    _h("-0x1.ab16db5288252p-41"),
)
# (cos(sqrt z) - 1 + z/2)/z^2, z in [0, 0.6169]
COS_C = (
    _h("0x1.5555555555555p-5"),
    _h("-0x1.6c16c16c16c16p-10"),
    _h("0x1.a01a01a019d06p-16"),
    _h("-0x1.27e4fb77125e8p-22"),
    _h("0x1.1eed8deadabefp-29"),
    _h("-0x1.9394b8fa4ece7p-37"),
    _h("0x1.ab77c5650a425p-45"),
)
# (atan(sqrt z)/sqrt z - 1)/z, z in [0, (7/16)^2]
ATAN_A = (
    _h("-0x1.5555555555555p-2"),
    _h("0x1.99999999998cap-3"),
    _h("-0x1.24924924858e8p-3"),
    _h("0x1.c71c71bd52eeap-4"),
    _h("-0x1.745d1552e8d71p-4"),
    _h("0x1.3b1376694c06dp-4"),
    _h("-0x1.110ca60ef6421p-4"),
    _h("0x1.e17131ed68ef0p-5"),
    _h("-0x1.ab5e0ae0eca25p-5"),
    _h("0x1.701255ab35430p-5"),
    _h("-0x1.0f87a102a636fp-5"),
    _h("0x1.e4a235b9c4a59p-7"),
)
# (asin(sqrt z)/sqrt z - 1)/z, z in [0, 1/4]
ASIN_R = (
    _h("0x1.5555555555555p-3"),
    _h("0x1.333333333338dp-4"),
    _h("0x1.6db6db6dab864p-5"),
    _h("0x1.f1c71c7abbd6fp-6"),
    _h("0x1.6e8ba110ffef0p-6"),
    _h("0x1.1c4efee7125cap-6"),
    _h("0x1.c9906663d6d13p-7"),
    _h("0x1.7b05bf281d434p-7"),
    _h("0x1.3b2fc027779e6p-7"),
    _h("0x1.31f2160e1e4f6p-7"),
    _h("0x1.86bc5d25f4c92p-9"),
    _h("0x1.42e3180325070p-6"),
    _h("-0x1.3ec09802d3b38p-6"),
    _h("0x1.e752255431ce2p-6"),
)
# (sinh(sqrt z)/sqrt z - 1)/z, z in [0, 1]
SINH_H = (
    _h("0x1.5555555555555p-3"),
    _h("0x1.1111111111111p-7"),
    _h("0x1.a01a01a01a001p-13"),
    _h("0x1.71de3a556f8e6p-19"),
    _h("0x1.ae64567936642p-26"),
    _h("0x1.612467c3f11f8p-33"),
    _h("0x1.ae7782967fd2ep-41"),
    _h("0x1.99eeaddf33ce1p-49"),
)
# (e^r - 1 - r - r^2/2)/r^3, r in [-0.34658, 0.34658]
EXP_P = (
    _h("0x1.5555555555555p-3"),
    _h("0x1.5555555555556p-5"),
    _h("0x1.1111111111110p-7"),
    _h("0x1.6c16c16c162dap-10"),
    _h("0x1.a01a01a01b3d5p-13"),
    _h("0x1.a01a01a6d56aap-16"),
    _h("0x1.71de3a5007d0dp-19"),
    _h("0x1.27e4db6d6a969p-22"),
    _h("0x1.ae645f12243e9p-26"),
    _h("0x1.1f72f2050aa81p-29"),
    _h("0x1.616b75e800303p-33"),
)
# (2 atanh(sqrt z)/sqrt z - 2)/z, z in [0, 0.02944]
LOG_G = (
    _h("0x1.5555555555555p-1"),
    _h("0x1.9999999999a38p-2"),
    _h("0x1.2492492476b5cp-2"),
    _h("0x1.c71c72017e746p-3"),
    _h("0x1.745cf8f61afcep-3"),
    _h("0x1.3b1c38386f2d9p-3"),
    _h("0x1.0fbe4dac856a8p-3"),
    _h("0x1.0c0665b09f2b8p-3"),
)

#: The functions, by name.
FUNCTIONS = (
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "floor",
    "ceil",
)

Bool = Union[bool, Value]


# ------------------------------------------------------------------------------------------------
# Math contexts
# ------------------------------------------------------------------------------------------------


class IRMath:
    """Emits the steps as IR ops on ``b`` (f64 values, i64 bit patterns, i1 predicates).

    Python floats / ints passed to an op are constants; a Python ``bool`` as a predicate
    selects at build time (nothing is emitted for the other side)."""

    def __init__(self, b: IRBuilder, anchor: Optional[Value] = None):
        self.b = b
        # ``anchor``: an i64 zero the compiler cannot see through (:func:`opaque_zero`); f64
        # constants are built on it so that they are not hoisted out of an enclosing loop
        self.anchor = anchor
        self.pool: dict = {}

    # values
    def f(self, v) -> Value:
        if not isinstance(v, (float, int)):
            return v
        if self.anchor is None:
            return self.b.const_f64(float(v))
        key = _bits(float(v))
        if key in self.pool:
            return self.pool[key]
        # built at its use (a value defined in one branch must not be used in another)
        return self.b.bitcast(self.b.lor(self.k64(key), self.anchor), F64)

    def materialize(self, constants: Sequence[float]) -> None:
        """Build the anchored ``constants`` here, once (they then dominate every later use,
        nested branches included): one constant, OR and bitcast each instead of one per use.
        """
        for c in constants:
            key = _bits(float(c))
            if key not in self.pool:
                self.pool[key] = self.b.bitcast(
                    self.b.lor(self.k64(key), self.anchor), F64
                )

    def k64(self, v: int) -> Value:
        """The i64 constant ``v``. Past 2^53 in magnitude it is built from its two 32-bit
        halves (``(zext hi << 32) | zext lo``, folded by the compiler to the same constant):
        portable recipes carry integers as doubles, which keep 53 bits."""
        v = int(v)
        if -_EXACT <= v <= _EXACT:
            return self.b.const_i64(v)
        u = v % (1 << 64)
        hi, lo = u >> 32, u & 0xFFFFFFFF
        b = self.b

        def half(h: int) -> Value:
            return b.zext(b.const_i32(h - (1 << 32) if h >= 1 << 31 else h), I64)

        return b.lor(b.shl(half(hi), b.const_i64(32)), half(lo))

    def i(self, v) -> Value:
        return self.k64(v) if isinstance(v, int) else v

    def add(self, a, c):
        return self.b.fadd(self.f(a), self.f(c))

    def sub(self, a, c):
        return self.b.fsub(self.f(a), self.f(c))

    def mul(self, a, c):
        return self.b.fmul(self.f(a), self.f(c))

    def div(self, a, c):
        return self.b.fdiv(self.f(a), self.f(c))

    def fma(self, a, c, d):
        return self.b.fma(self.f(a), self.f(c), self.f(d))

    def sqrt(self, a):
        return self.b.sqrt(self.f(a))

    def neg(self, a):
        return self.b.fneg(self.f(a))

    def abs(self, a):
        return self.b.fabs(self.f(a))

    # predicates
    def _cmp(self, pred, a, c):
        return self.b.fcmp(pred, self.f(a), self.f(c))

    def lt(self, a, c):
        return self._cmp("olt", a, c)

    def le(self, a, c):
        return self._cmp("ole", a, c)

    def gt(self, a, c):
        return self._cmp("ogt", a, c)

    def ge(self, a, c):
        return self._cmp("oge", a, c)

    def eq(self, a, c):
        return self._cmp("oeq", a, c)

    def isnan(self, a):
        return self._cmp("uno", a, a)

    def band(self, p: Bool, q: Bool) -> Bool:
        if isinstance(p, bool):
            return q if p else False
        if isinstance(q, bool):
            return p if q else False
        return self.b.land(p, q)

    def bor(self, p: Bool, q: Bool) -> Bool:
        if isinstance(p, bool):
            return True if p else q
        if isinstance(q, bool):
            return True if q else p
        return self.b.lor(p, q)

    def bnot(self, p: Bool) -> Bool:
        return (not p) if isinstance(p, bool) else self.b.lnot(p)

    def sel(self, p: Bool, a, c):
        if isinstance(p, bool):
            return a if p else c
        if isinstance(a, (float, int)) and isinstance(c, (float, int)):
            if isinstance(a, int) and isinstance(c, int):
                return self.b.select(p, self.i(a), self.i(c))
            return self.b.select(p, self.f(a), self.f(c))
        if (
            isinstance(a, Value)
            and a.type == I64
            or isinstance(c, Value)
            and c.type == I64
        ):
            return self.b.select(p, self.i(a), self.i(c))
        return self.b.select(p, self.f(a), self.f(c))

    # f64 encodings
    def bits(self, a):
        return self.b.bitcast(self.f(a), I64)

    def from_bits(self, n):
        return self.b.bitcast(self.i(n), F64)

    def iand(self, n, m):
        return self.b.land(self.i(n), self.i(m))

    def ior(self, n, m):
        return self.b.lor(self.i(n), self.i(m))

    def ixor(self, n, m):
        return self.b.xor(self.i(n), self.i(m))

    def ishl(self, n, s: int):
        return self.b.shl(self.i(n), self.b.const_i64(s))

    def ilshr(self, n, s: int):
        return self.b.lshr(self.i(n), self.b.const_i64(s))

    def iadd(self, n, m):
        return self.b.add(self.i(n), self.i(m))

    def isub(self, n, m):
        return self.b.sub(self.i(n), self.i(m))

    def ieq(self, n, m):
        return self.b.cmp_eq(self.i(n), self.i(m))

    # hardware
    def trig_preop(self, x, seg: int):
        """``v_trig_preop_f64``: 53 bits of ``2/pi`` from bit ``max(0, e - 1077) + 53 seg``
        (``e`` the biased exponent of ``x``), scaled by ``2^128`` when ``e >= 1968``."""
        return self.b.inline_asm(
            "v_trig_preop_f64 $0, $1, $2",
            "=v,v,v",
            [self.f(x), self.b.const_i32(seg)],
            F64,
            sideeffect=False,
            result_name_hint="preop",
        )

    # control
    def when(self, cond: Bool, vals: Sequence, fn: Callable[[List], List]) -> List:
        """``fn(vals)`` for the lanes where ``cond`` holds, else ``vals`` (a loop of
        ``cond ? 1 : 0`` trips: lanes without the condition skip it, waves without any such
        lane branch over it)."""
        if isinstance(cond, bool):
            return list(fn(list(vals))) if cond else list(vals)
        b = self.b
        vals = [self.f(v) if isinstance(v, float) else self.i(v) for v in vals]
        trips = b.select(cond, b.const_i32(1), b.const_i32(0))
        tag = b._fresh("el")[1:]
        loop = b.scf_for_iter(
            b.const_i32(0),
            trips,
            b.const_i32(1),
            [(f"{tag}_{j}", v) for j, v in enumerate(vals)],
            iv_name=f"{tag}_i",
        )
        with loop as (_it, cur):
            b.scf_yield(*fn(list(cur)))
        return list(loop.results)


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def _from_bits(n: int) -> float:
    n = ((n + (1 << 63)) % (1 << 64)) - (1 << 63)
    return struct.unpack("<d", struct.pack("<q", n))[0]


def _wrap64(n: int) -> int:
    return ((n + (1 << 63)) % (1 << 64)) - (1 << 63)


def two_over_pi_bits(nbits: int) -> int:
    """The first ``nbits`` fraction bits of ``2/pi`` as an integer (bit ``j`` from the top has
    the weight ``2^-(j+1)``)."""
    import mpmath

    with mpmath.workprec(nbits + 64):
        return int(mpmath.floor(2 / mpmath.pi * mpmath.mpf(2) ** nbits))


_TWO_OVER_PI_BITS = None


def trig_preop_reference(x: float, seg: int) -> float:
    """Host statement of ``v_trig_preop_f64(x, seg)``: the 53-bit window of ``2/pi`` starting
    at fraction bit ``max(0, e - 1077) + 53 seg`` as the value it has in ``2/pi``, times
    ``2^128`` when the biased exponent ``e >= 1968``."""
    global _TWO_OVER_PI_BITS
    total = 1400
    if _TWO_OVER_PI_BITS is None:
        _TWO_OVER_PI_BITS = two_over_pi_bits(total)
    e = (_bits(x) >> 52) & 0x7FF
    start = max(0, e - 1077) + 53 * seg
    window = (_TWO_OVER_PI_BITS >> (total - start - 53)) & ((1 << 53) - 1)
    scale = 128 if e >= 1968 else 0
    return float(Fraction(window, 1 << (start + 53)) * (Fraction(2) ** scale))


class HostMath:
    """Evaluates the steps on the host: Python floats in IEEE f64 (exact fma), Python ints as
    i64 bit patterns, Python bools."""

    @staticmethod
    def f(v):
        return float(v)

    @staticmethod
    def i(v):
        return int(v)

    def add(self, a, c):
        return float(a) + float(c)

    def sub(self, a, c):
        return float(a) - float(c)

    def mul(self, a, c):
        return float(a) * float(c)

    def div(self, a, c):
        from ..core.f64_math import _div

        return _div(float(a), float(c))

    def fma(self, a, c, d):
        from ..core.f64_math import _fma_exact

        return _fma_exact(float(a), float(c), float(d))

    def sqrt(self, a):
        from ..core.f64_math import _sqrt

        return _sqrt(float(a))

    def neg(self, a):
        return -float(a)

    def abs(self, a):
        return abs(float(a))

    def lt(self, a, c):
        return float(a) < float(c)

    def le(self, a, c):
        return float(a) <= float(c)

    def gt(self, a, c):
        return float(a) > float(c)

    def ge(self, a, c):
        return float(a) >= float(c)

    def eq(self, a, c):
        return float(a) == float(c)

    def isnan(self, a):
        return math.isnan(a)

    def band(self, p, q):
        return bool(p) and bool(q)

    def bor(self, p, q):
        return bool(p) or bool(q)

    def bnot(self, p):
        return not p

    def sel(self, p, a, c):
        return a if p else c

    def bits(self, a):
        return _bits(float(a))

    def from_bits(self, n):
        return _from_bits(int(n))

    def iand(self, n, m):
        return _wrap64(int(n) & int(m))

    def ior(self, n, m):
        return _wrap64(int(n) | int(m))

    def ixor(self, n, m):
        return _wrap64(int(n) ^ int(m))

    def ishl(self, n, s):
        return _wrap64(int(n) << s)

    def ilshr(self, n, s):
        return _wrap64((int(n) % (1 << 64)) >> s)

    def iadd(self, n, m):
        return _wrap64(int(n) + int(m))

    def isub(self, n, m):
        return _wrap64(int(n) - int(m))

    def ieq(self, n, m):
        return int(n) == int(m)

    def trig_preop(self, x, seg):
        return trig_preop_reference(float(x), seg)

    def when(self, cond, vals, fn):
        return list(fn(list(vals))) if cond else list(vals)


# ------------------------------------------------------------------------------------------------
# Error-free transformations and small pieces
# ------------------------------------------------------------------------------------------------


def two_sum(m, a, c):
    s = m.add(a, c)
    bb = m.sub(s, a)
    return s, m.add(m.sub(a, m.sub(s, bb)), m.sub(c, bb))


def fast_two_sum(m, a, c):
    """``|a| >= |c|`` (or ``a`` zero)."""
    s = m.add(a, c)
    return s, m.sub(c, m.sub(s, a))


def two_prod(m, a, c):
    p = m.mul(a, c)
    return p, m.fma(a, c, m.neg(p))


def horner(m, coefs: Sequence[float], z):
    acc = m.f(coefs[-1])
    for ck in reversed(coefs[:-1]):
        acc = m.fma(acc, z, ck)
    return acc


def rint_small(m, v):
    """The integer nearest ``v`` (ties to even), ``|v| < 2^51``."""
    return m.sub(m.add(v, _M52), _M52)


def int_bits(m, k):
    """An integer-valued f64 ``|k| < 2^51`` as an i64."""
    return m.isub(m.bits(m.add(k, _M52)), _bits(_M52))


def int_to_f64(m, n):
    """An i64 ``|n| < 2^51`` as an f64."""
    return m.sub(m.from_bits(m.iadd(n, _bits(_M52))), _M52)


def pow2(m, n):
    """``2^n`` for an i64 ``n`` in ``[-1022, 1023]``."""
    return m.from_bits(m.ishl(m.iadd(n, 1023), 52))


def copysign(m, v, x):
    return m.from_bits(m.ior(m.iand(m.bits(v), _ABS), m.iand(m.bits(x), _SIGN)))


def floor_nonneg(m, v):
    """``floor(v)`` for ``v >= 0`` (``v`` itself from ``2^52``)."""
    t = m.sub(m.add(v, 2.0**52), 2.0**52)
    t = m.sub(t, m.sel(m.gt(t, v), 1.0, 0.0))
    return m.sel(m.lt(v, 2.0**52), t, v)


def series_dd(m, x, zh, zl, coefs: Sequence[float]):
    """``x + x z Q(z)`` as a dd for ``z = zh + zl`` and ``Q`` with coefficients ``coefs``: the
    product ``x z Q0`` is exact, the rest of ``Q`` is evaluated at ``zh``."""
    ch, cl = two_prod(m, x, zh)
    cl = m.fma(x, zl, cl)
    th, tl = two_prod(m, ch, coefs[0])
    rest = m.fma(cl, coefs[0], tl)
    if len(coefs) > 1:
        q1 = horner(m, coefs[1:], zh)
        rest = m.fma(m.mul(ch, zh), q1, rest)
    sh, e = two_sum(m, x, th)
    return fast_two_sum(m, sh, m.add(e, rest))


# ------------------------------------------------------------------------------------------------
# floor / ceil
# ------------------------------------------------------------------------------------------------


def floor_ceil(m, x, is_ceil: Bool):
    """``floor(x)`` (``is_ceil`` false) or ``ceil(x)``."""
    a = m.abs(x)
    t = m.sub(m.add(a, 2.0**52), 2.0**52)  # |x| rounded to an integer
    fl = m.sub(t, m.sel(m.gt(t, a), 1.0, 0.0))
    ce = m.add(t, m.sel(m.lt(t, a), 1.0, 0.0))
    # floor(x) = floor(|x|) for x >= 0 and -ceil(|x|) for x < 0; ceil the other way round
    negative = _is_negative(m, x)
    up = m.sel(is_ceil, m.bnot(negative), negative)
    r = copysign(m, m.sel(up, ce, fl), x)
    return m.sel(m.lt(a, 2.0**52), r, x)


def _is_negative(m, x):
    """The sign bit of ``x`` (true for -0 and negative NaNs)."""
    return m.bnot(m.ieq(m.iand(m.bits(x), _SIGN), 0))


# ------------------------------------------------------------------------------------------------
# sin / cos / tan
# ------------------------------------------------------------------------------------------------


def _reduce_medium(m, a):
    """``a = k pi/2 + rh + rl`` for ``0 <= a < 2^30``; returns ``(rh, rl, k mod 4)``."""
    k = rint_small(m, m.mul(a, TWO_OVER_PI))
    xt = m.fma(m.neg(k), PIO2_0, a)  # exact
    ph, pl = two_prod(m, k, PIO2_1)
    s, e = two_sum(m, xt, m.neg(ph))
    t = m.fma(m.neg(k), PIO2_2, m.sub(e, pl))
    rh, rl = two_sum(m, s, t)
    q = m.iand(int_bits(m, k), 3)
    return rh, rl, q


def _reduce_large(m, a):
    """Payne-Hanek reduction of a finite ``a >= 2^30``; returns ``(rh, rl, k mod 4)``."""
    xs = m.mul(a, m.sel(m.ge(a, 2.0**945), 2.0**-128, 1.0))
    parts = []
    for seg in range(4):
        p = m.trig_preop(a, seg)
        parts.append(two_prod(m, xs, p))
    (h0, l0), (h1, l1), (h2, l2), (h3, l3) = parts
    # h0 modulo 4 (exact): every term of 2/pi before the windows contributed a multiple of 4
    h0 = m.sub(h0, m.mul(4.0, floor_nonneg(m, m.mul(h0, 0.25))))
    s1, e1 = two_sum(m, h0, l0)
    s2, e2 = two_sum(m, s1, h1)
    k = rint_small(m, s2)
    f1 = m.sub(s2, k)  # exact
    u, v1 = two_sum(m, f1, e2)
    u, v2 = two_sum(m, u, e1)
    u, v3 = two_sum(m, u, l1)
    u, v4 = two_sum(m, u, h2)
    lo = m.add(m.add(m.add(l3, h3), l2), m.add(m.add(v4, v3), m.add(v2, v1)))
    fh, fl = two_sum(m, u, lo)
    # r = (fh + fl) * pi/2
    rh = m.mul(fh, PIO2_0)
    rl = m.fma(fh, PIO2_0, m.neg(rh))
    rl = m.add(rl, m.fma(fl, PIO2_0, m.mul(fh, PIO2_1)))
    rh, rl = fast_two_sum(m, rh, rl)
    q = m.iand(int_bits(m, k), 3)
    return rh, rl, q


def reduce_pio2(m, a):
    """``a = k pi/2 + rh + rl``, ``|rh| <= pi/4`` (plus rounding), for a finite ``a >= 0``;
    returns ``(rh, rl, k mod 4)``. Other inputs give unspecified values."""
    rh, rl, q = _reduce_medium(m, a)
    large = m.band(m.ge(a, TRIG_LARGE), m.lt(a, math.inf))
    rh, rl, q = m.when(large, [rh, rl, q], lambda _v: list(_reduce_large(m, a)))
    return rh, rl, q


def sin_cos_kernel(m, rh, rl):
    """``sin(r)`` and ``cos(r)`` as dds for ``r = rh + rl``, ``|rh| <= pi/4``."""
    zh, zl = two_prod(m, rh, rh)
    # sin(rh + rl) = sin(rh) + rl (1 - z/2) + O(rl^2)
    sh, sl = series_dd(m, rh, zh, zl, SIN_S)
    sl = m.add(sl, m.mul(rl, m.fma(-0.5, zh, 1.0)))
    sh, sl = fast_two_sum(m, sh, sl)
    # cos(rh + rl) = 1 - z/2 + z^2 C(z) - rh rl
    hz = m.mul(0.5, zh)
    w = m.sub(1.0, hz)
    tail = m.sub(m.sub(1.0, w), hz)  # exact
    c4 = m.mul(m.mul(zh, zh), horner(m, COS_C, zh))
    rest = m.add(m.sub(tail, m.fma(0.5, zl, m.mul(rh, rl))), c4)
    ch, cl = fast_two_sum(m, w, rest)
    return (sh, sl), (ch, cl)


def dd_div(m, nh, nl, dh, dl):
    """``(nh + nl) / (dh + dl)`` rounded."""
    q = m.div(nh, dh)
    r = m.fma(m.neg(q), dh, nh)
    r = m.sub(m.add(r, nl), m.mul(q, dl))
    return m.add(q, m.div(r, dh))


def trig(m, x, is_sin: Bool, is_cos: Bool, is_tan: Bool):
    """``sin(x)``, ``cos(x)`` or ``tan(x)`` (exactly one predicate holds)."""
    a = m.abs(x)
    rh, rl, q = reduce_pio2(m, a)
    (sh, sl), (ch, cl) = sin_cos_kernel(m, rh, rl)
    odd = m.ieq(m.iand(q, 1), 1)
    out = None
    if is_sin is not False:
        v = m.sel(odd, m.add(ch, cl), m.add(sh, sl))
        v = m.sel(m.ieq(m.iand(q, 2), 2), m.neg(v), v)
        out = v
    if is_cos is not False:
        v = m.sel(odd, m.add(sh, sl), m.add(ch, cl))
        v = m.sel(m.ieq(m.iand(m.iadd(q, 1), 2), 2), m.neg(v), v)
        out = v if out is None else m.sel(is_cos, v, out)
    if is_tan is not False:

        def tan_q(_v):
            # sin / cos (even quadrants) or -cos / sin (odd ones): one dd division
            q = dd_div(
                m,
                m.sel(odd, ch, sh),
                m.sel(odd, cl, sl),
                m.sel(odd, sh, ch),
                m.sel(odd, sl, cl),
            )
            return [m.sel(odd, m.neg(q), q)]

        # a branch of its own when the predicate is a run-time value (the division is the
        # most expensive step; sin and cos do not need it)
        t = m.when(is_tan, [0.0], tan_q)[0]
        out = t if out is None else m.sel(is_tan, t, out)
    # sin and tan are odd
    odd_fn = m.bor(is_sin, is_tan)
    out = m.sel(odd_fn, m.from_bits(m.ixor(m.bits(out), m.iand(m.bits(x), _SIGN))), out)
    return m.sel(m.lt(a, math.inf), out, m.sub(x, x))  # NaN for +-inf and NaN


# ------------------------------------------------------------------------------------------------
# sinh / cosh
# ------------------------------------------------------------------------------------------------


def exp_core(m, a):
    """``e^a = 2^n (mh + ml)`` for ``0 <= a <= 711``: returns ``(n as i64, mh, ml)``."""
    nf = rint_small(m, m.mul(a, LOG2E))
    r0 = m.fma(m.neg(nf), LN2_42, a)  # exact
    ph, pl = two_prod(m, nf, LN2_42_1)
    rh, e = two_sum(m, r0, m.neg(ph))
    rl = m.fma(m.neg(nf), LN2_42_2, m.sub(e, pl))
    qh, ql = two_prod(m, rh, rh)
    c3 = m.mul(m.mul(rh, qh), horner(m, EXP_P, rh))
    a1, b1 = two_sum(m, 1.0, rh)
    a2, b2 = two_sum(m, a1, m.mul(0.5, qh))
    ml = m.add(m.add(b1, b2), m.add(m.fma(0.5, ql, c3), m.fma(rl, rh, rl)))
    mh, ml = fast_two_sum(m, a2, ml)
    return int_bits(m, nf), mh, ml


def hyp(m, x, is_sinh: Bool, is_cosh: Bool):
    """``sinh(x)`` or ``cosh(x)`` (exactly one predicate holds)."""
    a = m.abs(x)
    aa = m.sel(m.lt(a, 711.0), a, 711.0)
    n, mh, ml = exp_core(m, aa)
    # H = e^a / 2 = 2^(n-1) m, G = e^-a / 2 = 2^(-n-1) / m, each scale in two factors
    i = m.isub(n, 1)
    i1 = m.ilshr(m.iadd(i, 1024), 1)  # (i + 1024) // 2, i >= -1
    i1 = m.isub(i1, 512)
    s1, s2 = pow2(m, i1), pow2(m, m.isub(i, i1))
    hh = m.mul(m.mul(mh, s1), s2)
    hl = m.mul(m.mul(ml, s1), s2)
    gh = m.div(1.0, mh)
    gl = m.mul(gh, m.sub(m.fma(m.neg(gh), mh, 1.0), m.mul(gh, ml)))
    j = m.isub(m.isub(0, n), 1)
    j1 = m.isub(m.ilshr(m.iadd(j, 2048), 1), 1024)
    t1, t2 = pow2(m, j1), pow2(m, m.isub(j, j1))
    gh = m.mul(m.mul(gh, t1), t2)
    gl = m.mul(m.mul(gl, t1), t2)
    out = None
    if is_sinh is not False:
        u, v = two_sum(m, hh, m.neg(gh))
        big = m.add(u, m.add(v, m.sub(hl, gl)))

        def series(_v):
            zh, zl = two_prod(m, a, a)
            sh, sl = series_dd(m, a, zh, zl, SINH_H)
            return [m.add(sh, sl)]

        lt1 = m.lt(a, 1.0)
        small = m.when(m.band(is_sinh, lt1), [0.0], series)[0]
        v = m.sel(lt1, small, big)
        v = m.sel(m.eq(hh, math.inf), math.inf, v)
        out = copysign(m, v, x)
    if is_cosh is not False:
        u, v = two_sum(m, hh, gh)
        c = m.add(u, m.add(v, m.add(hl, gl)))
        c = m.sel(m.eq(hh, math.inf), math.inf, c)
        out = c if out is None else m.sel(is_cosh, c, out)
    return m.sel(m.isnan(x), x, out)


# ------------------------------------------------------------------------------------------------
# asin / acos / atan
# ------------------------------------------------------------------------------------------------


def asin_acos(m, x, is_asin: Bool, is_acos: Bool):
    """``asin(x)`` or ``acos(x)`` (exactly one predicate holds)."""
    a = m.abs(x)
    negative = _is_negative(m, x)
    small = m.le(a, 0.5)

    def near_zero(_v):
        # |x| <= 1/2: asin(a) = a + a z R(z), z = a^2
        zh, zl = two_prod(m, a, a)
        return list(series_dd(m, a, zh, zl, ASIN_R))

    def near_one(_v):
        # |x| > 1/2: asin(a) = pi/2 - 2 asin(s), s = sqrt(z), z = (1 - a)/2 (exact)
        z = m.mul(m.sub(1.0, a), 0.5)
        s = m.sqrt(z)
        s_l = m.sel(m.gt(s, 0.0), m.div(m.fma(m.neg(s), s, z), m.add(s, s)), 0.0)
        wh, wl = series_dd(m, s, z, 0.0, ASIN_R)
        wl = m.add(wl, m.mul(s_l, m.fma(z, ASIN_R[0], 1.0)))
        wh, wl = fast_two_sum(m, wh, wl)
        return [m.mul(2.0, wh), m.mul(2.0, wl)]

    # each path for the lanes that take it (waves whose lanes all take one skip the other)
    ph, pl = m.when(small, [0.0, 0.0], near_zero)
    wh2, wl2 = m.when(m.gt(a, 0.5), [0.0, 0.0], near_one)
    out = None
    if is_asin is not False:
        u, v = two_sum(m, PIO2_0, m.neg(wh2))
        big = m.add(u, m.add(v, m.sub(PIO2_1, wl2)))
        v = m.sel(small, m.add(ph, pl), big)
        out = copysign(m, v, x)
    if is_acos is not False:
        # |x| <= 1/2: pi/2 - asin(x)
        sph = m.sel(negative, m.neg(ph), ph)
        spl = m.sel(negative, m.neg(pl), pl)
        u, v = two_sum(m, PIO2_0, m.neg(sph))
        c_small = m.add(u, m.add(v, m.sub(PIO2_1, spl)))
        # x > 1/2: 2 asin(s); x < -1/2: pi - 2 asin(s)
        u, v = two_sum(m, PI_H, m.neg(wh2))
        c_neg = m.add(u, m.add(v, m.sub(PI_L, wl2)))
        c_pos = m.add(wh2, wl2)
        c = m.sel(small, c_small, m.sel(negative, c_neg, c_pos))
        out = c if out is None else m.sel(is_acos, c, out)
    return m.sel(m.le(a, 1.0), out, m.sub(m.add(x, math.inf), math.inf))


def atan(m, x):
    """``atan(x)``."""
    a = m.abs(x)
    b1 = m.gt(a, 7.0 / 16)
    b2 = m.gt(a, 11.0 / 16)
    b3 = m.gt(a, 19.0 / 16)
    b4 = m.gt(a, 39.0 / 16)

    def pick(v0, v1, v2, v3, v4):
        return m.sel(b4, v4, m.sel(b3, v3, m.sel(b2, v2, m.sel(b1, v1, v0))))

    # t = num / den around c: num = nA a + nB (exact), den = dA a + dB (as a dd)
    nA, nB = pick(1.0, 2.0, 1.0, 1.0, 0.0), pick(0.0, -1.0, -1.0, -1.5, -1.0)
    dA, dB = pick(0.0, 1.0, 1.0, 1.5, 1.0), pick(1.0, 2.0, 1.0, 1.0, 0.0)
    num = m.fma(nA, a, nB)
    p, pe = two_prod(m, dA, a)
    dh, de = two_sum(m, p, dB)
    dl = m.add(de, pe)
    th = m.div(num, dh)
    tl = m.div(m.sub(m.fma(m.neg(th), dh, num), m.mul(th, dl)), dh)
    zh, zl = two_prod(m, th, th)
    sh, sl = series_dd(m, th, zh, zl, ATAN_A)
    # atan(th + tl) = atan(th) + tl / (1 + th^2)
    sl = m.add(sl, m.mul(tl, m.sub(1.0, zh)))
    ch = pick(0.0, ATAN_HALF[0], ATAN_ONE[0], ATAN_3HALF[0], ATAN_INF[0])
    cl = pick(0.0, ATAN_HALF[1], ATAN_ONE[1], ATAN_3HALF[1], ATAN_INF[1])
    u, v = two_sum(m, ch, sh)
    r = m.add(u, m.add(v, m.add(sl, cl)))
    r = m.sel(m.eq(a, math.inf), PIO2_0, r)
    r = copysign(m, r, x)
    return m.sel(m.isnan(x), x, r)


# ------------------------------------------------------------------------------------------------
# asinh / acosh / atanh
# ------------------------------------------------------------------------------------------------


def log_dd(m, yh, yl):
    """``log(yh + yl)`` as a dd for a normal ``yh > 0``, ``|yl| <= ulp(yh)``."""
    bits = m.bits(yh)
    e = m.isub(m.ilshr(bits, 52), 1023)
    mm = m.from_bits(m.ior(m.iand(bits, _MANT_MASK), _ONE_BITS))
    up = m.gt(mm, SQRT2)
    mm = m.sel(up, m.mul(mm, 0.5), mm)
    e = m.iadd(e, m.sel(up, 1, 0))
    ef = int_to_f64(m, e)
    f = m.sub(mm, 1.0)  # exact
    fl = m.mul(yl, m.div(mm, yh))  # yl scaled like yh (the quotient is a power of two)
    s = m.div(f, m.add(2.0, f))
    z = m.mul(s, s)
    rz = m.mul(z, horner(m, LOG_G, z))
    qh, ql = two_prod(m, f, f)
    hf, hfl = m.mul(0.5, qh), m.mul(0.5, ql)
    corr = m.mul(s, m.add(hf, rz))
    l1, l1l = two_sum(m, f, m.neg(hf))
    tail = m.add(m.sub(l1l, hfl), m.add(corr, m.div(fl, mm)))
    th, tl = two_sum(m, m.mul(ef, LN2_42), l1)
    tl = m.add(tl, m.fma(ef, LN2_42_1, tail))
    return fast_two_sum(m, th, tl)


def _sqrt_dd(m, th, tl):
    sh = m.sqrt(th)
    sl = m.sel(
        m.gt(sh, 0.0), m.div(m.add(m.fma(m.neg(sh), sh, th), tl), m.add(sh, sh)), 0.0
    )
    return sh, sl


def ainvhyp(m, x, is_asinh: Bool, is_acosh: Bool, is_atanh: Bool):
    """``asinh(x)``, ``acosh(x)`` or ``atanh(x)`` (exactly one predicate holds)."""
    a = m.abs(x)
    # The argument y >= 1 of the log, as a dd; asinh and acosh past 2^28 take log(a) + ln 2.
    big = m.gt(a, 2.0**28)
    v = m.sel(is_acosh, x, a)

    def root_arg(_v):
        # asinh: |x| + sqrt(x^2 + 1), acosh: x + sqrt(x^2 - 1)
        ph, pl = two_prod(m, v, v)
        th, tl = two_sum(m, ph, m.sel(is_acosh, -1.0, 1.0))
        sh, sl = _sqrt_dd(m, th, m.add(tl, pl))
        yh, ye = two_sum(m, v, sh)
        return [yh, m.add(ye, sl)]

    def ratio_arg(_v):
        # atanh: 1 + 2|x| / (1 - |x|)
        dh, dl = two_sum(m, 1.0, m.neg(a))
        two_a = m.add(a, a)
        uh = m.div(two_a, dh)
        ul = m.div(m.sub(m.fma(m.neg(uh), dh, two_a), m.mul(uh, dl)), dh)
        zh, ze = two_sum(m, 1.0, uh)
        return [zh, m.add(ze, ul)]

    # one of the two arguments (a branch each when the predicate is a run-time value)
    yh, yl = m.when(m.bnot(is_atanh), [0.0, 0.0], root_arg)
    if is_atanh is not False:
        zh, zl = m.when(is_atanh, [0.0, 0.0], ratio_arg)
        yh = m.sel(is_atanh, zh, yh)
        yl = m.sel(is_atanh, zl, yl)
    use_big = m.band(big, m.bnot(is_atanh))
    yh = m.sel(use_big, v, yh)
    yl = m.sel(use_big, 0.0, yl)
    lh, ll = log_dd(m, yh, yl)
    u, w = two_sum(m, lh, LN2_42)
    bh, bl = u, m.add(w, m.add(ll, LN2_42_1))
    rh = m.sel(use_big, bh, lh)
    rl = m.sel(use_big, bl, ll)
    r = m.add(rh, rl)
    if is_atanh is not False:
        r = m.sel(is_atanh, m.mul(0.5, r), r)
    out = r
    if is_asinh is not False:
        s = m.sel(m.lt(a, 2.0**-28), a, r)
        s = m.sel(m.eq(a, math.inf), math.inf, s)
        s = copysign(m, s, x)
        out = s if is_asinh is True else m.sel(is_asinh, s, out)
    if is_acosh is not False:
        c = m.sel(m.eq(x, 1.0), 0.0, r)
        c = m.sel(m.eq(x, math.inf), math.inf, c)
        c = m.sel(m.ge(x, 1.0), c, math.nan)
        out = c if is_acosh is True else m.sel(is_acosh, c, out)
    if is_atanh is not False:
        t = m.sel(m.lt(a, 2.0**-28), a, r)
        t = m.sel(m.eq(a, 1.0), math.inf, t)
        t = m.sel(m.le(a, 1.0), t, math.nan)
        t = copysign(m, t, x)
        out = t if is_atanh is True else m.sel(is_atanh, t, out)
    return m.sel(m.isnan(x), x, out)


# ------------------------------------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------------------------------------

#: The functions grouped by the program that computes them (one evaluation per group).
FAMILIES = {
    "trig": ("sin", "cos", "tan"),
    "hyp": ("sinh", "cosh"),
    "asin_acos": ("asin", "acos"),
    "atan": ("atan",),
    "ainvhyp": ("asinh", "acosh", "atanh"),
    "round": ("floor", "ceil"),
}


def family(m, fam: str, x, preds: dict):
    """The function of family ``fam`` whose predicate in ``preds`` (name -> Python bool or
    i1 value; missing names are false) holds, applied to the f64 ``x``."""
    p = [preds.get(n, False) for n in FAMILIES[fam]]
    if fam == "trig":
        return trig(m, x, *p)
    if fam == "hyp":
        return hyp(m, x, *p)
    if fam == "asin_acos":
        return asin_acos(m, x, *p)
    if fam == "atan":
        return atan(m, x)
    if fam == "ainvhyp":
        return ainvhyp(m, x, *p)
    if fam == "round":
        return floor_ceil(m, x, p[1])
    raise ValueError(fam)


class _RecordingMath(IRMath):
    """IRMath on a scratch builder that records the f64 constants a program uses."""

    def __init__(self, b: IRBuilder):
        super().__init__(b)
        self.seen: List[float] = []
        self.uses: dict = {}

    def f(self, v) -> Value:
        if isinstance(v, (float, int)):
            key = _bits(float(v))
            if key not in self.uses:
                self.seen.append(float(v))
            self.uses[key] = self.uses.get(key, 0) + 1
        return super().f(v)


_CONSTANTS: dict = {}


def family_constants(fam: str, preds: dict) -> List[float]:
    """The f64 constants :func:`family` ``fam`` uses more than once, for predicates of the
    kinds in ``preds`` (Python bools as given, any other value an IR predicate), in first-use
    order (a build of the program on a scratch builder): the ones worth
    :meth:`IRMath.materialize`."""
    kinds = tuple(
        (n, preds[n] if isinstance(preds.get(n), bool) else "v")
        for n in FAMILIES[fam]
        if n in preds
    )
    key = (fam, kinds)
    if key not in _CONSTANTS:
        from ..core.ir import I1

        sb = IRBuilder("elementary_constants")
        x = sb.param("x", F64)
        p2 = {
            n: (k if isinstance(k, bool) else sb.param(f"p_{n}", I1)) for n, k in kinds
        }
        rec = _RecordingMath(sb)
        family(rec, fam, x, p2)
        # constants used once stay at their use: a pooled constant lives from the program's
        # start, and pooling the single-use ones (the polynomial coefficients) costs registers
        _CONSTANTS[key] = tuple(c for c in rec.seen if rec.uses[_bits(c)] >= 2)
    return list(_CONSTANTS[key])


def family_of(name: str) -> str:
    for fam, names in FAMILIES.items():
        if name in names:
            return fam
    raise ValueError(f"unknown elementary function {name!r}")


def apply(m, name: str, x):
    """``name(x)`` for an f64 ``x`` in the math context ``m``."""
    return family(m, family_of(name), x, {name: True})


def opaque_zero(b: IRBuilder) -> Value:
    """An i64 zero the compiler cannot see through (one scalar move with side effects), the
    ``anchor`` of :class:`IRMath`: the f64 constants of a program built on it are materialized
    where they are used. Without it the backend hoists every constant of a program that runs
    inside a loop (a row loop, a GEMM main loop) out of the loop into VGPRs, and a kernel with
    these programs needs several times the registers of the program itself."""
    return b.inline_asm(
        "s_mov_b64 $0, 0", "=s", [], I64, sideeffect=True, result_name_hint="ozero"
    )


def emit(b: IRBuilder, name: str, x: Value, anchored: bool = True) -> Value:
    """``name(x)`` as IR ops for an f32 or f64 value ``x`` (f32: evaluated in f64, rounded once
    to f32); ``anchored``: constants on an :func:`opaque_zero`."""
    m = IRMath(b, opaque_zero(b) if anchored else None)
    if anchored:
        m.materialize(family_constants(family_of(name), {name: True}))
    if x.type == F64:
        return apply(m, name, x)
    if x.type.name != "f32":
        raise ValueError(f"elementary.{name} takes f32 or f64, got {x.type.name}")
    return b.cvt_f64_to_f32(apply(m, name, b.cvt_f32_to_f64(x)))


def host(name: str, x: float) -> float:
    """``name(x)`` on the host, step for step what :func:`emit` computes for an f64 ``x``."""
    return apply(HostMath(), name, float(x))
